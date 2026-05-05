"""
FastAPI backend for Healthcare Misinformation Risk Classifier.
Supports single/batch inference, web scraping, and health checks.
"""

import sys
import time
from pathlib import Path
from contextlib import asynccontextmanager
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv

load_dotenv()

from api.schemas import (
    PredictRequest, PredictResponse, WordImportance,
    BatchPredictRequest, BatchPredictResponse,
    ScrapeRequest, ScrapeResponse, ScrapeResult,
    HealthCheckResponse,
)
from api.middleware import setup_cors, setup_logging, LatencyLoggingMiddleware
from training.config import LABEL_MAP, ONNX_MODEL_PATH, DISTILBERT_DIR

# Global predictor references
_onnx_predictor = None
_pt_predictor = None
_explainer = None
_inference_engine = "none"


def _load_models():
    """Load inference models on startup."""
    global _onnx_predictor, _pt_predictor, _explainer, _inference_engine

    # Try ONNX first (faster)
    if ONNX_MODEL_PATH.exists():
        try:
            from inference.predict_onnx import ONNXPredictor
            _onnx_predictor = ONNXPredictor()
            _inference_engine = "onnx_quantized"
            print("[OK] ONNX predictor loaded")
        except Exception as e:
            print(f"[!]  ONNX load failed: {e}")

    # Fallback to PyTorch
    best_model = DISTILBERT_DIR / "best"
    if best_model.exists():
        try:
            from inference.predict import PyTorchPredictor
            _pt_predictor = PyTorchPredictor(str(best_model))
            if _inference_engine == "none":
                _inference_engine = "pytorch"
            print("[OK] PyTorch predictor loaded")
        except Exception as e:
            print(f"[!]  PyTorch load failed: {e}")

    # Load explainer
    if best_model.exists():
        try:
            from explainability.attention import AttentionExplainer
            _explainer = AttentionExplainer(model_path=str(best_model))
            print("[OK] Attention explainer loaded")
        except Exception as e:
            print(f"[!]  Explainer load failed: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_models()
    yield


app = FastAPI(
    title="Healthcare Misinformation Risk Classifier",
    description="Detect and explain healthcare misinformation using DistilBERT",
    version="1.0.0",
    lifespan=lifespan,
)

logger = setup_logging()
setup_cors(app)
app.add_middleware(LatencyLoggingMiddleware)


def _get_predictor():
    if _onnx_predictor:
        return _onnx_predictor, "onnx"
    if _pt_predictor:
        return _pt_predictor, "pytorch"
    raise HTTPException(503, "No model loaded. Train the model first.")


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    return HealthCheckResponse(
        status="healthy",
        model_loaded=_onnx_predictor is not None or _pt_predictor is not None,
        inference_engine=_inference_engine,
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    start = time.perf_counter()
    predictor, engine = _get_predictor()

    # Get base prediction
    if engine == "pytorch" and request.explain and _pt_predictor:
        result = _pt_predictor.predict(request.text, explain=True)
    elif engine == "onnx" and request.explain and _explainer:
        onnx_result = _onnx_predictor.predict(request.text)
        attn_result = _explainer.get_attention_scores(request.text, top_k=8)
        result = {**onnx_result, "important_words": attn_result.get("important_words", [])}
    else:
        result = predictor.predict(request.text)

    # LLM explanation
    explanation = None
    if request.use_llm:
        try:
            from llm_layer.explainer import get_explanation_sync
            explanation = get_explanation_sync(
                request.text, result["label"], result["confidence"],
                result.get("important_words"),
            )
        except Exception as e:
            explanation = f"Explanation unavailable: {e}"

    latency = (time.perf_counter() - start) * 1000
    important = [WordImportance(word=w["word"], score=w["score"]) for w in result.get("important_words", [])]

    return PredictResponse(
        text=request.text,
        label=result["label"],
        label_id=result.get("label_id", 0),
        confidence=result["confidence"],
        confidence_level=result.get("confidence_level", "medium"),
        all_probabilities=result.get("all_probabilities", {}),
        important_words=important,
        explanation=explanation,
        latency_ms=round(latency, 2),
    )


@app.post("/predict/batch", response_model=BatchPredictResponse)
async def predict_batch(request: BatchPredictRequest):
    start = time.perf_counter()
    predictor, engine = _get_predictor()

    if engine == "onnx":
        results = _onnx_predictor.predict_batch(request.texts)
    else:
        results = _pt_predictor.predict_batch(request.texts, explain=False)

    total_latency = (time.perf_counter() - start) * 1000
    predictions = []
    for r in results:
        predictions.append(PredictResponse(
            text=r["text"], label=r["label"], label_id=r.get("label_id", 0),
            confidence=r["confidence"], confidence_level=r.get("confidence_level", "medium"),
            all_probabilities=r.get("all_probabilities", {}),
            important_words=[], explanation=None,
            latency_ms=round(total_latency / len(request.texts), 2),
        ))

    return BatchPredictResponse(
        predictions=predictions, total_latency_ms=round(total_latency, 2), count=len(predictions),
    )


@app.post("/scrape", response_model=ScrapeResponse)
async def scrape_and_classify(request: ScrapeRequest):
    predictor, _ = _get_predictor()
    scraped_data = []

    if request.platform == "reddit":
        from scraper.reddit_scraper import scrape_reddit
        df = scrape_reddit(
            subreddits=[request.query] if request.query else None,
            max_posts=request.max_items,
        )
        if len(df) > 0:
            scraped_data = df.to_dict("records")
    elif request.platform == "youtube":
        from scraper.youtube_scraper import scrape_youtube
        df = scrape_youtube(
            queries=[request.query] if request.query else None,
            max_results=request.max_items,
        )
        if len(df) > 0:
            scraped_data = df.to_dict("records")
    else:
        raise HTTPException(400, "Platform must be 'reddit' or 'youtube'")

    results = []
    label_counts = Counter()
    for item in scraped_data[:request.max_items]:
        pred = predictor.predict(item.get("text", ""))
        label_counts[pred["label"]] += 1
        results.append(ScrapeResult(
            text=item.get("text", "")[:500],
            source=item.get("source", "unknown"),
            platform=request.platform,
            url=item.get("url"),
            label=pred["label"],
            confidence=pred["confidence"],
        ))

    return ScrapeResponse(
        results=results, total_items=len(results), risk_summary=dict(label_counts),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)

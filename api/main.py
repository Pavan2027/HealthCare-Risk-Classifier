"""
FastAPI backend for Healthcare Misinformation Risk Classifier.
Powered by DeBERTa-v3 and LLM Dual-Check Reasoning.
"""

import sys
import time
import json
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException, Query
from dotenv import load_dotenv

load_dotenv()

from api.schemas import (
    PredictRequest, PredictResponse, WordImportance,
    BatchPredictRequest, BatchPredictResponse,
    ScrapeRequest, ScrapeResponse, ScrapeResult,
    HealthCheckResponse,
)
from api.middleware import setup_cors, setup_logging, LatencyLoggingMiddleware
from training.config import LABEL_MAP, ONNX_MODEL_PATH, MODELS_DIR

# Global predictor references
_onnx_predictor = None
_pt_predictor = None
_inference_engine = "none"

def _load_models():
    """Load inference models on startup."""
    global _onnx_predictor, _pt_predictor, _inference_engine

    # 1. Try ONNX (Fastest — no attention output)
    if ONNX_MODEL_PATH.exists():
        try:
            from inference.predict_onnx import ONNXPredictor
            _onnx_predictor = ONNXPredictor()
            _inference_engine = "onnx_quantized"
            print("[OK] ONNX DeBERTa predictor loaded")
        except Exception as e:
            print(f"[!]  ONNX load failed: {e}")

    # 2. Fallback to PyTorch (supports attention highlights)
    best_model = MODELS_DIR / "deberta" / "best"
    if best_model.exists() and _inference_engine == "none":
        try:
            from inference.predict import PyTorchPredictor
            _pt_predictor = PyTorchPredictor(str(best_model))
            _inference_engine = "pytorch"
            print("[OK] PyTorch DeBERTa predictor loaded")
        except Exception as e:
            print(f"[!]  PyTorch load failed: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    _load_models()
    yield

app = FastAPI(
    title="HealthGuard AI API",
    description="DeBERTa-v3 Healthcare Risk Classifier with LLM Verification and Attention Highlights",
    version="1.0.0",
    lifespan=lifespan,
)

setup_cors(app)
app.add_middleware(LatencyLoggingMiddleware)

def _get_predictor():
    if _onnx_predictor:
        return _onnx_predictor
    if _pt_predictor:
        return _pt_predictor
    raise HTTPException(503, "No model loaded. Check your model paths.")

def _supports_attention() -> bool:
    """Only the PyTorch predictor can output attention-based word scores."""
    return _pt_predictor is not None and _onnx_predictor is None

def _extract_important_words(result: dict) -> list[WordImportance]:
    """
    Convert raw attention word list from PyTorchPredictor into WordImportance
    schema objects, safely handling both dict and string word entries.
    """
    raw = result.get("important_words", [])
    words = []
    for w in raw[:8]:
        if isinstance(w, dict):
            words.append(WordImportance(word=w.get("word", ""), score=round(w.get("score", 0.0), 4)))
        elif isinstance(w, str):
            words.append(WordImportance(word=w, score=0.0))
    return words

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
    predictor = _get_predictor()

    # 1. Base Prediction (DeBERTa)
    #    Use explain=True only for PyTorch (ONNX has no attention output)
    use_explain = request.explain and _supports_attention()
    result = predictor.predict(request.text, explain=use_explain) if hasattr(predictor, 'predict') else predictor.predict(request.text)

    # 2. Extract attention-based important words (PyTorch only)
    important_words = _extract_important_words(result)

    # 3. LLM Reasoning & Verification (Dual-Check)
    explanation = None
    llm_label = result["label"]  # Default to same as model

    if request.use_llm:
        try:
            from llm_layer.explainer import get_llm_explanation
            explanation, llm_label = await get_llm_explanation(
                request.text, result["label"], result["confidence"],
                important_words=[w.word for w in important_words]
            )
        except Exception as e:
            explanation = f"Explanation service error: {e}"

    latency = (time.perf_counter() - start) * 1000

    # Disagreement flag: True when model and LLM reach different conclusions
    disagreement = result["label"] != llm_label if request.use_llm else None

    return PredictResponse(
        text=request.text,
        label=result["label"],
        label_id=result.get("label_id", 0),
        confidence=result["confidence"],
        confidence_level=result.get("confidence_level", "medium"),
        all_probabilities=result.get("all_probabilities", {}),
        important_words=important_words,
        explanation=explanation,
        llm_label=llm_label,
        disagreement=disagreement,
        latency_ms=round(latency, 2),
    )

@app.post("/predict/batch", response_model=BatchPredictResponse)
async def predict_batch(request: BatchPredictRequest):
    """
    Batch classify up to 32 texts. Used by the Chrome extension for
    efficient page scanning (no attention, no LLM for speed).
    """
    start = time.perf_counter()
    predictor = _get_predictor()

    texts = [t[:2000] for t in request.texts[:32]]

    if hasattr(predictor, "predict_batch"):
        raw_results = predictor.predict_batch(texts, explain=False)
    else:
        raw_results = [predictor.predict(t) for t in texts]

    predictions = []
    for i, res in enumerate(raw_results):
        predictions.append(PredictResponse(
            text=texts[i],
            label=res["label"],
            label_id=res.get("label_id", 0),
            confidence=res["confidence"],
            confidence_level=res.get("confidence_level", "medium"),
            all_probabilities=res.get("all_probabilities", {}),
            important_words=[],
            latency_ms=res.get("latency_ms", 0),
        ))

    total_latency = (time.perf_counter() - start) * 1000
    return BatchPredictResponse(
        predictions=predictions,
        total_latency_ms=round(total_latency, 2),
        count=len(predictions),
    )

@app.post("/scrape", response_model=ScrapeResponse)
async def scrape_and_analyze(request: ScrapeRequest):
    """
    Scrape a URL and classify its content using the double-check system.
    """
    from scraper.universal_scraper import extract_from_url

    # 1. Scrape Content
    scraped = extract_from_url(request.url)
    if "error" in scraped:
        raise HTTPException(400, scraped["error"])

    predictor = _get_predictor()

    # 2. Classify (DeBERTa)
    text_to_analyze = scraped["content"]
    pred = predictor.predict(text_to_analyze[:2000])
    important_words = _extract_important_words(pred)

    # 3. Explain (LLM)
    from llm_layer.explainer import get_llm_explanation
    explanation, llm_label = await get_llm_explanation(
        text_to_analyze[:1000], pred["label"], pred["confidence"],
        important_words=[w.word for w in important_words]
    )

    result = ScrapeResult(
        text=text_to_analyze[:500],
        source=scraped["platform"],
        platform=scraped["platform"],
        url=scraped["url"],
        label=pred["label"],
        confidence=pred["confidence"],
        explanation=explanation,
        llm_label=llm_label,
    )

    return ScrapeResponse(
        results=[result],
        total_items=1,
        risk_summary={pred["label"]: 1},
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)

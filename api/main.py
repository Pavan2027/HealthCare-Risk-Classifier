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

    # 1. Try ONNX (Fastest)
    if ONNX_MODEL_PATH.exists():
        try:
            from inference.predict_onnx import ONNXPredictor
            _onnx_predictor = ONNXPredictor()
            _inference_engine = "onnx_quantized"
            print("[OK] ONNX DeBERTa predictor loaded")
        except Exception as e:
            print(f"[!]  ONNX load failed: {e}")

    # 2. Fallback to PyTorch
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
    description="DeBERTa-v3 Healthcare Risk Classifier with LLM Verification",
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
    result = predictor.predict(request.text)
    
    # 2. LLM Reasoning & Verification (Dual-Check)
    explanation = None
    llm_label = result["label"] # Default to same as model
    
    if request.use_llm:
        try:
            from llm_layer.explainer import get_llm_explanation
            explanation, llm_label = await get_llm_explanation(
                request.text, result["label"], result["confidence"]
            )
        except Exception as e:
            explanation = f"Explanation service error: {e}"

    latency = (time.perf_counter() - start) * 1000
    
    # Check for disagreement (Model vs LLM)
    disagreement = result["label"] != llm_label
    
    return PredictResponse(
        text=request.text,
        label=result["label"],
        label_id=result.get("label_id", 0),
        confidence=result["confidence"],
        confidence_level=result.get("confidence_level", "medium"),
        all_probabilities=result.get("all_probabilities", {}),
        important_words=[], # DeBERTa attention extraction is upcoming
        explanation=explanation,
        llm_label=llm_label,
        latency_ms=round(latency, 2),
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
    # For long content, we take the first 1000 characters for now
    pred = predictor.predict(text_to_analyze[:2000])
    
    # 3. Explain (LLM)
    from llm_layer.explainer import get_llm_explanation
    explanation, llm_label = await get_llm_explanation(
        text_to_analyze[:1000], pred["label"], pred["confidence"]
    )
    
    result = ScrapeResult(
        text=text_to_analyze[:500],
        source=scraped["platform"],
        platform=scraped["platform"],
        url=scraped["url"],
        label=pred["label"],
        confidence=pred["confidence"],
        explanation=explanation,
        llm_label=llm_label
    )
    
    return ScrapeResponse(
        results=[result],
        total_items=1,
        risk_summary={pred["label"]: 1}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)

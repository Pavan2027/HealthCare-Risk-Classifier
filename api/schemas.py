"""
Pydantic models for FastAPI request/response validation.
"""

from pydantic import BaseModel, Field
from typing import Optional


class PredictRequest(BaseModel):
    """Single text prediction request."""
    text: str = Field(..., min_length=5, max_length=5000, description="Health-related text to classify")
    explain: bool = Field(default=True, description="Include attention-based explanations")
    use_llm: bool = Field(default=False, description="Include LLM-generated explanation")


class WordImportance(BaseModel):
    """A word with its attention importance score."""
    word: str
    score: float


class PredictResponse(BaseModel):
    """Single text prediction response."""
    text: str
    label: str
    label_id: int
    confidence: float
    confidence_level: str
    all_probabilities: dict[str, float]
    important_words: list[WordImportance] = []
    explanation: Optional[str] = None
    latency_ms: float


class BatchPredictRequest(BaseModel):
    """Batch prediction request."""
    texts: list[str] = Field(..., min_length=1, max_length=32, description="List of texts to classify")
    explain: bool = Field(default=False, description="Include explanations (slower for batch)")


class BatchPredictResponse(BaseModel):
    """Batch prediction response."""
    predictions: list[PredictResponse]
    total_latency_ms: float
    count: int


class ScrapeRequest(BaseModel):
    """Web scraping + classification request."""
    platform: str = Field(..., description="Platform to scrape: 'reddit' or 'youtube'")
    query: Optional[str] = Field(None, description="Search query or subreddit name")
    max_items: int = Field(default=20, ge=1, le=100, description="Maximum items to scrape")


class ScrapeResult(BaseModel):
    """Individual scraped item with classification."""
    text: str
    source: str
    platform: str
    url: Optional[str] = None
    label: str
    confidence: float
    explanation: Optional[str] = None


class ScrapeResponse(BaseModel):
    """Web scraping + classification response."""
    results: list[ScrapeResult]
    total_items: int
    risk_summary: dict[str, int]  # Count of each label


class HealthCheckResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    inference_engine: str
    version: str = "1.0.0"

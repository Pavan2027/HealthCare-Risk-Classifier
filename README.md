# HealthCare Risk Classifier (HRC)

> A state-of-the-art medical misinformation detection system powered by DeBERTa-v3. Classifies health claims into four risk levels with AI-generated reasoning, social media monitoring, and a real-time browser extension.

![DeBERTa](https://img.shields.io/badge/Model-DeBERTa--v3--base-blue?style=flat-square&logo=huggingface)
![FastAPI](https://img.shields.io/badge/FastAPI-0.134-009688?style=flat-square&logo=fastapi)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python)
![ONNX](https://img.shields.io/badge/Inference-ONNX-gray?style=flat-square&logo=onnx)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C?style=flat-square&logo=pytorch)
![Groq](https://img.shields.io/badge/LLM-Groq%20%2F%20Ollama-orange?style=flat-square)

---

## Overview

The **HealthCare Risk Classifier (HRC)** is a full-stack NLP framework built to combat medical misinformation. Unlike standard "True/False" classifiers, HRC focuses on the **risk profile** of a claim, distinguishing between verified medical facts, misleading nuances, and active medical harm.

The system uses a custom **Multi-Sample DeBERTa** architecture with **Layer-wise Learning Rate Decay (LLRD)** for high-precision classification under strict hardware constraints (6GB VRAM), paired with a **dual-check LLM reasoning layer** that independently verifies and explains each prediction.

---

## Features

### Core Modules (Completed)
| Module | Description |
|---|---|
| **DeBERTa-v3 Classifier** | Fine-tuned `microsoft/deberta-v3-base` with Multi-Sample Dropout for robustness. |
| **Data Augmentation Engine** | Seamless synthetic data injection to balance rare "Harmful" and "Misleading" classes. |
| **Probability Calibrator** | Post-training threshold optimizer to maximize Macro F1 score across imbalanced classes. |
| **ONNX Inference Engine** | Optimized model export for <25ms latency on standard CPUs. |
| **LLM Reasoning Layer** | Groq/Ollama-powered dual-check: explains *why* a claim is flagged and provides a second opinion label. |
| **Social Media Scraper** | Real-time monitoring of Reddit, YouTube transcripts/comments, and generic health blogs. |
| **FastAPI Backend** | Production-ready REST API with `/predict`, `/scrape`, and `/health` endpoints. |

### Upcoming
| Feature | Description |
|---|---|
| **Chrome Extension** | Browser overlay to highlight misinformation on health blogs in real-time. |
| **Attention Highlights** | Token-level importance scores surfaced through the `/predict` API. |

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Model Architecture** | DeBERTa-v3-base, Multi-Sample Dropout (3 samples) |
| **Training Strategy** | LLRD (Layer-wise LRD), Focal Loss, Weighted Sampler |
| **Frameworks** | PyTorch 2.x, HuggingFace Transformers, Accelerate |
| **Backend** | FastAPI 0.134 + Uvicorn |
| **LLM Explainer** | OpenAI-compatible client (Groq by default, Ollama-compatible) |
| **Scrapers** | PRAW (Reddit), YouTube Data API v3 + youtube-transcript-api, Trafilatura (web) |
| **Explainability** | Attention-based token scoring, Confidence Calibration |
| **Inference** | ONNX Runtime (CPU-optimized), PyTorch fallback |

---

## Project Structure

```
HealthCare-Risk-Classifier/
├── api/
│   ├── main.py            # FastAPI app: /predict, /scrape, /health
│   ├── schemas.py         # Pydantic request/response models
│   └── middleware.py      # CORS, latency logging
├── llm_layer/
│   ├── explainer.py       # Groq/Ollama dual-check reasoning engine
│   └── test_llm.py        # Standalone LLM layer test
├── scraper/
│   ├── universal_scraper.py   # Route-based dispatcher (YouTube/Reddit/Web)
│   ├── reddit_scraper.py      # PRAW-based Reddit post & comment scraper
│   ├── youtube_scraper.py     # YouTube Data API + transcript scraper
│   └── scraper_config.py      # Subreddits, queries, health keywords
├── inference/
│   ├── predict.py         # PyTorch inference with attention explainability
│   └── predict_onnx.py    # ONNX Runtime optimized inference (<25ms)
├── explainability/
│   ├── attention.py       # CLS-token attention score extractor
│   └── confidence.py      # Temperature-calibrated confidence scorer
├── training/
│   ├── train.py           # Multi-Sample DeBERTa fine-tuning loop
│   ├── config.py          # Centralized hyperparameters and paths
│   └── data_loader.py     # Weighted sampler + synthetic data injection
├── evaluation/
│   └── results/           # Confusion matrices and F1 reports
├── data/
│   ├── raw/               # PUBHEALTH base dataset
│   ├── processed/         # Augmented/Cleaned CSVs
│   └── scraped/           # Output from social media scrapers
├── models/
│   ├── deberta/best/      # Fine-tuned PyTorch weights
│   └── onnx/              # Exported ONNX model
├── extension/             # Chrome extension (popup, content, background)
├── export_onnx.py         # Torch → ONNX conversion script
├── optimize_thresholds.py # Class-bias optimization tool
└── run_training.py        # Main training entry point
```

---

## Getting Started

### 1. Prerequisites
- Python 3.10+
- NVIDIA GPU (RTX 3060+ recommended, 6GB VRAM minimum) for training
- API keys: Groq (for LLM) and optionally Reddit/YouTube (for scrapers)

### 2. Setup
```bash
# Clone and activate venv
git clone https://github.com/Pavan2027/HealthCare-Risk-Classifier.git
cd HealthCare-Risk-Classifier
python -m venv venv
./venv/Scripts/Activate.ps1  # Windows
# source venv/bin/activate    # Linux/macOS

# Install dependencies
pip install -r requirements.txt

# Copy and configure environment
cp .env.example .env
# Edit .env and fill in your API keys
```

### 3. Training & Optimization
```bash
# Start full fine-tuning with LLRD
python run_training.py

# Find the optimized thresholds for your model
python optimize_thresholds.py

# Export for the browser extension or CPU deployment
python export_onnx.py
```

### 4. Running the API Server
```bash
# Start the FastAPI backend
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Or use the built-in entrypoint
python api/main.py
```

The server will:
1. Try to load the **ONNX model** first (`models/onnx/model.onnx`) for fastest inference
2. Fall back to the **PyTorch model** (`models/deberta/best/`) if ONNX is unavailable
3. Expose interactive docs at **http://localhost:8000/docs**

### 5. API Endpoints

#### `GET /health`
```json
{
  "status": "healthy",
  "model_loaded": true,
  "inference_engine": "onnx_quantized",
  "version": "1.0.0"
}
```

#### `POST /predict`
Classify a single piece of health text with optional LLM dual-check.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Drinking bleach cures COVID-19.",
    "use_llm": true
  }'
```

```json
{
  "label": "Harmful",
  "confidence": 0.9612,
  "confidence_level": "high",
  "all_probabilities": { "Harmful": 0.9612, "Misleading": 0.031, ... },
  "explanation": "Bleach is a corrosive chemical that causes severe burns and organ failure...",
  "llm_label": "Harmful",
  "latency_ms": 4548.33
}
```

> **Dual-Check**: When `use_llm: true`, the Groq LLM independently classifies the claim. If `label` ≠ `llm_label`, the response signals a disagreement — a strong indicator to flag for human review.

#### `POST /scrape`
Scrape a URL and automatically classify its content.

```bash
curl -X POST http://localhost:8000/scrape \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.youtube.com/watch?v=<video_id>"}'
```

Supported sources: **YouTube** (transcript), **Reddit** (post + selftext), **Generic Web** (Trafilatura extraction).

### 6. Running the Scrapers Standalone

```bash
# Scrape Reddit health subreddits
python scraper/reddit_scraper.py

# Scrape YouTube health queries
python scraper/youtube_scraper.py

# Test the LLM reasoning layer independently
python llm_layer/test_llm.py
```

### 7. Data Augmentation (Optional)
To improve minority class performance (`Harmful`, `Misleading`), generate synthetic data:

```bash
python data/augment_data.py
```

The `training/data_loader.py` automatically detects and injects `data/processed/synthetic_data.csv` on the next training run.

---

## LLM Configuration

The reasoning layer is provider-agnostic — it uses any OpenAI-compatible API.

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `groq` | Identifier label (for logging) |
| `LLM_MODEL` | `llama-3.3-70b-versatile` | Model name sent to the API |
| `LLM_API_BASE` | `https://api.groq.com/openai/v1` | Base URL of the inference API |
| `LLM_API_KEY` | *(your key)* | API key |

**To use Ollama (local):**
```dotenv
LLM_PROVIDER=ollama
LLM_MODEL=llama3.2
LLM_API_BASE=http://localhost:11434/v1
LLM_API_KEY=ollama
```

**To use Google AI Studio (Gemini):** Use the OpenAI-compatible endpoint from AI Studio and update `LLM_API_BASE` and `LLM_MODEL` accordingly.

---

## Results & Performance

The model has been optimized for imbalanced medical data, where identifying the rare "Harmful" class is more important than raw accuracy.

### Model Comparison
| Model | Test F1 (Macro) | Improvement |
|---|---|---|
| Linear SVM | 0.395 | Baseline |
| Logistic Regression | 0.473 | Baseline |
| **DeBERTa-v3 (Ours)** | **0.648** | **+37.0% (Macro)** |

### Key Metrics (DeBERTa-v3)
| Metric | Score | Impact |
|---|---|---|
| **Weighted F1** | **0.752** | High overall reliability for health claims |
| **Accuracy** | **0.740** | Precise classification across 4 classes |
| **Harmful Recall** | **0.771** | Successfully flags dangerous misinformation |
| **Inference (CPU/ONNX)** | **~23ms** | Optimized with ONNX for real-time use |

---

## Configuration

The project uses a centralized `training/config.py` for all hyperparameters.

### Optimized Biases
After running `optimize_thresholds.py`, the following biases are applied to the final layer to maximize Macro F1:
```python
BEST_BIASES = [0.8, 1.0, 0.9, 1.0]  # [Harmful, Misleading, Verified, Irrelevant]
```

---

## License

This project is developed for educational and research purposes. Model weights and synthetic data injection methods are subject to the project's internal licensing.

---

<p align="center">Built with DeBERTa + PyTorch · Healthcare AI · 2026</p>

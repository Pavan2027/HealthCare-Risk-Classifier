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
| **FastAPI Backend** | Production-ready REST API with `/predict`, `/predict/batch`, `/scrape`, and `/health` endpoints. |
| **Chrome Extension** | Browser overlay — inline word-level attention highlights, tooltip explanations, and floating risk badge. |
| **Attention Highlights** | DeBERTa CLS-token attention scores surface key risk words in both the API and the extension. |

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

### 3. Training Pipeline

> **Data augmentation is run first** — it significantly improves minority class (Harmful / Misleading) recall and should always precede a training run.

```bash
# Step 1 — Generate synthetic data for rare classes
#   Requires Ollama running locally (llama3.2) or a Groq/Gemini API key.
#   Output → data/processed/synthetic_data.csv (auto-injected by data_loader.py)
python data/augment_data.py

# Step 2 — Fine-tune DeBERTa-v3 with LLRD + augmented data
python run_training.py

# Step 3 — Find optimal per-class decision thresholds
python optimize_thresholds.py

# Step 4 (optional) — Export to ONNX for CPU deployment / browser extension
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
  "inference_engine": "pytorch",
  "version": "1.0.0"
}
```

#### `POST /predict`
Classify a single piece of health text. Returns attention-based word highlights when the PyTorch model is active.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Drinking bleach cures COVID-19.",
    "explain": true,
    "use_llm": true
  }'
```

```json
{
  "label": "Harmful",
  "confidence": 0.9612,
  "confidence_level": "high",
  "important_words": [{"word": "bleach", "score": 0.412}, {"word": "cures", "score": 0.287}],
  "all_probabilities": { "Harmful": 0.9612, "Misleading": 0.031, "Verified": 0.005, "Irrelevant": 0.003 },
  "explanation": "Bleach is a corrosive chemical that causes severe burns...",
  "llm_label": "Harmful",
  "disagreement": false,
  "latency_ms": 4548.33
}
```

> **Dual-Check**: When `use_llm: true`, the Groq LLM independently classifies the claim. If `disagreement: true`, the two models disagree — a strong indicator to flag for human review.

#### `POST /predict/batch`
Batch-classify up to 32 texts efficiently (used by the Chrome extension for page scanning).

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Vaccines cause autism.", "Exercise improves heart health."], "explain": false}'
```

#### `POST /scrape`
Scrape a URL and automatically classify its content.

```bash
curl -X POST http://localhost:8000/scrape \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.youtube.com/watch?v=<video_id>"}'
```

Supported sources: **YouTube** (transcript), **Reddit** (post + selftext), **Generic Web** (Trafilatura extraction).

### 6. Chrome Extension

```bash
# No build step needed — the extension is plain JS/HTML/CSS.
# Load it in Chrome:
# 1. Open chrome://extensions
# 2. Enable "Developer mode" (top right)
# 3. Click "Load unpacked" → select the extension/ folder
```

Once loaded:
- Open any health article (e.g., healthline.com, WebMD)
- The extension scans paragraphs automatically and adds left-border risk indicators
- **Flagged words** are underlined with a wavy colored mark (attention highlights)
- Hover over a paragraph to see the tooltip with confidence, key signals, and LLM explanation
- Click the extension icon to open the popup for manual claim checking

### 7. Running the Scrapers Standalone

```bash
# Scrape Reddit health subreddits
python scraper/reddit_scraper.py

# Scrape YouTube health queries
python scraper/youtube_scraper.py

# Test the LLM reasoning layer independently
python llm_layer/test_llm.py
```

---

## Testing Guide

### Layer 1 — LLM Reasoning
```bash
# Tests all 4 labels against the Groq API and prints explanation + second-opinion label
python llm_layer/test_llm.py
```
Expected: each label prints `Primary Model: X`, `LLM Second Opinion: Y`, and a 2-sentence explanation.

### Layer 2 — API Backend
```bash
# Start the server
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000

# In a separate terminal:
# Health check
curl http://localhost:8000/health

# Single predict (with attention highlights)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Essential oils can cure cancer.", "explain": true, "use_llm": true}'

# Batch predict (extension page scan)
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Vaccines are safe.", "Bleach cures COVID."], "explain": false}'

# Or use interactive Swagger UI:
start http://localhost:8000/docs
```
Expected: `/predict` returns `important_words` as a list of `{word, score}` objects (PyTorch model).

### Layer 3 — Chrome Extension
1. Load the extension (see step 6 above)
2. Open `https://www.healthline.com/nutrition/vitamin-c-benefits`
3. Wait ~2 seconds for the auto-scan
4. Verify: red/orange left borders on flagged paragraphs, wavy underlines on key words
5. Hover a flagged paragraph → tooltip appears with label, confidence, and word chips
6. Open the popup → type `"Drinking bleach cures COVID"` → click **Analyze Claim**
7. Toggle **🤖 LLM Second Opinion** ON → re-run → verify explanation block appears

### Layer 4 — Social Media Scrapers
```bash
# Requires REDDIT_CLIENT_ID + REDDIT_CLIENT_SECRET in .env
python scraper/reddit_scraper.py

# Requires YOUTUBE_API_KEY in .env
python scraper/youtube_scraper.py
```
Expected: CSV files written to `data/scraped/` with `text`, `source`, `platform`, `url` columns.

---

## GitHub Releases

The recommended release strategy for this project:

### Option A — GitHub Releases (Recommended)
Because the model weights are large (>400MB), they are **not committed to the repo**. Instead:

1. **Tag a release** after training:
   ```bash
   git tag -a v1.0.0 -m "DeBERTa-v3 trained — F1 Macro 0.648"
   git push origin v1.0.0
   ```
2. On GitHub → **Releases → Create a release from tag**
3. Upload the following as release **Assets**:
   - `models/deberta/best/` (zip) — PyTorch weights
   - `models/onnx/model.onnx` — ONNX model for CPU deployment
4. **End users** can then download and place these in the correct paths before running the API.

### Option B — Hugging Face Hub
Push the fine-tuned model to your HuggingFace account so end users can pull it automatically:
```bash
pip install huggingface_hub
huggingface-cli login
python -c "
from transformers import AutoModelForSequenceClassification, AutoTokenizer
model = AutoModelForSequenceClassification.from_pretrained('models/deberta/best')
tokenizer = AutoTokenizer.from_pretrained('models/deberta/best')
model.push_to_hub('Pavan2027/healthcare-risk-deberta-v3')
tokenizer.push_to_hub('Pavan2027/healthcare-risk-deberta-v3')
"
```
Then users can load directly: `AutoModel.from_pretrained('Pavan2027/healthcare-risk-deberta-v3')`

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

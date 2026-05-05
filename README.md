# HealthCare Risk Classifier (HRC)

> A state-of-the-art medical misinformation detection system powered by DeBERTa-v3. Designed to classify health claims into four risk levels and provide AI-generated reasoning.

![DeBERTa](https://img.shields.io/badge/Model-DeBERTa--v3--base-blue?style=flat-square&logo=huggingface)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=flat-square&logo=fastapi)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python)
![ONNX](https://img.shields.io/badge/Inference-ONNX-gray?style=flat-square&logo=onnx)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C?style=flat-square&logo=pytorch)

---

## Overview

The **HealthCare Risk Classifier (HRC)** is an end-to-end NLP framework built to combat medical misinformation. Unlike standard "True/False" classifiers, HRC focuses on the **risk profile** of a claim, distinguishing between verified medical facts, misleading nuances, and active medical harm.

The system utilizes a custom **Multi-Sample DeBERTa** architecture with **Layer-wise Learning Rate Decay (LLRD)** to achieve high precision in the medical domain, even under strict hardware constraints (6GB VRAM).

---

## Features

### Core Modules (Completed)
| Module | Description |
|---|---|
| **DeBERTa-v3 Classifier** | Fine-tuned `microsoft/deberta-v3-base` with Multi-Sample Dropout for robustness. |
| **Data Augmentation Engine** | Seamless synthetic data injection to balance rare "Harmful" and "Misleading" classes. |
| **Probability Calibrator** | Post-training threshold optimizer to maximize Macro F1 score across imbalanced classes. |
| **ONNX Inference Engine** | Optimized model export for <25ms latency on standard CPUs. |

### Advanced Layers (In Progress / Left)
| Feature | Status | Description |
|---|---|---|
| **LLM Reasoning Layer** | 🟡 Left | Uses Gemini/Ollama to explain *why* a claim was flagged as harmful. |
| **Social Media Scraper** | 🟡 Left | Real-time monitoring of Twitter (X), Reddit, and YouTube transcripts. |
| **API Application Layer** | 🟡 Left | FastAPI backend for multi-user reports and historical tracking. |
| **Chrome Extension** | 🟡 Left | Browser overlay to highlight misinformation on health blogs in real-time. |

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Model Architecture** | DeBERTa-v3-base, Multi-Sample Dropout (3 samples) |
| **Training Strategy** | LLRD (Layer-wise LRD), Focal Loss, Weighted Sampler |
| **Frameworks** | PyTorch 2.2, HuggingFace Transformers, Accelerate |
| **Backend** | FastAPI (Upcoming) |
| **Explainability** | Integrated Gradients, LLM Reasoning (Upcoming) |
| **Inference** | ONNX Runtime Web/Python |

---

## Project Structure

```
HealthCare-Risk-Classifier/
├── training/
│   ├── train.py           # Custom Multi-Sample DeBERTa implementation
│   ├── config.py          # LLRD and Hyperparameter configuration
│   └── data_loader.py     # Weighted sampler and synthetic data injection
├── evaluation/
│   ├── results/           # Confusion matrices and F1 reports
│   └── evaluate.py        # Comprehensive metric suite
├── data/
│   ├── raw/               # PUBHEALTH base dataset
│   └── processed/         # Augmented/Cleaned CSVs
├── models/
│   ├── deberta/           # Fine-tuned PyTorch weights
│   └── onnx/              # Exported edge-ready model
├── llm_layer/             # LLM reasoning and explanation (Placeholder)
├── export_onnx.py         # Torch -> ONNX conversion script
├── optimize_thresholds.py # Class-bias optimization tool
└── run_training.py        # Main entry point for stable training
```

---

## Getting Started

### 1. Prerequisites
- Python 3.10+
- NVIDIA GPU (RTX 3060+ recommended, 6GB VRAM minimum)

### 2. Setup
```bash
# Clone and activate venv
git clone https://github.com/Pavan2027/HealthCare-Risk-Classifier.git
cd HealthCare-Risk-Classifier
python -m venv venv
./venv/Scripts/Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 3. Training & Optimization
```bash
# Start full fine-tuning with LLRD
python run_training.py

# Find the optimized thresholds for your model
python optimize_thresholds.py

# Export for the browser extension
python export_onnx.py
```

### 4. Data Augmentation (Optional)
If you want to improve the model's performance on minority classes (like `Harmful` or `Misleading`), you can generate synthetic data using a local or cloud-based LLM.

```bash
# Run the augmentation script
python data/augment_data.py
```

#### **How it works:**
The script iterates through the rare classes in your training set and uses an LLM to paraphrase and create distinct variations of existing claims. The `training/data_loader.py` will automatically detect and inject any data found in `data/processed/synthetic_data.csv` during the next training run.

#### **Configuration:**
You can modify the constants at the top of `data/augment_data.py`:
- **Target Counts**: Modify `SAMPLES_TO_GENERATE` to set your desired dataset size.
- **Local LLM (Ollama)**: By default, it uses `llama3.2` on `localhost:11434`. Ensure Ollama is running (`ollama serve`).
- **Cloud LLMs (Groq / Google AI Studio)**: 
    - To use **Groq**, update `OLLAMA_URL` to `https://api.groq.com/openai/v1/chat/completions` and modify the `generate_synthetic_data` function to use the OpenAI-compatible payload and your API Key.
    - To use **Google AI Studio**, use the Gemini API endpoint and update the payload structure accordingly.

---

## Results & Performance

The model has been optimized for imbalanced medical data, where identifying the rare "Harmful" class is more important than raw accuracy.

### **Model Comparison**
| Model | Test F1 (Macro) | Improvement |
|---|---|---|
| Linear SVM | 0.395 | Baseline |
| Logistic Regression | 0.473 | Baseline |
| **DeBERTa-v3 (Ours)** | **0.648** | **+37.0% (Macro)** |

### **Key Metrics (DeBERTa-v3)**
| Metric | Score | Impact |
|---|---|---|
| **Weighted F1** | **0.752** | High overall reliability for health claims |
| **Accuracy** | **0.740** | Precise classification across 4 classes |
| **Harmful Recall** | **0.771** | Successfully flags dangerous misinformation |
| **Inference (CPU)** | **~23ms** | Optimized with ONNX for real-time use |

---

## Configuration

The project uses a centralized `training/config.py` for all hyperparameters. 

### Optimized Biases
After running `optimize_thresholds.py`, the following biases are applied to the final layer to maximize Macro F1:
```python
BEST_BIASES = [0.8, 1.0, 0.9, 1.0] # [Harmful, Misleading, Verified, Irrelevant]
```

---

## License

This project is developed for educational and research purposes. Model weights and synthetic data injection methods are subject to the project's internal licensing.

---

<p align="center">Built with DeBERTa + PyTorch · Healthcare AI · 2026</p>

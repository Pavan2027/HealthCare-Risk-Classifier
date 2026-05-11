"""
Training configuration for Healthcare Misinformation Risk Classifier.
Central place for all hyperparameters and settings.
"""

import os
from pathlib import Path

# --- Paths -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SCRAPED_DATA_DIR = DATA_DIR / "scraped"
MODELS_DIR = PROJECT_ROOT / "models"
DEBERTA_DIR = MODELS_DIR / "deberta"
ONNX_DIR = MODELS_DIR / "onnx"
EVAL_RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"

# Create directories
for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, SCRAPED_DATA_DIR,
          DEBERTA_DIR, ONNX_DIR, EVAL_RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# --- Model -------------------------------------------------------------------
MODEL_NAME = "microsoft/deberta-v3-base"
NUM_LABELS = 4
MAX_LENGTH = 384

# --- Label Mapping -----------------------------------------------------------
# PUBHEALTH original: {0: false, 1: mixture, 2: true, 3: unproven}
# Our mapping:
LABEL_MAP = {
    0: "Harmful",      # false claims ? harmful misinformation
    1: "Misleading",   # mixture ? partially misleading
    2: "Verified",     # true ? verified information
    3: "Irrelevant",   # unproven ? irrelevant / unverifiable
}
LABEL_TO_ID = {v: k for k, v in LABEL_MAP.items()}

# Human-readable names for display
LABEL_DESCRIPTIONS = {
    "Harmful": "Contains dangerous health misinformation that could cause harm",
    "Misleading": "Contains partially true but misleading health information",
    "Verified": "Contains verified, factually accurate health information",
    "Irrelevant": "Unverifiable or not directly related to actionable health claims",
}

# --- Training Hyperparameters ------------------------------------------------
LEARNING_RATE = 3e-5
NUM_EPOCHS = 10
BATCH_SIZE = 2
EVAL_BATCH_SIZE = 8
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.05
LABEL_SMOOTHING = 0.1
EARLY_STOPPING_PATIENCE = 10
GRADIENT_ACCUMULATION_STEPS = 12
FP16 = True  # Use mixed precision for speed on RTX 4050
GRADIENT_CHECKPOINTING = False  # Disabled to resolve backward pass conflict

# Seed for reproducibility
SEED = 42

# --- Inference ---------------------------------------------------------------
ONNX_MODEL_PATH = ONNX_DIR / "model.onnx"
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for high-confidence predictions
TEMPERATURE = 1.0  # Temperature scaling (calibrated post-training)

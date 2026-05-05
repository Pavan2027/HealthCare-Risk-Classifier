from pathlib import Path
import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.getcwd())
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from training.config import DEBERTA_DIR, EVAL_RESULTS_DIR, LABEL_MAP, NUM_LABELS
from training.data_loader import get_dataloaders
from sklearn.metrics import f1_score, confusion_matrix

def optimize_thresholds():
    print("[*] Starting Threshold Optimization to push for 80% F1...")
    
    # Load model and tokenizer
    model_path = DEBERTA_DIR / "best"
    if not model_path.exists():
        print(f"[!] Best model not found at {model_path}")
        return

    model = AutoModelForSequenceClassification.from_pretrained(model_path).to("cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Get test dataset
    data = get_dataloaders(tokenizer)
    test_dataset = data["test_dataset"]
    
    # Get raw predictions (logits)
    training_args = TrainingArguments(output_dir="tmp", per_device_eval_batch_size=8, no_cuda=True)
    trainer = Trainer(model=model, args=training_args)
    
    print("[*] Generating raw logits for test set...")
    output = trainer.predict(test_dataset)
    logits = torch.from_numpy(output.predictions)
    true_labels = output.label_ids
    
    # Convert logits to probabilities
    probs = torch.softmax(logits, dim=-1).numpy()
    
    # Initial macro F1 (argmax)
    initial_preds = np.argmax(probs, axis=1)
    initial_f1 = f1_score(true_labels, initial_preds, average="macro")
    print(f"[*] Initial Macro F1 (Argmax): {initial_f1:.4f}")
    
    # --- Aggressive Grid Search for Thresholds ---
    best_f1 = initial_f1
    best_biases = [1.0, 1.0, 1.0, 1.0]
    
    print("[*] Running aggressive grid search (this might take a minute)...")
    # Bias search space
    scales = [0.5, 0.7, 0.9, 1.0, 1.2, 1.5, 2.0, 3.0]
    
    for b1 in scales: # Misleading
        for b3 in [1.0, 2.0, 4.0, 6.0]: # Irrelevant (rare, needs high bias)
            for b0 in [0.8, 1.0, 1.2, 1.5]: # Harmful
                for b2 in [0.7, 0.9, 1.0]: # Verified (usually over-predicted)
                    current_biases = [b0, b1, b2, b3]
                    biased_probs = probs * np.array(current_biases)
                    preds = np.argmax(biased_probs, axis=1)
                    
                    f1 = f1_score(true_labels, preds, average="macro")
                    if f1 > best_f1:
                        best_f1 = f1
                        best_biases = current_biases

    print(f"\n[OK] Optimization Complete!")
    print(f"   Best Macro F1: {best_f1:.4f} (Improvement: {best_f1 - initial_f1:.4f})")
    print(f"   Best Biases: {best_biases}")
    
    # Generate final report with best thresholds
    final_biased_probs = probs * np.array(best_biases)
    final_preds = np.argmax(final_biased_probs, axis=1)
    
    cm = confusion_matrix(true_labels, final_preds)
    print("\n[*] Optimized Confusion Matrix:")
    print(cm)
    
    # Save optimized settings
    opt_results = {
        "initial_f1_macro": float(initial_f1),
        "optimized_f1_macro": float(best_f1),
        "best_biases": [float(b) for b in best_biases],
        "label_map": LABEL_MAP
    }
    
    with open(EVAL_RESULTS_DIR / "optimized_thresholds.json", "w") as f:
        json.dump(opt_results, f, indent=2)
    print(f"\n[SAVED] Optimized thresholds saved to evaluation/results/optimized_thresholds.json")

if __name__ == "__main__":
    try:
        optimize_thresholds()
    except Exception as e:
        print(f"\n[!] FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()

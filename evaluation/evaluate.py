"""
Comprehensive evaluation framework for the Health Risk Classifier.
Generates metrics, confusion matrices, and comparison reports.
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)

from training.config import LABEL_MAP, NUM_LABELS, EVAL_RESULTS_DIR


def evaluate_predictions(true_labels, pred_labels, save_name="evaluation"):
    """
    Run full evaluation and generate reports.

    Args:
        true_labels: List/array of true label IDs
        pred_labels: List/array of predicted label IDs
        save_name: Prefix for saved files

    Returns:
        dict with all metrics
    """
    label_names = [LABEL_MAP[i] for i in range(NUM_LABELS)]

    # -- Compute metrics --------------------------------------------------
    metrics = {
        "accuracy": accuracy_score(true_labels, pred_labels),
        "f1_macro": f1_score(true_labels, pred_labels, average="macro"),
        "f1_weighted": f1_score(true_labels, pred_labels, average="weighted"),
        "precision_macro": precision_score(
            true_labels, pred_labels, average="macro"
        ),
        "recall_macro": recall_score(
            true_labels, pred_labels, average="macro"
        ),
    }

    # Per-class F1
    per_class_f1 = f1_score(true_labels, pred_labels, average=None)
    for i, name in enumerate(label_names):
        metrics[f"f1_{name}"] = per_class_f1[i]

    # Classification report
    report = classification_report(
        true_labels, pred_labels,
        target_names=label_names,
        digits=4,
        output_dict=True,
    )
    report_text = classification_report(
        true_labels, pred_labels,
        target_names=label_names,
        digits=4,
    )

    print("=" * 60)
    print(f"? EVALUATION RESULTS: {save_name}")
    print("=" * 60)
    print(report_text)

    # -- Confusion Matrix -------------------------------------------------
    cm = confusion_matrix(true_labels, pred_labels)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Raw counts
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=label_names, yticklabels=label_names,
        ax=axes[0],
    )
    axes[0].set_title("Confusion Matrix (Counts)")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    # Normalized
    sns.heatmap(
        cm_normalized, annot=True, fmt=".2%", cmap="Oranges",
        xticklabels=label_names, yticklabels=label_names,
        ax=axes[1],
    )
    axes[1].set_title("Confusion Matrix (Normalized)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    plt.tight_layout()
    cm_path = EVAL_RESULTS_DIR / f"{save_name}_confusion_matrix.png"
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   ? Confusion matrix saved to {cm_path}")

    # -- Per-class F1 bar chart -------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#e74c3c", "#f39c12", "#2ecc71", "#95a5a6"]
    bars = ax.bar(label_names, per_class_f1, color=colors, edgecolor="white", linewidth=1.5)

    for bar, f1 in zip(bars, per_class_f1):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{f1:.3f}", ha="center", va="bottom", fontweight="bold", fontsize=12,
        )

    ax.set_ylabel("F1-Score", fontsize=12)
    ax.set_title("Per-Class F1 Scores", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    f1_path = EVAL_RESULTS_DIR / f"{save_name}_per_class_f1.png"
    plt.savefig(f1_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   ? Per-class F1 chart saved to {f1_path}")

    # -- Save JSON results ------------------------------------------------
    results = {
        "metrics": {k: round(v, 4) for k, v in metrics.items()},
        "report": report,
        "confusion_matrix": cm.tolist(),
    }
    json_path = EVAL_RESULTS_DIR / f"{save_name}_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"   ? Results saved to {json_path}")

    return results


def compare_models(baseline_results_path=None, distilbert_results_path=None):
    """
    Generate a comparison table between baseline and DistilBERT.
    Loads saved results from evaluation/results/.
    """
    baseline_path = baseline_results_path or EVAL_RESULTS_DIR / "baseline_results.json"
    distilbert_path = distilbert_results_path or EVAL_RESULTS_DIR / "training_results.json"

    if not baseline_path.exists() or not distilbert_path.exists():
        print("[!]  Results files not found. Run baseline.py and train.py first.")
        return

    with open(baseline_path) as f:
        baseline = json.load(f)
    with open(distilbert_path) as f:
        distilbert = json.load(f)

    print("\n" + "=" * 60)
    print("? MODEL COMPARISON")
    print("=" * 60)

    # Get best baseline F1
    lr_f1 = baseline.get("logistic_regression", {}).get("test_f1_macro", 0)
    svm_f1 = baseline.get("linear_svm", {}).get("test_f1_macro", 0)
    best_baseline = max(lr_f1, svm_f1)

    # DistilBERT F1
    db_f1 = distilbert.get("test_metrics", {}).get("eval_f1_macro", 0)

    print(f"\n{'Model':<25} {'Test F1 (Macro)':<20} {'Improvement':<15}")
    print("-" * 60)
    print(f"{'Logistic Regression':<25} {lr_f1:<20.4f} {'(baseline)':<15}")
    print(f"{'Linear SVM':<25} {svm_f1:<20.4f} {'(baseline)':<15}")
    print(f"{'DistilBERT (Ours)':<25} {db_f1:<20.4f} {f'+{(db_f1-best_baseline):.4f}':<15}")

    improvement_pct = (db_f1 - best_baseline) / best_baseline * 100
    print(f"\n? DistilBERT improvement: +{improvement_pct:.1f}% over best baseline")
    print(f"\n? Resume line:")
    print(f"   Fine-tuned DistilBERT on PUBHEALTH for multi-class risk")
    print(f"   classification, achieving F1={db_f1:.2f} vs {best_baseline:.2f}")
    print(f"   baseline (LR/SVM).")

    # Generate comparison chart
    fig, ax = plt.subplots(figsize=(10, 5))
    models = ["Logistic\nRegression", "Linear\nSVM", "DistilBERT\n(Ours)"]
    f1_scores = [lr_f1, svm_f1, db_f1]
    colors = ["#bdc3c7", "#bdc3c7", "#3498db"]

    bars = ax.bar(models, f1_scores, color=colors, edgecolor="white", linewidth=2)
    for bar, f1 in zip(bars, f1_scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{f1:.3f}", ha="center", va="bottom", fontweight="bold", fontsize=13,
        )

    ax.set_ylabel("Macro F1-Score", fontsize=12)
    ax.set_title("Model Comparison: Baseline vs DistilBERT", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    chart_path = EVAL_RESULTS_DIR / "model_comparison.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n   ? Comparison chart saved to {chart_path}")

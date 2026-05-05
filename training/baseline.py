"""
Baseline models (Logistic Regression + SVM) for comparison.
Uses TF-IDF features. This establishes the [X] baseline F1-score for your resume.
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    classification_report,
    f1_score,
    confusion_matrix,
)
from sklearn.calibration import CalibratedClassifierCV

from training.config import LABEL_MAP, EVAL_RESULTS_DIR
from training.data_loader import load_data


def train_baselines():
    """Train and evaluate baseline models."""
    print("=" * 60)
    print("? Training Baseline Models (LR + SVM)")
    print("=" * 60)

    # Load data
    train_texts, train_labels = load_data("train")
    val_texts, val_labels = load_data("validation")
    test_texts, test_labels = load_data("test")

    # TF-IDF Vectorization
    print("\n? Fitting TF-IDF vectorizer...")
    tfidf = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )
    X_train = tfidf.fit_transform(train_texts)
    X_val = tfidf.transform(val_texts)
    X_test = tfidf.transform(test_texts)
    print(f"   Vocabulary size: {len(tfidf.vocabulary_)}")
    print(f"   Feature matrix: {X_train.shape}")

    label_names = [LABEL_MAP[i] for i in range(len(LABEL_MAP))]
    results = {}

    # -- Logistic Regression ----------------------------------------------
    print("\n? Training Logistic Regression...")
    lr = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        C=1.0,
        solver="lbfgs",
        random_state=42,
    )
    lr.fit(X_train, train_labels)

    lr_val_preds = lr.predict(X_val)
    lr_test_preds = lr.predict(X_test)

    lr_val_f1 = f1_score(val_labels, lr_val_preds, average="macro")
    lr_test_f1 = f1_score(test_labels, lr_test_preds, average="macro")

    print(f"\n   Validation Macro F1: {lr_val_f1:.4f}")
    print(f"   Test Macro F1:       {lr_test_f1:.4f}")
    print(f"\n   Test Classification Report:")
    lr_report = classification_report(
        test_labels, lr_test_preds, target_names=label_names, digits=4
    )
    print(lr_report)

    results["logistic_regression"] = {
        "val_f1_macro": round(lr_val_f1, 4),
        "test_f1_macro": round(lr_test_f1, 4),
        "test_report": classification_report(
            test_labels, lr_test_preds, target_names=label_names,
            digits=4, output_dict=True
        ),
    }

    # -- Linear SVM -------------------------------------------------------
    print("\n? Training Linear SVM...")
    svm_base = LinearSVC(
        max_iter=2000,
        class_weight="balanced",
        C=1.0,
        random_state=42,
    )
    # Wrap with CalibratedClassifierCV for probability estimates
    svm = CalibratedClassifierCV(svm_base, cv=3)
    svm.fit(X_train, train_labels)

    svm_val_preds = svm.predict(X_val)
    svm_test_preds = svm.predict(X_test)

    svm_val_f1 = f1_score(val_labels, svm_val_preds, average="macro")
    svm_test_f1 = f1_score(test_labels, svm_test_preds, average="macro")

    print(f"\n   Validation Macro F1: {svm_val_f1:.4f}")
    print(f"   Test Macro F1:       {svm_test_f1:.4f}")
    print(f"\n   Test Classification Report:")
    svm_report = classification_report(
        test_labels, svm_test_preds, target_names=label_names, digits=4
    )
    print(svm_report)

    results["linear_svm"] = {
        "val_f1_macro": round(svm_val_f1, 4),
        "test_f1_macro": round(svm_test_f1, 4),
        "test_report": classification_report(
            test_labels, svm_test_preds, target_names=label_names,
            digits=4, output_dict=True
        ),
    }

    # -- Summary ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("? BASELINE SUMMARY")
    print("=" * 60)
    print(f"   Logistic Regression  ?  Test F1: {lr_test_f1:.4f}")
    print(f"   Linear SVM           ?  Test F1: {svm_test_f1:.4f}")
    best_baseline = max(lr_test_f1, svm_test_f1)
    print(f"   Best Baseline F1:                {best_baseline:.4f}")
    print(f"\n   This is your [X] baseline for the resume comparison.")

    # Save results
    results_path = EVAL_RESULTS_DIR / "baseline_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n   ? Results saved to {results_path}")

    return results


if __name__ == "__main__":
    train_baselines()

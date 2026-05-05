"""
Standard PyTorch inference engine for DistilBERT predictions.
Supports single and batch inference with explainability.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from training.config import DISTILBERT_DIR, MAX_LENGTH, LABEL_MAP, NUM_LABELS
from explainability.attention import AttentionExplainer
from explainability.confidence import ConfidenceScorer


class PyTorchPredictor:
    """
    PyTorch-based inference for health risk classification.
    Includes attention explainability and confidence scoring.
    """

    def __init__(self, model_path=None):
        self.model_path = model_path or str(DISTILBERT_DIR / "best")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path, output_attentions=True
        )
        self.model.eval()

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Explainability & confidence
        self.explainer = AttentionExplainer(
            model=self.model, tokenizer=self.tokenizer
        )
        self.confidence_scorer = ConfidenceScorer()

        print(f"[OK] PyTorch predictor loaded on {self.device}")

    def predict(self, text, explain=True, top_k=5):
        """
        Predict risk classification for a single text.

        Args:
            text: Input text to classify
            explain: Whether to include attention explanations
            top_k: Number of important words to return

        Returns:
            dict with label, confidence, explanation, latency
        """
        start_time = time.perf_counter()

        if explain:
            result = self.explainer.get_attention_scores(text, top_k=top_k)
        else:
            # Fast path without attention
            inputs = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=MAX_LENGTH,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            score_result = self.confidence_scorer.score(
                outputs.logits.cpu().numpy()
            )
            result = {
                "text": text,
                **score_result,
                "important_words": [],
            }

        latency_ms = (time.perf_counter() - start_time) * 1000
        result["latency_ms"] = round(latency_ms, 2)

        return result

    def predict_batch(self, texts, explain=False):
        """
        Predict risk classification for a batch of texts.

        Args:
            texts: List of input texts
            explain: Whether to include attention explanations

        Returns:
            list of prediction dicts
        """
        start_time = time.perf_counter()

        if explain:
            # Explain mode: process individually (attention is per-sample)
            results = [self.predict(text, explain=True) for text in texts]
        else:
            # Fast batch mode
            inputs = self.tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                max_length=MAX_LENGTH,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            scores = self.confidence_scorer.batch_score(
                outputs.logits.cpu().numpy()
            )

            results = []
            for text, score in zip(texts, scores):
                results.append({
                    "text": text,
                    **score,
                    "important_words": [],
                })

        total_latency = (time.perf_counter() - start_time) * 1000
        for r in results:
            r["batch_latency_ms"] = round(total_latency, 2)

        return results


if __name__ == "__main__":
    predictor = PyTorchPredictor()

    test_texts = [
        "Drinking bleach can cure COVID-19",
        "Vaccines have been proven effective in preventing diseases",
        "Essential oils can replace chemotherapy for cancer treatment",
        "Regular exercise improves cardiovascular health",
    ]

    for text in test_texts:
        result = predictor.predict(text)
        print(f"\n? '{text}'")
        print(f"   Label: {result['label']} ({result['confidence']:.2%})")
        if result.get("important_words"):
            words = [w["word"] for w in result["important_words"][:5]]
            print(f"   Key words: {', '.join(words)}")
        print(f"   Latency: {result['latency_ms']:.1f}ms")

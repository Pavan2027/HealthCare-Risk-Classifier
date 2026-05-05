"""
ONNX Runtime optimized inference engine.
Provides low-latency predictions with input bucketing and warm-up.
Target: <40ms per inference on CPU.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

from training.config import (
    ONNX_DIR, ONNX_MODEL_PATH, MAX_LENGTH, LABEL_MAP, NUM_LABELS
)
from explainability.confidence import ConfidenceScorer


class ONNXPredictor:
    """
    ONNX Runtime-based inference for health risk classification.
    Optimized for low latency with quantized model and input bucketing.
    """

    def __init__(self, model_path=None, tokenizer_path=None):
        model_path = model_path or str(ONNX_MODEL_PATH)
        tokenizer_path = tokenizer_path or str(ONNX_DIR)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        # Configure ONNX Runtime session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        sess_options.intra_op_num_threads = 4
        sess_options.inter_op_num_threads = 1
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        # Create session
        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )

        # Confidence scorer
        self.confidence_scorer = ConfidenceScorer()

        # Input buckets (pad to nearest bucket to avoid recompilation)
        self.buckets = [32, 64, 128, 256]

        # Warm-up
        self._warmup()

        print(f"[OK] ONNX predictor loaded (quantized)")

    def _warmup(self, n_warmup=3):
        """Run warm-up inferences to stabilize latency."""
        dummy_text = "Warm-up inference for ONNX Runtime session initialization."
        for _ in range(n_warmup):
            self._run_inference(dummy_text)

    def _get_bucket_length(self, seq_length):
        """Find the smallest bucket that fits the sequence."""
        for bucket in self.buckets:
            if seq_length <= bucket:
                return bucket
        return MAX_LENGTH

    def _run_inference(self, text, max_length=None):
        """Run raw ONNX inference."""
        # Tokenize with dynamic length
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="np",
        )

        # Bucket padding for consistent performance
        actual_length = encoding["input_ids"].shape[1]
        bucket_length = max_length or self._get_bucket_length(actual_length)

        # Re-tokenize with bucket-padded length
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=bucket_length,
            return_tensors="np",
        )

        ort_inputs = {
            "input_ids": encoding["input_ids"].astype(np.int64),
            "attention_mask": encoding["attention_mask"].astype(np.int64),
        }

        logits = self.session.run(None, ort_inputs)[0]
        return logits

    def predict(self, text):
        """
        Predict risk classification for a single text.

        Returns:
            dict with label, confidence, latency_ms
        """
        start_time = time.perf_counter()

        logits = self._run_inference(text)
        result = self.confidence_scorer.score(logits)
        result["text"] = text

        latency_ms = (time.perf_counter() - start_time) * 1000
        result["latency_ms"] = round(latency_ms, 2)

        return result

    def predict_batch(self, texts):
        """
        Predict for a batch of texts.

        Returns:
            list of prediction dicts
        """
        start_time = time.perf_counter()

        # Tokenize all together
        encoding = self.tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="np",
        )

        ort_inputs = {
            "input_ids": encoding["input_ids"].astype(np.int64),
            "attention_mask": encoding["attention_mask"].astype(np.int64),
        }

        logits = self.session.run(None, ort_inputs)[0]
        scores = self.confidence_scorer.batch_score(logits)

        total_latency = (time.perf_counter() - start_time) * 1000
        per_sample_latency = total_latency / len(texts)

        results = []
        for text, score in zip(texts, scores):
            score["text"] = text
            score["latency_ms"] = round(per_sample_latency, 2)
            score["batch_latency_ms"] = round(total_latency, 2)
            results.append(score)

        return results


if __name__ == "__main__":
    predictor = ONNXPredictor()

    test_texts = [
        "Drinking bleach can cure COVID-19",
        "Vaccines have been proven effective in preventing diseases",
        "Essential oils can replace chemotherapy for cancer treatment",
        "Regular exercise improves cardiovascular health",
    ]

    print("\n? ONNX Inference Results:")
    for text in test_texts:
        result = predictor.predict(text)
        print(f"\n? '{text}'")
        print(f"   Label: {result['label']} ({result['confidence']:.2%})")
        print(f"   Latency: {result['latency_ms']:.1f}ms")

    # Batch inference
    print(f"\n\n? Batch Inference ({len(test_texts)} texts):")
    batch_results = predictor.predict_batch(test_texts)
    print(f"   Total batch latency: {batch_results[0]['batch_latency_ms']:.1f}ms")
    for r in batch_results:
        print(f"   {r['label']}: {r['confidence']:.2%}")

"""
Latency benchmarking for PyTorch vs ONNX vs ONNX-Quantized inference.
Measures p50, p95, p99 latencies for single and batch inference.
"""

import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from training.config import EVAL_RESULTS_DIR, DISTILBERT_DIR, ONNX_MODEL_PATH


def benchmark_latency(n_iterations=100, warmup=10):
    """
    Benchmark inference latency for different backends.

    Args:
        n_iterations: Number of inference iterations
        warmup: Number of warm-up iterations (excluded from timing)
    """
    print("=" * 60)
    print("[*] LATENCY BENCHMARK")
    print("=" * 60)

    test_texts = [
        "Drinking bleach can cure COVID-19",
        "Vaccines have been proven safe and effective by multiple clinical trials",
        "Essential oils can replace chemotherapy for cancer treatment according to some natural health advocates",
        "Regular exercise improves cardiovascular health",
        "Taking vitamin C in high doses prevents all viral infections",
    ]

    results = {}

    # -- ONNX Inference ---------------------------------------------------
    try:
        from inference.predict_onnx import ONNXPredictor

        print(f"\n? Benchmarking ONNX (Quantized)...")
        onnx_predictor = ONNXPredictor()

        latencies = []
        for i in range(warmup + n_iterations):
            text = test_texts[i % len(test_texts)]
            start = time.perf_counter()
            onnx_predictor.predict(text)
            elapsed = (time.perf_counter() - start) * 1000

            if i >= warmup:
                latencies.append(elapsed)

        latencies = np.array(latencies)
        results["onnx_quantized"] = {
            "mean": round(np.mean(latencies), 2),
            "median": round(np.median(latencies), 2),
            "p50": round(np.percentile(latencies, 50), 2),
            "p95": round(np.percentile(latencies, 95), 2),
            "p99": round(np.percentile(latencies, 99), 2),
            "min": round(np.min(latencies), 2),
            "max": round(np.max(latencies), 2),
            "std": round(np.std(latencies), 2),
        }
        print(f"   Mean:   {results['onnx_quantized']['mean']:.2f} ms")
        print(f"   Median: {results['onnx_quantized']['median']:.2f} ms")
        print(f"   P95:    {results['onnx_quantized']['p95']:.2f} ms")
        print(f"   P99:    {results['onnx_quantized']['p99']:.2f} ms")

        # Batch inference benchmark
        print(f"\n   Batch inference (batch_size={len(test_texts)})...")
        batch_latencies = []
        for i in range(warmup + n_iterations):
            start = time.perf_counter()
            onnx_predictor.predict_batch(test_texts)
            elapsed = (time.perf_counter() - start) * 1000

            if i >= warmup:
                batch_latencies.append(elapsed)

        batch_latencies = np.array(batch_latencies)
        results["onnx_quantized_batch"] = {
            "batch_size": len(test_texts),
            "total_mean": round(np.mean(batch_latencies), 2),
            "per_sample_mean": round(np.mean(batch_latencies) / len(test_texts), 2),
            "p95": round(np.percentile(batch_latencies, 95), 2),
        }
        print(f"   Total mean:      {results['onnx_quantized_batch']['total_mean']:.2f} ms")
        print(f"   Per-sample mean: {results['onnx_quantized_batch']['per_sample_mean']:.2f} ms")

    except Exception as e:
        print(f"   [ERROR] ONNX benchmark failed: {e}")

    # -- PyTorch Inference ------------------------------------------------
    try:
        from inference.predict import PyTorchPredictor

        print(f"\n? Benchmarking PyTorch...")
        pt_predictor = PyTorchPredictor()

        latencies = []
        for i in range(warmup + n_iterations):
            text = test_texts[i % len(test_texts)]
            start = time.perf_counter()
            pt_predictor.predict(text, explain=False)
            elapsed = (time.perf_counter() - start) * 1000

            if i >= warmup:
                latencies.append(elapsed)

        latencies = np.array(latencies)
        results["pytorch"] = {
            "mean": round(np.mean(latencies), 2),
            "median": round(np.median(latencies), 2),
            "p50": round(np.percentile(latencies, 50), 2),
            "p95": round(np.percentile(latencies, 95), 2),
            "p99": round(np.percentile(latencies, 99), 2),
            "min": round(np.min(latencies), 2),
            "max": round(np.max(latencies), 2),
            "std": round(np.std(latencies), 2),
        }
        print(f"   Mean:   {results['pytorch']['mean']:.2f} ms")
        print(f"   Median: {results['pytorch']['median']:.2f} ms")
        print(f"   P95:    {results['pytorch']['p95']:.2f} ms")
        print(f"   P99:    {results['pytorch']['p99']:.2f} ms")

    except Exception as e:
        print(f"   [ERROR] PyTorch benchmark failed: {e}")

    # -- Summary ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("? LATENCY SUMMARY")
    print("=" * 60)

    if "onnx_quantized" in results and "pytorch" in results:
        speedup = results["pytorch"]["mean"] / results["onnx_quantized"]["mean"]
        print(f"\n   PyTorch mean:       {results['pytorch']['mean']:.2f} ms")
        print(f"   ONNX Quantized mean: {results['onnx_quantized']['mean']:.2f} ms")
        print(f"   Speedup:            {speedup:.1f}x faster")

    # -- Generate chart ---------------------------------------------------
    if results:
        backends = []
        means = []
        p95s = []

        for name, data in results.items():
            if "batch" not in name and "mean" in data:
                backends.append(name.replace("_", "\n").title())
                means.append(data["mean"])
                p95s.append(data["p95"])

        if backends:
            fig, ax = plt.subplots(figsize=(10, 5))
            x = np.arange(len(backends))
            width = 0.35

            bars1 = ax.bar(x - width/2, means, width, label="Mean", color="#3498db", edgecolor="white")
            bars2 = ax.bar(x + width/2, p95s, width, label="P95", color="#e74c3c", edgecolor="white")

            ax.set_ylabel("Latency (ms)", fontsize=12)
            ax.set_title("Inference Latency Comparison", fontsize=14, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(backends)
            ax.legend()
            ax.grid(axis="y", alpha=0.3)

            for bar in bars1:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=10)
            for bar in bars2:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=10)

            plt.tight_layout()
            chart_path = EVAL_RESULTS_DIR / "latency_benchmark.png"
            plt.savefig(chart_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"\n   ? Latency chart saved to {chart_path}")

    # Save results
    json_path = EVAL_RESULTS_DIR / "latency_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"   ? Results saved to {json_path}")

    return results


if __name__ == "__main__":
    benchmark_latency(n_iterations=50, warmup=5)

"""
Export fine-tuned DistilBERT to ONNX format with quantization.
Produces an optimized model for fast CPU inference.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from training.config import DISTILBERT_DIR, ONNX_DIR, MAX_LENGTH, NUM_LABELS


def export_to_onnx():
    """Export PyTorch model to ONNX with optimization and quantization."""
    print("=" * 60)
    print("? Exporting DistilBERT to ONNX")
    print("=" * 60)

    model_path = str(DISTILBERT_DIR / "best")
    onnx_path = ONNX_DIR / "model.onnx"
    onnx_quantized_path = ONNX_DIR / "model_quantized.onnx"

    # Load model and tokenizer
    print(f"\n? Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

    # Create dummy input
    dummy_text = "This is a test health claim for ONNX export."
    dummy_input = tokenizer(
        dummy_text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    # Export to ONNX
    print(f"\n? Exporting to ONNX...")
    torch.onnx.export(
        model,
        (dummy_input["input_ids"], dummy_input["attention_mask"]),
        str(onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size"},
        },
        opset_version=14,
        do_constant_folding=True,
    )
    print(f"   [OK] ONNX model saved to {onnx_path}")

    # Quantize (INT8 dynamic quantization for CPU)
    print(f"\n? Applying INT8 dynamic quantization...")
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType

        quantize_dynamic(
            str(onnx_path),
            str(onnx_quantized_path),
            weight_type=QuantType.QUInt8,
        )
        print(f"   [OK] Quantized model saved to {onnx_quantized_path}")

        # Compare sizes
        original_size = onnx_path.stat().st_size / (1024 * 1024)
        quantized_size = onnx_quantized_path.stat().st_size / (1024 * 1024)
        print(f"\n? Size comparison:")
        print(f"   Original:  {original_size:.1f} MB")
        print(f"   Quantized: {quantized_size:.1f} MB")
        print(f"   Reduction: {(1 - quantized_size/original_size)*100:.1f}%")

    except ImportError:
        print("   [!]  onnxruntime.quantization not available, skipping quantization")
        print("   Using non-quantized model instead")
        import shutil
        shutil.copy2(str(onnx_path), str(onnx_quantized_path))

    # Verify ONNX model
    print(f"\n? Verifying ONNX model...")
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_quantized_path))
    ort_inputs = {
        "input_ids": dummy_input["input_ids"].numpy(),
        "attention_mask": dummy_input["attention_mask"].numpy(),
    }
    ort_outputs = session.run(None, ort_inputs)

    # Compare with PyTorch output
    with torch.no_grad():
        pt_output = model(
            dummy_input["input_ids"],
            dummy_input["attention_mask"],
        )

    pt_logits = pt_output.logits.numpy()
    ort_logits = ort_outputs[0]

    max_diff = np.max(np.abs(pt_logits - ort_logits))
    print(f"   Max difference (PyTorch vs ONNX): {max_diff:.6f}")
    if max_diff < 0.01:
        print(f"   [OK] Outputs match! ONNX export verified.")
    else:
        print(f"   [!]  Some difference detected (expected with quantization)")

    # Save tokenizer alongside ONNX model
    tokenizer.save_pretrained(str(ONNX_DIR))
    print(f"\n? Tokenizer saved to {ONNX_DIR}")
    print(f"\n[OK] ONNX export complete!")


if __name__ == "__main__":
    export_to_onnx()


import torch
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from training.config import DEBERTA_DIR, ONNX_DIR

def export_to_onnx():
    print("[*] Preparing model for ONNX export...")
    
    model_path = DEBERTA_DIR / "best"
    if not model_path.exists():
        print(f"[!] Best model not found at {model_path}")
        return

    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()

    # Create dummy input for tracing
    dummy_text = "Is this health claim true?"
    inputs = tokenizer(dummy_text, return_tensors="pt")
    
    onnx_path = ONNX_DIR / "model.onnx"
    print(f"[*] Exporting to {onnx_path}...")
    
    torch.onnx.export(
        model,
        (inputs["input_ids"], inputs["attention_mask"]),
        str(onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size"},
        },
        opset_version=14,
    )
    
    print(f"\n[OK] Export complete! Model saved to {onnx_path}")
    print("[TIP] You can now use this file in your browser extension with ONNX Runtime Web.")

if __name__ == "__main__":
    export_to_onnx()

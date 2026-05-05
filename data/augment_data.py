import os
import sys
import json
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATA_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = DATA_DIR / "processed"
TRAIN_CSV = PROCESSED_DIR / "train.csv"
SYNTHETIC_CSV = PROCESSED_DIR / "synthetic_data.csv"

# Daily Target Goals (Unlimited via Local Ollama!)
SAMPLES_TO_GENERATE = {
    "Harmful": 1000,
    "Misleading": 2100,
    "Irrelevant": 2000
}

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2"  # Explicitly using 3B model to fit in 6GB VRAM

def generate_synthetic_data(original_claim: str, original_context: str, label_name: str = None) -> tuple:
    """Generate synthetic data with label-specific strategies for better quality."""
    
    if label_name == "Misleading":
        # Special handling for Misleading: emphasize partial truths and ambiguities
        prompt = f"""
    You are an expert AI data augmentor specializing in healthcare misinformation. Your task is to create a challenging variation of this MISLEADING health claim that:
    1. Contains partially true elements (like the original)
    2. Makes exaggerated or incomplete claims
    3. Lacks proper context or nuance
    4. Could confuse readers despite being somewhat factually grounded
    
    IMPORTANT RULES:
    1. Output ONLY a valid JSON object with keys: "synthetic_claim" and "synthetic_context".
    2. Do NOT output any markdown formatting or code blocks.
    3. Use single quotes (') instead of double quotes (").
    
    ORIGINAL MISLEADING CLAIM: {original_claim}
    ORIGINAL CONTEXT: {original_context[:1000]}...
    
    Create a NEW misleading claim that maintains the partially-true nature.
    """
    else:
        # Standard augmentation for other classes
        prompt = f"""
    You are an expert AI data augmentor. Your task is to paraphrase the following health claim and its context to create a new, distinct variation while preserving its original meaning and risk classification.
    
    IMPORTANT RULES:
    1. Output ONLY a valid JSON object with exactly two keys: "synthetic_claim" and "synthetic_context".
    2. Do NOT output any markdown formatting or code blocks.
    3. Do NOT use double quotes (") inside your generated text. Use single quotes (') instead.
    
    ORIGINAL CLAIM: {original_claim}
    ORIGINAL CONTEXT: {original_context[:1000]}...
    """
    
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "format": "json",
        "stream": False,
        "options": {
            "temperature": 0.7,
            "num_predict": 500,
            "num_gpu": 99
        }
    }
    
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=60)
        response.raise_for_status()
        
        raw_text = response.json()["response"].strip()
        # Clean markdown code blocks if the model hallucinates them
        if raw_text.startswith("```json"):
            raw_text = raw_text[7:]
        elif raw_text.startswith("```"):
            raw_text = raw_text[3:]
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3]
            
        result = json.loads(raw_text.strip())
        return result.get("synthetic_claim", original_claim), result.get("synthetic_context", original_context)
    except Exception as e:
        print(f"\n[!] Local API Error: {e}")
        return None, None

def get_existing_counts():
    """Reads the synthetic cache to see how much we've already generated across sessions."""
    if not SYNTHETIC_CSV.exists():
        return {}
    df = pd.read_csv(SYNTHETIC_CSV)
    return df["label_name"].value_counts().to_dict()

def main():
    print(f"[*] Loading raw dataset from {TRAIN_CSV}")
    df_raw = pd.read_csv(TRAIN_CSV)
    
    existing_counts = get_existing_counts()
    if existing_counts:
        print(f"[*] Found existing synthetic data: {existing_counts}")
        
    for label_name, target_count in SAMPLES_TO_GENERATE.items():
        already_generated = existing_counts.get(label_name, 0)
        remaining = target_count - already_generated
        
        if remaining <= 0:
            print(f"\n[*] '{label_name}' already hit target ({already_generated}/{target_count}). Skipping.")
            continue
            
        print(f"\n[*] Augmenting '{label_name}' class. Need {remaining} more samples to reach {target_count} target...")
        minority_df = df_raw[df_raw["label_name"] == label_name]
        
        if minority_df.empty:
            print(f"   [!] No raw samples found for {label_name}. Skipping.")
            continue
        
        successful_generations = 0
        attempts = 0
        
        with tqdm(total=remaining) as pbar:
            while successful_generations < remaining and attempts < (remaining * 3):
                row = minority_df.sample(1).iloc[0]
                syn_claim, syn_context = generate_synthetic_data(
                    str(row["text"]), str(row.get("main_text", "")), label_name=label_name
                )
                
                if syn_claim and syn_claim != str(row["text"]):
                    new_row = pd.DataFrame([{
                        "text": syn_claim,
                        "main_text": syn_context,
                        "label": row["label"],
                        "label_name": row["label_name"]
                    }])
                    
                    # Immediately write to disk (Crash-Proof)
                    header = not SYNTHETIC_CSV.exists()
                    new_row.to_csv(SYNTHETIC_CSV, mode='a', header=header, index=False)
                    
                    successful_generations += 1
                    pbar.update(1)
                
                attempts += 1
                
                # Notice: No more time.sleep()! Your GPU is running at max speed.

    print(f"\n[OK] Augmentation session complete. Total synthetic samples on disk: {get_existing_counts()}")

if __name__ == "__main__":
    main()

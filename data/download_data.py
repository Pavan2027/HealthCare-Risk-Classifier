"""
Download and preprocess the PUBHEALTH (health_fact) dataset.
Downloads from Google Drive, extracts TSV files, maps labels, and saves CSVs.
"""

import sys
import os
import io
import zipfile
import pandas as pd
from pathlib import Path
from urllib.request import urlopen, Request

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, LABEL_MAP
)

# Google Drive direct download URL (from HuggingFace dataset script)
GDRIVE_URL = "https://drive.google.com/uc?export=download&id=1eTtRs5cUlBP5dXsx-FTAlmXuB6JQi2qj"

# Files inside the zip
SPLIT_FILES = {
    "train": "PUBHEALTH/train.tsv",
    "validation": "PUBHEALTH/dev.tsv",
    "test": "PUBHEALTH/test.tsv",
}

# PUBHEALTH label mapping
PUBHEALTH_LABEL_TO_ID = {
    "false": 0,       # -> Harmful
    "mixture": 1,     # -> Misleading
    "true": 2,        # -> Verified
    "unproven": 3,    # -> Irrelevant
}


def download_and_process():
    """Download PUBHEALTH dataset and process it for training."""
    print("=" * 60)
    print("[*] Downloading PUBHEALTH dataset from Google Drive...")
    print("=" * 60)

    # Download zip file
    print("[*] Fetching data archive...")
    req = Request(GDRIVE_URL, headers={"User-Agent": "Mozilla/5.0"})
    response = urlopen(req)
    zip_data = response.read()
    print(f"   Downloaded {len(zip_data) / 1024 / 1024:.1f} MB")

    # Extract from zip
    zf = zipfile.ZipFile(io.BytesIO(zip_data))
    print(f"   Archive contents: {zf.namelist()[:10]}")

    for split_name, filename in SPLIT_FILES.items():
        print(f"\n[*] Processing {split_name} ({filename})...")

        try:
            with zf.open(filename) as f:
                df_raw = pd.read_csv(f, sep="\t", on_bad_lines="skip")
        except KeyError:
            print(f"   [ERROR] File {filename} not found in archive")
            # Try alternative paths
            for name in zf.namelist():
                if split_name.replace("validation", "dev") in name.lower() and name.endswith(".tsv"):
                    print(f"   [*] Trying alternative: {name}")
                    with zf.open(name) as f:
                        df_raw = pd.read_csv(f, sep="\t", on_bad_lines="skip")
                    break
            else:
                continue

        print(f"   Raw rows: {len(df_raw)}")

        # Identify columns
        text_col = "claim" if "claim" in df_raw.columns else df_raw.columns[0]
        label_col = "label" if "label" in df_raw.columns else None
        explanation_col = "explanation" if "explanation" in df_raw.columns else None
        main_text_col = "main_text" if "main_text" in df_raw.columns else None

        if label_col is None:
            print(f"   [ERROR] No 'label' column found")
            continue

        # Build clean dataframe
        df = pd.DataFrame({
            "text": df_raw[text_col].astype(str).str.strip(),
            "label_str": df_raw[label_col].astype(str).str.strip().str.lower(),
        })

        if explanation_col and explanation_col in df_raw.columns:
            df["explanation"] = df_raw[explanation_col].astype(str).str.strip()
        else:
            df["explanation"] = ""

        if main_text_col and main_text_col in df_raw.columns:
            df["main_text"] = df_raw[main_text_col].astype(str).str.strip()
        else:
            df["main_text"] = ""

        # Map string labels to numeric IDs
        df["label"] = df["label_str"].map(PUBHEALTH_LABEL_TO_ID)

        # Filter out rows with unmappable labels
        original_len = len(df)
        df = df.dropna(subset=["label"]).reset_index(drop=True)
        df["label"] = df["label"].astype(int)
        filtered = original_len - len(df)
        if filtered > 0:
            print(f"   [!] Filtered {filtered} rows with unknown labels")

        # Map to our risk labels
        df["label_name"] = df["label"].map(LABEL_MAP)

        # Remove rows with empty/short text
        df = df[df["text"].str.len() > 5].reset_index(drop=True)

        # Save raw
        raw_path = RAW_DATA_DIR / f"{split_name}.csv"
        df.to_csv(raw_path, index=False)

        # Save processed
        processed_path = PROCESSED_DATA_DIR / f"{split_name}.csv"
        df[["text", "main_text", "label", "label_name"]].to_csv(processed_path, index=False)

        print(f"   [SAVED] {split_name}: {len(df)} samples -> {processed_path}")

        # Print label distribution
        dist = df["label_name"].value_counts()
        print(f"   Distribution:")
        for label, count in dist.items():
            pct = count / len(df) * 100
            print(f"      {label}: {count} ({pct:.1f}%)")

    print("")
    print("[OK] Data preprocessing complete!")
    print(f"   Raw data: {RAW_DATA_DIR}")
    print(f"   Processed data: {PROCESSED_DATA_DIR}")


if __name__ == "__main__":
    download_and_process()

"""
PyTorch Dataset and DataLoader utilities for DistilBERT training.
Includes tokenization, weighted sampling for class imbalance, and data collation.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer
from collections import Counter

from training.config import (
    MODEL_NAME, MAX_LENGTH, BATCH_SIZE, EVAL_BATCH_SIZE,
    PROCESSED_DATA_DIR, NUM_LABELS, SEED
)


class HealthClaimDataset(Dataset):
    """PyTorch Dataset for health claim classification with sentence-pair encoding."""

    def __init__(self, claims, contexts, labels, tokenizer, max_length=MAX_LENGTH):
        self.claims = claims
        self.contexts = contexts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.claims)

    def __getitem__(self, idx):
        claim = str(self.claims[idx])
        context = str(self.contexts[idx])
        label = self.labels[idx]

        # Use sentence-pair encoding (Claim, Context)
        encoding = self.tokenizer(
            claim,
            context,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def compute_class_weights(labels):
    """
    Compute inverse-frequency class weights for weighted loss.
    weight_i = total_samples / (num_classes * count_i)
    """
    counter = Counter(labels)
    total = len(labels)
    weights = []
    for i in range(NUM_LABELS):
        count = counter.get(i, 1)  # Avoid division by zero
        weights.append(total / (NUM_LABELS * count))

    # Normalize so weights sum to NUM_LABELS
    weight_sum = sum(weights)
    weights = [w * NUM_LABELS / weight_sum for w in weights]

    return torch.tensor(weights, dtype=torch.float32)


def create_weighted_sampler(labels):
    """
    Create a WeightedRandomSampler to oversample minority classes.
    Each sample gets a weight inversely proportional to its class frequency.
    """
    counter = Counter(labels)
    total = len(labels)
    class_weights = {
        cls: total / count for cls, count in counter.items()
    }
    sample_weights = [class_weights[label] for label in labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    return sampler


def load_data(split="train"):
    """Load processed CSV data for a given split."""
    path = PROCESSED_DATA_DIR / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Data not found at {path}. Run 'python data/download_data.py' first."
        )
    df = pd.read_csv(path)
    
    # Dynamically inject synthetic data if training
    if split == "train":
        synthetic_path = PROCESSED_DATA_DIR / "synthetic_data.csv"
        if synthetic_path.exists():
            print(f"[*] Seamlessly injecting synthetic data from {synthetic_path.name}")
            syn_df = pd.read_csv(synthetic_path)
            df = pd.concat([df, syn_df], ignore_index=True)
            df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
            
    # Fill NaN main_text with empty string to avoid formatting issues
    df["main_text"] = df["main_text"].fillna("")
    
    # Return claims and contexts separately for sentence-pair encoding
    return df["text"].tolist(), df["main_text"].tolist(), df["label"].tolist()


def get_tokenizer():
    """Load the DistilBERT tokenizer."""
    return AutoTokenizer.from_pretrained(MODEL_NAME)


def get_dataloaders(tokenizer=None):
    """
    Create train, validation, and test DataLoaders.
    Training uses WeightedRandomSampler for class balance.
    """
    if tokenizer is None:
        tokenizer = get_tokenizer()

    # Load data (Unpack 3 values: claims, contexts, labels)
    train_claims, train_contexts, train_labels = load_data("train")
    val_claims, val_contexts, val_labels = load_data("validation")
    test_claims, test_contexts, test_labels = load_data("test")

    # Create datasets
    train_dataset = HealthClaimDataset(train_claims, train_contexts, train_labels, tokenizer)
    val_dataset = HealthClaimDataset(val_claims, val_contexts, val_labels, tokenizer)
    test_dataset = HealthClaimDataset(test_claims, test_contexts, test_labels, tokenizer)

    # Weighted sampler for training (oversampling minority classes)
    train_sampler = create_weighted_sampler(train_labels)

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # Compute class weights for loss function
    class_weights = compute_class_weights(train_labels)

    print(f"[*] DataLoaders created:")
    print(f"   Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"   Val:   {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"   Test:  {len(test_dataset)} samples, {len(test_loader)} batches")
    print(f"   Class weights: {class_weights.tolist()}")

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "class_weights": class_weights,
        "tokenizer": tokenizer,
    }

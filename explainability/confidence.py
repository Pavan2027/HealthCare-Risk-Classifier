"""
Confidence scoring and temperature calibration for predictions.
Ensures model confidence is well-calibrated (matches actual accuracy).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
from scipy.optimize import minimize_scalar

from training.config import LABEL_MAP, CONFIDENCE_THRESHOLD


class ConfidenceScorer:
    """
    Calibrate and score model prediction confidence.
    Uses temperature scaling fitted on validation set.
    """

    def __init__(self, temperature=1.0):
        self.temperature = temperature

    def calibrate(self, logits, labels):
        """
        Find optimal temperature via validation set.
        Minimizes negative log-likelihood on validation data.

        Args:
            logits: Raw model logits (numpy array, shape [N, C])
            labels: True labels (numpy array, shape [N])
        """
        logits_tensor = torch.tensor(logits, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.long)

        def nll_with_temp(temp):
            temp = max(temp, 0.01)  # Avoid division by zero
            scaled = logits_tensor / temp
            log_probs = torch.log_softmax(scaled, dim=-1)
            nll = -log_probs[
                torch.arange(len(labels_tensor)), labels_tensor
            ].mean()
            return nll.item()

        # Optimize temperature
        result = minimize_scalar(
            nll_with_temp,
            bounds=(0.1, 10.0),
            method="bounded",
        )
        self.temperature = result.x
        print(f"??  Calibrated temperature: {self.temperature:.4f}")
        return self.temperature

    def score(self, logits):
        """
        Compute calibrated confidence from raw logits.

        Args:
            logits: Raw model logits (numpy array or tensor)

        Returns:
            dict with label, confidence, all probabilities, and confidence level
        """
        if isinstance(logits, np.ndarray):
            logits = torch.tensor(logits, dtype=torch.float32)

        if logits.dim() == 1:
            logits = logits.unsqueeze(0)

        # Apply temperature scaling
        scaled_logits = logits / self.temperature
        probs = torch.softmax(scaled_logits, dim=-1).squeeze()

        predicted_id = torch.argmax(probs).item()
        confidence = probs[predicted_id].item()

        # Determine confidence level
        if confidence >= 0.85:
            confidence_level = "high"
        elif confidence >= 0.6:
            confidence_level = "medium"
        else:
            confidence_level = "low"

        return {
            "label": LABEL_MAP[predicted_id],
            "label_id": predicted_id,
            "confidence": round(confidence, 4),
            "confidence_level": confidence_level,
            "all_probabilities": {
                LABEL_MAP[i]: round(probs[i].item(), 4)
                for i in range(len(LABEL_MAP))
            },
            "is_high_confidence": confidence >= CONFIDENCE_THRESHOLD,
        }

    def batch_score(self, logits_batch):
        """Score a batch of logits."""
        results = []
        if isinstance(logits_batch, np.ndarray):
            logits_batch = torch.tensor(logits_batch, dtype=torch.float32)

        for logits in logits_batch:
            results.append(self.score(logits))

        return results

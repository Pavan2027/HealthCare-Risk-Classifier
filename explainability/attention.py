"""
Attention-based explainability for DistilBERT predictions.
Extracts attention weights to identify which tokens influenced the classification.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from training.config import DISTILBERT_DIR, MAX_LENGTH, LABEL_MAP


class AttentionExplainer:
    """
    Extract attention-based explanations from DistilBERT.
    Uses the last attention layer's weights to identify important tokens.
    """

    def __init__(self, model=None, tokenizer=None, model_path=None):
        if model is not None and tokenizer is not None:
            self.model = model
            self.tokenizer = tokenizer
        else:
            model_path = model_path or str(DISTILBERT_DIR / "best")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_path, output_attentions=True
            )

        self.model.eval()
        self.device = next(self.model.parameters()).device

    def get_attention_scores(self, text, top_k=10):
        """
        Extract attention-based token importance scores.

        Args:
            text: Input text to analyze
            top_k: Number of top important tokens to return

        Returns:
            dict with prediction, important words, and attention details
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Forward pass with attention
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)

        logits = outputs.logits
        attentions = outputs.attentions  # Tuple of (batch, heads, seq, seq)

        # Get prediction
        probs = torch.softmax(logits, dim=-1).squeeze()
        predicted_label_id = torch.argmax(probs).item()
        confidence = probs[predicted_label_id].item()

        # Extract attention from the last layer
        # Shape: (batch, num_heads, seq_len, seq_len)
        last_layer_attention = attentions[-1].squeeze(0)  # (heads, seq, seq)

        # Average across all attention heads
        avg_attention = last_layer_attention.mean(dim=0)  # (seq, seq)

        # Get attention from [CLS] token to all other tokens
        # [CLS] is at index 0 and aggregates information for classification
        cls_attention = avg_attention[0]  # (seq,)

        # Get tokens
        tokens = self.tokenizer.convert_ids_to_tokens(
            inputs["input_ids"].squeeze()
        )
        attention_mask = inputs["attention_mask"].squeeze()

        # Filter out special tokens and padding
        token_scores = []
        for i, (token, attn, mask) in enumerate(
            zip(tokens, cls_attention, attention_mask)
        ):
            if mask.item() == 0:
                continue  # Skip padding
            if token in ["[CLS]", "[SEP]", "[PAD]"]:
                continue  # Skip special tokens

            # Clean up subword tokens
            clean_token = token.replace("##", "")
            if len(clean_token) <= 1:
                continue  # Skip single characters

            token_scores.append({
                "token": clean_token,
                "score": attn.item(),
                "position": i,
            })

        # Sort by attention score (descending)
        token_scores.sort(key=lambda x: x["score"], reverse=True)

        # Merge subword tokens and aggregate scores
        merged_words = self._merge_subwords(tokens, cls_attention, attention_mask)
        merged_words.sort(key=lambda x: x["score"], reverse=True)

        # Get top-K important words
        top_words = merged_words[:top_k]

        return {
            "text": text,
            "label": LABEL_MAP[predicted_label_id],
            "label_id": predicted_label_id,
            "confidence": round(confidence, 4),
            "all_probabilities": {
                LABEL_MAP[i]: round(probs[i].item(), 4)
                for i in range(len(LABEL_MAP))
            },
            "important_words": [
                {"word": w["word"], "score": round(w["score"], 4)}
                for w in top_words
            ],
            "token_scores": token_scores[:20],  # Raw token scores
        }

    def _merge_subwords(self, tokens, attention_scores, attention_mask):
        """
        Merge WordPiece subword tokens back into full words
        and aggregate their attention scores.
        """
        words = []
        current_word = ""
        current_score = 0.0
        token_count = 0

        for i, (token, score, mask) in enumerate(
            zip(tokens, attention_scores, attention_mask)
        ):
            if mask.item() == 0:
                break
            if token in ["[CLS]", "[SEP]", "[PAD]"]:
                if current_word:
                    words.append({
                        "word": current_word,
                        "score": current_score / max(token_count, 1),
                    })
                    current_word = ""
                    current_score = 0.0
                    token_count = 0
                continue

            if token.startswith("##"):
                # Subword continuation
                current_word += token[2:]
                current_score += score.item()
                token_count += 1
            else:
                # New word
                if current_word:
                    words.append({
                        "word": current_word,
                        "score": current_score / max(token_count, 1),
                    })
                current_word = token
                current_score = score.item()
                token_count = 1

        # Don't forget the last word
        if current_word:
            words.append({
                "word": current_word,
                "score": current_score / max(token_count, 1),
            })

        return words

    def highlight_text(self, text, top_k=5):
        """
        Return text with important words marked for display.
        Useful for the browser extension overlay.
        """
        result = self.get_attention_scores(text, top_k=top_k)
        important_words = {
            w["word"].lower() for w in result["important_words"]
        }

        # Create word-level highlights
        words = text.split()
        highlighted = []
        for word in words:
            clean = word.lower().strip(".,!?;:\"'()[]")
            if clean in important_words:
                highlighted.append({"word": word, "highlighted": True})
            else:
                highlighted.append({"word": word, "highlighted": False})

        result["highlighted_words"] = highlighted
        return result

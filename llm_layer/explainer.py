"""
LLM-based reasoning layer for generating human-readable explanations.
Uses Groq API (fast inference) with fallback to rule-based explanations.
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

from training.config import LABEL_MAP, LABEL_DESCRIPTIONS

load_dotenv()


# --- In-memory cache for LLM responses --------------------------------------
_explanation_cache = {}


def _get_rule_based_explanation(text, label, confidence, important_words=None):
    """
    Generate a rule-based explanation when LLM API is unavailable.
    Provides useful fallback explanations based on label and keywords.
    """
    words_str = ""
    if important_words:
        word_list = [w["word"] if isinstance(w, dict) else w for w in important_words[:5]]
        words_str = f" Key indicators: {', '.join(word_list)}."

    explanations = {
        "Harmful": (
            f"This content has been classified as HARMFUL with {confidence:.0%} confidence. "
            f"It contains claims that contradict established medical evidence and could lead to "
            f"dangerous health decisions if followed.{words_str} "
            f"Please consult qualified healthcare professionals for medical advice."
        ),
        "Misleading": (
            f"This content has been classified as MISLEADING with {confidence:.0%} confidence. "
            f"While it may contain some factual elements, the overall claim is partially "
            f"inaccurate or presented in a way that could lead to incorrect conclusions.{words_str} "
            f"Cross-reference with trusted medical sources for accuracy."
        ),
        "Verified": (
            f"This content has been classified as VERIFIED with {confidence:.0%} confidence. "
            f"The claim aligns with current medical evidence and established scientific consensus.{words_str}"
        ),
        "Irrelevant": (
            f"This content has been classified as IRRELEVANT/UNVERIFIABLE with {confidence:.0%} "
            f"confidence. The claim lacks sufficient evidence to verify or disprove, "
            f"or is not directly related to actionable health information.{words_str}"
        ),
    }

    return explanations.get(label, f"Classification: {label} ({confidence:.0%})")


async def get_llm_explanation(text, label, confidence, important_words=None):
    """
    Generate an LLM-powered explanation for a classification result.
    Uses Groq with caching and fallback.

    Args:
        text: The classified text
        label: Predicted label
        confidence: Confidence score
        important_words: List of important words/dicts from attention

    Returns:
        str: Human-readable explanation
    """
    # Check cache
    cache_key = f"{text[:100]}_{label}_{confidence:.2f}"
    if cache_key in _explanation_cache:
        return _explanation_cache[cache_key]

    # Prepare important words string
    if important_words:
        words = [w["word"] if isinstance(w, dict) else w for w in important_words[:5]]
        words_str = ", ".join(words)
    else:
        words_str = "N/A"

    # Try Groq API
    api_key = os.getenv("GROQ_API_KEY")
    if api_key and api_key != "your_gemini_api_key" and api_key != "your_groq_api_key":
        try:
            from groq import AsyncGroq

            client = AsyncGroq(api_key=api_key)
            model_name = "llama-3.3-70b-versatile"

            prompt = (
                f"You are a medical fact-checking assistant. A health claim has been "
                f"classified as [{label}] with {confidence:.0%} confidence.\n\n"
                f'Claim: "{text}"\n'
                f"Key indicators: {words_str}\n\n"
                f"Provide a concise 2-3 sentence explanation of why this classification "
                f"is appropriate, referencing specific medical/scientific reasoning. "
                f"Do NOT use markdown formatting. Be direct and informative."
            )

            response = await client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful medical fact-checker."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_completion_tokens=150,
            )
            
            explanation = response.choices[0].message.content.strip()

            # Cache the result
            _explanation_cache[cache_key] = explanation
            return explanation

        except Exception as e:
            print(f"[*] Groq API error: {e}. Using rule-based fallback.")

    # Fallback to rule-based
    explanation = _get_rule_based_explanation(
        text, label, confidence, important_words
    )
    _explanation_cache[cache_key] = explanation
    return explanation


def get_explanation_sync(text, label, confidence, important_words=None):
    """
    Synchronous wrapper for get_llm_explanation.
    Use this when async is not available.
    """
    import asyncio

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Already in an async context, use rule-based
            return _get_rule_based_explanation(
                text, label, confidence, important_words
            )
        return loop.run_until_complete(
            get_llm_explanation(text, label, confidence, important_words)
        )
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                get_llm_explanation(text, label, confidence, important_words)
            )
        finally:
            loop.close()


def clear_cache():
    """Clear the explanation cache."""
    global _explanation_cache
    _explanation_cache = {}

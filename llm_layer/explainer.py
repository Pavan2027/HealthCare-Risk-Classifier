"""
LLM-based reasoning layer for generating human-readable explanations.
Uses Groq API (fast inference) with fallback to rule-based explanations.
"""

import sys
import os
import json
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
    Generate an LLM-powered explanation and cross-check the classification.
    Returns: (explanation, secondary_label)
    """
    # Check cache
    cache_key = f"{text[:100]}_{label}_{confidence:.2f}"
    if cache_key in _explanation_cache:
        return _explanation_cache[cache_key]

    # Configuration from .env
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    model_name = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
    api_base = os.getenv("LLM_API_BASE")
    api_key = os.getenv("LLM_API_KEY")

    if api_key and api_key != "your_api_key_here":
        try:
            from openai import AsyncOpenAI
            
            # Use context manager to ensure client closure
            async with AsyncOpenAI(api_key=api_key, base_url=api_base) as client:
                prompt = (
                    f"You are a medical fact-checking assistant.\n"
                    f"A primary model classified this claim as [{label}] with {confidence:.0%} confidence.\n\n"
                    f"Claim: \"{text}\"\n\n"
                    f"Task:\n"
                    f"1. Provide a concise 2-sentence medical explanation.\n"
                    f"2. Independent Labeling: Classify this as [Harmful, Misleading, Verified, or Irrelevant].\n\n"
                    f"Format: JSON ONLY with keys 'explanation' and 'llm_label'."
                )

                response = await client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are a professional medical fact-checker. Respond in valid JSON only."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    response_format={"type": "json_object"}
                )
                
                res_content = json.loads(response.choices[0].message.content)
                explanation = res_content.get("explanation", "No explanation provided.")
                llm_label = res_content.get("llm_label", label)

                result = (explanation, llm_label)
                _explanation_cache[cache_key] = result
                return result

        except Exception as e:
            print(f"[*] {provider.upper()} API error: {e}. Falling back to rule-based.")

    # Fallback
    explanation = _get_rule_based_explanation(text, label, confidence, important_words)
    result = (explanation, label) # Secondary label is same as primary in fallback
    _explanation_cache[cache_key] = result
    return result


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

"""LLM-based translation with multiple provider support."""
from __future__ import annotations
import os
import json
import logging
import httpx

log = logging.getLogger(__name__)


def _get_proxy():
    return os.environ.get("MANGA_PROXY") or os.environ.get("HTTP_PROXY") or os.environ.get("HTTPS_PROXY")


def translate_with_llm(
    texts: list[str],
    source_lang: str = "ja",
    target_lang: str = "ru",
    provider: str = "auto",
) -> list[str]:
    """Translate texts using LLM. Returns list of same length as input."""
    if not texts:
        return []

    # Auto-select provider
    if provider == "auto":
        if os.environ.get("GROQ_API_KEY"):
            provider = "groq"
        elif os.environ.get("GEMINI_API_KEY"):
            provider = "gemini"
        elif os.environ.get("OPENROUTER_API_KEY"):
            provider = "openrouter"
        else:
            log.warning("No API key found, returning originals")
            return texts

    # Build prompt
    lang_names = {"ja": "Japanese", "en": "English", "ko": "Korean", "zh": "Chinese", "ru": "Russian"}
    src_name = lang_names.get(source_lang, source_lang)
    tgt_name = lang_names.get(target_lang, target_lang)

    numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))
    prompt = f"""Translate the following {src_name} manga dialogue lines to natural {tgt_name}.
Keep the tone and style of the original. Return ONLY a JSON array of strings, same order, no explanation.

{numbered}"""

    try:
        if provider == "groq":
            result = _call_groq(prompt)
        elif provider == "gemini":
            result = _call_gemini(prompt)
        elif provider == "openrouter":
            result = _call_openrouter(prompt)
        else:
            return texts

        # Parse JSON array from response
        translations = _extract_json_array(result)
        if translations and len(translations) == len(texts):
            return translations
        log.warning("Translation count mismatch: got %d, expected %d", len(translations), len(texts))
        return texts
    except Exception as e:
        log.error("Translation failed: %s", e)
        return texts


def _call_groq(prompt: str) -> str:
    api_key = os.environ["GROQ_API_KEY"]
    proxy = _get_proxy()
    with httpx.Client(proxy=proxy, timeout=30) as c:
        r = c.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
            },
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


def _call_gemini(prompt: str) -> str:
    api_key = os.environ["GEMINI_API_KEY"]
    proxy = _get_proxy()
    with httpx.Client(proxy=proxy, timeout=30) as c:
        r = c.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-lite:generateContent?key={api_key}",
            json={"contents": [{"parts": [{"text": prompt}]}]},
        )
        r.raise_for_status()
        return r.json()["candidates"][0]["content"]["parts"][0]["text"]


def _call_openrouter(prompt: str) -> str:
    api_key = os.environ["OPENROUTER_API_KEY"]
    proxy = _get_proxy()
    with httpx.Client(proxy=proxy, timeout=30) as c:
        r = c.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "google/gemini-2.0-flash-001",
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


def _extract_json_array(text: str) -> list[str]:
    """Extract JSON array from LLM response."""
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find array in text
    import re
    match = re.search(r'\[[\s\S]*?\]', text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return []

"""LLM-curated glossary built from chapter translation pairs using Groq (free).

Refines the naive frequency-based glossary in cfg/memory.py: instead of only
counting ko->ru pairs, a free LLM (Groq, reachable without a proxy) extracts
character names and recurring terms, maps Korean spelling to the Russian names
used, and merges them into cfg/glossary.json for stable terminology.
"""

import json
import os
import re

import httpx

from .log import log
from cfg import GROQ_API_KEY, GLOSSARY, GLOSSARY_PATH

GLOSSARY_PROMPT = """You are a manga glossarist working on a Korean (ko) -> Russian (ru) scanlation.
From the list of Korean source bubbles and their Russian translations, extract a glossary.

Return ONLY JSON (no markdown fences, no commentary):
{"characters": {"korean_spelling": "russian_name", ...}, "terms": {"korean_phrase": "russian_translation", ...}}

Rules:
- characters: proper names of people. Map the Korean spelling to the Russian name actually used in the translations.
- terms: recurring story nouns — places, organizations, techniques, systems, titles, honorifics.
- Every Korean key MUST appear verbatim in at least one source line below.
- Russian values must be short (1-4 words) and consistent with the translations.
- Skip generic words, single dashes, and OCR garbage.
- If nothing qualifies, return {"characters": {}, "terms": {}}."""


def _sample_pairs(pairs, limit=80):
    seen = set()
    out = []
    for p in pairs:
        ko = (p.get("ko") or "").strip()
        ru = (p.get("ru") or "").strip()
        if not ko or not ru:
            continue
        key = (ko, ru)
        if key in seen:
            continue
        seen.add(key)
        out.append((ko, ru))
        if len(out) >= limit:
            break
    return out


def _parse_glossary_json(text: str) -> dict:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    data = None
    try:
        data = json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                data = json.loads(m.group(0))
            except Exception:
                data = None
    if not isinstance(data, dict):
        return {"characters": {}, "terms": {}}
    chars = data.get("characters") or {}
    terms = data.get("terms") or {}
    return {
        "characters": chars if isinstance(chars, dict) else {},
        "terms": terms if isinstance(terms, dict) else {},
    }


def _valid_entry(ko, ru, corpus):
    ko = (ko or "").strip()
    ru = (ru or "").strip()
    if not ko or not ru or len(ru) > 40:
        return None
    if len(ko) > 30:
        return None
    if not any(ko in line for line in corpus):
        return None
    return (ko, ru)


def _save_glossary():
    try:
        tmp = GLOSSARY_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(GLOSSARY, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(GLOSSARY_PATH)
    except Exception as e:
        log.warning("Glossary save failed: %s", e)


def merge_glossary(new: dict) -> int:
    """Merge extracted entries into the global cfg GLOSSARY. Returns count added."""
    chars = GLOSSARY.setdefault("characters", {})
    terms = GLOSSARY.setdefault("terms", {})
    added = 0
    for section, store in (("characters", chars), ("terms", terms)):
        for ko, ru in (new.get(section) or {}).items():
            ko = (ko or "").strip()
            ru = (ru or "").strip()
            if not ko or not ru:
                continue
            if store.get(ko) != ru:
                store[ko] = ru
                added += 1
    if added:
        _save_glossary()
    return added


async def build_glossary(pairs: list[dict], source_lang: str = "ko", max_samples: int = 80) -> dict:
    """Extract a curated glossary from translation pairs via Groq (free)."""
    samples = _sample_pairs(pairs, max_samples)
    if not samples or not GROQ_API_KEY:
        return {}
    corpus = [ko for ko, _ in samples]
    body = "\n".join(f"[{i + 1}] {ko} -> {ru}" for i, (ko, ru) in enumerate(samples))
    prompt = GLOSSARY_PROMPT + f"\n\nSOURCE->RU PAIRS:\n{body}"
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 2048,
    }
    client_kwargs = {"timeout": 60.0}
    proxy = os.environ.get("HTTP_PROXY") or os.environ.get("HTTPS_PROXY") or os.environ.get("http_proxy") or os.environ.get("https_proxy")
    if proxy:
        client_kwargs["proxy"] = proxy
        client_kwargs["verify"] = False
    try:
        async with httpx.AsyncClient(**client_kwargs) as client:
            resp = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
                json=payload,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        log.warning("Glossary LLM call failed: %s", e)
        return {}

    parsed = _parse_glossary_json(content)
    cleaned = {"characters": {}, "terms": {}}
    for section in ("characters", "terms"):
        for ko, ru in parsed[section].items():
            ent = _valid_entry(ko, ru, corpus)
            if ent:
                cleaned[section][ent[0]] = ent[1]
    log.info(
        "Glossary LLM: extracted %d characters, %d terms",
        len(cleaned["characters"]),
        len(cleaned["terms"]),
    )
    return cleaned
import asyncio
import json
import os
import httpx
from cfg import GLOSSARY, CONFIG, COLAB_URL, OPENROUTER_API_KEY, GROQ_API_KEY, GEMINI_API_KEY

from cfg.memory import get_context as get_memory_context, get_glossary as get_memory_glossary
def _get_proxy():
    return (
        os.environ.get("HTTP_PROXY")
        or os.environ.get("HTTPS_PROXY")
        or os.environ.get("http_proxy")
        or os.environ.get("https_proxy")
    )


def _make_client(timeout=60.0):
    proxy = _get_proxy()
    if proxy:
        return httpx.AsyncClient(timeout=timeout, proxy=proxy, verify=False)
    return httpx.AsyncClient(timeout=timeout)


LANG_NAMES = {
    "ko": ("Korean", "корейский"),
    "en": ("English", "английский"),
    "ja": ("Japanese", "японский"),
    "zh": ("Chinese", "китайский"),
}


def _system_prompt(source_lang: str = "ko") -> str:
    src_name, tgt_name = LANG_NAMES.get(source_lang, ("Korean", "русский"))
    tgt_ru = "русский"
    return f"""You are a professional manga/manhwa translator ({src_name}→{tgt_ru}).

Translate each numbered {src_name} text bubble into natural {tgt_ru}.

CRITICAL — ANTI-HALLUCINATION RULES:
1. OCR output may contain garbage characters. Distinguish real words from OCR noise (random chars like "DF", "VV", "ABHCH", "asdkj", single letters alone).
2. If part is unreadable garbage — DISCARD it completely. Do NOT invent/hallucinate a meaning for nonsense.
3. If a bubble has only garbage/no real words, return it as an empty string "".
4. If original text is garbled but you can guess the intended word, translate your best guess.
5. DO NOT invent plot info that is not in the text.

RULES:
1. Each translation must be 1-2 short sentences max. Keep it brief.
2. Use natural conversational {tgt_ru}. Short words, simple grammar.
3. If a bubble has 1-2 words, expand into a short natural phrase from context.
4. Character speech must match their personality (polite, rude, excited, etc.)
5. SFX (sound effects like ヒヒ, バキ, DOOM, WHOOSH, etc.) — DO NOT translate. Return original text as-is.
6. If original is {src_name} with real words — NEVER return empty text or original. If original is garbage OCR noise — return empty string "".
7. DO NOT add narrator marks, quotes, or explanations.

CRITICAL — length limit: maximum 50 characters per bubble. Shorter is better.

Respond ONLY with JSON array:
[{{"id": 1, "ru": "translation"}}, {{"id": 2, "ru": "translation"}}, ...]"""


def _build_prompt(korean_texts, english_texts, page_number, context, glossary, memory_context="", memory_glossary=None, source_lang="ko"):
    parts = []
    if memory_context:
        parts.append("PREVIOUS CHAPTERS (translations of this manga from earlier chapters):\n" + memory_context)
    if context:
        parts.append("STORY SO FAR (Korean texts from previous pages for context):\n" + "\n".join(context[-5:]))
    parts.append(f"=== PAGE {page_number} ===")
    if memory_glossary:
        lines = ["Known terms (Korean -> Russian) from previous chapters — MUST USE these translations:"]
        for ko, ru in memory_glossary.items():
            lines.append(f"  {ko} → {ru}")
        parts.append("\n".join(lines))
    if glossary:
        chars = glossary.get("characters", {})
        terms = glossary.get("terms", {})
        lines = []
        if chars:
            lines.append("Characters (always use these names):")
            for ko, ru in chars.items():
                lines.append(f"  {ko} → {ru}")
        if terms:
            lines.append("Terms (always use these translations):")
            for ko, ru in terms.items():
                lines.append(f"  {ko} → {ru}")
        if lines:
            parts.append("GLOSSARY:\n" + "\n".join(lines))
            parts.append("RULE: If a Korean term appears in GLOSSARY above, you MUST use the listed Russian translation. Do NOT translate it differently.")
    if english_texts:
        en_block = "\n".join(f"[{i+1}] {t}" for i, t in enumerate(english_texts))
        parts.append("ENGLISH REFERENCE (for context):\n" + en_block)
    texts_block = json.dumps(
        [{f"bubble_{i+1}": t} for i, t in enumerate(korean_texts)],
        ensure_ascii=False, indent=2,
    )
    parts.append("TEXTS TO TRANSLATE (speech bubbles on this page):\n" + texts_block)
    return "\n\n".join(parts)


def _parse_json_response(text: str) -> list[dict] | None:
    try:
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(l for l in lines if not l.startswith("```"))
        text = text.strip()
        if text.startswith("json"):
            text = text[4:].strip()
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return None


class LLMTranslator:
    def __init__(self):
        self.colab_url = (COLAB_URL or "").rstrip("/")
        self.colab_available = False
        self.context_pages: list[str] = []
        self.colab_client = None
        self.model = CONFIG["llm"]["model"]
        self.temperature = CONFIG["llm"]["temperature"]
        self._memory_context = ""
        self._memory_glossary = {}

        if self.colab_url:
            self._test_colab()

        self._providers = []
        self._init_providers()
        self._print_chain()

    def _test_colab(self):
        try:
            kwargs = {"timeout": 5.0}
            proxy = _get_proxy()
            if proxy:
                kwargs["proxy"] = proxy
            resp = httpx.get(f"{self.colab_url}/health", **kwargs)
            if resp.status_code == 200:
                self.colab_available = True
                self.colab_client = _make_client(60.0)
                print(f"[LLM] Colab OK — {self.colab_url}")
            else:
                print(f"[LLM] Colab health check returned {resp.status_code}, skipping")
        except Exception:
            print("[LLM] Colab unreachable, skipping")

    def _init_providers(self):
        if GEMINI_API_KEY:
            self._providers.append(("Gemini 2.0 Flash", self._call_gemini))
        if self.colab_available:
            self._providers.append(("Colab Server", self._call_colab))
        if GROQ_API_KEY:
            self._providers.append(("Groq (Llama 70B)", self._call_groq))
        if OPENROUTER_API_KEY:
            self._providers.append(("OpenRouter Free", self._call_openrouter_free))
        if OPENROUTER_API_KEY:
            self._providers.append(("OpenRouter Paid", self._call_openrouter_paid))
        self._providers.append(("deep-translator", self._call_fallback))

    def _print_chain(self):
        names = [p[0] for p in self._providers]
        print(f"[LLM] Chain: {' -> '.join(names)}")

    def clear_context(self):
        self.context_pages.clear()

    def add_context(self, page_texts: list[str]):
        self.context_pages.append("\n".join(page_texts))
        if len(self.context_pages) > 10:
            self.context_pages = self.context_pages[-10:]

    def _build_glossary_dict(self) -> dict:
        glossary = {"characters": {}, "terms": {}}
        for ko, ru in GLOSSARY.get("characters", {}).items():
            glossary["characters"][ko] = ru
        for ko, ru in GLOSSARY.get("terms", {}).items():
            glossary["terms"][ko] = ru
        return glossary

    async def _call_openrouter(self, korean_texts, english_texts, page_number, model, timeout=120.0) -> list[dict] | None:
        sl = getattr(self, '_source_lang', 'ko')
        prompt = _build_prompt(korean_texts, english_texts, page_number, self.context_pages, self._build_glossary_dict(), self._memory_context, self._memory_glossary, source_lang=sl)
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": _system_prompt(sl)},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": 4096,
        }
        async with _make_client(timeout) as client:
            resp = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            if resp.status_code == 429:
                return None
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            result = _parse_json_response(content)
            if result and len(result) == len(korean_texts):
                return result
            return None

    async def _call_openrouter_free(self, korean_texts, english_texts, page_number) -> list[dict] | None:
        try:
            return await asyncio.wait_for(
                self._call_openrouter(korean_texts, english_texts, page_number, "openrouter/free"),
                timeout=30.0
            )
        except asyncio.TimeoutError:
            return None

    async def _call_openrouter_paid(self, korean_texts, english_texts, page_number) -> list[dict] | None:
        return await self._call_openrouter(korean_texts, english_texts, page_number, self.model)

    async def _call_groq(self, korean_texts, english_texts, page_number) -> list[dict] | None:
        sl = getattr(self, '_source_lang', 'ko')
        prompt = _build_prompt(korean_texts, english_texts, page_number, self.context_pages, self._build_glossary_dict(), self._memory_context, self._memory_glossary, source_lang=sl)
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {"role": "system", "content": _system_prompt(sl)},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": 4096,
        }
        async with _make_client(120.0) as client:
            resp = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            if resp.status_code == 429:
                return None
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            result = _parse_json_response(content)
            if result and len(result) == len(korean_texts):
                return result
            return None

    async def _call_gemini(self, korean_texts, english_texts, page_number) -> list[dict] | None:
        sl = getattr(self, '_source_lang', 'ko')
        prompt = _build_prompt(korean_texts, english_texts, page_number, self.context_pages, self._build_glossary_dict(), self._memory_context, self._memory_glossary, source_lang=sl)
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
            "systemInstruction": {
                "parts": [{"text": _system_prompt(sl)}],
            },
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": 4096,
            },
        }
        async with _make_client(60.0) as client:
            resp = await client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}",
                headers={"Content-Type": "application/json"},
                json=payload,
            )
            if resp.status_code == 429:
                return None
            if resp.status_code == 403:
                print("  [LLM] Gemini → 403 (quota/blocked), skipping")
                return None
            resp.raise_for_status()
            data = resp.json()
            candidates = data.get("candidates", [])
            if not candidates:
                return None
            content = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            result = _parse_json_response(content)
            if result and len(result) == len(korean_texts):
                return result
            return None

    async def _call_colab(self, korean_texts, english_texts, page_number) -> list[dict] | None:
        try:
            payload = {
                "korean_texts": korean_texts,
                "english_texts": english_texts or [],
                "page_number": page_number,
                "model": self.model,
                "temperature": self.temperature,
                "context": {
                    "glossary": self._build_glossary_dict(),
                    "previous_pages": self.context_pages[-5:],
                },
            }
            resp = await self.colab_client.post(f"{self.colab_url}/translate", json=payload, timeout=60.0)
            resp.raise_for_status()
            data = resp.json()
            return data.get("translations", [])
        except Exception:
            return None

    async def _call_fallback(self, korean_texts, english_texts=None, page_number=0) -> list[dict]:
        try:
            from deep_translator import GoogleTranslator
            sl = getattr(self, '_source_lang', 'ko')
            deepl_src = {"ko": "ko", "en": "en", "ja": "ja", "zh": "zh-CN"}.get(sl, "ko")
            translator = GoogleTranslator(source=deepl_src, target="ru")
        except ImportError:
            print("[LLM] deep-translator not installed, returning originals")
            return [{"id": i + 1, "ru": text} for i, text in enumerate(korean_texts)]
        results = []
        for i, text in enumerate(korean_texts):
            try:
                translated = translator.translate(text)
                results.append({"id": i + 1, "ru": translated})
            except Exception:
                results.append({"id": i + 1, "ru": text})
        return results

    async def translate_page(self, korean_texts, english_texts=None, page_number=1, manga_id=None, chapter=None, source_lang="ko") -> list[dict]:
        if not korean_texts:
            return []
        self._source_lang = source_lang

        self._memory_context = get_memory_context(manga_id) if manga_id else ""
        self._memory_glossary = get_memory_glossary(manga_id) if manga_id else {}

        for name, call_fn in self._providers:
            try:
                result = await call_fn(korean_texts, english_texts or [], page_number)
                if result is None:
                    print(f"  [LLM] {name} → rate limited, next...")
                    continue
                self.add_context(korean_texts)
                for i, ko in enumerate(korean_texts):
                    if i < len(result):
                        result[i]["ko"] = ko
                result = self._apply_post_replace(result)
                return result
            except Exception as e:
                print(f"  [LLM] {name} → error: {e}")
                continue

        print(f"  [LLM] All providers failed, returning originals")
        return [{"id": i + 1, "ko": t, "ru": t} for i, t in enumerate(korean_texts)]

    async def translate_sfx(self, korean_sfx, english_sfx="") -> str:
        if english_sfx:
            return english_sfx
        for name, call_fn in self._providers:
            if name == "deep-translator":
                continue
            try:
                result = await call_fn([korean_sfx], [], 0)
                if result and result[0].get("ru"):
                    return result[0]["ru"]
            except Exception:
                continue
        return korean_sfx

    def _apply_post_replace(self, result: list[dict]) -> list[dict]:
        glossary = CONFIG.get("translation", {}).get("post_replace", {})
        if not glossary or not result:
            return result
        for entry in result:
            ru = entry.get("ru", "")
            if not ru:
                continue
            for ko, ru_replacement in glossary.items():
                ru = ru.replace(ko, ru_replacement)
            entry["ru"] = ru
        return result

    async def close(self):
        if self.colab_client:
            await self.colab_client.aclose()

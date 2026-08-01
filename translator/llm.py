import asyncio
import json
import os
import hashlib
import logging
import re
import httpx
import time
import random
from functools import wraps
from cfg import GLOSSARY, CONFIG, COLAB_URL, OPENROUTER_API_KEY, GROQ_API_KEY, GEMINI_API_KEY

from cfg.memory import get_context as get_memory_context, get_glossary as get_memory_glossary, get_character_profiles
from translator.rag import RAGIndex, load_translations_from_memory
from translator.validator import validate_translation, fix_translation
from .log import log


def retry_async(max_attempts=3, base_delay=1.0, max_delay=10.0, jitter=True):
    """
    Retry decorator for async functions with exponential backoff.
    
    Args:
        max_attempts: Maximum number of attempts
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        jitter: Whether to add random jitter to delay
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt == max_attempts - 1:  # Last attempt
                        break
                    
                    # Calculate delay with exponential backoff
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    if jitter:
                        delay *= (0.5 + random.random() * 0.5)  # 0.5 to 1.0 multiplier
                    
                    await asyncio.sleep(delay)
            
            raise last_exception
        return wrapper
    return decorator


class RateLimiter:
    """Token bucket rate limiter for async functions."""
    def __init__(self, rate: float, capacity: int = 1):
        self.rate = rate  # tokens per second
        self.capacity = capacity  # max tokens
        self.tokens = capacity
        self.last_refill = time.monotonic()
        self._lock = asyncio.Lock()
    
    async def acquire(self):
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_refill
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self.last_refill = now
            if self.tokens >= 1:
                self.tokens -= 1
                return True
            wait_time = (1 - self.tokens) / self.rate
            return False
    
    async def wait(self):
        while True:
            if await self.acquire():
                return
            wait_time = (1 - self.tokens) / self.rate
            await asyncio.sleep(wait_time)


class CircuitBreaker:
    """Circuit breaker pattern to prevent cascading failures."""
    def __init__(self, max_failures: int = 3, reset_timeout: float = 30.0):
        self.max_failures = max_failures
        self.reset_timeout = reset_timeout
        self.failures = 0
        self.last_failure_time = 0
        self.state = "CLOSED"
        self._lock = asyncio.Lock()
    
    async def call(self, func, *args, **kwargs):
        async with self._lock:
            now = time.monotonic()
            if self.state == "OPEN":
                if now - self.last_failure_time > self.reset_timeout:
                    self.state = "HALF-OPEN"
                    self.failures = 0
                else:
                    raise Exception("Circuit breaker is open")
        
        try:
            result = await func(*args, **kwargs)
            async with self._lock:
                if self.state == "HALF-OPEN":
                    self.state = "CLOSED"
                    self.failures = 0
            return result
        except Exception as e:
            async with self._lock:
                self.failures += 1
                self.last_failure_time = time.monotonic()
                if self.failures >= self.max_failures:
                    self.state = "OPEN"
            raise


def circuit_breaker(max_failures: int = 3, reset_timeout: float = 30.0):
    """Circuit breaker decorator for async functions."""
    breaker = CircuitBreaker(max_failures, reset_timeout)
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await breaker.call(func, *args, **kwargs)
        return wrapper
    return decorator


def timeout_async(timeout: float = 60.0):
    """Timeout decorator for async functions."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)
        return wrapper
    return decorator


# Global circuit breakers
openrouter_breaker = CircuitBreaker(max_failures=3, reset_timeout=30)
colab_breaker = CircuitBreaker(max_failures=3, reset_timeout=30)
gemini_breaker = CircuitBreaker(max_failures=3, reset_timeout=30)
groq_breaker = CircuitBreaker(max_failures=3, reset_timeout=30)

# Global bulkhead semaphores for limiting concurrent requests
translation_semaphore = asyncio.Semaphore(3)  # Max 3 concurrent translations


class TranslationCache:
    _cache: dict[str, list[dict]] = {}
    _max_size = 500

    @classmethod
    def _key(cls, texts: list[str], source_lang: str) -> str:
        raw = json.dumps(texts, sort_keys=True, ensure_ascii=False) + f"|{source_lang}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    @classmethod
    def get(cls, texts: list[str], source_lang: str) -> list[dict] | None:
        k = cls._key(texts, source_lang)
        return cls._cache.get(k)

    @classmethod
    def put(cls, texts: list[str], source_lang: str, result: list[dict]):
        k = cls._key(texts, source_lang)
        if len(cls._cache) >= cls._max_size:
            oldest = next(iter(cls._cache))
            del cls._cache[oldest]
        cls._cache[k] = result
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


def _build_prompt(korean_texts, english_texts, page_number, context, glossary, memory_context="", memory_glossary=None, character_profiles=None, source_lang="ko", rag_context=""):
    parts = []
    if memory_context:
        parts.append("PREVIOUS CHAPTERS (translations of this manga from earlier chapters):\n" + memory_context)
    if rag_context:
        parts.append("SIMILAR PHRASES FROM EARLIER CHAPTERS (match style/terminology):\n" + rag_context)
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
    # Character voice profiles
    if character_profiles:
        lines = ["CHARACTER VOICES (match their tone & politeness):"]
        for name, prof in character_profiles.items():
            tone = prof.get("tone", "")
            pol = prof.get("politeness", "neutral")
            ex = "; ".join(prof.get("examples_ru", []))
            lines.append(f"  {name}: {tone} ({pol}). Examples: {ex}")
        parts.append("\n".join(lines))
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


_UNTRANSLATED_RE = re.compile(r'[\uAC00-\uD7AF\u3040-\u30FF\u4E00-\u9FFF]')


def _has_untranslated(text: str) -> bool:
    """True if text still contains CJK/Korean script (not yet translated to RU)."""
    return bool(_UNTRANSLATED_RE.search(text))


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
        self._character_profiles = {}
        self._rag_index = None
        self._rag_context = ""

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
                log.info("Colab OK — %s", self.colab_url)
            else:
                log.warning("Colab health check returned %d, skipping", resp.status_code)
        except Exception:
            log.warning("Colab unreachable, skipping")

    def _init_providers(self):
        # Colab Server (если доступен) - имеет приоритет, так как локальный
        if self.colab_available:
            self._providers.append(("Colab Server", self._call_colab))
        # Groq (Llama 70B) - бесплатно и качественно
        if GROQ_API_KEY:
            self._providers.append(("Groq (Llama 70B)", self._call_groq))
        # Gemini 2.0 Flash - если Groq не сработал
        if GEMINI_API_KEY:
            self._providers.append(("Gemini 2.0 Flash", self._call_gemini))
        # OpenRouter - если предыдущие не сработали
        if OPENROUTER_API_KEY:
            self._providers.append(("OpenRouter Free", self._call_openrouter_free))
        if OPENROUTER_API_KEY:
            self._providers.append(("OpenRouter Paid", self._call_openrouter_paid))
        # deep-translator (Google Translate) - как последний fallback
        self._providers.append(("deep-translator", self._call_fallback))

    def _print_chain(self):
        names = [p[0] for p in self._providers]
        log.info("Chain: %s", " -> ".join(names))

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

    def _ensure_rag_index(self, manga_id: str):
        if self._rag_index is not None:
            return self._rag_index
        try:
            texts = load_translations_from_memory(manga_id)
            idx = RAGIndex()
            idx.rebuild(texts)
            self._rag_index = idx
            if texts:
                log.info("RAG index built for %s (%d phrases)", manga_id, len(texts))
        except Exception as e:
            log.warning("RAG index build failed: %s", e)
            self._rag_index = RAGIndex()
        return self._rag_index

    def _build_rag_context(self, korean_texts: list[str], manga_id: str | None) -> str:
        if not manga_id or not korean_texts:
            return ""
        try:
            idx = self._ensure_rag_index(manga_id)
            found: list[str] = []
            seen = set()
            for t in korean_texts:
                for match, score in idx.search(t, k=2):
                    if match not in seen and match != t:
                        seen.add(match)
                        found.append(f"  {match}")
            if not found:
                return ""
            return "\n".join(found[:10])
        except Exception:
            return ""

    @timeout_async(timeout=120.0)
    @circuit_breaker(max_failures=3, reset_timeout=30)
    @retry_async(max_attempts=3, base_delay=1.0, max_delay=10.0)
    async def _call_openrouter(self, korean_texts, english_texts, page_number, model, timeout=120.0) -> list[dict] | None:
        async with translation_semaphore:
            await openrouter_limiter.wait()
            sl = getattr(self, '_source_lang', 'ko')
        prompt = _build_prompt(korean_texts, english_texts, page_number, self.context_pages, self._build_glossary_dict(), self._memory_context, self._memory_glossary, source_lang=sl, character_profiles=self._character_profiles, rag_context=getattr(self, '_rag_context', ''))
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

    @retry_async(max_attempts=3, base_delay=1.0, max_delay=10.0)
    async def _call_openrouter_free(self, korean_texts, english_texts, page_number) -> list[dict] | None:
        try:
            return await asyncio.wait_for(
                self._call_openrouter(korean_texts, english_texts, page_number, "openrouter/free"),
                timeout=30.0
            )
        except asyncio.TimeoutError:
            return None

    @retry_async(max_attempts=3, base_delay=1.0, max_delay=10.0)
    async def _call_openrouter_paid(self, korean_texts, english_texts, page_number) -> list[dict] | None:
        return await self._call_openrouter(korean_texts, english_texts, page_number, self.model)

    @timeout_async(timeout=120.0)
    @circuit_breaker(max_failures=3, reset_timeout=30)
    @retry_async(max_attempts=3, base_delay=1.0, max_delay=10.0)
    async def _call_groq(self, korean_texts, english_texts, page_number) -> list[dict] | None:
        async with translation_semaphore:
            await groq_limiter.wait()
            sl = getattr(self, '_source_lang', 'ko')
        prompt = _build_prompt(korean_texts, english_texts, page_number, self.context_pages, self._build_glossary_dict(), self._memory_context, self._memory_glossary, source_lang=sl, character_profiles=self._character_profiles, rag_context=getattr(self, '_rag_context', ''))
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

    @timeout_async(timeout=60.0)
    @circuit_breaker(max_failures=3, reset_timeout=30)
    @retry_async(max_attempts=3, base_delay=1.0, max_delay=10.0)
    async def _call_gemini(self, korean_texts, english_texts, page_number) -> list[dict] | None:
        async with translation_semaphore:
            await gemini_limiter.wait()
            sl = getattr(self, '_source_lang', 'ko')
        prompt = _build_prompt(korean_texts, english_texts, page_number, self.context_pages, self._build_glossary_dict(), self._memory_context, self._memory_glossary, source_lang=sl, character_profiles=self._character_profiles, rag_context=getattr(self, '_rag_context', ''))
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
                log.warning("Gemini → 403 (quota/blocked), skipping")
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

    @timeout_async(timeout=60.0)
    @circuit_breaker(max_failures=3, reset_timeout=30)
    @retry_async(max_attempts=3, base_delay=1.0, max_delay=10.0)
    async def _call_colab(self, korean_texts, english_texts, page_number) -> list[dict] | None:
        async with translation_semaphore:
            await colab_limiter.wait()
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
                        "character_profiles": self._character_profiles,
                    },
                }
                resp = await self.colab_client.post(f"{self.colab_url}/translate", json=payload, timeout=60.0)
                resp.raise_for_status()
                data = resp.json()
                return data.get("translations", [])
            except Exception:
                return None

    @timeout_async(timeout=30.0)
    @retry_async(max_attempts=3, base_delay=1.0, max_delay=10.0)
    async def _call_fallback(self, korean_texts, english_texts=None, page_number=0) -> list[dict]:
        async with translation_semaphore:
            await openrouter_limiter.wait()  # Use same limiter as openrouter for fallback
            try:
                from deep_translator import GoogleTranslator
                sl = getattr(self, '_source_lang', 'ko')
                deepl_src = {"ko": "ko", "en": "en", "ja": "ja", "zh": "zh-CN"}.get(sl, "ko")
                translator = GoogleTranslator(source=deepl_src, target="ru")
            except ImportError:
                log.warning("deep-translator not installed, returning originals")
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
        self._character_profiles = get_character_profiles(manga_id) if manga_id else {}
        self._rag_context = self._build_rag_context(korean_texts, manga_id)

        cached = TranslationCache.get(korean_texts, source_lang)
        if cached is not None:
            log.debug("Cache HIT for %d bubbles", len(korean_texts))
            self.add_context(korean_texts)
            return cached

        # Hard glossary pre-replace: inject known RU terms inline as hints
        hard_glossary = self._build_glossary_dict()
        prompt_texts = self._apply_hard_glossary(korean_texts, hard_glossary)

        for name, call_fn in self._providers:
            try:
                if name == "deep-translator":
                    result = await call_fn(korean_texts, english_texts or [], page_number)
                else:
                    result = await call_fn(prompt_texts, english_texts or [], page_number)
                if result is None:
                    log.warning("%s → rate limited, next...", name)
                    continue
                self.add_context(korean_texts)
                result = self._apply_hard_glossary_cleanup(result)
                result = await self._self_correct(korean_texts, result, source_lang)
                # Post-validation: fix trivial issues, else fallback to deep-translator
                result = self._validate_and_fix(korean_texts, result, source_lang)
                for i, ko in enumerate(korean_texts):
                    if i < len(result):
                        result[i]["ko"] = ko
                result = self._apply_post_replace(result)
                TranslationCache.put(korean_texts, source_lang, result)
                return result
            except Exception as e:
                log.error("%s → error: %s", name, e)
                continue

        log.error("All providers failed, returning originals")
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

    async def _self_correct(self, korean_texts, result: list[dict], source_lang: str) -> list[dict]:
        """Retry translation for bubbles the LLM left untranslated."""
        untranslated_idx = []
        for i, entry in enumerate(result):
            ru = (entry.get("ru") or "").strip()
            if not ru or _has_untranslated(ru):
                untranslated_idx.append(i)

        if not untranslated_idx:
            return result

        log.warning("Self-correction: %d/%d bubbles untranslated, retrying", len(untranslated_idx), len(result))

        # Filter to problematic texts
        problems = [korean_texts[i] for i in untranslated_idx]

        for name, call_fn in self._providers:
            if name == "deep-translator":
                continue
            try:
                fixed = await call_fn(problems, [], 0)
                if not fixed:
                    continue
                ok = True
                for j, i in enumerate(untranslated_idx):
                    ru = (fixed[j].get("ru") or "").strip() if j < len(fixed) else ""
                    if not ru or _has_untranslated(ru):
                        ok = False
                        continue
                    result[i]["ru"] = ru
                if ok:
                    break
            except Exception:
                continue

        return result

    def _validate_and_fix(self, korean_texts, result: list[dict], source_lang: str) -> list[dict]:
        """Validate each translation; apply trivial fixes and mark unfixable ones."""
        for i, entry in enumerate(result):
            src = korean_texts[i] if i < len(korean_texts) else ""
            ru = (entry.get("ru") or "").strip()
            if not ru:
                continue
            check = validate_translation(src, ru, source_lang)
            if check["ok"]:
                continue
            fixed = fix_translation(src, ru, source_lang)
            if fixed is not None and validate_translation(src, fixed, source_lang)["ok"]:
                log.info("Validator fixed bubble %d: %r -> %r", i, ru, fixed)
                entry["ru"] = fixed
            elif any(x in check["issues"] for x in ("untranslated-script", "korean-left", "empty")):
                log.warning("Validator: bubble %d has issues %s, leaving as-is", i, check["issues"])
        return result

    def _apply_hard_glossary(self, texts: list[str], glossary: dict) -> list[str]:
        """Hard-replace exact glossary terms in source text with bracketed RU hints.

        This guarantees consistent terminology even if the LLM misses the prompt.
        The RU translation is embedded inline as a hint, then extracted back after
        the LLM returns its JSON.
        """
        terms = {}
        if glossary:
            for section in ("characters", "terms"):
                for ko, ru in glossary.get(section, {}).items():
                    if ko and ru:
                        terms[ko] = ru
        if not terms:
            return list(texts)

        replaced = []
        for t in texts:
            out = t
            for ko, ru in sorted(terms.items(), key=lambda kv: -len(kv[0])):
                if ko in out:
                    out = out.replace(ko, f"{ko}（{ru}）")
            replaced.append(out)
        return replaced

    def _apply_hard_glossary_cleanup(self, result: list[dict]) -> list[dict]:
        """Remove injected RU hints from LLM output (括号 format)."""
        for entry in result:
            ru = entry.get("ru", "")
            if not ru:
                continue
            cleaned = re.sub(r'（[^）]*）', '', ru).strip()
            entry["ru"] = cleaned or ru
        return result

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

import asyncio
import aiohttp
import httpx
import os
import logging
import time
import random
from functools import wraps
from .base import BaseSource, MangaResult, Chapter, Page


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
                    if attempt == max_attempts - 1:
                        break
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    if jitter:
                        delay *= (0.5 + random.random() * 0.5)
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


# Rate limiter for proxy calls
proxy_limiter = RateLimiter(rate=5, capacity=10)  # ~5 requests per second

# Circuit breakers for proxy
proxy_breaker = CircuitBreaker(max_failures=3, reset_timeout=30)


log = logging.getLogger("manga_translator")

def _get_proxy():
    p = os.environ.get("MANGA_PROXY") or os.environ.get("HTTP_PROXY") or os.environ.get("HTTPS_PROXY") or os.environ.get("TG_PROXY_URL") or os.environ.get("http_proxy") or os.environ.get("https_proxy")
    if not p:
        try:
            from cfg import TG_PROXY_URL
            p = TG_PROXY_URL
        except Exception:
            p = ""
    return p or None


def _httpx_client(**kwargs):
    proxy = _get_proxy()
    if proxy:
        kwargs["proxy"] = proxy
        kwargs["verify"] = False
    return httpx.AsyncClient(**kwargs)


class MangaDexSource(BaseSource):
    BASE = "https://api.mangadex.org"
    MAX_RETRIES = 5
    _rate_limit_sem = asyncio.Semaphore(2)
    _last_request_time = 0
    _rate_lock = asyncio.Lock()

    @staticmethod
    def _is_valid_proxy_url(url: str) -> bool:
        """Проверяет, что URL выглядит как валидный прокси (не railway/render/localhost)."""
        bad_patterns = [
            "railway.app", "up.railway.app", "railway.internal",
            "render.com", "onrender.com",
            "localhost", "127.0.0.1", "0.0.0.0",
            ".local", ".internal",
        ]
        url_lower = url.lower()
        return not any(bad in url_lower for bad in bad_patterns)

    def __init__(self):
        self._own_session: aiohttp.ClientSession | None = None
        self.proxy_url = os.environ.get("REMOTE_SERVER_URL", "").rstrip("/")
        # Если REMOTE_SERVER_URL задан, но это не настоящий прокси — игнорируем
        if self.proxy_url and not self._is_valid_proxy_url(self.proxy_url):
            log.warning("REMOTE_SERVER_URL looks invalid (%s), ignoring proxy", self.proxy_url)
            self.proxy_url = ""
        self._proxy = _get_proxy()

    async def _rate_limit(self):
        async with self._rate_lock:
            now = asyncio.get_event_loop().time()
            elapsed = now - self.__class__._last_request_time
            if elapsed < 1.0:
                await asyncio.sleep(1.0 - elapsed)
            self.__class__._last_request_time = asyncio.get_event_loop().time()

    async def _get_json(self, url: str, params: dict | None = None) -> dict:
        if self.proxy_url:
            try:
                async with _httpx_client(timeout=15.0) as c:
                    r = await c.get(url, params=params)
                    r.raise_for_status()
                    return r.json()
            except Exception as e:
                log.warning("MangaDex proxy_url error: %s", e)

        async with self._rate_limit_sem:
            await self._rate_limit()
            for attempt in range(self.MAX_RETRIES):
                # 1) Try httpx (handles proxies, SSL, HTTP/2, DNS gracefully)
                try:
                    async with _httpx_client(timeout=30.0) as client:
                        resp = await client.get(url, params=params)
                        if resp.status_code == 429:
                            wait = 2 ** (attempt + 1)
                            log.warning("MangaDex 429 (httpx), жду %dс...", wait)
                            await asyncio.sleep(wait)
                            continue
                        resp.raise_for_status()
                        return resp.json()
                except Exception as e1:
                    log.warning("MangaDex httpx попытка %d/%d ошибка: %s", attempt + 1, self.MAX_RETRIES, e1)

                # 2) Fallback to aiohttp with IPv4 connector
                try:
                    import socket
                    conn = aiohttp.TCPConnector(family=socket.AF_INET)
                    async with aiohttp.ClientSession(connector=conn) as session:
                        async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                            if resp.status == 429:
                                wait = 2 ** (attempt + 1)
                                log.warning("MangaDex 429 (aiohttp), жду %dс...", wait)
                                await asyncio.sleep(wait)
                                continue
                            resp.raise_for_status()
                            return await resp.json()
                except Exception as e2:
                    log.warning("MangaDex aiohttp попытка %d/%d ошибка: %s", attempt + 1, self.MAX_RETRIES, e2)

                await asyncio.sleep(2 ** attempt)

        return {}

    @timeout_async(timeout=30.0)
    @circuit_breaker(max_failures=3, reset_timeout=30)
    @retry_async(max_attempts=3, base_delay=1.0, max_delay=10.0)
    async def _proxy_get(self, path: str, params: dict | None = None) -> dict:
        await proxy_limiter.wait()
        if not self.proxy_url:
            return {}
        try:
            async with _httpx_client(timeout=30.0) as c:
                r = await c.get(f"{self.proxy_url}{path}", params=params)
                r.raise_for_status()
                return r.json()
        except Exception as e:
            log.error("MangaDex proxy Error: %s", e)
            return {}

    @timeout_async(timeout=60.0)
    @circuit_breaker(max_failures=3, reset_timeout=30)
    @retry_async(max_attempts=3, base_delay=1.0, max_delay=10.0)
    async def _proxy_download(self, url: str) -> bytes:
        await proxy_limiter.wait()
        if not self.proxy_url:
            return b""
        try:
            async with _httpx_client(timeout=60.0) as c:
                r = await c.get(f"{self.proxy_url}/mangadex/download", params={"url": url})
                r.raise_for_status()
                import base64
                return base64.b64decode(r.json()["image_b64"])
        except Exception as e:
            log.error("MangaDex proxy download Error: %s", e)
            return b""

    async def search(self, title: str) -> list[MangaResult]:
        data = await self._proxy_get("/mangadex/search", {"q": title, "limit": 10})
        if not data:
            data = await self._get_json(f"{self.BASE}/manga", params={
                "title": title,
                "limit": 10,
                "hasAvailableChapters": "true",
                "includes[]": "cover_art",
                "order[relevance]": "desc",
            })
            if not data or not data.get("data"):
                data = await self._get_json(f"{self.BASE}/manga", params={
                    "title": title,
                    "limit": 10,
                    "includes[]": "cover_art",
                })
        results = []
        for item in data.get("data", []):
            attrs = item["attributes"]
            title_en = attrs.get("title", {}).get("en", "")
            alt_titles = [v for t in attrs.get("altTitles", []) for v in t.values()]
            cover_url = None
            for rel in item.get("relationships", []):
                if rel["type"] == "cover_art":
                    fname = rel.get("attributes", {}).get("fileName", "")
                    if fname:
                        cover_url = f"https://uploads.mangadex.org/covers/{item['id']}/{fname}.256.jpg"
            results.append(MangaResult(
                id=item["id"],
                title=title_en or (alt_titles[0] if alt_titles else "Unknown"),
                alt_titles=alt_titles,
                description=attrs.get("description", {}).get("en", ""),
                status=attrs.get("status", ""),
                year=attrs.get("year"),
                cover_url=cover_url,
                source="mangadex",
                original_language=attrs.get("originalLanguage", "ko"),
            ))
        return results

    async def get_chapters(self, manga_id: str, lang: str = "en") -> list[Chapter]:
        chapters = []
        offset = 0
        limit = 500  # увеличен с 100 до 500 (макс. MangaDex)
        seen_numbers = set()
        total_fetched = 0
        while True:
            data = await self._proxy_get(
                f"/mangadex/{manga_id}/chapters",
                {"lang": lang, "limit": limit, "offset": offset},
            )
            if not data:
                data = await self._get_json(f"{self.BASE}/manga/{manga_id}/feed", params={
                    "translatedLanguage[]": lang,
                    "limit": limit,
                    "offset": offset,
                    "order[chapter]": "asc",
                    "includes[]": "scanlation_group",
                })
            items = data.get("data", [])
            if not items:
                break
            page_count = 0
            for item in items:
                attrs = item["attributes"]
                ch_num = attrs.get("chapter", "")
                try:
                    ch_float = float(ch_num)
                    # Дедупликация по номеру главы (MangaDex может возвращать дубликаты от разных групп)
                    if ch_float in seen_numbers:
                        continue
                    seen_numbers.add(ch_float)
                    page_count += 1
                    chapters.append(Chapter(
                        id=item["id"],
                        number=ch_num,
                        title=attrs.get("title", ""),
                        volume=attrs.get("volume"),
                        pages_count=attrs.get("pages", 0),
                        translated_language=attrs.get("translatedLanguage", lang),
                    ))
                except (ValueError, TypeError):
                    continue
            total_fetched += page_count
            log.info("get_chapters: manga=%s lang=%s offset=%d fetched=%d unique=%d total=%d",
                     manga_id, lang, offset, len(items), page_count, total_fetched)
            offset += limit
            if len(items) < limit:
                break
        log.info("get_chapters DONE: manga=%s lang=%s total_chapters=%d", manga_id, lang, len(chapters))
        return chapters

    async def find_chapter_by_number(self, manga_id: str, chapter_number: str, lang: str) -> Chapter | None:
        chapters = await self.get_chapters(manga_id, lang)
        for ch in chapters:
            if ch.number == chapter_number:
                return ch
        return None

    async def get_available_languages(self, manga_id: str) -> list[str]:
        data = await self._proxy_get(
            f"/mangadex/{manga_id}/chapters",
            {"lang": "", "limit": 100, "offset": 0},
        )
        if not data:
            data = await self._get_json(f"{self.BASE}/manga/{manga_id}/feed", params={
                "limit": 100,
                "order[chapter]": "desc",
            })
        langs = set()
        for item in data.get("data", []):
            lang = item.get("attributes", {}).get("translatedLanguage", "")
            if lang:
                langs.add(lang)
        return sorted(langs)

    async def get_pages(self, chapter_id: str) -> list[Page]:
        data = await self._proxy_get(f"/mangadex/chapter/{chapter_id}/pages")
        if data and "pages" in data:
            return [
                Page(url=p["url"], index=p["index"], width=0, height=0)
                for p in data["pages"]
            ]
        data = await self._get_json(f"{self.BASE}/at-home/server/{chapter_id}")
        base_url = data.get("baseUrl", "")
        chapter_hash = data.get("chapter", {}).get("hash", "")
        ch_data = data.get("chapter", {})
        filenames = ch_data.get("data") or ch_data.get("dataSaver", [])
        pages = []
        for i, fname in enumerate(filenames):
            url = f"{base_url}/data/{chapter_hash}/{fname}"
            pages.append(Page(url=url, index=i, width=0, height=0))
        return pages

    async def download_page(self, page: Page) -> bytes:
        img_data = await self._proxy_download(page.url)
        if img_data:
            return img_data
        async with self._rate_limit_sem:
            await self._rate_limit()
            for attempt in range(self.MAX_RETRIES):
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(page.url, timeout=aiohttp.ClientTimeout(total=120)) as resp:
                            if resp.status == 429:
                                await asyncio.sleep(2 ** (attempt + 1))
                                continue
                            resp.raise_for_status()
                            return await resp.read()
                except (aiohttp.ClientError, TimeoutError, asyncio.TimeoutError):
                    if attempt == self.MAX_RETRIES - 1:
                        raise
                    await asyncio.sleep(1)
        return b""

    async def close(self):
        if self._own_session and not self._own_session.closed:
            await self._own_session.close()

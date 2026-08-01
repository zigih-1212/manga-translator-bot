import asyncio
import json
import os
import io
import logging
import time
import random
import httpx
import cv2
import numpy as np
from PIL import Image
from functools import wraps
from .log import log
from .manga_ocr import get_manga_ocr, is_available as manga_ocr_available


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


# Rate limiter for OCR calls
ocr_limiter = RateLimiter(rate=5, capacity=10)  # ~5 requests per second


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


# Circuit breakers for OCR
ocr_paddle_breaker = CircuitBreaker(max_failures=3, reset_timeout=30)
ocr_space_breaker = CircuitBreaker(max_failures=3, reset_timeout=30)


def retry_async(max_attempts=3, base_delay=1.0, max_delay=10.0, jitter=True):
    """
    Retry decorator for async functions with exponential backoff for async functions.
    
    Args:
        attempts: Maximum number of attempts
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


def _get_proxy():
    return os.environ.get("HTTP_PROXY") or os.environ.get("HTTPS_PROXY") or os.environ.get("http_proxy") or os.environ.get("https_proxy")


OCR_SPACE_API = "https://api.ocr.space/parse/image"
OCR_SPACE_KEY = "helloworld"


PADDLE_LANG_MAP = {
    "ko": "korean",
    "kor": "korean",
    "en": "english",
    "eng": "english",
    "ja": "japanese",
    "jpn": "japanese",
    "ru": "russian",
    "rus": "russian",
}


class ColabClient:
    def __init__(self, base_url: str = ""):
        proxy = _get_proxy()
        self.client = httpx.AsyncClient(timeout=120.0, proxy=proxy, verify=False) if proxy else httpx.AsyncClient(timeout=120.0)
        self._connected = True
        self._paddle_available = False # PaddleOCR отключен для экономии RAM

    async def init(self):
        # Инициализация PaddleOCR отключена
        log.info("PaddleOCR отключен, используем ocr.space")

    @staticmethod
    def _init_paddle(lang: str):
        # Функция инициализации PaddleOCR не используется
        return None

    def _get_paddle(self, lang: str):
        # Функция получения PaddleOCR не используется
        return None

    @property
    def is_connected(self) -> bool:
        # Всегда возвращаем True, так как ocr.space доступен по сети
        return True

    async def ocr_manga_local(self, page_data: bytes, lang: str = "ja") -> list[dict]:
        """Local manga-ocr fallback for Japanese pages (offline, free)."""
        try:
            engine = get_manga_ocr()
            if not engine.available:
                await asyncio.to_thread(engine._load)
            if not engine.available:
                return []
            img = Image.open(io.BytesIO(page_data)).convert("RGB")
            text = await asyncio.to_thread(engine.ocr, img)
            if not text.strip():
                return []
            return [{
                "bbox": [[0, 0], [img.width, 0], [img.width, img.height], [0, img.height]],
                "text": text.strip(),
                "word_bboxes": [[0, 0, img.width, img.height]],
                "confidence": 1.0,
            }]
        except Exception as e:
            log.error("manga-ocr local failed: %s", e)
            return []

    async def ocr_pages(self, pages: list[bytes], lang: str = "kor") -> list[list[dict]]:
        # Всегда используем ocr.space
        return await self._ocr_space(pages, lang)

    @staticmethod
    def _preprocess_for_ocr(img_np: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY) if img_np.ndim == 3 else img_np
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        denoised = cv2.fastNlMeansDenoising(enhanced, h=10, templateWindowSize=7, searchWindowSize=21)
        sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(denoised, -1, sharpen_kernel)
        return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2RGB)

    @timeout_async(timeout=120.0)
    @circuit_breaker(max_failures=3, reset_timeout=30)
    @retry_async(max_attempts=3, base_delay=1.0, max_delay=10.0)
    async def _ocr_paddle(self, pages: list[bytes], lang: str = "korean") -> list[list[dict]]:
        await ocr_limiter.wait()
        paddle = self._get_paddle(lang)
        if not paddle:
            return await self._ocr_space(pages, "kor" if lang == "korean" else "eng")
        all_results = []
        loop = asyncio.get_event_loop()
        for page_data in pages:
            try:
                img = Image.open(io.BytesIO(page_data)).convert("RGB")
                ow, oh = img.size
                scale = 2
                img_big = img.resize((ow * scale, oh * scale), Image.LANCZOS)
                img_np = np.array(img_big)
                if img_np.ndim == 3 and img_np.shape[2] == 4:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
                img_np = self._preprocess_for_ocr(img_np)
                result = paddle.ocr(img_np, cls=True)
                if result and len(result) > 0:
                    all_results.append(result[0])
                else:
                    all_results.append([])
            except Exception:
                all_results.append([])
        return all_results

    @timeout_async(timeout=120.0)
    @circuit_breaker(max_failures=3, reset_timeout=30)
    @retry_async(max_attempts=3, base_delay=1.0, max_delay=10.0)
    async def _ocr_space(self, pages: list[bytes], lang: str) -> list[list[dict]]:
        await ocr_limiter.wait()

        async def _ocr_one(page_data: bytes) -> list[dict]:
            try:
                if len(page_data) > 4.5 * 1024 * 1024:
                    img = Image.open(io.BytesIO(page_data))
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", quality=85)
                    page_data = buf.getvalue()

                ext = "jpg" if page_data[:3] == b"\xff\xd8\xff" else "png"
                resp = await self.client.post(
                    OCR_SPACE_API,
                    files={"file": (f"page.{ext}", page_data, f"image/{ext}")},
                    data={"apikey": OCR_SPACE_KEY, "language": lang, "isOverlayRequired": "true"},
                    timeout=60.0,
                )
                resp.raise_for_status()
                data = resp.json()

                if data.get("IsErroredOnProcessing"):
                    log.error("OCR Error: %s", data.get('ErrorMessage', ''))
                    return []

                results = []
                for parsed in data.get("ParsedResults", []):
                    overlay = parsed.get("TextOverlay", {})
                    for line in overlay.get("Lines", []):
                        line_text = line.get("LineText", "").strip()
                        if not line_text:
                            continue
                        words = line.get("Words", [])
                        if not words:
                            continue
                        left = min(w["Left"] for w in words)
                        top = min(w["Top"] for w in words)
                        right = max(w["Left"] + w["Width"] for w in words)
                        bottom = max(w["Top"] + w["Height"] for w in words)
                        word_bboxes = [
                            [w["Left"], w["Top"], w["Left"] + w["Width"], w["Top"] + w["Height"]]
                            for w in words
                        ]
                        confs = [w.get("Confidence") for w in words]
                        confs = [c for c in confs if c is not None]
                        confidence = sum(confs) / len(confs) if confs else 1.0
                        results.append({
                            "bbox": [[left, top], [right, top], [right, bottom], [left, bottom]],
                            "text": line_text,
                            "word_bboxes": word_bboxes,
                            "confidence": confidence,
                        })
                return results
            except Exception as e:
                log.error("OCR Request failed: %s", e)
                return []

        # Batch: process up to CONCURRENCY pages concurrently (bounded)
        CONCURRENCY = 4
        sem = asyncio.Semaphore(CONCURRENCY)

        async def _bounded(page_data):
            async with sem:
                return await _ocr_one(page_data)

        return await asyncio.gather(*(_bounded(p) for p in pages))

    async def inpaint_pages(self, pages, masks):
        return pages

    async def close(self):
        await self.client.aclose()

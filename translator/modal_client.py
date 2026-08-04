import os
import base64
import logging
import time
import random
import asyncio
import threading
from functools import wraps
log = logging.getLogger("manga_translator")


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


# Rate limiter for Modal calls
modal_limiter = RateLimiter(rate=2, capacity=5)  # ~2 requests per second


class SyncRateLimiter:
    """Token bucket rate limiter for synchronous functions."""
    def __init__(self, rate: float, capacity: int = 1):
        self.rate = rate  # tokens per second
        self.capacity = capacity  # max tokens
        self.tokens = capacity
        self.last_refill = time.monotonic()
        self._lock = threading.Lock()
    
    def acquire(self):
        with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_refill
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self.last_refill = now
            if self.tokens >= 1:
                self.tokens -= 1
                return True
            wait_time = (1 - self.tokens) / self.rate
            return False
    
    def wait(self):
        while True:
            if self.acquire():
                return
            wait_time = (1 - self.tokens) / self.rate
            time.sleep(wait_time)


modal_sync_limiter = SyncRateLimiter(rate=2, capacity=5)


class CircuitBreaker:
    """Circuit breaker pattern to prevent cascading failures."""
    def __init__(self, max_failures: int = 3, reset_timeout: float = 30.0):
        self.max_failures = max_failures
        self.reset_timeout = reset_timeout
        self.failures = 0
        self.last_failure_time = 0
        self.state = "CLOSED"
        self._lock = threading.Lock()
    
    def call(self, func, *args, **kwargs):
        with self._lock:
            now = time.monotonic()
            if self.state == "OPEN":
                if now - self.last_failure_time > self.reset_timeout:
                    self.state = "HALF-OPEN"
                    self.failures = 0
                else:
                    raise Exception("Circuit breaker is open")
        
        try:
            result = func(*args, **kwargs)
            with self._lock:
                if self.state == "HALF-OPEN":
                    self.state = "CLOSED"
                    self.failures = 0
            return result
        except Exception as e:
            with self._lock:
                self.failures += 1
                self.last_failure_time = time.monotonic()
                if self.failures >= self.max_failures:
                    self.state = "OPEN"
            raise


def circuit_breaker(max_failures: int = 3, reset_timeout: float = 30.0):
    """Circuit breaker decorator for synchronous functions."""
    breaker = CircuitBreaker(max_failures, reset_timeout)
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return breaker.call(func, *args, **kwargs)
        return wrapper
    return decorator


def timeout_sync(timeout: float = 60.0):
    """Timeout decorator for synchronous functions."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(func, *args, **kwargs)
                try:
                    return future.result(timeout=timeout)
                except concurrent.futures.TimeoutError:
                    future.cancel()
                    raise TimeoutError(f"Function timed out after {timeout} seconds")
        return wrapper
    return decorator


# Circuit breaker for Modal
modal_breaker = CircuitBreaker(max_failures=3, reset_timeout=30)


def retry_sync(max_attempts=3, base_delay=1.0, max_delay=10.0, jitter=True):
    """
    Retry decorator for synchronous functions with exponential backoff.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt == max_attempts - 1:
                        break
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    if jitter:
                        delay *= (0.5 + random.random() * 0.5)
                    time.sleep(delay)
            raise last_exception
        return wrapper
    return decorator


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

# Старый формат персонального токена (ak-/as-), слался как X-Modal-Token-Id/Secret
MODAL_TOKEN_ID = os.getenv("MODAL_TOKEN_ID")
MODAL_TOKEN_SECRET = os.getenv("MODAL_TOKEN_SECRET")
# Новый формат workspace-ключей для вебхука/прокси (wk-/ws-),
# слается как Modal-Key/Modal-Secret или Authorization: Bearer key.secret
MODAL_KEY = os.getenv("MODAL_KEY")
MODAL_SECRET = os.getenv("MODAL_SECRET")
MODAL_APP_ID = os.getenv("MODAL_APP_ID", "dvybornyh332/manga-inpaint")
MODAL_APP_URL = os.getenv("MODAL_APP_URL", "")

if not MODAL_APP_URL and MODAL_APP_ID and "/" in MODAL_APP_ID:
    workspace, app_name = MODAL_APP_ID.split("/", 1)
    MODAL_APP_URL = f"https://{workspace}--{app_name}-inpaint-api.modal.run"

_MODAL_SDK_AVAILABLE = False
try:
    import modal
    _MODAL_SDK_AVAILABLE = True
except ImportError:
    log.warning("Modal SDK not installed, using REST API fallback")

# Доступен, если задан любой действующий набор учётных данных
MODAL_AVAILABLE = bool(
    (MODAL_KEY and MODAL_SECRET) or (MODAL_TOKEN_ID and MODAL_TOKEN_SECRET)
)


def inpaint_batch_sync(images: list[bytes], dilation: int = 5, radius: int = 10, masks: list[bytes] | None = None) -> list[bytes] | None:
    if not MODAL_AVAILABLE:
        log.warning("Modal not configured (missing MODAL_TOKEN_ID/SECRET or MODAL_KEY/SECRET)")
        return None

    images_b64 = [base64.b64encode(img).decode() for img in images]
    masks_b64 = None
    if masks:
        masks_b64 = [base64.b64encode(m).decode() if m else None for m in masks]

    # SDK path
    if _MODAL_SDK_AVAILABLE:
        try:
            import modal
            app_name = MODAL_APP_ID.split("/")[-1] if "/" in MODAL_APP_ID else "manga-inpaint"
            f = modal.Function.from_name(app_name, "inpaint_batch")
            result_b64 = f.remote(images_b64, dilation=dilation, radius=radius, masks_b64=masks_b64)
            log.info("Modal GPU inpaint OK (%d images via SDK)", len(images))
            return [base64.b64decode(b64) for b64 in result_b64]
        except Exception as e:
            log.warning("Modal SDK call failed: %s, trying REST API", e)

    # REST API path (no SDK needed)
    return _inpaint_batch_rest_sync(images_b64, dilation, radius, masks_b64)


@timeout_sync(timeout=600.0)
@circuit_breaker(max_failures=3, reset_timeout=30)
@retry_sync(max_attempts=3, base_delay=1.0, max_delay=10.0)
def _inpaint_batch_rest_sync(images_b64: list[str], dilation: int, radius: int, masks_b64: list[str] | None = None) -> list[bytes] | None:
    """Retry wrapper for REST API call to Modal."""
    modal_sync_limiter.wait()
    try:
        import httpx
        payload = {"images_b64": images_b64, "dilation": dilation, "radius": radius}
        if masks_b64:
            payload["masks_b64"] = masks_b64
        headers = {}
        if MODAL_KEY and MODAL_SECRET:
            # Новый формат вебхук/прокси-авторизации (wk-/ws-)
            headers["Modal-Key"] = MODAL_KEY
            headers["Modal-Secret"] = MODAL_SECRET
        elif MODAL_TOKEN_ID and MODAL_TOKEN_SECRET:
            # Legacy формат персонального токена
            headers["X-Modal-Token-Id"] = MODAL_TOKEN_ID
            headers["X-Modal-Token-Secret"] = MODAL_TOKEN_SECRET
        with httpx.Client(timeout=600) as client:
            resp = client.post(
                MODAL_APP_URL,
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
            result_b64 = resp.json()
            log.info("Modal GPU inpaint OK (%d images via REST)", len(images_b64))
            return [base64.b64decode(b64) for b64 in result_b64]
    except Exception as e:
        log.error("Modal REST API failed: %s", e)
        raise


async def inpaint_batch(images: list[bytes], dilation: int = 5, radius: int = 10, masks: list[bytes] | None = None) -> list[bytes] | None:
    return inpaint_batch_sync(images, dilation, radius, masks)

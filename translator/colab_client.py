import base64
import logging
import os
import random
import time
import threading
import urllib.request
import json
from functools import wraps

log = logging.getLogger("manga_translator")

COLAB_URL = os.getenv("COLAB_URL") or os.getenv("REMOTE_SERVER_URL") or ""
COLAB_AVAILABLE = bool(COLAB_URL)


def timeout_sync(timeout: float = 120.0):
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


def retry_sync(max_attempts=3, base_delay=1.0, max_delay=10.0, jitter=True):
    """Retry decorator for synchronous functions with exponential backoff."""
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


class ColabClient:
    """Client for the Kaggle/Colab GPU server (inpaint + OCR fallback)."""

    def __init__(self, base_url: str = ""):
        self.base_url = (base_url or COLAB_URL).rstrip("/")
        self._available = bool(self.base_url)
        if self._available:
            self._test_health()
        else:
            log.info("Colab server not configured (COLAB_URL empty)")

    @timeout_sync(timeout=5.0)
    def _test_health(self):
        try:
            with urllib.request.urlopen(f"{self.base_url}/health", timeout=5) as r:
                ok = r.status == 200
                self._available = ok
                log.info("Colab server health: %s (%s)", r.status, self.base_url)
        except Exception as e:
            self._available = False
            log.warning("Colab server unreachable: %s", e)

    @property
    def available(self) -> bool:
        return self._available

    @retry_sync(max_attempts=2, base_delay=1.0)
    @timeout_sync(timeout=300.0)
    def inpaint_batch(self, images: list[bytes], dilation: int = 5, radius: int = 10,
                      masks: list[bytes] | None = None) -> list[bytes] | None:
        """Inpaint pages on the Colab GPU server. Returns cleaned page PNGs."""
        if not self._available:
            return None
        import urllib.request
        multipart = self._build_multipart(images, masks)
        req = urllib.request.Request(
            f"{self.base_url}/inpaint_batch",
            data=multipart,
            headers={"Content-Type": self._content_type},
        )
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                data = json.loads(r.read().decode("utf-8"))
            b64s = data.get("clean_pages_b64", [])
            if not b64s:
                log.warning("Colab inpaint returned no pages")
                return None
            log.info("Colab GPU inpaint OK (%d pages)", len(b64s))
            return [base64.b64decode(b) for b in b64s]
        except Exception as e:
            log.warning("Colab inpaint failed: %s", e)
            raise

    @retry_sync(max_attempts=2, base_delay=1.0)
    @timeout_sync(timeout=120.0)
    def ocr_pages(self, pages: list[bytes]) -> list[list[dict]] | None:
        """OCR pages on the Colab GPU server (easyocr)."""
        if not self._available:
            return None
        import urllib.request
        multipart = self._build_multipart(pages)
        req = urllib.request.Request(
            f"{self.base_url}/ocr",
            data=multipart,
            headers={"Content-Type": self._content_type},
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                data = json.loads(r.read().decode("utf-8"))
            results = data.get("results", [])
            if not results:
                return None
            log.info("Colab GPU OCR OK (%d pages)", len(results))
            return results
        except Exception as e:
            log.warning("Colab OCR failed: %s", e)
            raise

    @staticmethod
    def _append_field(buf: list[bytes], boundary: str, name: str, value: str):
        buf.append(f"--{boundary}\r\n".encode())
        buf.append(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        buf.append(value.encode("utf-8"))
        buf.append(b"\r\n")

    @staticmethod
    def _mask_to_bbox_payload(mask_bytes: bytes) -> list[dict]:
        """Convert a binary PNG mask into bbox JSON for old Colab servers."""
        try:
            import cv2
            import numpy as np

            mask_np = np.frombuffer(mask_bytes, np.uint8)
            mask = cv2.imdecode(mask_np, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                return []

            _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            payload = []
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if w <= 1 or h <= 1:
                    continue
                payload.append({"bbox": [int(x), int(y), int(x + w), int(y + h)]})
            return payload
        except Exception as e:
            log.warning("Failed to convert mask to bbox payload: %s", e)
            return []

    def _build_masks_data(self, pages: list[bytes], masks: list[bytes] | None = None) -> str | None:
        if not masks:
            return None
        payload: list[list[dict]] = []
        for i in range(len(pages)):
            mask_bytes = masks[i] if i < len(masks) else None
            payload.append(self._mask_to_bbox_payload(mask_bytes) if mask_bytes else [])
        return json.dumps(payload, ensure_ascii=False)

    def _build_multipart(self, pages: list[bytes], masks: list[bytes] | None = None) -> bytes:
        import uuid
        boundary = uuid.uuid4().hex
        self._content_type = f"multipart/form-data; boundary={boundary}"
        buf = []
        for i, p in enumerate(pages):
            buf.append(f"--{boundary}\r\n".encode())
            buf.append(f'Content-Disposition: form-data; name="pages"; filename="page{i}.png"\r\n'.encode())
            buf.append(b"Content-Type: image/png\r\n\r\n")
            buf.append(p)
            buf.append(b"\r\n")
        if masks:
            for i, m in enumerate(masks):
                if not m:
                    continue
                buf.append(f"--{boundary}\r\n".encode())
                buf.append(f'Content-Disposition: form-data; name="masks"; filename="mask{i}.png"\r\n'.encode())
                buf.append(b"Content-Type: image/png\r\n\r\n")
                buf.append(m)
                buf.append(b"\r\n")
        masks_data = self._build_masks_data(pages, masks)
        if masks_data is not None:
            self._append_field(buf, boundary, "masks_data", masks_data)
        buf.append(f"--{boundary}--\r\n".encode())
        return b"".join(buf)

    def close(self):
        self._available = False

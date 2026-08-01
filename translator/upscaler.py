import io
import os
import logging
import time
import random
from functools import wraps
import numpy as np
import cv2
from pathlib import Path

log = logging.getLogger("manga_translator")


def retry_sync(max_attempts=3, base_delay=1.0, max_delay=10.0, jitter=True):
    """
    Retry decorator for synchronous functions with exponential backoff.
    
    Args:
        max_attempts: Maximum number of attempts
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        jitter: Whether to add random jitter to delay
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

MODEL_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.onnx"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODEL_DIR / "RealESRGAN_x4plus.onnx"


@retry_sync(max_attempts=3, base_delay=1.0, max_delay=10.0)
def _ensure_model():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.exists():
        return True
    try:
        import urllib.request
        log.info("Downloading model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        log.info("Downloaded OK")
        return True
    except Exception as e:
        log.error("Download failed: %s", e)
        return False


class RealESRGANUpscaler:
    def __init__(self):
        self.session = None
        self.available = False
        if _ensure_model():
            self._load()

    def _load(self):
        try:
            import onnxruntime
            import numpy as np
            self.session = onnxruntime.InferenceSession(
                str(MODEL_PATH), providers=["CPUExecutionProvider"]
            )
            self.available = True
            log.info("ESRGAN Model loaded on CPU")
        except Exception as e:
            log.error("ESRGAN Failed: %s", e)
            self.available = False

    def upscale(self, img: np.ndarray, scale: int = 2) -> np.ndarray:
        if not self.available or img is None:
            return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
        h, w = img.shape[:2]
        padded_h = (scale - h % scale) % scale
        padded_w = (scale - w % scale) % scale
        if padded_h or padded_w:
            img = cv2.copyMakeBorder(img, 0, padded_h, 0, padded_w, cv2.BORDER_REFLECT)
        inp = img.astype(np.float32) / 255.0
        inp = inp.transpose(2, 0, 1)[np.newaxis, ...]
        out = self.session.run(None, {"input": inp})[0]
        out = out[0].transpose(1, 2, 0)
        out = (out * 255).clip(0, 255).astype(np.uint8)
        out = out[: h * scale, : w * scale]
        return out
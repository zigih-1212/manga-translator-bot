import io
import os
import numpy as np
import cv2
from pathlib import Path

MODEL_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.onnx"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODEL_DIR / "RealESRGAN_x4plus.onnx"


def _ensure_model():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.exists():
        return True
    try:
        import urllib.request
        print("[ESRGAN] Downloading model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[ESRGAN] Downloaded OK")
        return True
    except Exception as e:
        print(f"[ESRGAN] Download failed: {e}")
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
            print("[ESRGAN] Model loaded on CPU")
        except Exception as e:
            print(f"[ESRGAN] Failed: {e}")
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
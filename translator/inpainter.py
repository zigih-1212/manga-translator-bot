import os
import urllib.request
from pathlib import Path
import numpy as np
import cv2

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_URLS = [
    "https://huggingface.co/botp/big-lama/resolve/main/lama.onnx",
    "https://github.com/Sanster/lama-onnx/releases/download/v0.1.0/lama.onnx",
]
MODEL_PATH = MODEL_DIR / "lama.onnx"


class LaMaInpainter:
    def __init__(self):
        self.session = None
        self._available = False
        self._load()

    def _load(self):
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        if not MODEL_PATH.exists():
            for url in MODEL_URLS:
                try:
                    print(f"[LaMa] Downloading {url}...")
                    urllib.request.urlretrieve(url, MODEL_PATH)
                    print("[LaMa] Downloaded")
                    break
                except Exception as e:
                    print(f"[LaMa] Download failed: {e}")
                    if MODEL_PATH.exists():
                        MODEL_PATH.unlink()
            if not MODEL_PATH.exists():
                print("[LaMa] No model available, using cv2.inpaint fallback")
                return
        try:
            import onnxruntime
            self.session = onnxruntime.InferenceSession(
                str(MODEL_PATH), providers=["CPUExecutionProvider"]
            )
            self._available = True
            print("[LaMa] ONNX model loaded on CPU")
        except Exception as e:
            print(f"[LaMa] Failed to load: {e}")

    @property
    def available(self) -> bool:
        return self._available

    def _pad(self, img, mask):
        h, w = img.shape[:2]
        ph = (8 - h % 8) % 8
        pw = (8 - w % 8) % 8
        if ph or pw:
            img = cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_REFLECT)
            mask = cv2.copyMakeBorder(mask, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=0)
        return img, mask, ph, pw

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if not self._available:
            return cv2.inpaint(image, mask, 5, cv2.INPAINT_NS)
        h, w = image.shape[:2]
        img_pad, mask_pad, ph, pw = self._pad(image, mask)
        inp = img_pad.astype(np.float32) / 127.5 - 1.0
        inp = inp.transpose(2, 0, 1)[np.newaxis, ...]
        msk = (mask_pad > 127).astype(np.float32)[np.newaxis, np.newaxis, ...]
        inp_name = self.session.get_inputs()[0].name
        msk_name = self.session.get_inputs()[1].name
        out = self.session.run(None, {inp_name: inp, msk_name: msk})[0]
        out = out[0].transpose(1, 2, 0)
        out = ((out + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        if ph or pw:
            out = out[:h, :w]
        return out

import os
import threading
import urllib.request
from pathlib import Path
import numpy as np
import cv2

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_URLS = [
    "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.onnx",
    "https://huggingface.co/botp/big-lama/resolve/main/lama.onnx",
]
MODEL_PATH = MODEL_DIR / "lama.onnx"


class LaMaInpainter:
    def __init__(self):
        self.session = None
        self._available = False
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        if MODEL_PATH.exists():
            self._load_model()
        else:
            threading.Thread(target=self._download_async, daemon=True).start()

    def _download_async(self):
        for url in MODEL_URLS:
            try:
                print(f"[LaMa] Downloading {url}...")
                urllib.request.urlretrieve(url, MODEL_PATH)
                print("[LaMa] Downloaded OK")
                self._load_model()
                return
            except Exception as e:
                print(f"[LaMa] Download failed: {e}")
                if MODEL_PATH.exists():
                    MODEL_PATH.unlink()

    def _load_model(self):
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

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if self._available:
            return self._inpaint_lama(image, mask)
        return self._inpaint_cv(image, mask)

    def _inpaint_lama(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        ph = (8 - h % 8) % 8
        pw = (8 - w % 8) % 8
        if ph or pw:
            img = cv2.copyMakeBorder(image, 0, ph, 0, pw, cv2.BORDER_REFLECT)
            msk = cv2.copyMakeBorder(mask, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=0)
        else:
            img, msk = image, mask
        inp = img.astype(np.float32) / 127.5 - 1.0
        inp = inp.transpose(2, 0, 1)[np.newaxis, ...]
        msk_in = (msk > 127).astype(np.float32)[np.newaxis, np.newaxis, ...]
        inp_name = self.session.get_inputs()[0].name
        msk_name = self.session.get_inputs()[1].name
        out = self.session.run(None, {inp_name: inp, msk_name: msk_in})[0]
        out = out[0].transpose(1, 2, 0)
        out = ((out + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        if ph or pw:
            out = out[:h, :w]
        return out

    def _inpaint_cv(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=3)
        result = cv2.inpaint(image, mask, 10, cv2.INPAINT_NS)
        return result

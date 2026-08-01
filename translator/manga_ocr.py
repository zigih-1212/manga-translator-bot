import io
import json
import os
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

from .log import log

MODEL_REPO = os.environ.get("MANGA_OCR_REPO", "kha-white/manga-ocr-base")
MODEL_REVISION = "main"
MODEL_CACHE = Path(os.environ.get("MODEL_CACHE", "models")) / "manga-ocr"
MODEL_FILE = "model.onnx"
VOCAB_FILE = "vocab.json"
HF_BASE = "https://huggingface.co"


def _hf_url(repo: str, filename: str, revision: str = MODEL_REVISION) -> str:
    return f"{HF_BASE}/{repo}/resolve/{revision}/{filename}"


class MangaOCREngine:
    """Local ONNX manga-ocr engine for Japanese (and some Korean) text.

    Uses kha-white/manga-ocr-base exported to ONNX. Downloads the model on
    first use and caches it under MODEL_CACHE. Falls back gracefully (returns
    empty results) if the model cannot be downloaded or onnxruntime is missing.
    """

    def __init__(self, repo: str = MODEL_REPO):
        self.repo = repo
        self._session = None
        self._tokenizer = None
        self._load_lock = threading.Lock()
        self.available = False
        self._load_attempted = False

    def _download(self, filename: str, dest: Path) -> bool:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and dest.stat().st_size > 0:
            return True
        url = _hf_url(self.repo, filename)
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        log.info("Downloading %s from %s ...", filename, url)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "manga-translator/1.0"})
            with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as out:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
            if tmp.stat().st_size > 0:
                tmp.rename(dest)
                log.info("Downloaded %s (%d bytes)", filename, dest.stat().st_size)
                return True
            tmp.unlink(missing_ok=True)
            return False
        except Exception as e:
            log.warning("Failed to download %s: %s", filename, e)
            tmp.unlink(missing_ok=True)
            return False

    def _load(self) -> bool:
        if self._load_attempted:
            return self.available
        self._load_attempted = True
        with self._load_lock:
            try:
                import onnxruntime as ort
            except ImportError:
                log.warning("onnxruntime not installed, manga-ocr disabled")
                return False

            model_path = MODEL_CACHE / MODEL_FILE
            vocab_path = MODEL_CACHE / VOCAB_FILE

            if not self._download(MODEL_FILE, model_path):
                return False
            if not self._download(VOCAB_FILE, vocab_path):
                return False

            try:
                self._tokenizer = self._load_tokenizer(vocab_path)
                providers = ["CPUExecutionProvider"]
                self._session = ort.InferenceSession(
                    str(model_path), providers=providers
                )
                self.available = True
                log.info("manga-ocr ONNX loaded successfully")
                return True
            except Exception as e:
                log.warning("manga-ocr load failed: %s", e)
                self._session = None
                self.available = False
                return False

    @staticmethod
    def _load_tokenizer(vocab_path: Path):
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        if isinstance(vocab, dict):
            # {"id": "token"} or {"token": "id"}
            if any(isinstance(k, str) and v.isdigit() for k, v in list(vocab.items())[:20]):
                id_to_token = {int(v): k for k, v in vocab.items()}
            else:
                id_to_token = {int(k): v for k, v in vocab.items()}
        elif isinstance(vocab, list):
            id_to_token = {i: t for i, t in enumerate(vocab)}
        else:
            raise ValueError("Unknown vocab format")

        max_id = max(id_to_token)
        return id_to_token, max_id

    def _decode(self, token_ids) -> str:
        id_to_token, _ = self._tokenizer
        # Replace special tokens with markers for spacing (manga-ocr convention)
        out = []
        for tid in token_ids:
            tok = id_to_token.get(int(tid), "")
            if tok in ("<pad>", "<unk>", "<s>", "</s>", "<mask>", ""):
                continue
            if tok.startswith("▁") and out:
                out.append(" ")
                out.append(tok[1:])
            elif tok in ("</s>",):
                continue
            else:
                out.append(tok)
        return "".join(out).strip()

    @staticmethod
    def _preprocess(img: Image.Image, target: int = 960):
        """Convert image to model input following manga-ocr preprocessing."""
        gray = img.convert("L")
        w, h = gray.size
        scale = target / max(w, h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        resized = gray.resize((nw, nh), Image.BILINEAR)

        canvas = Image.new("L", (target, target), 255)
        canvas.paste(resized, (0, 0))

        arr = np.array(canvas, dtype=np.float32)
        # grayscale -> 3 channel
        arr = np.stack([arr] * 3, axis=-1)
        arr = arr / 255.0
        mean = np.array([0.7931, 0.7931, 0.7931], dtype=np.float32)
        std = np.array([0.1738, 0.1738, 0.1738], dtype=np.float32)
        arr = (arr - mean) / std
        # NCHW
        arr = arr.transpose(2, 0, 1)
        return arr[None, ...]

    def ocr(self, image: Image.Image, max_new_tokens: int = 256) -> str:
        """Recognize text in a full page image. Returns '' on failure."""
        if not self._load():
            return ""
        try:
            inputs = self._preprocess(image)
            input_name = self._session.get_inputs()[0].name
            outputs = self._session.run(None, {input_name: inputs})
            logits = outputs[0]
            token_ids = np.argmax(logits, axis=-1)[0]
            return self._decode(token_ids)
        except Exception as e:
            log.warning("manga-ocr inference failed: %s", e)
            return ""


_manga_ocr = None


def get_manga_ocr() -> MangaOCREngine:
    global _manga_ocr
    if _manga_ocr is None:
        _manga_ocr = MangaOCREngine()
    return _manga_ocr


def is_available() -> bool:
    return get_manga_ocr().available

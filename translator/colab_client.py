import json
import os
import io
import asyncio
import numpy as np
from PIL import Image
import httpx


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
        self._paddle_instances: dict[str, any] = {}
        self._paddle_available = False

    async def init(self):
        try:
            loop = asyncio.get_event_loop()
            self._paddle_instances["korean"] = await loop.run_in_executor(None, self._init_paddle, "korean")
            self._paddle_available = True
            print("[OCR] PaddleOCR (korean) on CPU")
        except Exception as e:
            print(f"[OCR] PaddleOCR not available: {e}, fallback to ocr.space")

    @staticmethod
    def _init_paddle(lang: str):
        from paddleocr import PaddleOCR
        return PaddleOCR(lang=lang, use_angle_cls=False, use_gpu=False, show_log=False)

    def _get_paddle(self, lang: str):
        if lang not in self._paddle_instances:
            try:
                self._paddle_instances[lang] = self._init_paddle(lang)
                print(f"[OCR] PaddleOCR ({lang}) initialized")
            except Exception as e:
                print(f"[OCR] PaddleOCR ({lang}) failed: {e}")
                return self._paddle_instances.get("korean")
        return self._paddle_instances[lang]

    @property
    def is_connected(self) -> bool:
        return True

    async def ocr_pages(self, pages: list[bytes], lang: str = "kor") -> list[list[dict]]:
        if self._paddle_available:
            paddle_lang = PADDLE_LANG_MAP.get(lang, "korean")
            return await self._ocr_paddle(pages, paddle_lang)
        return await self._ocr_space(pages, lang)

    async def _ocr_paddle(self, pages: list[bytes], lang: str = "korean") -> list[list[dict]]:
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
                result = await loop.run_in_executor(
                    None, lambda: paddle.ocr(img_np, cls=False)
                )
                results = []
                if result and result[0]:
                    for line in result[0]:
                        poly, (text, conf) = line
                        text = text.strip()
                        if not text or conf < 0.3:
                            continue
                        poly = [[p[0] / scale, p[1] / scale] for p in poly]
                        xs = [int(p[0]) for p in poly]
                        ys = [int(p[1]) for p in poly]
                        l, t, r, b = min(xs), min(ys), max(xs), max(ys)
                        results.append({
                            "bbox": [[l, t], [r, t], [r, b], [l, b]],
                            "polygon": [[int(p[0]), int(p[1])] for p in poly],
                            "text": text,
                            "word_bboxes": [[l, t, r, b]],
                            "confidence": conf,
                        })
                all_results.append(results)
            except Exception as e:
                print(f"[OCR] Paddle error: {e}")
                all_results.append([])
        return all_results

    async def _ocr_space(self, pages: list[bytes], lang: str) -> list[list[dict]]:
        all_results = []
        for page_data in pages:
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
                    print(f"[OCR] Error: {data.get('ErrorMessage', '')}")
                    all_results.append([])
                    continue

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
                        results.append({
                            "bbox": [[left, top], [right, top], [right, bottom], [left, bottom]],
                            "text": line_text,
                            "word_bboxes": word_bboxes,
                        })
                all_results.append(results)
            except Exception as e:
                print(f"[OCR] Request failed: {e}")
                all_results.append([])
        return all_results

    async def inpaint_pages(self, pages, masks):
        return pages

    async def close(self):
        await self.client.aclose()

import json
import os
import httpx
import io
from PIL import Image


def _get_proxy():
    return os.environ.get("HTTP_PROXY") or os.environ.get("HTTPS_PROXY") or os.environ.get("http_proxy") or os.environ.get("https_proxy")


OCR_SPACE_API = "https://api.ocr.space/parse/image"
OCR_SPACE_KEY = "helloworld"


class ColabClient:
    def __init__(self, base_url: str = ""):
        proxy = _get_proxy()
        self.client = httpx.AsyncClient(timeout=120.0, proxy=proxy, verify=False) if proxy else httpx.AsyncClient(timeout=120.0)
        self._connected = True

    async def init(self):
        print("[OCR] ocr.space API (free tier)")

    @property
    def is_connected(self) -> bool:
        return True

    async def ocr_pages(self, pages: list[bytes], lang: str = "kor") -> list[list[dict]]:
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
                    data={
                        "apikey": OCR_SPACE_KEY,
                        "language": lang,
                        "isOverlayRequired": "true",
                    },
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

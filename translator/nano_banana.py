"""Nano Banana (gemini image models) client for high-quality manga text removal.

Nanobanana-style image models regenerate masked areas naturally, often better
than LaMa for complex backgrounds. Wired into the inpaint chain behind a config
flag because image-generation quota is NOT covered by a free Gemini key (429)
and may cost money on a paid plan.

To enable: cfg/config.json ->  "inpaint": {"nano_banana": true, "nano_banana_model": "gemini-3.1-flash-image"}
"""

import base64
import logging
import os
import threading

import httpx

log = logging.getLogger("manga_translator")


class NanoBananaClient:
    def __init__(self, model: str = "gemini-3.1-flash-image", api_key: str = ""):
        self.model = model
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.available = bool(self.api_key)

    @staticmethod
    def _overlay_mask(page: bytes, mask: bytes) -> bytes:
        """Return a PNG where masked (text) areas are highlighted red, to guide the model."""
        import cv2
        import numpy as np

        try:
            page_np = np.frombuffer(page, np.uint8)
            img = cv2.imdecode(page_np, cv2.IMREAD_COLOR)
            mask_np = np.frombuffer(mask, np.uint8)
            m = cv2.imdecode(mask_np, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return page
            if m is not None and m.shape[:2] != img.shape[:2]:
                m = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            overlay = img.copy()
            if m is not None:
                red = np.full_like(img, (0, 0, 255))
                alpha = (m > 127).astype(np.uint8)
                overlay = cv2.addWeighted(img, 0.5, red, 0.5, 0)
                overlay[m <= 127] = img[m <= 127]
            ok, buf = cv2.imencode(".png", overlay)
            return buf.tobytes() if ok else page
        except Exception as e:
            log.warning("nano overlay failed: %s", e)
            return page

    def clean_page(self, page: bytes, mask: bytes | None = None) -> bytes | None:
        """Remove text from a single page. Returns cleaned PNG bytes or None on failure.

        Mask-aware: when a mask is provided, we highlight it and ask for a local fix,
        otherwise we ask for a full-page text removal.
        """
        if not self.api_key:
            return None
        if mask is not None:
            img_b64 = self._overlay_mask(page, mask)
        else:
            img_b64 = page
        prompt = (
            "Clean this manga page: erase all the text/speech-bubble text completely "
            "and repaint the original background and artwork underneath exactly as if "
            "the text had never been there. Keep colors, shapes and layout identical. "
            "Do not leave any letters or traces of text."
        )
        if mask is not None:
            prompt = (
                "Clean this manga page: remove ONLY the text inside the red-highlighted areas "
                "and restore the original background/artwork there. Do not touch the rest of the page."
            )
        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": "image/png", "data": base64.b64encode(img_b64).decode()}},
                ]
            }],
            "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]},
        }
        try:
            resp = httpx.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=180.0,
            )
            if resp.status_code in (429, 403, 400):
                log.warning("Nano Banana → %s (quota/model unavailable), skipping", resp.status_code)
                return None
            resp.raise_for_status()
            cands = resp.json().get("candidates", [])
            if not cands:
                return None
            for part in cands[0].get("content", {}).get("parts", []):
                if "inlineData" in part:
                    return base64.b64decode(part["inlineData"]["data"])
            return None
        except Exception as e:
            log.warning("Nano Banana clean failed: %s", e)
            return None
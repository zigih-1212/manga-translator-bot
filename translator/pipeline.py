import json
import gc
import asyncio
import signal
from pathlib import Path
from PIL import Image
import io
import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from sources.router import SourceRouter
from translator.llm import LLMTranslator
from translator.renderer import TextRenderer, _classify_font_style, _is_caption_region, _polygon_angle
from translator.kaggle_client import KaggleClient
from translator.inpainter import LaMaInpainter
from translator.bubbles import get_bubble_bounds, build_mask, build_smart_mask
from translator.modal_client import inpaint_batch_sync, MODAL_AVAILABLE
from translator.colab_client import ColabClient
from translator.sfx_detector import annotate_sfx, is_sfx_text
from translator.preprocess import preprocess_page, sauvola
# from translator.upscaler import RealESRGANUpscaler  # РћС‚РєР»СЋС‡РµРЅРѕ РґР»СЏ СЌРєРѕРЅРѕРјРёРё RAM
from cfg import TEMP_DIR, CONFIG

from cfg.memory import save_translations, _extract_speaker
from translator.health import inc_metric, record_error, record_ocr, record_llm
from translator.log import log
from editor import store as editor_store

# Chunk size for processing chapters in chunks to avoid OOM
CHUNK_SIZE = 15  # pages per chunk

def _filter_text_regions(ocr_texts: list[dict], img_w: int, img_h: int) -> list[dict]:
    filtered = []
    page_area = img_w * img_h
    for r in ocr_texts:
        bb = r.get("bbox", [])
        if not bb:
            continue
        text = r.get("text", "")
        conf = r.get("confidence", 1.0)
        if conf < 0.5:
            continue
        if isinstance(bb[0], (list, tuple)):
            xs = [p[0] for p in bb]
            ys = [p[1] for p in bb]
            x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
        else:
            x1, y1, x2, y2 = [int(v) for v in bb[:4]]
        bw = max(x2 - x1, 1)
        bh = max(y2 - y1, 1)
        area = bw * bh
        aspect = bw / bh
        if area > 0.35 * page_area:
            continue
        if area < 300:
            continue
        if aspect > 8 or aspect < 0.15:
            continue
        if text and len(text.strip()) <= 2 and area > page_area * 0.02:
            continue
        filtered.append(r)
    return filtered


CREDIT_KEYWORDS = [
    "contents", "copyright", "all rights reserved", "author", "artist",
    "illustrator", "story by", "art by", "special thanks", "first published",
    "editor", "designer", "translation", "production", "originally published",
    "no part of", "permission", "license", "printed in",
    "содержание", "автор", "художник", "издательство", "тираж",
]


def _has_dark_content(img: np.ndarray, threshold: float = 0.02) -> bool:
    """True if a meaningful fraction of pixels are dark (likely text/lines)."""
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        dark = float((gray < 128).mean())
        return dark >= threshold
    except Exception:
        return True


def _is_credit_page(ocr_texts: list[dict], img_w: int, img_h: int) -> bool:
    if not ocr_texts:
        return False
    all_text = " ".join(r.get("text", "") for r in ocr_texts).lower()
    for kw in CREDIT_KEYWORDS:
        if kw in all_text:
            return True
    total_area = 0
    for r in ocr_texts:
        bb = r.get("bbox", [])
        if not bb:
            continue
        if isinstance(bb[0], (list, tuple)):
            xs = [int(p[0]) for p in bb]
            ys = [int(p[1]) for p in bb]
            bw = max(xs) - min(xs)
            bh = max(ys) - min(ys)
        else:
            bw = bb[2] - bb[0]
            bh = bb[3] - bb[1]
        total_area += bw * bh
    if total_area > 0.15 * img_w * img_h:
        return True
    if len(ocr_texts) > 20:
        return True
    return False


def _get_bbox_y(r):
    bb = r.get("bbox", [])
    if not bb:
        return 0
    if isinstance(bb[0], (list, tuple)):
        return int(min(p[1] for p in bb))
    return int(bb[1])


def _get_bbox_x(r):
    bb = r.get("bbox", [])
    if not bb:
        return 0
    if isinstance(bb[0], (list, tuple)):
        xs = [int(p[0]) for p in bb]
        return (min(xs) + max(xs)) // 2
    return (bb[0] + bb[2]) // 2


def _group_texts_by_bubble(ocr_texts: list[dict], y_gap: int = 30) -> list[list[dict]]:
    sorted_y = sorted(ocr_texts, key=_get_bbox_y)
    rows = []
    current = []
    prev_bottom = None
    for r in sorted_y:
        top = _get_bbox_y(r)
        if prev_bottom is not None and top - prev_bottom > y_gap:
            if current:
                rows.append(sorted(current, key=_get_bbox_x, reverse=True))
            current = [r]
        else:
            current.append(r)
        bottom_vals = []
        for item in current:
            bb = item.get("bbox", [])
            if isinstance(bb[0], (list, tuple)):
                bottom_vals.append(int(max(p[1] for p in bb)))
            else:
                bottom_vals.append(int(bb[3]))
        prev_bottom = max(bottom_vals) if bottom_vals else None
    if current:
        rows.append(sorted(current, key=_get_bbox_x, reverse=True))
    return rows


@dataclass
class PageRaw:
    index: int
    src_data: bytes
    en_data: Optional[bytes]
    src_url: str
    en_url: Optional[str]
    img_w: int = 0
    img_h: int = 0


@dataclass
class PageOCR:
    index: int
    src_data: bytes
    en_data: Optional[bytes]
    ocr_texts: List[Dict]
    en_texts: List[str]
    groups: List[List[Dict]]
    grouped_ko: List[str]
    img_w: int = 0
    img_h: int = 0
    is_credit: bool = False


@dataclass
class PageTranslated:
    index: int
    src_data: bytes
    en_data: Optional[bytes]
    translations: List[Dict]
    ocr_texts: List[Dict]
    groups: List[List[Dict]]
    grouped_ko: List[str]
    img_w: int = 0
    img_h: int = 0


@dataclass
class PageRendered:
    index: int
    out_data: bytes
    translations: List[Dict]


class TranslationPipeline:
    def __init__(self, source: str = "mangadex"):
        self.router = SourceRouter()
        self.source_name = source
        self.translator = LLMTranslator()
        self.renderer = TextRenderer()
        self.kaggle = KaggleClient()
        self.colab = ColabClient()
        self.inpainter = LaMaInpainter()
        self.progress_callback = None
        self._bg_mask = None
        # Nano Banana (Gemini image model) — optional high-quality cleanup, off by default
        self.nano_banana = None
        try:
            inpaint_cfg = CONFIG.get("inpaint", {})
            if inpaint_cfg.get("nano_banana"):
                from translator.nano_banana import NanoBananaClient
                self.nano_banana = NanoBananaClient(model=inpaint_cfg.get("nano_banana_model", "gemini-3.1-flash-image"))
                log.info("Nano Banana inpainting enabled (%s)", self.nano_banana.model)
        except Exception as e:
            log.warning("Nano Banana init failed: %s", e)
        # self._upscaler = RealESRGANUpscaler()  # РћС‚РєР»СЋС‡РµРЅРѕ РґР»СЏ СЌРєРѕРЅРѕРјРёРё RAM

    def on_progress(self, callback):
        self.progress_callback = callback

    async def _report(self, message: str, current: int = 0, total: int = 0):
        if self.progress_callback:
            await self.progress_callback(message, current, total)

    @staticmethod
    def _fill_bubble_bg(cv_img: np.ndarray, bubble_bboxes: list[tuple], iterations: int = 2) -> tuple[np.ndarray, np.ndarray]:
        filled = cv_img.copy()
        bg_mask = np.zeros((cv_img.shape[0], cv_img.shape[1]), dtype=np.uint8)
        for (x1, y1, x2, y2) in bubble_bboxes:
            x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, cv_img.shape[1]), min(y2, cv_img.shape[0])
            if x2 - x1 < 10 or y2 - y1 < 10:
                continue
            border_pixels = []
            for x in range(x1, x2, max(1, (x2 - x1) // 5)):
                border_pixels.append((x, y1))
                border_pixels.append((x, y2 - 1))
            for y in range(y1, y2, max(1, (y2 - y1) // 5)):
                border_pixels.append((x1, y))
                border_pixels.append((x2 - 1, y))
            if not border_pixels:
                continue
            colors = [filled[p[1], p[0]].tolist() for p in border_pixels]
            avg_color = [int(sum(c) / len(c)) for c in zip(*colors)]
            mask = np.zeros((cv_img.shape[0] + 2, cv_img.shape[1] + 2), dtype=np.uint8)
            seed = (x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2)
            lo = (10, 10, 10)
            hi = (30, 30, 30)
            try:
                cv2.floodFill(filled, mask, seed, tuple(avg_color), lo, hi)
                cv2.floodFill(bg_mask, mask, seed, 255, lo, hi)
            except Exception:
                pass
        return filled, bg_mask

    @staticmethod
    def _postprocess_inpaint(img_bgr: np.ndarray, inpainted_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Blend inpainted region with original outside the mask; normalize brightness
        of the inpainted area to match its immediate border pixels."""
        result = inpainted_bgr.copy()
        m = (mask > 127).astype(np.uint8)

        # Dilate mask slightly to define border ring around inpainted region
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        border = cv2.dilate(m, kernel, iterations=2)
        ring = border - m
        if cv2.countNonZero(ring) > 0:
            # Reference color from border of the ORIGINAL image
            orig_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
            inp_lab = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
            ref_l = orig_lab[:, :, 0][ring > 0]
            inp_l = inp_lab[:, :, 0][ring > 0]
            if len(ref_l) > 10:
                # Ratio to shift inpainted lightness toward border lightness
                ratio = float(ref_l.mean()) / max(float(inp_l.mean()), 1.0)
                ratio = np.clip(ratio, 0.7, 1.3)
                inp_lab[:, :, 0] = np.clip(inp_lab[:, :, 0] * ratio, 0, 255)
                corrected = cv2.cvtColor(inp_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
                # Only apply corrected pixels inside the inpainted mask
                inside = m > 0
                result[inside] = corrected[inside]

        # Soft-blend the seam (5px feather)
        seam_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        soft = cv2.dilate(m, seam_kernel, iterations=1)
        soft = cv2.GaussianBlur(soft.astype(np.float32), (0, 0), 2.0)
        soft = (soft / 255.0)[..., None]
        return (result.astype(np.float32) * soft + img_bgr.astype(np.float32) * (1.0 - soft)).clip(0, 255).astype(np.uint8)

    def _inpaint_page(self, filled_img: np.ndarray, remaining: np.ndarray) -> np.ndarray:
        """Inpaint chain: Modal (GPU) -> Colab (GPU) -> Nano Banana -> LaMa (CPU)."""
        _, img_bytes = cv2.imencode(".png", filled_img)
        _, mask_bytes = cv2.imencode(".png", remaining)
        if MODAL_AVAILABLE:
            try:
                modal_result = inpaint_batch_sync(
                    [img_bytes.tobytes()],
                    masks=[mask_bytes.tobytes()],
                )
                if modal_result:
                    return cv2.imdecode(np.frombuffer(modal_result[0], np.uint8), cv2.IMREAD_COLOR)
            except Exception as e:
                log.warning("Modal inpainting failed: %s", e)
        if self.colab.available:
            try:
                colab_result = self.colab.inpaint_batch(
                    [img_bytes.tobytes()],
                    masks=[mask_bytes.tobytes()],
                )
                if colab_result:
                    log.info("Colab inpainting used as fallback")
                    return cv2.imdecode(np.frombuffer(colab_result[0], np.uint8), cv2.IMREAD_COLOR)
            except Exception as e:
                log.warning("Colab inpainting failed: %s", e)
        if self.nano_banana is not None:
            try:
                nb_result = self.nano_banana.clean_page(img_bytes.tobytes(), mask_bytes.tobytes())
                if nb_result:
                    log.info("Nano Banana inpainting used")
                    return cv2.imdecode(np.frombuffer(nb_result, np.uint8), cv2.IMREAD_COLOR)
            except Exception as e:
                log.warning("Nano Banana inpainting failed: %s", e)
        return self.inpainter.inpaint(filled_img, remaining)

    @staticmethod
    def _auto_rotate(page_data: bytes) -> bytes:
        try:
            np_img = np.frombuffer(page_data, np.uint8)
            cv_img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
            if cv_img is None:
                return page_data
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            coords = np.column_stack(np.where(binary > 0))
            if len(coords) < 100:
                return page_data
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            if abs(angle) > 1.0:
                h, w = cv_img.shape[:2]
                M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
                cv_img = cv2.warpAffine(cv_img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
                _, buf = cv2.imencode(".png", cv_img)
                return buf.tobytes()
        except Exception:
            pass
        return page_data

    async def process_chapter(
        self,
        mangadex_manga_id: str,
        chapter_number: str,
        source_lang: str = "ko",
        target_lang: str = "en",
        source: str | None = None,
    ) -> list[bytes]:
        source = source or self.source_name
        work_dir = TEMP_DIR / f"chapter_{chapter_number}"
        work_dir.mkdir(parents=True, exist_ok=True)

        await self.kaggle.init()

        await self._report(f"РџРѕРёСЃРє {source_lang} РіР»Р°РІС‹ {chapter_number}...", 0, 100)
        src_chapter = await self.router.find_chapter_by_number(
            source, mangadex_manga_id, chapter_number, source_lang
        )
        if not src_chapter:
            await self._report(f"Р“Р»Р°РІР° {chapter_number} ({source_lang}) РЅРµ РЅР°Р№РґРµРЅР°!", 0, 1)
            return []

        await self._report(f"РџРѕРёСЃРє {target_lang} РіР»Р°РІС‹ {chapter_number}...", 2, 100)
        en_chapter = await self.router.find_chapter_by_number(
            source, mangadex_manga_id, chapter_number, target_lang
        )

        await self._report(f"РџРѕР»СѓС‡Р°РµРј СЃРїРёСЃРѕРє СЃС‚СЂР°РЅРёС† {source_lang}...", 5, 100)
        src_pages = await self.router.get_pages(source, src_chapter.id, mangadex_manga_id)
        total_pages = len(src_pages)
        if not total_pages:
            await self._report("РќРµС‚ СЃС‚СЂР°РЅРёС†!", 0, 1)
            return []

        en_page_map = {}
        if en_chapter:
            await self._report(f"РџРѕР»СѓС‡Р°РµРј СЃРїРёСЃРѕРє СЃС‚СЂР°РЅРёС† {target_lang}...", 7, 100)
            en_pages = await self.router.get_pages(source, en_chapter.id, mangadex_manga_id)
            en_page_map = {p.index: p for p in en_pages}

        self.translator.clear_context()

        # ========== ASYNC PIPELINE STAGES ==========
        # Stage 1: Downloader -> raw_queue
        # Stage 2: Preprocessor (auto-rotate, upscale, OCR) -> ocr_queue
        # Stage 3: Translator -> translated_queue
        # Stage 4: Inpainter + Renderer -> rendered_queue
        # Stage 5: Saver -> done

        # Process in chunks to avoid OOM
        all_final_paths = []
        chapter_translations: list[dict] = []

        for chunk_start in range(0, total_pages, CHUNK_SIZE):
            chunk_end = min(chunk_start + CHUNK_SIZE, total_pages)
            chunk_size = chunk_end - chunk_start
            
            await self._report(f"РћР±СЂР°Р±Р°С‚С‹РІР°РµРј С‡Р°РЅРє {chunk_start//CHUNK_SIZE + 1}/{(total_pages + CHUNK_SIZE - 1)//CHUNK_SIZE} (СЃС‚СЂ. {chunk_start+1}-{chunk_end})", 
                             5 + int(90 * chunk_start / total_pages), 100)

            # Initialize queues for this chunk
            raw_queue = asyncio.Queue()
            ocr_queue = asyncio.Queue()
            translated_queue = asyncio.Queue()
            rendered_queue = asyncio.Queue()
            done_event = asyncio.Event()

            # Results storage for final collection
            rendered_results = {}
            results_lock = asyncio.Lock()

            # Store page indices mapping for this chunk
            chunk_page_indices = list(range(chunk_start, chunk_end))

            # ---- Stage 1: Downloader (fetches pages) ----
            async def stage_downloader():
                # РЈРјРµРЅСЊС€Р°РµРј РїР°СЂР°Р»Р»РµР»СЊРЅРѕСЃС‚СЊ РґР»СЏ СЌРєРѕРЅРѕРјРёРё RAM
                sem = asyncio.Semaphore(1)  # max 1 concurrent download
                async def download_one(i: int):
                    async with sem:
                        try:
                            global_index = chunk_start + i
                            src_data = await self.router.download_page(source, src_pages[global_index])
                            en_data = None
                            if global_index in en_page_map:
                                en_data = await self.router.download_page(source, en_page_map[global_index])
                            # Get image dimensions
                            tmp_img = Image.open(io.BytesIO(src_data))
                            img_w, img_h = tmp_img.size
                            tmp_img.close()
                            return {
                                "index": i,  # local index within chunk
                                "src_data": src_data,
                                "en_data": en_data,
                                "img_w": img_w,
                                "img_h": img_h,
                            }
                        except Exception as e:
                            await self._report(f"РћС€РёР±РєР° СЃРєР°С‡РёРІР°РЅРёСЏ СЃС‚СЂ. {chunk_start+i+1}: {e}", 
                                             10 + int(80 * (chunk_start+i) / total_pages), 100)
                            return None

                # Launch all downloads with limited concurrency
                tasks = [asyncio.create_task(download_one(i)) for i in range(chunk_size)]
                for i, task in enumerate(tasks):
                    result = await task
                    if result:
                        await raw_queue.put(result)
                    else:
                        await raw_queue.put({"index": i, "error": True})
                await raw_queue.put(None)  # sentinel

            # ---- Stage 2: Preprocessor (auto-rotate, upscale, OCR) ----
            async def stage_preprocessor():
                # РџР°СЂР°Р»Р»РµР»СЊРЅС‹Р№ OCR: РЅРµСЃРєРѕР»СЊРєРѕ СЃС‚СЂР°РЅРёС† РѕРґРЅРѕРІСЂРµРјРµРЅРЅРѕ (Р»РёРјРёС‚РµСЂ Р·Р°С‰РёС‰Р°РµС‚)
                sem = asyncio.Semaphore(3)
                async def preprocess(item):
                    if item.get("error"):
                        await ocr_queue.put({"index": item["index"], "error": True})
                        return
                    async with sem:
                        idx = item["index"]
                        src_data = item["src_data"]
                        en_data = item["en_data"]
                        img_w = item["img_w"]
                        img_h = item["img_h"]

                        try:
                            # Auto-rotate
                            src_data = self._auto_rotate(src_data)

                            # Upscale (РѕС‚РєР»СЋС‡РµРЅРѕ РґР»СЏ СЌРєРѕРЅРѕРјРёРё RAM)
                            # if self._upscaler.available:
                            #     try:
                            #         np_img = np.frombuffer(src_data, np.uint8)
                            #         cv_img_up = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
                            #         cv_img_up = self._upscaler.upscale(cv_img_up)
                            #         _, src_data = cv2.imencode(".png", cv_img_up)
                            #         src_data = src_data.tobytes()
                            #     except Exception:
                            #         pass

                            # OCR source
                            ocr_texts = []
                            if self.kaggle.is_connected:
                                try:
                                    preprocessed = src_data
                                    try:
                                        arr = np.frombuffer(src_data, np.uint8)
                                        cv_page = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                                        if cv_page is not None:
                                            cv_page = preprocess_page(cv_page)
                                            ok, enc = cv2.imencode(".png", cv_page)
                                            if ok:
                                                preprocessed = enc.tobytes()
                                    except Exception:
                                        pass
                                    ocr_result = await self.kaggle.ocr_pages([preprocessed], lang=source_lang)
                                    ocr_texts = ocr_result[0] if ocr_result else []

                                    # Auto-retry: page has visible dark pixels but OCR found nothing
                                    if not ocr_texts:
                                        try:
                                            arr = np.frombuffer(src_data, np.uint8)
                                            cv_page = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                                            if cv_page is not None and _has_dark_content(cv_page):
                                                # Retry with Sauvola binarization for better contrast
                                                cv_binary = sauvola(cv_page)
                                                cv_retry = cv2.cvtColor(cv_binary, cv2.COLOR_GRAY2BGR)
                                                ok, enc = cv2.imencode(".png", cv_retry)
                                                if ok:
                                                    ocr_result = await self.kaggle.ocr_pages([enc.tobytes()], lang=source_lang)
                                                    ocr_texts = ocr_result[0] if ocr_result else []
                                        except Exception:
                                            pass
                                except Exception:
                                    ocr_texts = []
                            ocr_texts = _filter_text_regions(ocr_texts, img_w, img_h)
                            annotate_sfx(ocr_texts, source_lang=source_lang)

                            # Credit page detection
                            is_credit = False
                            if ocr_texts and _is_credit_page(ocr_texts, img_w, img_h):
                                is_credit = True
                                ocr_texts = []

                            # Group texts
                            groups = _group_texts_by_bubble(ocr_texts)
                            grouped_ko = [" ".join(r.get("text", "") for r in g if r.get("text")) for g in groups]
                            sfx_flags = [bool(g) and all(r.get("sfx") for r in g) for g in groups]

                            # Detect speakers for character voice consistency
                            # Build a temporary glossary from existing memory for speaker matching
                            from cfg.memory import _load as _load_memory
                            memory_data = _load_memory()
                            manga_memory = memory_data.get(mangadex_manga_id, {})
                            glossary_for_speaker = manga_memory.get("glossary", {})
                            speakers = [_extract_speaker(ko, glossary_for_speaker) for ko in grouped_ko]

                            # OCR English reference
                            en_texts = []
                            if en_data and ocr_texts and self.kaggle.is_connected:
                                try:
                                    en_ocr = await self.kaggle.ocr_pages([en_data], lang=target_lang)
                                    en_texts = [r.get("text", "") for r in (en_ocr[0] if en_ocr else []) if r.get("text")]
                                except Exception:
                                    pass

                            await ocr_queue.put({
                                "index": item["index"],
                                "src_data": src_data,
                                "en_data": en_data,
                                "ocr_texts": ocr_texts,
                                "en_texts": en_texts,
                                "groups": groups,
                                "grouped_ko": grouped_ko,
                                "sfx_flags": sfx_flags,
                                "speakers": speakers,
                                "img_w": img_w,
                                "img_h": img_h,
                                "is_credit": is_credit,
                            })
                        except Exception as e:
                            await self._report(f"Preprocess error page {chunk_start+idx+1}: {e}", 
                                             10 + int(80 * (chunk_start+idx) / total_pages), 100)
                            await ocr_queue.put({"index": item["index"], "error": True})

                while True:
                    item = await raw_queue.get()
                    if item is None:  # sentinel
                        await ocr_queue.put(None)
                        raw_queue.task_done()
                        break
                    await preprocess(item)
                    raw_queue.task_done()

            # ---- Stage 3: Translator ----
            async def stage_translator():
                while True:
                    item = await ocr_queue.get()
                    if item is None:  # sentinel
                        await translated_queue.put(None)
                        ocr_queue.task_done()
                        break

                    if item.get("error") or item.get("is_credit"):
                        await translated_queue.put({
                            "index": item["index"],
                            "src_data": item.get("src_data"),
                            "en_data": item.get("en_data"),
                            "translations": [],
                            "ocr_texts": item.get("ocr_texts", []),
                            "groups": item.get("groups", []),
                            "grouped_ko": item.get("grouped_ko", []),
                            "sfx_flags": item.get("sfx_flags", []),
                            "speakers": item.get("speakers", []),
                            "img_w": item.get("img_w", 0),
                            "img_h": item.get("img_h", 0),
                            "skip_render": True,
                        })
                        ocr_queue.task_done()
                        continue

                    idx = item["index"]
                    grouped_ko = item["grouped_ko"]
                    en_texts = item["en_texts"]
                    img_w = item["img_w"]
                    img_h = item["img_h"]
                    speakers = item.get("speakers", [])

                    try:
                        await self._report(f"РџРµСЂРµРІРѕРґ СЃС‚СЂ. {chunk_start+idx+1}/{total_pages}", 
                                         10 + int(80 * (chunk_start+idx) / total_pages), 100)
                        translations = await self.translator.translate_page(
                            korean_texts=grouped_ko,
                            english_texts=item["en_texts"],
                            page_number=chunk_start + idx + 1,
                            manga_id=mangadex_manga_id,
                            chapter=chapter_number,
                            source_lang=source_lang,
                            page_image=item["src_data"],
                        )
                    except Exception as e:
                        await self._report(f"РџРµСЂРµРІРѕРґ РѕС€РёР±РєР° СЃС‚СЂ. {chunk_start+idx+1}: {e}", 
                                         10 + int(80 * (chunk_start+idx) / total_pages), 100)
                        translations = []

                    # Attach detected speakers to translations for memory/profile update
                    for t, sp in zip(translations, speakers):
                        if sp:
                            t["speaker"] = sp

                    # Preserve SFX regions verbatim (do not render translation over them)
                    sfx_flags = item.get("sfx_flags", [])
                    for t, ko, is_sfx in zip(translations, grouped_ko, sfx_flags):
                        if is_sfx:
                            t["ru"] = ko
                            t["sfx"] = True

                    chapter_translations.extend(translations)

                    await translated_queue.put({
                        "index": idx,
                        "src_data": item["src_data"],
                        "en_data": item.get("en_data"),
                        "translations": translations,
                        "ocr_texts": item["ocr_texts"],
                        "groups": item["groups"],
                        "grouped_ko": grouped_ko,
                        "sfx_flags": sfx_flags,
                        "speakers": speakers,
                        "img_w": img_w,
                        "img_h": img_h,
                    })
                    ocr_queue.task_done()

            # ---- Stage 4: Inpainter + Renderer ----
            async def stage_inpainter_renderer():
                # РЈРјРµРЅСЊС€Р°РµРј РїР°СЂР°Р»Р»РµР»СЊРЅРѕСЃС‚СЊ РґР»СЏ СЌРєРѕРЅРѕРјРёРё RAM
                sem = asyncio.Semaphore(1)
                while True:
                    item = await translated_queue.get()
                    if item is None:  # sentinel
                        await rendered_queue.put(None)
                        translated_queue.task_done()
                        break

                    if item.get("skip_render") or not item["translations"]:
                        # Just save original
                        idx = item["index"]
                        src_data = item.get("src_data", b"")
                        async with results_lock:
                            rendered_results[idx] = src_data
                        translated_queue.task_done()
                        continue

                    async with sem:
                        idx = item["index"]
                        src_data = item["src_data"]
                        translations = item["translations"]
                        ocr_texts = item["ocr_texts"]
                        groups = item["groups"]

                        try:
                            img = Image.open(io.BytesIO(src_data)).convert("RGB")
                            cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                            h, w = cv_img.shape[:2]

                            bubble_pairs = []
                            all_bubble_bboxes = []
                            for g_idx, group in enumerate(groups):
                                ru = translations[g_idx].get("ru", "").strip() if g_idx < len(translations) else ""
                                if not ru:
                                    continue
                                # Skip rendering SFX text (preserve original artwork)
                                if translations[g_idx].get("sfx"):
                                    continue
                                xs, ys = [], []
                                for r in group:
                                    bb = r.get("bbox", [])
                                    if not bb:
                                        continue
                                    if isinstance(bb[0], (list, tuple)):
                                        xs.extend(int(p[0]) for p in bb)
                                        ys.extend(int(p[1]) for p in bb)
                                    else:
                                        xs.extend([bb[0], bb[2]])
                                        ys.extend([bb[1], bb[3]])
                                if not xs:
                                    continue
                                text_bbox = (min(xs), min(ys), max(xs), max(ys))
                                bubble_bbox, is_bubble = get_bubble_bounds(cv_img, text_bbox, w, h)
                                # Estimate text skew from OCR polygons for warp-rendering
                                angle = 0.0
                                for r in group:
                                    poly = r.get("polygon")
                                    if poly:
                                        angle = _polygon_angle(poly)
                                        break
                                all_bubble_bboxes.append(bubble_bbox)
                                ko = translations[g_idx].get("ko", "") or (item.get("grouped_ko") or [""])[min(g_idx, len(item.get("grouped_ko") or [""]) - 1)]
                                bubble_pairs.append((g_idx, bubble_bbox, ru, ko, is_bubble, text_bbox, angle))

                            if not bubble_pairs:
                                async with results_lock:
                                    rendered_results[idx] = src_data
                                translated_queue.task_done()
                                continue

                            filled_img, self._bg_mask = self._fill_bubble_bg(cv_img, all_bubble_bboxes)

                            mask = build_mask(h, w, all_bubble_bboxes)
                            # Smart mask: exactly cover text glyphs for cleaner inpaint
                            try:
                                text_boxes = []
                                for r in item["ocr_texts"]:
                                    bb = r.get("bbox", [])
                                    if not bb:
                                        continue
                                    if isinstance(bb[0], (list, tuple)):
                                        xs = [int(p[0]) for p in bb]
                                        ys = [int(p[1]) for p in bb]
                                        text_boxes.append((min(xs), min(ys), max(xs), max(ys)))
                                    else:
                                        text_boxes.append((int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])))
                                if text_boxes:
                                    smart = build_smart_mask(cv_img, text_boxes)
                                    # Combine smart (precise) with rectangular (safe coverage)
                                    mask = cv2.bitwise_or(mask, smart)
                            except Exception:
                                pass
                            for r in item["ocr_texts"]:
                                poly = r.get("polygon")
                                if poly:
                                    pts = np.array([[(p[0], p[1]) for p in poly]], dtype=np.int32)
                                    cv2.fillPoly(mask, pts, 255)

                            remaining = cv2.bitwise_and(mask, cv2.bitwise_not(self._bg_mask))

                            if cv2.countNonZero(remaining) == 0:
                                clean_img = Image.fromarray(cv2.cvtColor(filled_img, cv2.COLOR_BGR2RGB))
                            else:
                                inpainted_bgr = self._inpaint_page(filled_img, remaining)
                                # Post-process: match inpainted lightness to bubble border
                                inpainted_bgr = self._postprocess_inpaint(cv_img, inpainted_bgr, remaining)
                                clean_img = Image.fromarray(cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB))
                                del inpainted_bgr
                            del mask, filled_img

                            bubbles_meta = []
                            for g_idx, bubble_bbox, text, ko, is_bubble, text_bbox, angle in bubble_pairs:
                                try:
                                    # Classify caption (rectangular) vs dialogue (round bubble)
                                    font_type = "dialogue"
                                    if not is_bubble or _is_caption_region(cv_img, text_bbox):
                                        font_type = "narration"
                                    else:
                                        # Font-style matching: handwritten vs clean
                                        bubble_mask_img = self._bg_mask
                                        if _classify_font_style(img, bubble_mask_img, text_bbox) == "narration":
                                            font_type = "narration"
                                    clean_img = self.renderer.render_bubble_text(
                                        clean_img, bubble_bbox, text, font_type=font_type, is_bubble=is_bubble,
                                        original_img=img, angle=angle,
                                    )
                                    bubbles_meta.append({
                                        "id": g_idx,
                                        "bbox": list(bubble_bbox),
                                        "text_bbox": list(text_bbox),
                                        "is_bubble": bool(is_bubble),
                                        "angle": angle,
                                        "font_type": font_type,
                                        "text": text,
                                        "ko": ko,
                                        "edited": False,
                                    })
                                except Exception:
                                    continue

                            buf = io.BytesIO()
                            clean_img.save(buf, format="PNG")
                            out_data = buf.getvalue()
                            del img, clean_img, buf

                            # Сохранение данных для веб-редактора
                            try:
                                await asyncio.to_thread(
                                    editor_store.save_page,
                                    mangadex_manga_id, chapter_number,
                                    chunk_start + idx,
                                    src_data, out_data,
                                    bubbles_meta,
                                    w, h,
                                )
                            except Exception:
                                pass

                            async with results_lock:
                                rendered_results[idx] = out_data
                        except Exception:
                            # Auto-retry: transient inpaint/render errors should not drop the page
                            try:
                                img = Image.open(io.BytesIO(src_data)).convert("RGB")
                                cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                                filled_img, self._bg_mask = self._fill_bubble_bg(cv_img, all_bubble_bboxes)
                                mask = build_mask(h, w, all_bubble_bboxes)
                                remaining = cv2.bitwise_and(mask, cv2.bitwise_not(self._bg_mask))
                                if cv2.countNonZero(remaining) == 0:
                                    clean_img = Image.fromarray(cv2.cvtColor(filled_img, cv2.COLOR_BGR2RGB))
                                else:
                                    inpainted_bgr = self._inpaint_page(filled_img, remaining)
                                    inpainted_bgr = self._postprocess_inpaint(cv_img, inpainted_bgr, remaining)
                                    clean_img = Image.fromarray(cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB))
                                    del inpainted_bgr
                                del mask, filled_img
                                for g_idx, bubble_bbox, text, ko, is_bubble, text_bbox, angle in bubble_pairs:
                                    try:
                                        font_type = "dialogue"
                                        if not is_bubble or _is_caption_region(cv_img, text_bbox):
                                            font_type = "narration"
                                        else:
                                            bubble_mask_img = self._bg_mask
                                            if _classify_font_style(img, bubble_mask_img, text_bbox) == "narration":
                                                font_type = "narration"
                                        clean_img = self.renderer.render_bubble_text(
                                            clean_img, bubble_bbox, text, font_type=font_type, is_bubble=is_bubble,
                                            original_img=img, angle=angle,
                                        )
                                    except Exception:
                                        continue
                                buf = io.BytesIO()
                                clean_img.save(buf, format="PNG")
                                out_data = buf.getvalue()
                                del img, clean_img, buf
                                async with results_lock:
                                    rendered_results[idx] = out_data
                            except Exception:
                                record_error()
                                async with results_lock:
                                    rendered_results[idx] = src_data
                        translated_queue.task_done()

            # ---- Stage 5: Saver ----
            async def stage_saver():
                for i in range(chunk_size):
                    # Wait for this page's result
                    while i not in rendered_results:
                        await asyncio.sleep(0.05)

                    async with results_lock:
                        out_data = rendered_results.pop(i)

                    out_path = work_dir / f"{(chunk_start+i):03d}.png"
                    out_path.write_bytes(out_data)
                    await self._report(f"РЎРѕС…СЂР°РЅРµРЅРѕ СЃС‚СЂ. {chunk_start+i+1}/{total_pages}", 
                                     10 + int(80 * (chunk_start+i) / total_pages), 100)

                done_event.set()

            # Launch all stages
            await self._report("Р—Р°РїСѓСЃРє РєРѕРЅРІРµР№РµСЂР°...", 
                             10 + int(80 * chunk_start / total_pages), 100)

            stage_tasks = [
                asyncio.create_task(stage_downloader()),
                asyncio.create_task(stage_preprocessor()),
                asyncio.create_task(stage_translator()),
                asyncio.create_task(stage_inpainter_renderer()),
                asyncio.create_task(stage_saver()),
            ]

            # Wait for downloader to finish feeding
            await stage_tasks[0]

            # Wait for all queues to be processed
            await raw_queue.join()
            await ocr_queue.join()
            await translated_queue.join()
            await rendered_queue.join()

            # Wait for saver to finish
            await done_event.wait()
            await stage_tasks[4]

            # Collect final pages in order for this chunk
            chunk_final_paths = []
            for i in range(chunk_size):
                path = work_dir / f"{(chunk_start+i):03d}.png"
                if path.exists():
                    chunk_final_paths.append(str(path))

            all_final_paths.extend(chunk_final_paths)

            # Clear memory between chunks
            gc.collect()

            # Clear translator context to prevent memory buildup
            self.translator.clear_context()

            # Report progress
            await self._report(f"Р§Р°РЅРє {chunk_start//CHUNK_SIZE + 1} Р·Р°РІРµСЂС€РµРЅ. Р’СЃРµРіРѕ РѕР±СЂР°Р±РѕС‚Р°РЅРѕ: {chunk_end}/{total_pages} СЃС‚СЂ.", 
                             10 + int(90 * chunk_end / total_pages), 100)

        # РЎРѕС…СЂР°РЅСЏРµРј РїРµСЂРµРІРѕРґС‹ РІ РїР°РјСЏС‚СЊ
        if all_final_paths:
            await asyncio.to_thread(save_translations, mangadex_manga_id, mangadex_manga_id, chapter_number, chapter_translations)

        # LLM-курируемый глоссарий главы (Groq, бесплатно) — стабилизирует имена/термы
        try:
            from translator.glossary_builder import build_glossary, merge_glossary
            if chapter_translations:
                new_glossary = await build_glossary(chapter_translations, source_lang)
                added = merge_glossary(new_glossary)
                if added:
                    log.info("Glossary updated: +%d entries", added)
        except Exception as e:
            log.warning("Chapter glossary build failed: %s", e)

        await self._report("Р“РѕС‚РѕРІРѕ!", 100, 100)
        if all_final_paths:
            inc_metric("chapters_processed")
            inc_metric("pages_processed", len(all_final_paths))
        return all_final_paths

    async def render_single_page(
        self,
        src_data: bytes,
        bubbles_meta: list[dict],
        img_w: int,
        img_h: int,
    ) -> bytes:
        """Перерисовать одну страницу с учётом правок (для веб-редактора).

        Принимает список пузырей с полями: bbox, text_bbox, is_bubble, angle,
        font_type, text. Использует те же алгоритмы, что и основной пайплайн.
        """
        try:
            img = Image.open(io.BytesIO(src_data)).convert("RGB")
            cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            h, w = cv_img.shape[:2]

            all_bubble_bboxes = []
            text_boxes = []
            active = []
            for b in bubbles_meta:
                bb = b.get("bbox")
                if not bb or not (b.get("text") or "").strip():
                    continue
                all_bubble_bboxes.append(tuple(int(v) for v in bb))
                tb = b.get("text_bbox") or bb
                text_boxes.append(tuple(int(v) for v in tb))
                active.append(b)

            if not all_bubble_bboxes:
                return src_data

            filled_img, self._bg_mask = self._fill_bubble_bg(cv_img, all_bubble_bboxes)
            mask = build_mask(h, w, all_bubble_bboxes)
            try:
                if text_boxes:
                    smart = build_smart_mask(cv_img, text_boxes)
                    mask = cv2.bitwise_or(mask, smart)
            except Exception:
                pass

            remaining = cv2.bitwise_and(mask, cv2.bitwise_not(self._bg_mask))

            if cv2.countNonZero(remaining) == 0:
                clean_img = Image.fromarray(cv2.cvtColor(filled_img, cv2.COLOR_BGR2RGB))
            else:
                inpainted_bgr = self._inpaint_page(filled_img, remaining)
                inpainted_bgr = self._postprocess_inpaint(cv_img, inpainted_bgr, remaining)
                clean_img = Image.fromarray(cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB))
                del inpainted_bgr
            del mask, filled_img

            for b in active:
                try:
                    clean_img = self.renderer.render_bubble_text(
                        clean_img,
                        tuple(int(v) for v in b["bbox"]),
                        b["text"],
                        font_type=b.get("font_type", "dialogue"),
                        is_bubble=b.get("is_bubble", True),
                        original_img=img,
                        angle=b.get("angle", 0.0),
                    )
                except Exception:
                    continue

            buf = io.BytesIO()
            clean_img.save(buf, format="PNG")
            return buf.getvalue()
        except Exception:
            record_error()
            return src_data

    async def close(self):
        await self.router.close()
        await self.kaggle.close()
        await self.translator.close()
        self.inpainter.close()  # Ensure inpainter is closed

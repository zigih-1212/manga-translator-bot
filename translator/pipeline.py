import json
import gc
import asyncio
from pathlib import Path
from PIL import Image
import io
import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from sources.mangadex import MangaDexSource
from translator.llm import LLMTranslator
from translator.renderer import TextRenderer, _classify_font_style, _is_caption_region
from translator.colab_client import ColabClient
from translator.inpainter import LaMaInpainter
from translator.bubbles import get_bubble_bounds, build_mask
from translator.modal_client import inpaint_batch_sync, MODAL_AVAILABLE
# from translator.upscaler import RealESRGANUpscaler  # Отключено для экономии RAM
from cfg import TEMP_DIR

from cfg.memory import save_translations

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
    def __init__(self):
        self.mangadex = MangaDexSource()
        self.translator = LLMTranslator()
        self.renderer = TextRenderer()
        self.colab = ColabClient()
        self.inpainter = LaMaInpainter()
        self.progress_callback = None
        self._bg_mask = None
        # self._upscaler = RealESRGANUpscaler()  # Отключено для экономии RAM

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
    ) -> list[bytes]:
        work_dir = TEMP_DIR / f"chapter_{chapter_number}"
        work_dir.mkdir(parents=True, exist_ok=True)

        await self.colab.init()

        await self._report(f"Поиск {source_lang} главы {chapter_number}...", 0, 100)
        src_chapter = await self.mangadex.find_chapter_by_number(
            mangadex_manga_id, chapter_number, source_lang
        )
        if not src_chapter:
            await self._report(f"Глава {chapter_number} ({source_lang}) не найдена!", 0, 1)
            return []

        await self._report(f"Поиск {target_lang} главы {chapter_number}...", 2, 100)
        en_chapter = await self.mangadex.find_chapter_by_number(
            mangadex_manga_id, chapter_number, target_lang
        )

        await self._report(f"Получаем список страниц {source_lang}...", 5, 100)
        src_pages = await self.mangadex.get_pages(src_chapter.id)
        total_pages = len(src_pages)
        if not total_pages:
            await self._report("Нет страниц!", 0, 1)
            return []

        en_page_map = {}
        if en_chapter:
            await self._report(f"Получаем список страниц {target_lang}...", 7, 100)
            en_pages = await self.mangadex.get_pages(en_chapter.id)
            en_page_map = {p.index: p for p in en_pages}

        self.translator.clear_context()

        # ========== ASYNC PIPELINE STAGES ==========
        # Stage 1: Downloader -> raw_queue
        # Stage 2: Preprocessor (auto-rotate, upscale, OCR) -> ocr_queue
        # Stage 3: Translator -> translated_queue
        # Stage 4: Inpainter + Renderer -> rendered_queue
        # Stage 5: Saver -> done

        raw_queue = asyncio.Queue()
        ocr_queue = asyncio.Queue()
        translated_queue = asyncio.Queue()
        rendered_queue = asyncio.Queue()
        done_event = asyncio.Event()

        # Results storage for final collection
        rendered_results = {}
        results_lock = asyncio.Lock()

        # ---- Stage 1: Downloader (fetches pages) ----
        async def stage_downloader():
            # Уменьшаем параллельность для экономии RAM
            sem = asyncio.Semaphore(1)  # max 1 concurrent download
            async def download_one(i: int):
                async with sem:
                    try:
                        src_data = await self.mangadex.download_page(src_pages[i])
                        en_data = None
                        if i in en_page_map:
                            en_data = await self.mangadex.download_page(en_page_map[i])
                        # Get image dimensions
                        tmp_img = Image.open(io.BytesIO(src_data))
                        img_w, img_h = tmp_img.size
                        tmp_img.close()
                        return {
                            "index": i,
                            "src_data": src_data,
                            "en_data": en_data,
                            "img_w": img_w,
                            "img_h": img_h,
                        }
                    except Exception as e:
                        await self._report(f"Ошибка скачивания стр. {i+1}: {e}", 10 + int(80 * i / total_pages), 100)
                        return None

            # Launch all downloads with limited concurrency
            tasks = [asyncio.create_task(download_one(i)) for i in range(total_pages)]
            for i, task in enumerate(tasks):
                result = await task
                if result:
                    await raw_queue.put(result)
                else:
                    await raw_queue.put({"index": i, "error": True})
            await raw_queue.put(None)  # sentinel

        # ---- Stage 2: Preprocessor (auto-rotate, upscale, OCR) ----
        async def stage_preprocessor():
            # Уменьшаем параллельность для экономии RAM
            sem = asyncio.Semaphore(1)  # max 1 concurrent OCR
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

                        # Upscale (отключено для экономии RAM)
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
                        if self.colab.is_connected:
                            try:
                                ocr_result = await self.colab.ocr_pages([src_data], lang=source_lang)
                                ocr_texts = ocr_result[0] if ocr_result else []
                            except Exception:
                                ocr_texts = []
                        ocr_texts = _filter_text_regions(ocr_texts, img_w, img_h)

                        # Credit page detection
                        is_credit = False
                        if ocr_texts and _is_credit_page(ocr_texts, img_w, img_h):
                            is_credit = True
                            ocr_texts = []

                        # Group texts
                        groups = _group_texts_by_bubble(ocr_texts)
                        grouped_ko = [" ".join(r.get("text", "") for r in g if r.get("text")) for g in groups]

                        # Detect speakers for character voice consistency
                        # Build a temporary glossary from existing memory for speaker matching
                        from cfg.memory import _load as _load_memory
                        memory_data = _load_memory()
                        manga_memory = memory_data.get(mangadex_manga_id, {})
                        glossary_for_speaker = manga_memory.get("glossary", {})
                        speakers = [_extract_speaker(ko, glossary_for_speaker) for ko in grouped_ko]

                        # OCR English reference
                        en_texts = []
                        if en_data and ocr_texts and self.colab.is_connected:
                            try:
                                en_ocr = await self.colab.ocr_pages([en_data], lang=target_lang)
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
                            "speakers": speakers,
                            "img_w": img_w,
                            "img_h": img_h,
                            "is_credit": is_credit,
                        })
                    except Exception as e:
                        await self._report(f"Preprocess error page {idx+1}: {e}", 10 + int(80 * idx / total_pages), 100)
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
                    await self._report(f"Перевод стр. {idx+1}/{total_pages}", 10 + int(80 * idx / total_pages), 100)
                    translations = await self.translator.translate_page(
                        korean_texts=grouped_ko,
                        english_texts=item["en_texts"],
                        page_number=idx + 1,
                        manga_id=mangadex_manga_id,
                        chapter=chapter_number,
                        source_lang=source_lang,
                    )
                except Exception as e:
                    await self._report(f"Перевод ошибка стр. {idx+1}: {e}", 10 + int(80 * idx / total_pages), 100)
                    translations = []

                # Attach detected speakers to translations for memory/profile update
                for t, sp in zip(translations, speakers):
                    if sp:
                        t["speaker"] = sp

                await translated_queue.put({
                    "index": idx,
                    "src_data": item["src_data"],
                    "en_data": item.get("en_data"),
                    "translations": translations,
                    "ocr_texts": item["ocr_texts"],
                    "groups": item["groups"],
                    "grouped_ko": grouped_ko,
                    "speakers": speakers,
                    "img_w": img_w,
                    "img_h": img_h,
                })
                ocr_queue.task_done()

        # ---- Stage 4: Inpainter + Renderer ----
        async def stage_inpainter_renderer():
            # Уменьшаем параллельность для экономии RAM
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
                            all_bubble_bboxes.append(bubble_bbox)
                            bubble_pairs.append((bubble_bbox, ru, is_bubble, text_bbox))

                        if not bubble_pairs:
                            async with results_lock:
                                rendered_results[idx] = src_data
                            translated_queue.task_done()
                            continue

                        filled_img, self._bg_mask = self._fill_bubble_bg(cv_img, all_bubble_bboxes)

                        mask = build_mask(h, w, all_bubble_bboxes)
                        for r in item["ocr_texts"]:
                            poly = r.get("polygon")
                            if poly:
                                pts = np.array([[(p[0], p[1]) for p in poly]], dtype=np.int32)
                                cv2.fillPoly(mask, pts, 255)

                        remaining = cv2.bitwise_and(mask, cv2.bitwise_not(self._bg_mask))

                        if cv2.countNonZero(remaining) == 0:
                            clean_img = Image.fromarray(cv2.cvtColor(filled_img, cv2.COLOR_BGR2RGB))
                        elif MODAL_AVAILABLE:
                            _, img_bytes = cv2.imencode(".png", filled_img)
                            modal_result = inpaint_batch_sync([img_bytes.tobytes()])
                            if modal_result:
                                modal_img = cv2.imdecode(np.frombuffer(modal_result[0], np.uint8), cv2.IMREAD_COLOR)
                                clean_img = Image.fromarray(cv2.cvtColor(modal_img, cv2.COLOR_BGR2RGB))
                                del modal_img
                            else:
                                c = self.inpainter.inpaint(filled_img, remaining)
                                clean_img = Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))
                                del c
                        else:
                            c = self.inpainter.inpaint(filled_img, remaining)
                            clean_img = Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))
                            del c
                        del cv_img, mask, filled_img

                        for bubble_bbox, text, is_bubble, text_bbox in bubble_pairs:
                            try:
                                # Classify caption (rectangular) vs dialogue (round bubble)
                                font_type = "dialogue"
                                if not is_bubble or _is_caption_region(cv_img, text_bbox):
                                    font_type = "narration"
                                else:
                                    # Font-style matching: handwritten vs clean
                                    bubble_mask_img = self._bg_mask
                                    if bubble_mask_img is not None:
                                        mask_pil = Image.fromarray(bubble_mask_img)
                                        style = _classify_font_style(img, mask_pil, bubble_bbox)
                                        if style == "narration":
                                            font_type = "narration"
                                clean_img = self.renderer.render_bubble_text(
                                    clean_img, bubble_bbox, text, font_type=font_type, is_bubble=is_bubble,
                                    original_img=img,
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
                        async with results_lock:
                            rendered_results[idx] = src_data
                    translated_queue.task_done()

        # ---- Stage 5: Saver ----
        async def stage_saver():
            for i in range(total_pages):
                # Wait for this page's result
                while i not in rendered_results:
                    await asyncio.sleep(0.05)

                async with results_lock:
                    out_data = rendered_results.pop(i)

                out_path = work_dir / f"page_{i:03d}.png"
                out_path.write_bytes(out_data)
                await self._report(f"Сохранено стр. {i+1}/{total_pages}", 10 + int(80 * i / total_pages), 100)

            done_event.set()

        # Launch all stages
        await self._report("Запуск конвейера...", 10, 100)

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

        # Collect final pages in order
        final_paths = []
        for i in range(total_pages):
            path = work_dir / f"page_{i:03d}.png"
            if path.exists():
                final_paths.append(str(path))

        # Сохраняем переводы в память
        if final_paths:
from cfg.memory import save_translations, _extract_speaker
            current_chapter_translations = []
            for item in translated_queue._queue:
                if item and not item.get("skip_render") and item.get("translations"):
                    current_chapter_translations.extend(item["translations"])
            await asyncio.to_thread(save_translations, mangadex_manga_id, mangadex_manga_id, chapter_number, current_chapter_translations)

        await self._report("Готово!", 100, 100)
        return final_paths

    async def close(self):
        await self.mangadex.close()
        await self.colab.close()
        await self.translator.close()

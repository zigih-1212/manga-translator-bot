import json
import gc
from pathlib import Path
from PIL import Image
import io
import numpy as np
import cv2

from sources.mangadex import MangaDexSource
from translator.llm import LLMTranslator
from translator.renderer import TextRenderer
from translator.colab_client import ColabClient
from translator.inpainter import LaMaInpainter
from translator.bubbles import get_bubble_bounds, build_mask
from translator.modal_client import inpaint_batch_sync, MODAL_AVAILABLE
from translator.upscaler import RealESRGANUpscaler
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


class TranslationPipeline:
    def __init__(self):
        self.mangadex = MangaDexSource()
        self.translator = LLMTranslator()
        self.renderer = TextRenderer()
        self.colab = ColabClient()
        self.inpainter = LaMaInpainter()
        self.progress_callback = None
        self._bg_mask = None
        self._upscaler = RealESRGANUpscaler()

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

        en_pages = []
        en_page_map = {}
        if en_chapter:
            await self._report(f"Получаем список страниц {target_lang}...", 7, 100)
            en_pages = await self.mangadex.get_pages(en_chapter.id)
            en_page_map = {p.index: p for p in en_pages}

        self.translator.clear_context()
        translated_page_paths = []

        for i in range(total_pages):
            progress = 10 + int(80 * i / total_pages)
            await self._report(
                f"Страница {i + 1}/{total_pages}",
                progress, 100
            )

            await self._report(
                f"Скачиваем стр. {i + 1}/{total_pages} ({source_lang})",
                progress, 100
            )
            try:
                src_data = await self.mangadex.download_page(src_pages[i])
                src_data = self._auto_rotate(src_data)
                if self._upscaler.available:
                    try:
                        np_img = np.frombuffer(src_data, np.uint8)
                        cv_img_up = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
                        cv_img_up = self._upscaler.upscale(cv_img_up)
                        _, src_data = cv2.imencode(".png", cv_img_up)
                        src_data = src_data.tobytes()
                    except Exception:
                        pass
            except Exception as e:
                await self._report(f"Ошибка скачивания стр. {i+1}: {e}", progress, 100)
                continue

            en_data = None
            if i in en_page_map:
                try:
                    en_data = await self.mangadex.download_page(en_page_map[i])
                except Exception:
                    pass

            if self.colab.is_connected:
                await self._report(f"OCR стр. {i + 1}/{total_pages}", progress, 100)
                try:
                    ocr_result = await self.colab.ocr_pages([src_data], lang=source_lang)
                    ocr_texts = ocr_result[0] if ocr_result else []
                except Exception as e:
                    await self._report(f"OCR недоступен: {e}", progress, 100)
                    ocr_texts = []
                try:
                    tmp_img = Image.open(io.BytesIO(src_data))
                    img_w, img_h = tmp_img.size
                    tmp_img.close()
                    ocr_texts = _filter_text_regions(ocr_texts, img_w, img_h)
                    if ocr_texts and _is_credit_page(ocr_texts, img_w, img_h):
                        await self._report(f"Credit-страница {i + 1}, пропускаю", progress, 100)
                        ocr_texts = []
                except Exception:
                    pass
            else:
                ocr_texts = []

            ko_texts = [r.get("text", "") for r in ocr_texts if r.get("text")]

            en_texts = []
            if en_data and ocr_texts:
                if self.colab.is_connected:
                    try:
                        en_ocr = await self.colab.ocr_pages([en_data], lang=target_lang)
                        en_texts = [r.get("text", "") for r in (en_ocr[0] if en_ocr else []) if r.get("text")]
                    except Exception:
                        pass

            if not ko_texts:
                out_path = work_dir / f"page_{i:03d}.png"
                out_path.write_bytes(src_data)
                translated_page_paths.append(str(out_path))
                del src_data, en_data
                gc.collect()
                continue

            groups = _group_texts_by_bubble(ocr_texts)
            grouped_ko = [" ".join(r.get("text", "") for r in g if r.get("text")) for g in groups]

            await self._report(f"Перевод стр. {i + 1}/{total_pages}", progress, 100)
            translations = await self.translator.translate_page(
                korean_texts=grouped_ko,
                english_texts=en_texts,
                page_number=i + 1,
                manga_id=mangadex_manga_id,
                chapter=chapter_number,
                source_lang=source_lang,
            )

            await self._report(f"Обработка стр. {i + 1}/{total_pages}", progress, 100)
            try:
                img = Image.open(io.BytesIO(src_data)).convert("RGB")
                cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                h, w = cv_img.shape[:2]

                bubble_pairs = []
                all_bubble_bboxes = []
                is_bubble_flags = []
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
                    bubble_pairs.append((bubble_bbox, ru, is_bubble))
                    is_bubble_flags.append(is_bubble)

                if not bubble_pairs:
                    out_path = work_dir / f"page_{i:03d}.png"
                    out_path.write_bytes(src_data)
                    translated_page_paths.append(str(out_path))
                    del src_data, en_data, img, cv_img
                    gc.collect()
                    continue

                filled_img, self._bg_mask = self._fill_bubble_bg(cv_img, all_bubble_bboxes)

                mask = build_mask(h, w, all_bubble_bboxes)
                for r in ocr_texts:
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

                for bubble_bbox, text, is_bubble in bubble_pairs:
                    try:
                        clean_img = self.renderer.render_bubble_text(
                            clean_img, bubble_bbox, text, font_type="dialogue", is_bubble=is_bubble
                        )
                    except Exception:
                        continue

                buf = io.BytesIO()
                clean_img.save(buf, format="PNG")
                out_data = buf.getvalue()
                del img, clean_img, buf
            except Exception:
                out_data = src_data

            out_path = work_dir / f"page_{i:03d}.png"
            out_path.write_bytes(out_data)
            translated_page_paths.append(str(out_path))

            pairs_for_memory = [{"ko": t.get("ko", ""), "ru": t.get("ru", "")} for t in translations if t.get("ko") and t.get("ru")]
            if pairs_for_memory:
                save_translations(mangadex_manga_id, mangadex_manga_id[:8], chapter_number, pairs_for_memory)

            del src_data, en_data, out_data
            gc.collect()

        await self._report("Готово!", 100, 100)
        return translated_page_paths

    async def close(self):
        await self.mangadex.close()
        await self.colab.close()
        await self.translator.close()

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
from config import TEMP_DIR


def _filter_text_regions(ocr_texts: list[dict], img_w: int, img_h: int) -> list[dict]:
    filtered = []
    page_area = img_w * img_h
    for r in ocr_texts:
        bb = r.get("bbox", [])
        if not bb:
            continue
        text = r.get("text", "")
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


def _group_texts_by_bubble(ocr_texts: list[dict], y_gap: int = 30) -> list[list[dict]]:
    sorted_items = sorted(ocr_texts, key=lambda r: (r.get("bbox", [{}])[0][1] if isinstance(r.get("bbox"), list) and r["bbox"] and isinstance(r["bbox"][0], (list, tuple)) else r.get("bbox", [0])[1], r.get("bbox", [{}])[0][0] if isinstance(r.get("bbox"), list) and r["bbox"] and isinstance(r["bbox"][0], (list, tuple)) else r.get("bbox", [0])[0]))
    groups = []
    current = []
    prev_bottom = None
    for r in sorted_items:
        bb = r.get("bbox", [])
        if not bb:
            continue
        if isinstance(bb[0], (list, tuple)):
            top, bottom = int(min(p[1] for p in bb)), int(max(p[1] for p in bb))
        else:
            top, bottom = int(bb[1]), int(bb[3])
        if prev_bottom is not None and top - prev_bottom > y_gap:
            if current:
                groups.append(current)
            current = [r]
        else:
            current.append(r)
        prev_bottom = bottom
    if current:
        groups.append(current)
    return groups


class TranslationPipeline:
    def __init__(self):
        self.mangadex = MangaDexSource()
        self.translator = LLMTranslator()
        self.renderer = TextRenderer()
        self.colab = ColabClient()
        self.inpainter = LaMaInpainter()
        self.progress_callback = None

    def on_progress(self, callback):
        self.progress_callback = callback

    async def _report(self, message: str, current: int = 0, total: int = 0):
        if self.progress_callback:
            await self.progress_callback(message, current, total)

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
                    ocr_result = await self.colab.ocr_pages([src_data])
                    ocr_texts = ocr_result[0] if ocr_result else []
                except Exception as e:
                    await self._report(f"OCR недоступен: {e}", progress, 100)
                    ocr_texts = []
                try:
                    tmp_img = Image.open(io.BytesIO(src_data))
                    img_w, img_h = tmp_img.size
                    tmp_img.close()
                    ocr_texts = _filter_text_regions(ocr_texts, img_w, img_h)
                except Exception:
                    pass
            else:
                ocr_texts = []

            ko_texts = [r.get("text", "") for r in ocr_texts if r.get("text")]

            en_texts = []
            if en_data and ocr_texts:
                if self.colab.is_connected:
                    try:
                        en_ocr = await self.colab.ocr_pages([en_data])
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
            )

            await self._report(f"Обработка стр. {i + 1}/{total_pages}", progress, 100)
            try:
                pairs = []
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
                    if xs:
                        combined_bbox = (min(xs), min(ys), max(xs), max(ys))
                        pairs.append((combined_bbox, ru))

                if not pairs:
                    out_path = work_dir / f"page_{i:03d}.png"
                    out_path.write_bytes(src_data)
                    translated_page_paths.append(str(out_path))
                    del src_data, en_data
                    gc.collect()
                    continue

                img = Image.open(io.BytesIO(src_data)).convert("RGB")
                cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                h, w = cv_img.shape[:2]
                mask = np.zeros((h, w), dtype=np.uint8)
                PAD = 5
                for r in ocr_texts:
                    word_bboxes = r.get("word_bboxes", [])
                    if word_bboxes:
                        for wb in word_bboxes:
                            x1, y1, x2, y2 = [int(v) for v in wb]
                            x1, y1 = max(0, x1 - PAD), max(0, y1 - PAD)
                            x2, y2 = min(w, x2 + PAD), min(h, y2 + PAD)
                            mask[y1:y2, x1:x2] = 255
                    else:
                        bb = r.get("bbox", [])
                        if not bb:
                            continue
                        if isinstance(bb[0], (list, tuple)):
                            xs = [int(p[0]) for p in bb]
                            ys = [int(p[1]) for p in bb]
                            x1, y1 = max(0, min(xs) - PAD), max(0, min(ys) - PAD)
                            x2, y2 = min(w, max(xs) + PAD), min(h, max(ys) + PAD)
                        elif len(bb) >= 4:
                            x1, y1, x2, y2 = [int(v) for v in bb[:4]]
                            x1, y1 = max(0, x1 - PAD), max(0, y1 - PAD)
                            x2, y2 = min(w, x2 + PAD), min(h, y2 + PAD)
                        else:
                            continue
                        mask[y1:y2, x1:x2] = 255
                kernel = np.ones((3, 3), np.uint8)
                mask = cv2.dilate(mask, kernel, iterations=1)
                clean_cv = self.inpainter.inpaint(cv_img, mask)
                clean_img = Image.fromarray(cv2.cvtColor(clean_cv, cv2.COLOR_BGR2RGB))
                del cv_img, mask, clean_cv

                for bbox, text in pairs:
                    try:
                        clean_img = self.renderer.render_bubble_text(
                            clean_img, bbox, text, font_type="dialogue"
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

            del src_data, en_data, out_data
            gc.collect()

        await self._report("Готово!", 100, 100)
        return translated_page_paths

    async def close(self):
        await self.mangadex.close()
        await self.colab.close()
        await self.translator.close()

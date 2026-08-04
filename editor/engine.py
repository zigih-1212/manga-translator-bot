"""Движок веб-редактора: перерисовка страницы с учётом правок перевода."""
from __future__ import annotations

import asyncio
from typing import List, Dict

from editor import store
from translator.pipeline import TranslationPipeline

_pipeline: TranslationPipeline | None = None
_pipeline_lock = asyncio.Lock()


async def _get_pipeline() -> TranslationPipeline:
    global _pipeline
    async with _pipeline_lock:
        if _pipeline is None:
            _pipeline = TranslationPipeline()
    return _pipeline


async def rerender_page(manga_id: str, chapter: str, page: int, edits: List[Dict]) -> bytes:
    """Перерисовывает страницу с учётом внесённых правок.

    edits: список {"id": bubble_id, "text": new_text}
    Возвращает bytes нового изображения и обновляет хранилище.
    """
    page_data = store.load_page(manga_id, chapter, page)
    if not page_data:
        raise ValueError(f"Страница {page} главы {chapter} манги {manga_id} не найдена в хранилище.")

    # Применяем правки к текущим метаданным пузырей
    bubbles = page_data.get("bubbles", [])
    edited_bubbles = []
    for bubble in bubbles:
        matched = next((e for e in edits if e.get("id") == bubble.get("id")), None)
        if matched is not None:
            bubble["text"] = (matched.get("text") or "").strip()
            bubble["edited"] = bool(bubble["text"])
        edited_bubbles.append(bubble)

    src_data = open(page_data["_src_path"], "rb").read()

    pipeline = await _get_pipeline()
    out_data = await pipeline.render_single_page(
        src_data=src_data,
        bubbles_meta=edited_bubbles,
        img_w=page_data.get("img_w", 0),
        img_h=page_data.get("img_h", 0),
    )

    store.update_out(manga_id, chapter, page, out_data)
    store.update_bubbles(manga_id, chapter, page, edited_bubbles)
    return out_data

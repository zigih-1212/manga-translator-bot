"""Хранение промежуточных данных переведённых страниц для веб-редактора.

Во время обработки главы пайплайн сохраняет для каждой страницы:
  - оригинальное изображение (src)
  - итоговый PNG (out)  — тот же, что отправляется в Telegram
  - JSON с пузырями (bbox, строки до/после, шрифт, угол) для правки
"""
from __future__ import annotations

import json
import re
import zipfile
from io import BytesIO
from pathlib import Path

from cfg import TEMP_DIR

EDITOR_ROOT = TEMP_DIR / "editor"


def _safe(name: str) -> str:
    name = re.sub(r"[^0-9A-Za-zА-Яа-яЁё_.-]", "_", str(name))
    return name or "x"


def chapter_dir(manga_id: str, chapter: str) -> Path:
    d = EDITOR_ROOT / _safe(manga_id) / _safe(chapter)
    d.mkdir(parents=True, exist_ok=True)
    return d


def page_name(page: int) -> str:
    return f"{page:03d}"


def _paths(manga_id: str, chapter: str, page: int):
    d = chapter_dir(manga_id, chapter)
    return {
        "meta": d / f"{page_name(page)}.json",
        "src": d / f"{page_name(page)}.src.png",
        "out": d / f"{page_name(page)}.out.png",
    }


def save_page(manga_id: str, chapter: str, page: int, src_data: bytes, out_data: bytes, page_meta: dict):
    p = _paths(manga_id, chapter, page)
    p["src"].write_bytes(src_data)
    p["out"].write_bytes(out_data)
    with open(p["meta"], "w", encoding="utf-8") as f:
        json.dump({
            "manga_id": manga_id,
            "chapter": chapter,
            "page": page,
            **page_meta,
        }, f, ensure_ascii=False, indent=1)


def update_out(manga_id: str, chapter: str, page: int, out_data: bytes):
    p = _paths(manga_id, chapter, page)
    p["out"].write_bytes(out_data)


def load_page(manga_id: str, chapter: str, page: int) -> dict | None:
    p = _paths(manga_id, chapter, page)
    if not p["meta"].exists():
        return None
    with open(p["meta"], encoding="utf-8") as f:
        data = json.load(f)
    data["_src_path"] = str(p["src"])
    data["_out_path"] = str(p["out"])
    return data


def save_edits(manga_id: str, chapter: str, page: int, bubble_id: int, ru: str):
    """Применить одну правку перевода и обновить rendered-страницу."""
    if not ru.strip() and bubble_id >= (bubble_edit["edited"] and None):
        pass
    p = _paths(manga_id, chapter, page)
    if not p["meta"].exists():
        return None
    with open(p["meta"], encoding="utf-8") as f:
        data = json.load(f)
    bubbles = data["bubbles"]
    for b in bubbles:
        if b.get("id") == bubble_id:
            b["text"] = ru.strip()
            b["edited"] = True
            break
    with open(p["meta"], "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    return data


def list_chapters() -> list[dict]:
    out = []
    if not EDITOR_ROOT.exists():
        return out
    for m in sorted(EDITOR_ROOT.iterdir()):
        if not m.is_dir():
            continue
        for ch in sorted(m.iterdir()):
            if not ch.is_dir():
                continue
            metas = list(ch.glob("*.json"))
            if not metas:
                continue
            pages = sorted(int(mm.stem.split(".")[0]) for mm in metas)
            out.append({"manga_id": m.name, "chapter": ch.name, "pages": pages})
    return out


def build_zip(manga_id: str, chapter: str, session_id: int | None = None) -> bytes:
    """ZIP текущих (возможно отредактированных) страниц."""
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        d = chapter_dir(manga_id, chapter)
        for meta_f in sorted(d.glob("*.json"), key=lambda p: int(p.stem)):
            page = int(meta_f.stem)
            out = d / f"{page_name(page)}.out.png"
            if out.exists():
                zf.write(out, arcname=f"{page_name(page)}.png")
    buf.seek(0)
    return buf.getvalue()
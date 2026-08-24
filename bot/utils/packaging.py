"""CBZ packaging with ComicInfo.xml — reader-friendly chapter archives.

Replaces bare ZIP export: Tachiyomi/Mihon, CDisplayEx, YACReader and most
manga readers understand .cbz + metadata out of the box.
"""
from __future__ import annotations

import zipfile
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape


def _comicinfo_xml(
    series: str,
    chapter: str,
    title: str = "",
    language: str = "ru",
    page_count: int = 0,
    manga: str = "YesAndRightToLeft",
) -> str:
    try:
        num_f = float(chapter)
        number = ("%g" % num_f)
        if "." not in number:
            number = str(int(float(number)))
    except ValueError:
        number = escape(chapter)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return f"""<?xml version="1.0" encoding="utf-8"?>
<ComicInfo xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <Series>{escape(series)}</Series>
  <Number>{number}</Number>
  <Title>{escape(title or f"Chapter {chapter}")}</Title>
  <LanguageISO>{escape(language)}</LanguageISO>
  <PageCount>{page_count}</PageCount>
  <Manga>{manga}</Manga>
  <Year>{now[:4]}</Year>
  <Month>{int(now[5:7])}</Month>
  <Day>{int(now[8:10])}</Day>
</ComicInfo>
"""


def pack_chapter_cbz(
    page_paths: list[str],
    out_path: Path,
    series: str,
    chapter: str,
    title: str = "",
) -> Path:
    """Pack rendered pages into a .cbz with ComicInfo.xml. Pages sorted by index."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(page_paths, key=lambda p: Path(p).name)
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_STORED) as zf:  # STORED: jpg already compressed
        zf.writestr("ComicInfo.xml", _comicinfo_xml(
            series=series, chapter=chapter, title=title, page_count=len(files),
        ))
        for p in files:
            zf.write(p, arcname=Path(p).name)
    return out_path

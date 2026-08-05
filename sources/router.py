"""Роутер источников манги: Mangakakalot, MangaNelo + MangaDex fallback."""
import asyncio
import logging
from .base import BaseSource, MangaResult, Chapter, Page
from .mangakakalot import MangakakalotSource
from .manganelo import MangaNeloSource
from .mangadex import MangaDexSource

log = logging.getLogger("manga_translator")


class SourceRouter:
    """Выбирает нужный источник по имени и делегирует ему вызовы.

    Источники создаются лениво и переиспользуются до close().
    """

    _registry: dict[str, type[BaseSource]] = {
        "mangakakalot": MangakakalotSource,
        "manganelo": MangaNeloSource,
        "mangadex": MangaDexSource,
    }

    def __init__(self):
        self._sources: dict[str, BaseSource] = {}
        self._lock = asyncio.Lock()

    async def get(self, name: str) -> BaseSource:
        key = (name or "mangakakalot").lower()
        if key not in self._registry:
            log.warning("SourceRouter: неизвестный источник '%s', использую mangakakalot", name)
            key = "mangakakalot"
        if key not in self._sources:
            async with self._lock:
                if key not in self._sources:
                    self._sources[key] = self._registry[key]()
        return self._sources[key]

    def available(self) -> list[str]:
        return list(self._registry)

    async def search(self, title: str) -> list[MangaResult]:
        """Search across all sources, return results sorted by chapter count."""
        results = []
        for name in self._registry:
            try:
                src = await self.get(name)
                src_results = await src.search(title)
                log.info("SourceRouter: %s found %d results for '%s'", name, len(src_results), title)
                results.extend(src_results)
            except Exception as e:
                log.warning("SourceRouter: поиск в %s упал: %s", name, e)
        # Deduplicate by title, keep the one with most chapters
        seen = {}
        for r in results:
            key = r.title.lower().strip()
            if key not in seen or (r.chapters_count or 0) > (seen[key].chapters_count or 0):
                seen[key] = r
        # Sort by chapter count descending (most chapters first)
        sorted_results = sorted(seen.values(), key=lambda x: x.chapters_count or 0, reverse=True)
        return sorted_results

    async def find_chapter_by_number(
        self, source: str, manga_id: str, chapter_number: str, lang: str
    ) -> Chapter | None:
        src = await self.get(source)
        if hasattr(src, "find_chapter_by_number"):
            return await src.find_chapter_by_number(manga_id, chapter_number, lang)
        chapters = await src.get_chapters(manga_id, lang)
        for ch in chapters:
            if ch.number == chapter_number:
                return ch
        return None

    async def get_chapters(self, source: str, manga_id: str, lang: str) -> list[Chapter]:
        return await (await self.get(source)).get_chapters(manga_id, lang)

    async def get_pages(self, source: str, chapter_id: str, manga_id: str = "") -> list[Page]:
        src = await self.get(source)
        return await src.get_pages(chapter_id)

    async def download_page(self, source: str, page: Page) -> bytes:
        return await (await self.get(source)).download_page(page)

    async def close(self):
        async with self._lock:
            for name, src in list(self._sources.items()):
                try:
                    await src.close()
                except Exception as e:
                    log.warning("SourceRouter: close %s: %s", name, e)
            self._sources.clear()

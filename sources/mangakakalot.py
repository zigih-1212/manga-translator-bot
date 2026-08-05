"""Mangakakalot source via MangaHook API — Solo Leveling 200 chapters."""
import asyncio
import logging
import os
import httpx
from .base import BaseSource, MangaResult, Chapter, Page

log = logging.getLogger("manga_translator")

API_BASE = "https://mangahook-api.vercel.app/api"
# Fallback: direct Mangakakalot API
MK_API = "https://api.mangakakalot.tv"


class MangakakalotSource:
    def __init__(self):
        self._client: httpx.AsyncClient | None = None

    def _get_proxy(self):
        """Get proxy URL from environment."""
        return os.environ.get("HTTP_PROXY") or os.environ.get("HTTPS_PROXY") or os.environ.get("MANGA_PROXY")

    async def _client_get(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            proxy = self._get_proxy()
            timeout = httpx.Timeout(30.0, connect=10.0)
            if proxy:
                self._client = httpx.AsyncClient(timeout=timeout, proxy=proxy, verify=False,
                    headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"})
            else:
                self._client = httpx.AsyncClient(timeout=timeout,
                    headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"})
        return self._client

    async def search(self, title: str) -> list[MangaResult]:
        """Search manga by title using Mangakakalot API."""
        try:
            client = await self._client_get()
            # Try Mangakakalot API directly
            r = await client.get(f"{MK_API}/search", params={"query": title})
            if r.status_code == 200:
                data = r.json()
                results = []
                for item in data.get("results", data.get("manga_list", [])):
                    results.append(MangaResult(
                        id=item.get("id", item.get("slug", "")),
                        title=item.get("title", item.get("name", "")),
                        alt_titles=item.get("alt_names", item.get("alt_titles", [])),
                        description=item.get("description", ""),
                        status=item.get("status", ""),
                        year=item.get("year"),
                        cover_url=item.get("image", item.get("cover_url", item.get("thumb"))),
                        source="mangakakalot",
                        original_language="ko",
                    ))
                if results:
                    return results
            # Fallback to MangaHook
            r = await client.get(f"{API_BASE}/search", params={"query": title})
            if r.status_code == 200:
                data = r.json()
                results = []
                for item in data.get("results", []):
                    results.append(MangaResult(
                        id=item.get("id", ""),
                        title=item.get("title", ""),
                        alt_titles=item.get("alt_titles", []),
                        description=item.get("description", ""),
                        status=item.get("status", ""),
                        year=item.get("year"),
                        cover_url=item.get("image", item.get("cover_url")),
                        source="mangakakalot",
                        original_language="ko",
                    ))
                return results
            log.warning("Mangakakalot search failed: HTTP %d", r.status_code)
            return []
        except Exception as e:
            log.warning("Mangakakalot search error: %s", e)
            return []

    async def get_chapters(self, manga_id: str, lang: str = "en") -> list[Chapter]:
        """Get chapter list."""
        try:
            client = await self._client_get()
            # Try Mangakakalot API
            r = await client.get(f"{MK_API}/manga/{manga_id}/chapters")
            if r.status_code == 200:
                data = r.json()
                chapters = []
                for item in data.get("chapters", data.get("data", [])):
                    chapters.append(Chapter(
                        id=item.get("id", item.get("slug", "")),
                        number=item.get("number", item.get("chapter", "")),
                        title=item.get("title", f"Chapter {item.get('number', '?')}"),
                        volume=item.get("volume"),
                        pages_count=item.get("pages_count", 0),
                        translated_language=lang,
                    ))
                if chapters:
                    return chapters
            # Fallback to MangaHook
            r = await client.get(f"{API_BASE}/manga/{manga_id}/chapters")
            if r.status_code == 200:
                data = r.json()
                chapters = []
                for item in data.get("chapters", []):
                    chapters.append(Chapter(
                        id=item.get("id", ""),
                        number=item.get("number", ""),
                        title=item.get("title", ""),
                        volume=item.get("volume"),
                        pages_count=item.get("pages_count", 0),
                        translated_language=lang,
                    ))
                return chapters
            log.warning("Mangakakalot get_chapters failed: HTTP %d", r.status_code)
            return []
        except Exception as e:
            log.warning("Mangakakalot get_chapters error: %s", e)
            return []

    async def get_pages(self, chapter_id: str) -> list[Page]:
        """Get page images for chapter."""
        try:
            client = await self._client_get()
            # Try Mangakakalot API
            r = await client.get(f"{MK_API}/chapter/{chapter_id}")
            if r.status_code == 200:
                data = r.json()
                pages = []
                for i, url in enumerate(data.get("pages", data.get("images", data.get("data", [])))):
                    if isinstance(url, dict):
                        url = url.get("url", url.get("image", url.get("src", "")))
                    pages.append(Page(
                        url=url,
                        index=i,
                        width=0,
                        height=0,
                    ))
                if pages:
                    return pages
            # Fallback to MangaHook
            r = await client.get(f"{API_BASE}/chapter/{chapter_id}")
            if r.status_code == 200:
                data = r.json()
                pages = []
                for i, url in enumerate(data.get("pages", data.get("images", []))):
                    pages.append(Page(
                        url=url,
                        index=i,
                        width=0,
                        height=0,
                    ))
                return pages
            log.warning("Mangakakalot get_pages failed: HTTP %d", r.status_code)
            return []
        except Exception as e:
            log.warning("Mangakakalot get_pages error: %s", e)
            return []

    async def download_page(self, page: Page) -> bytes:
        """Download page image."""
        client = await self._client_get()
        r = await client.get(page.url)
        return r.content

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()

"""Mangakakalot source — direct HTML scraping (no dead third-party APIs)."""
import asyncio
import logging
import os
import re
import httpx
from .base import BaseSource, MangaResult, Chapter, Page

log = logging.getLogger("manga_translator")

DOMAINS = [
    "https://mangakakalot.com",
    "https://mangakakalottt.com",
    "https://www.mangakakalot.gg",
]

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

_BLOCK_MARKERS = ("SpinzyWheel", "Just a moment", "challenge-platform")


class MangakakalotSource(BaseSource):
    def __init__(self):
        self._client: httpx.AsyncClient | None = None
        self._working_domain: str | None = None

    @staticmethod
    def _get_proxy():
        return (os.environ.get("MANGA_PROXY")
                or os.environ.get("HTTP_PROXY")
                or os.environ.get("HTTPS_PROXY"))

    async def _client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            kwargs = dict(
                timeout=httpx.Timeout(30.0, connect=10.0),
                headers=_HEADERS,
                follow_redirects=True,
            )
            proxy = self._get_proxy()
            if proxy:
                kwargs["proxy"] = proxy
                kwargs["verify"] = False
            self._client = httpx.AsyncClient(**kwargs)
        return self._client

    async def _fetch(self, url: str) -> str | None:
        client = await self._client()
        try:
            r = await client.get(url)
            if r.status_code == 200 and not any(m in r.text for m in _BLOCK_MARKERS):
                return r.text
            log.debug("Mangakakalot %s -> HTTP %d", url, r.status_code)
        except Exception as e:
            log.debug("Mangakakalot %s error: %s", url, e)
        return None

    async def search(self, title: str) -> list[MangaResult]:
        """Search via GET /search/story/{query} (space->underscore)."""
        slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
        for domain in DOMAINS:
            html = await self._fetch(f"{domain}/search/story/{slug}")
            if not html:
                continue
            self._working_domain = domain
            results = []
            pattern = re.compile(
                r'<a[^>]+href="([^"]+)"[^>]*title="([^"]+)"[^>]*>\s*'
                r'<(?:span|img)[^>]*src="([^"]+)"',
                re.DOTALL,
            )
            seen: set[str] = set()
            for m in pattern.finditer(html):
                url, name, img = m.groups()
                mid = url.rstrip("/").rsplit("/", 1)[-1]
                if mid in seen:
                    continue
                seen.add(mid)
                results.append(MangaResult(
                    id=mid,
                    title=name.strip(),
                    alt_titles=[],
                    description="",
                    status="",
                    year=None,
                    cover_url=img,
                    source="mangakakalot",
                    original_language="en",
                ))
            if results:
                log.info("Mangakakalot.search('%s') via %s -> %d", title, domain, len(results))
                return results[:15]
        log.info("Mangakakalot.search('%s') -> 0 results", title)
        return []

    async def find_chapter_by_number(self, manga_id: str, chapter_number: str, lang: str) -> Chapter | None:
        chapters = await self.get_chapters(manga_id, lang)
        for ch in chapters:
            if ch.number == chapter_number:
                return ch
        try:
            want = float(chapter_number)
            for ch in chapters:
                try:
                    if float(ch.number) == want:
                        return ch
                except ValueError:
                    pass
        except ValueError:
            pass
        return None

    async def get_chapters(self, manga_id: str, lang: str = "en") -> list[Chapter]:
        base = self._working_domain or DOMAINS[0]
        html = await self._fetch(f"{base}/manga/{manga_id}")
        if not html:
            # retry remaining domains
            for d in DOMAINS:
                if d == base:
                    continue
                html = await self._fetch(f"{d}/manga/{manga_id}")
                if html:
                    self._working_domain = d
                    break
        if not html:
            return []
        chapters = []
        pattern = re.compile(r'<a[^>]+href="([^"]+/chapter[^"]*)"[^>]*>([^<]+)</a>', re.I)
        seen_nums: set[str] = set()
        for m in pattern.finditer(html):
            url, title = m.groups()
            cid = url.rstrip("/").rsplit("/", 1)[-1]
            num_m = re.search(r"(\d+(?:\.\d+)?)", cid) or re.search(r"(\d+(?:\.\d+)?)", title)
            number = num_m.group(1) if num_m else cid
            if number in seen_nums:
                continue
            seen_nums.add(number)
            chapters.append(Chapter(
                id=cid,
                number=number,
                title=title.strip(),
                volume=None,
                pages_count=0,
                translated_language="en",
            ))
        chapters.reverse()
        log.info("Mangakakalot.get_chapters(%s) -> %d", manga_id, len(chapters))
        return chapters

    async def get_pages(self, chapter_id: str) -> list[Page]:
        base = self._working_domain or DOMAINS[0]
        html = await self._fetch(f"{base}/chapter/{chapter_id}")
        if not html:
            return []
        pages = []
        pattern = re.compile(r'<img[^>]+src="([^"]+)"[^>]*(?:class="img-loading"|loading="lazy")', re.I)
        alt = re.compile(r'class="img-loading"[^>]*src="([^"]+)"', re.I)
        urls = [m.group(1) for m in pattern.finditer(html)] or [m.group(1) for m in alt.finditer(html)]
        for i, url in enumerate(urls):
            pages.append(Page(url=url.strip(), index=i, width=0, height=0))
        log.info("Mangakakalot.get_pages(%s) -> %d pages", chapter_id, len(pages))
        return pages

    async def download_page(self, page: Page) -> bytes:
        client = await self._client()
        r = await client.get(page.url)
        r.raise_for_status()
        return r.content

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()

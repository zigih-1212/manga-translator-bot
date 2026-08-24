"""MangaNelo/Nato source — HTML scraping with multiple domain fallbacks."""
import asyncio
import logging
import re
import os
import time
import httpx
from .base import BaseSource, MangaResult, Chapter, Page

log = logging.getLogger("manga_translator")

# Working domains of the Manganato/Mangakakalot family (order = priority).
# NOTE: from RU IPs these are Cloudflare-gated; set MANGA_PROXY to route around.
DOMAINS = [
    "https://natomanga.com",
    "https://chapmanganato.to",
    "https://manganato.com",
    "https://chapmanganato.com",
    "https://mangakakalot.com",
    "https://mangakakalottt.com",
]

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

_BLOCK_MARKERS = ("SpinzyWheel", "Just a moment", "challenge-platform", "cf-browser-verification")


class MangaNeloSource(BaseSource):
    def __init__(self):
        self._client: httpx.AsyncClient | None = None
        self._working_domain: str | None = None

    # ---------- infra ----------

    @staticmethod
    def _get_proxy():
        return (os.environ.get("MANGA_PROXY")
                or os.environ.get("HTTP_PROXY")
                or os.environ.get("HTTPS_PROXY"))

    async def _client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            proxy = self._get_proxy()
            kwargs = dict(
                timeout=httpx.Timeout(30.0, connect=10.0),
                headers=_HEADERS,
                follow_redirects=True,
            )
            if proxy:
                kwargs["proxy"] = proxy
                kwargs["verify"] = False
            self._client = httpx.AsyncClient(**kwargs)
        return self._client

    async def _fetch(self, path: str) -> str | None:
        """GET first healthy domain. Caches the one that works."""
        client = await self._client()
        candidates = ([self._working_domain] if self._working_domain else []) + \
                     [d for d in DOMAINS if d != self._working_domain]
        for domain in candidates:
            url = f"{domain}{path}"
            try:
                r = await client.get(url)
                if r.status_code == 200 and not any(m in r.text for m in _BLOCK_MARKERS):
                    self._working_domain = domain
                    return r.text
                log.debug("MangaNelo %s -> HTTP %d (blocked or miss)", domain, r.status_code)
            except Exception as e:
                log.debug("MangaNelo %s error: %s", domain, e)
        return None

    @staticmethod
    def _is_blocked(html: str) -> bool:
        return any(m in html for m in _BLOCK_MARKERS)

    # ---------- API ----------

    async def search(self, title: str) -> list[MangaResult]:
        """Search by title across the nato/mangakakalot family."""
        slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
        html = await self._fetch(f"/search/story/{slug}")
        if not html:
            return []
        results = []
        pattern = re.compile(
            r'<a[^>]+href="([^"]+/(?:manga|read)[^"]*)"[^>]*title="([^"]+)"[^>]*>.*?'
            r'<img[^>]+src="([^"]+)"',
            re.DOTALL,
        )
        seen: set[str] = set()
        for m in pattern.finditer(html):
            url, name, img = m.groups()
            manga_id = url.rstrip("/").rsplit("/", 1)[-1]
            if manga_id in seen:
                continue
            seen.add(manga_id)
            status = ""
            sm = re.search(r"(Ongoing|Completed|Hiatus)", html[m.end():m.end() + 400])
            if sm:
                status = sm.group(1)
            results.append(MangaResult(
                id=manga_id,
                title=name.strip(),
                alt_titles=[],
                description="",
                status=status,
                year=None,
                cover_url=img,
                source="manganelo",
                original_language="en",
            ))
        log.info("MangaNelo.search('%s') -> %d results", title, len(results))
        return results[:15]

    async def find_chapter_by_number(self, manga_id: str, chapter_number: str, lang: str) -> Chapter | None:
        chapters = await self.get_chapters(manga_id, lang)
        want = float(chapter_number) if chapter_number.replace(".", "", 1).isdigit() else None
        for ch in chapters:
            if ch.number == chapter_number:
                return ch
            if want is not None:
                try:
                    if float(ch.number) == want:
                        return ch
                except ValueError:
                    pass
        return None

    async def get_chapters(self, manga_id: str, lang: str = "en") -> list[Chapter]:
        """Chapter list for a manga. lang kept for interface compat (site is EN)."""
        html = await self._fetch(f"/manga/{manga_id}")
        if not html:
            return []
        chapters = []
        pattern = re.compile(r'<a[^>]+href="([^"]+)"[^>]*>([^<]*chapter\s*[\d.]+[^<]*)</a>', re.I)
        seen_nums: set[str] = set()
        for m in pattern.finditer(html):
            url, title = m.groups()
            cid = url.rstrip("/").rsplit("/", 1)[-1]
            num_m = re.search(r"(\d+(?:\.\d+)?)", cid)
            if not num_m:
                num_m = re.search(r"(\d+(?:\.\d+)?)", title)
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
        chapters.reverse()  # ascending chapter order
        log.info("MangaNelo.get_chapters(%s) -> %d chapters", manga_id, len(chapters))
        return chapters

    async def get_pages(self, chapter_id: str) -> list[Page]:
        html = await self._fetch(f"/chapter/{chapter_id}")
        if not html:
            return []
        pages = []
        pattern = re.compile(r'<img[^>]+(?:class="img-loading[^"]*"|loading="lazy")[^>]*src="([^"]+)"', re.I)
        fallback = re.compile(r'<div class="container-chapter-reader"><img[^>]+src="([^"]+)"')
        urls = [u for u, in pattern.findall_all(html)] if hasattr(pattern, "find_all") else \
               [m.group(1) for m in pattern.finditer(html)]
        if not urls:
            urls = [m.group(1) for m in fallback.finditer(html)]
        for i, url in enumerate(urls):
            pages.append(Page(url=url.strip(), index=i, width=0, height=0))
        log.info("MangaNelo.get_pages(%s) -> %d pages", chapter_id, len(pages))
        return pages

    async def download_page(self, page: Page) -> bytes:
        client = await self._client()
        r = await client.get(page.url)
        r.raise_for_status()
        return r.content

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()

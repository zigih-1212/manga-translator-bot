import aiohttp
import re
from .base import BaseSource, MangaResult, Chapter, Page


class NaverSource(BaseSource):
    WEBTOON_API = "https://comic.naver.com/api"
    WEBTOON_WEB = "https://comic.naver.com/webtoon"
    MAX_RETRIES = 3

    def __init__(self):
        self.session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    "Referer": "https://comic.naver.com/",
                }
            )
        return self.session

    def _extract_title_id(self, url_or_id: str) -> str:
        match = re.search(r"titleId=(\d+)", url_or_id)
        if match:
            return match.group(1)
        if url_or_id.isdigit():
            return url_or_id
        match = re.search(r"/(\d+)$", url_or_id)
        if match:
            return match.group(1)
        return url_or_id

    async def search(self, title: str) -> list[MangaResult]:
        session = await self._get_session()
        try:
            async with session.get(
                f"{self.WEBTOON_API}/search/all",
                params={"keyword": title, "page": 1},
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()
                results = []
                for item in data.get("titleList", {}).get("titleInfoList", []):
                    results.append(MangaResult(
                        id=str(item.get("titleId", "")),
                        title=item.get("title", ""),
                        alt_titles=[],
                        description=item.get("description", ""),
                        status="ongoing" if item.get("state") == "ONGOING" else "completed",
                        year=None,
                        cover_url=item.get("thumbnail", ""),
                        source="naver",
                    ))
                return results
        except Exception:
            return []

    async def get_chapters(self, manga_id: str, lang: str = "ko") -> list[Chapter]:
        title_id = self._extract_title_id(manga_id)
        session = await self._get_session()
        try:
            async with session.get(
                f"{self.WEBTOON_API}/article/list",
                params={"titleId": title_id, "page": 1, "size": 100},
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()
                chapters = []
                for item in data.get("articleList", {}).get("articleInfoList", []):
                    chapters.append(Chapter(
                        id=str(item.get("no", "")),
                        number=str(item.get("no", "")),
                        title=item.get("title", ""),
                        volume=None,
                        pages_count=0,
                        translated_language="ko",
                    ))
                return list(reversed(chapters))
        except Exception:
            return []

    async def get_pages(self, chapter_id: str, title_id: str = "") -> list[Page]:
        session = await self._get_session()
        try:
            async with session.get(
                f"{self.WEBTOON_API}/article/list/info",
                params={"titleId": title_id, "no": chapter_id},
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()
                pages = []
                for i, item in enumerate(data.get("imageList", [])):
                    url = item.get("url", "")
                    if url:
                        pages.append(Page(url=url, index=i, width=0, height=0))
                return pages
        except Exception:
            return []

    async def download_page(self, page: Page) -> bytes:
        session = await self._get_session()
        async with session.get(page.url, timeout=aiohttp.ClientTimeout(total=60)) as resp:
            resp.raise_for_status()
            return await resp.read()

    async def close(self):
        if self.session and not self.session.closed:
            await self.session.close()

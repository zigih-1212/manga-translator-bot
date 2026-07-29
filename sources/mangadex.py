import asyncio
import aiohttp
import httpx
import os
from .base import BaseSource, MangaResult, Chapter, Page


def _get_proxy():
    return os.environ.get("HTTP_PROXY") or os.environ.get("HTTPS_PROXY") or os.environ.get("http_proxy") or os.environ.get("https_proxy")


def _httpx_client(**kwargs):
    proxy = _get_proxy()
    if proxy:
        kwargs["proxy"] = proxy
        kwargs["verify"] = False
    return httpx.AsyncClient(**kwargs)


class MangaDexSource(BaseSource):
    BASE = "https://api.mangadex.org"
    MAX_RETRIES = 5
    _rate_limit_sem = asyncio.Semaphore(2)
    _last_request_time = 0
    _rate_lock = asyncio.Lock()

    def __init__(self):
        self._own_session: aiohttp.ClientSession | None = None
        self.proxy_url = os.environ.get("COLAB_URL", "").rstrip("/")
        self._proxy = _get_proxy()

    async def _rate_limit(self):
        async with self._rate_lock:
            now = asyncio.get_event_loop().time()
            elapsed = now - self.__class__._last_request_time
            if elapsed < 1.0:
                await asyncio.sleep(1.0 - elapsed)
            self.__class__._last_request_time = asyncio.get_event_loop().time()

    async def _get_json(self, url: str, params: dict | None = None) -> dict:
        if self.proxy_url:
            try:
                async with _httpx_client(timeout=15.0) as c:
                    r = await c.get(url, params=params)
                    r.raise_for_status()
                    return r.json()
            except Exception:
                pass
        async with self._rate_limit_sem:
            await self._rate_limit()
            for attempt in range(self.MAX_RETRIES):
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                            if resp.status == 429:
                                wait = 2 ** (attempt + 1)
                                print(f"[MangaDex] 429, жду {wait}с...")
                                await asyncio.sleep(wait)
                                continue
                            resp.raise_for_status()
                            return await resp.json()
                except (aiohttp.ClientError, TimeoutError, asyncio.TimeoutError) as e:
                    print(f"[MangaDex] попытка {attempt+1}/{self.MAX_RETRIES} ошибка: {e}")
                    if attempt == self.MAX_RETRIES - 1:
                        raise
                    await asyncio.sleep(2 ** attempt)
        return {}

    async def _proxy_get(self, path: str, params: dict | None = None) -> dict:
        if not self.proxy_url:
            return {}
        try:
            async with _httpx_client(timeout=30.0) as c:
                r = await c.get(f"{self.proxy_url}{path}", params=params)
                r.raise_for_status()
                return r.json()
        except Exception as e:
            print(f"[MangaDex proxy] Error: {e}")
            return {}

    async def _proxy_download(self, url: str) -> bytes:
        if not self.proxy_url:
            return b""
        try:
            async with _httpx_client(timeout=60.0) as c:
                r = await c.get(f"{self.proxy_url}/mangadex/download", params={"url": url})
                r.raise_for_status()
                import base64
                return base64.b64decode(r.json()["image_b64"])
        except Exception as e:
            print(f"[MangaDex proxy download] Error: {e}")
            return b""

    async def search(self, title: str) -> list[MangaResult]:
        data = await self._proxy_get("/mangadex/search", {"q": title, "limit": 10})
        if not data:
            data = await self._get_json(f"{self.BASE}/manga", params={
                "title": title,
                "limit": 10,
                "hasAvailableChapters": "true",
                "includes[]": "cover_art",
                "order[relevance]": "desc",
            })
        results = []
        for item in data.get("data", []):
            attrs = item["attributes"]
            title_en = attrs.get("title", {}).get("en", "")
            alt_titles = [v for t in attrs.get("altTitles", []) for v in t.values()]
            cover_url = None
            for rel in item.get("relationships", []):
                if rel["type"] == "cover_art":
                    fname = rel.get("attributes", {}).get("fileName", "")
                    if fname:
                        cover_url = f"https://uploads.mangadex.org/covers/{item['id']}/{fname}.256.jpg"
            results.append(MangaResult(
                id=item["id"],
                title=title_en or (alt_titles[0] if alt_titles else "Unknown"),
                alt_titles=alt_titles,
                description=attrs.get("description", {}).get("en", ""),
                status=attrs.get("status", ""),
                year=attrs.get("year"),
                cover_url=cover_url,
                source="mangadex",
            ))
        return results

    async def get_chapters(self, manga_id: str, lang: str = "en") -> list[Chapter]:
        chapters = []
        offset = 0
        limit = 100
        while True:
            data = await self._proxy_get(
                f"/mangadex/{manga_id}/chapters",
                {"lang": lang, "limit": limit, "offset": offset},
            )
            if not data:
                data = await self._get_json(f"{self.BASE}/manga/{manga_id}/feed", params={
                    "translatedLanguage[]": lang,
                    "limit": limit,
                    "offset": offset,
                    "order[chapter]": "asc",
                    "includes[]": "scanlation_group",
                })
            items = data.get("data", [])
            if not items:
                break
            for item in items:
                attrs = item["attributes"]
                ch_num = attrs.get("chapter", "")
                try:
                    float(ch_num)
                except (ValueError, TypeError):
                    continue
                chapters.append(Chapter(
                    id=item["id"],
                    number=ch_num,
                    title=attrs.get("title", ""),
                    volume=attrs.get("volume"),
                    pages_count=attrs.get("pages", 0),
                    translated_language=attrs.get("translatedLanguage", lang),
                ))
            offset += limit
            if len(items) < limit:
                break
        return chapters

    async def find_chapter_by_number(self, manga_id: str, chapter_number: str, lang: str) -> Chapter | None:
        chapters = await self.get_chapters(manga_id, lang)
        for ch in chapters:
            if ch.number == chapter_number:
                return ch
        return None

    async def get_available_languages(self, manga_id: str) -> list[str]:
        data = await self._proxy_get(
            f"/mangadex/{manga_id}/chapters",
            {"lang": "", "limit": 100, "offset": 0},
        )
        if not data:
            data = await self._get_json(f"{self.BASE}/manga/{manga_id}/feed", params={
                "limit": 100,
                "order[chapter]": "desc",
            })
        langs = set()
        for item in data.get("data", []):
            lang = item.get("attributes", {}).get("translatedLanguage", "")
            if lang:
                langs.add(lang)
        return sorted(langs)

    async def get_pages(self, chapter_id: str) -> list[Page]:
        data = await self._proxy_get(f"/mangadex/chapter/{chapter_id}/pages")
        if data and "pages" in data:
            return [
                Page(url=p["url"], index=p["index"], width=0, height=0)
                for p in data["pages"]
            ]
        data = await self._get_json(f"{self.BASE}/at-home/server/{chapter_id}")
        base_url = data.get("baseUrl", "")
        chapter_hash = data.get("chapter", {}).get("hash", "")
        ch_data = data.get("chapter", {})
        filenames = ch_data.get("data") or ch_data.get("dataSaver", [])
        pages = []
        for i, fname in enumerate(filenames):
            url = f"{base_url}/data/{chapter_hash}/{fname}"
            pages.append(Page(url=url, index=i, width=0, height=0))
        return pages

    async def download_page(self, page: Page) -> bytes:
        img_data = await self._proxy_download(page.url)
        if img_data:
            return img_data
        async with self._rate_limit_sem:
            await self._rate_limit()
            for attempt in range(self.MAX_RETRIES):
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(page.url, timeout=aiohttp.ClientTimeout(total=120)) as resp:
                            if resp.status == 429:
                                await asyncio.sleep(2 ** (attempt + 1))
                                continue
                            resp.raise_for_status()
                            return await resp.read()
                except (aiohttp.ClientError, TimeoutError, asyncio.TimeoutError):
                    if attempt == self.MAX_RETRIES - 1:
                        raise
                    await asyncio.sleep(1)
        return b""

    async def close(self):
        if self._own_session and not self._own_session.closed:
            await self._own_session.close()

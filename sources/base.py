from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class MangaResult:
    id: str
    title: str
    alt_titles: list[str]
    description: str
    status: str
    year: int | None
    cover_url: str | None
    source: str


@dataclass
class Chapter:
    id: str
    number: str
    title: str
    volume: str | None
    pages_count: int
    translated_language: str


@dataclass
class Page:
    url: str
    index: int
    width: int
    height: int


class BaseSource(ABC):
    @abstractmethod
    async def search(self, title: str) -> list[MangaResult]:
        pass

    @abstractmethod
    async def get_chapters(self, manga_id: str, lang: str) -> list[Chapter]:
        pass

    @abstractmethod
    async def get_pages(self, chapter_id: str) -> list[Page]:
        pass

    async def close(self):
        pass

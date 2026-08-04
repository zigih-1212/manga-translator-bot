"""Webhooks: внешние HTTP-уведомления о событиях перевода."""
import asyncio
import os

import httpx

from cfg import CONFIG
from .log import log


def _urls() -> list[str]:
    """Источники webhook-URL из config.json -> webbhooks.urls и env WEBHOOK_URLS."""
    urls = list(CONFIG.get("webhooks", {}).get("urls", []) or [])
    env = os.environ.get("WEBHOOK_URLS", "")
    for u in env.split(","):
        u = u.strip()
        if u:
            urls.append(u)
    # Дедуп, сохраняя порядок
    seen, out = set(), []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def webhooks_enabled() -> bool:
    return bool(_urls())


async def _post_once(url: str, payload: dict) -> bool:
    proxy = os.environ.get("HTTP_PROXY") or os.environ.get("https_proxy")
    try:
        async with httpx.AsyncClient(proxy=proxy, timeout=httpx.Timeout(15.0)) as client:
            r = await client.post(url, json=payload, headers={"Content-Type": "application/json"})
            if r.status_code >= 400:
                log.warning("Webhook %s -> HTTP %s: %s", url, r.status_code, r.text[:200])
                return False
            return True
    except Exception as e:
        log.warning("Webhook %s error: %s", url, e)
        return False


async def _post_with_retry(url: str, payload: dict, attempts: int = 3):
    for i in range(attempts):
        if await _post_once(url, payload):
            return True
        if i < attempts - 1:
            await asyncio.sleep(1 + i)
    return False


async def notify(event: str, payload: dict) -> list[asyncio.Task]:
    """Асинхронно разослать уведомление по всем webhook URL (без блокировки).

    Возвращает созданные задачи, чтобы вызывающий мог при необходимости их дождаться.
    """
    urls = _urls()
    if not urls:
        return []
    body = {"event": event, **payload}
    tasks = []
    for url in urls:
        tasks.append(asyncio.create_task(_post_with_retry(url, body)))
        log.info("Webhook queued: %s (%s)", url, event)
    return tasks


async def notify_chapter_done(manga_title: str, chapter: str, zip_url: str = ""):
    return await notify("chapter_done", {
        "manga": manga_title,
        "chapter": chapter,
        "zip_url": zip_url,
        "status": "done",
    })


async def notify_chapter_failed(manga_title: str, chapter: str, error: str = ""):
    return await notify("chapter_failed", {
        "manga": manga_title,
        "chapter": chapter,
        "error": error,
        "status": "failed",
    })
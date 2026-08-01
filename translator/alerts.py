import asyncio
import os
import time
from collections import defaultdict

import httpx

from cfg import TG_BOT_TOKEN, CONFIG
from .log import log

_ALERT_INTERVAL = 300  # seconds
_last_sent: dict[str, float] = defaultdict(float)
_alert_lock = asyncio.Lock()


def _get_chat_id():
    return CONFIG.get("telegram", {}).get("chat_id")


async def _send_raw(text: str):
    if not TG_BOT_TOKEN:
        return False
    chat_id = _get_chat_id()
    if not chat_id:
        return False
    proxy = os.environ.get("HTTP_PROXY") or os.environ.get("https_proxy")
    try:
        timeout = httpx.Timeout(15.0)
        async with httpx.AsyncClient(timeout=timeout, proxy=proxy) as client:
            r = await client.post(
                f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage",
                json={"chat_id": chat_id, "text": text, "disable_web_page_preview": True},
            )
            if r.status_code != 200:
                log.warning("Alert send failed: %s %s", r.status_code, r.text[:200])
                return False
            return True
    except Exception as e:
        log.warning("Alert send error: %s", e)
        return False


async def send_alert(message: str, tag: str = "generic", force: bool = False, cooldown: float = _ALERT_INTERVAL):
    """Send a rate-limited alert to the configured Telegram chat."""
    now = time.monotonic()
    async with _alert_lock:
        if not force and now - _last_sent[tag] < cooldown:
            return False
        _last_sent[tag] = now
    ok = await _send_raw(f"⚠️ *{tag}*\n{message}")
    if ok:
        log.info("Alert sent: %s", tag)
    return ok

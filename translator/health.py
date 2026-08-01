import asyncio
import json
import os
import time
from datetime import datetime, timezone

from aiohttp import web

from .log import log

HEALTH_PORT = int(os.environ.get("HEALTH_PORT", "8080"))

START_TIME = time.monotonic()
_bot_uptime_start = None
_last_activity = time.monotonic()
_monitor_task = None
_alert_cooldown_until = 0.0
_metrics = {
    "chapters_processed": 0,
    "pages_processed": 0,
    "errors": 0,
    "llm_success": 0,
    "llm_fail": 0,
    "ocr_success": 0,
    "ocr_fail": 0,
}


def record_activity():
    global _last_activity
    _last_activity = time.monotonic()


def inc_metric(name: str, value: int = 1):
    if name in _metrics:
        _metrics[name] += value


def record_llm(success: bool):
    inc_metric("llm_success" if success else "llm_fail")
    record_activity()


def record_ocr(success: bool):
    inc_metric("ocr_success" if success else "ocr_fail")
    record_activity()


def record_error():
    inc_metric("errors")
    record_activity()


def mark_bot_started():
    global _bot_uptime_start
    _bot_uptime_start = time.monotonic()


def _health_payload() -> dict:
    uptime = time.monotonic() - START_TIME
    payload = {
        "status": "ok",
        "uptime_seconds": round(uptime, 1),
        "last_activity_seconds_ago": round(time.monotonic() - _last_activity, 1),
        "metrics": dict(_metrics),
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if _bot_uptime_start is not None:
        payload["bot_uptime_seconds"] = round(time.monotonic() - _bot_uptime_start, 1)
    return payload


async def _health_handler(request: web.Request) -> web.Response:
    return web.json_response(_health_payload())


async def _metrics_handler(request: web.Request) -> web.Response:
    return web.json_response(_metrics)


async def _index_handler(request: web.Request) -> web.Response:
    return web.json_response({"service": "manga-translator-bot", "endpoints": ["/health", "/metrics"]})


def _build_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/", _index_handler)
    app.router.add_get("/health", _health_handler)
    app.router.add_get("/metrics", _metrics_handler)
    return app


async def start_health_server(port: int = HEALTH_PORT):
    app = _build_app()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host="0.0.0.0", port=port)
    await site.start()
    log.info("Health server listening on 0.0.0.0:%s", port)
    global _monitor_task
    if _monitor_task is None:
        _monitor_task = asyncio.create_task(_monitor_loop())
    return runner


async def stop_health_server(runner):
    if runner:
        await runner.cleanup()
    global _monitor_task
    if _monitor_task:
        _monitor_task.cancel()
        _monitor_task = None


async def _monitor_loop(interval: float = 60.0):
    global _alert_cooldown_until
    last_errors = 0
    while True:
        await asyncio.sleep(interval)
        try:
            errors = _metrics.get("errors", 0)
            llm_fail = _metrics.get("llm_fail", 0)
            ocr_fail = _metrics.get("ocr_fail", 0)
            llm_success = _metrics.get("llm_success", 0)
            idle = time.monotonic() - _last_activity
            now = time.monotonic()

            new_errors = errors - last_errors
            last_errors = errors

            triggers = []
            if new_errors >= 3:
                triggers.append(f"+{new_errors} ошибок за последние {int(interval)}с")
            if llm_success > 0 and llm_fail >= llm_success * 3:
                triggers.append(f"LLM падения ({llm_fail} fail / {llm_success} success)")
            if ocr_fail >= 3 and ocr_success == 0:
                triggers.append(f"OCR полностью падает ({ocr_fail} fail)")

            if triggers and now > _alert_cooldown_until:
                _alert_cooldown_until = now + 300
                from .alerts import send_alert
                asyncio.create_task(send_alert(
                    "Критические метрики:\n" + "\n".join(f"- {t}" for t in triggers),
                    tag="health",
                ))
        except Exception as e:
            log.error("Monitor loop error: %s", e)

import asyncio
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from aiohttp import web

from .log import log
from cfg import DATA_DIR

HEALTH_PORT = int(os.environ.get("HEALTH_PORT", "8080"))

START_TIME = time.monotonic()
_bot_uptime_start = None
_last_activity = time.monotonic()
_monitor_task = None
_report_task = None
_alert_cooldown_until = 0.0
_metrics = {
    "chapters_processed": 0,
    "pages_processed": 0,
    "errors": 0,
    "llm_success": 0,
    "llm_fail": 0,
    "ocr_success": 0,
    "ocr_fail": 0,
    "self_corrections": 0,
    "validator_fixes": 0,
}

_METRICS_JSON = DATA_DIR / "metrics.json"
_REPORT_TAG = "metrics_report"
_REPORT_INTERVAL = int(os.environ.get("METRICS_REPORT_INTERVAL", "3600"))


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


def record_self_correction():
    inc_metric("self_corrections")
    record_activity()


def record_validator_fix():
    inc_metric("validator_fixes")
    record_activity()


def mark_bot_started():
    global _bot_uptime_start
    _bot_uptime_start = time.monotonic()


def _persist_metrics():
    """Save current metrics snapshot to DATA_DIR/metrics.json (survives restarts)."""
    try:
        data = {
            "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "metrics": dict(_metrics),
            "uptime_seconds": round(time.monotonic() - START_TIME, 1),
        }
        _METRICS_JSON.parent.mkdir(parents=True, exist_ok=True)
        with open(_METRICS_JSON, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log.error("Failed to persist metrics: %s", e)


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
    global _monitor_task, _report_task
    if _monitor_task is None:
        _monitor_task = asyncio.create_task(_monitor_loop())
    if _report_task is None:
        _report_task = asyncio.create_task(_persist_loop())
    return runner


async def stop_health_server(runner):
    if runner:
        await runner.cleanup()
    global _monitor_task, _report_task
    for task_name in ("_monitor_task", "_report_task"):
        task = globals()[task_name]
        if task:
            task.cancel()
            globals()[task_name] = None


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


async def _persist_loop(interval: float = 300.0):
    """Periodically dump metrics to JSON + send a periodic Telegram summary."""
    while True:
        await asyncio.sleep(interval)
        try:
            _persist_metrics()
        except Exception as e:
            log.error("Persist loop error: %s", e)
        if _REPORT_INTERVAL > 0:
            try:
                await _send_periodic_report()
            except Exception as e:
                log.error("Report loop error: %s", e)


def _format_duration(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}ч {m}м"
    if m:
        return f"{m}м {s}с"
    return f"{s}с"


async def _send_periodic_report():
    """Send a JSON-style summary to Telegram with rate-limit tag."""
    from .alerts import send_alert
    uptime = _format_duration(time.monotonic() - START_TIME)
    payload = {
        "uptime": uptime,
        "metrics": dict(_metrics),
    }
    msg = "📊 Метрики бота:\n" + json.dumps(payload, ensure_ascii=False, indent=2)
    # send_alert applies per-tag cooldown; force=True to skip cooldown for the scheduled report
    await send_alert(msg, tag=_REPORT_TAG, force=True)

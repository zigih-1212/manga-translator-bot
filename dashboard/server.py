"""Web Dashboard: статус бота, метрики, очередь и логи в браузере."""
import asyncio
import json
import os
from pathlib import Path

from aiohttp import web

from translator.health import _health_payload
from cfg import DATA_DIR

DASHBOARD_PORT = int(os.environ.get("DASHBOARD_PORT", "8091"))
STATIC_DIR = Path(__file__).parent / "static"


def _active_task_counts() -> dict:
    """Собирает число активных задач из обработчиков (без жёстких импортов)."""
    total = 0
    by_module = {}
    try:
        from bot.handlers import manga_translate, translate
        for mod_name, mod in (("manga", manga_translate), ("translate", translate)):
            tasks = getattr(mod, "active_tasks", None) or {}
            live = sum(1 for t in tasks.values() if t and not t.done())
            by_module[mod_name] = live
            total += live
    except Exception:
        pass
    return {"total": total, "by_module": by_module}


def _queue_stats() -> dict:
    """Числа задач в очереди SQLite по статусам."""
    counts = {"pending": 0, "processing": 0, "completed": 0, "failed": 0}
    tasks = []
    try:
        from cfg.db import TranslationQueueDB
        db = TranslationQueueDB()
        tasks = db.get_all_tasks()
        db.close()
    except Exception:
        tasks = []
    for t in tasks:
        counts[t.get("status", "pending")] = counts.get(t.get("status", "pending"), 0) + 1
    counts["total"] = len(tasks)
    return counts


def _recent_errors(limit: int = 8) -> list[dict]:
    """Последние записи об ошибках из лога приложений (в обратном порядке)."""
    log_file = Path(DATA_DIR) / "logs" / "app.jsonl"
    if not log_file.exists():
        return []
    lines = []
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except Exception:
                    continue
                if entry.get("level") == "ERROR" or entry.get("exc"):
                    lines.append({
                        "ts": entry.get("ts", ""),
                        "msg": entry.get("msg", ""),
                        "exc": entry.get("exc"),
                    })
    except Exception:
        return []
    return lines[-limit:][::-1]


async def index_handler(request: web.Request) -> web.Response:
    return web.FileResponse(STATIC_DIR / "index.html")


async def overview_handler(request: web.Request) -> web.Response:
    payload = {
        "health": _health_payload(),
        "queue": _queue_stats(),
        "active": _active_task_counts(),
        "recent_errors": _recent_errors(),
        "endpoints": ["/api/overview", "/api/metrics"],
    }
    return web.json_response(payload)


async def metrics_handler(request: web.Request) -> web.Response:
    return web.json_response(_health_payload().get("metrics", {}))


def _build_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/", index_handler)
    app.router.add_get("/api/overview", overview_handler)
    app.router.add_get("/api/metrics", metrics_handler)
    app.router.add_static("/static", STATIC_DIR)
    return app


async def start_dashboard_server(port: int = DASHBOARD_PORT):
    """Запустить web-дашборд в фоне (для вызова из bot.main)."""
    app = _build_app()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    return runner


async def main():
    runner = await start_dashboard_server()
    print(f"Starting dashboard on http://0.0.0.0:{DASHBOARD_PORT}")
    while True:
        await asyncio.sleep(3600)


if __name__ == "__main__":
    asyncio.run(main())
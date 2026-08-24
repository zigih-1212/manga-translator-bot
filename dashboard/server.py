"""Web Dashboard: управление очередью, тайтлами, настройками, логи в реальном времени."""
import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from aiohttp import web
from aiohttp import WSMsgType

from translator.health import _health_payload
from cfg import DATA_DIR, CONFIG, save_config, load_glossary, save_glossary
from cfg.db import TranslationQueueDB
from bot.utils.telegram_helpers import _bot as get_bot

DASHBOARD_PORT = int(os.environ.get("DASHBOARD_PORT", "8091"))
STATIC_DIR = Path(__file__).parent / "static"
# Auth token: if set, all /api/* requests must carry "Authorization: Bearer <token>"
# or ?token=<token>. Leave empty to disable auth (local-only setups).
DASHBOARD_TOKEN = os.environ.get("DASHBOARD_TOKEN", "")

# WebSocket connections for real-time updates
ws_clients: set[web.WebSocketResponse] = set()

logger = logging.getLogger("dashboard")


@web.middleware
async def auth_middleware(request: web.Request, handler):
    """Bearer-token gate for API routes (static page stays open)."""
    if DASHBOARD_TOKEN and request.path.startswith("/api/") and request.path != "/api/ws":
        auth = request.headers.get("Authorization", "")
        token = auth[7:] if auth.startswith("Bearer ") else request.query.get("token", "")
        if token != DASHBOARD_TOKEN:
            return web.json_response({"error": "unauthorized"}, status=401)
    return await handler(request)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

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


async def _broadcast_ws(data: dict):
    """Отправить данные всем WebSocket-клиентам."""
    if not ws_clients:
        return
    text = json.dumps(data, ensure_ascii=False)
    for ws in list(ws_clients):
        try:
            await ws.send_str(text)
        except Exception:
            ws_clients.discard(ws)


# ──────────────────────────────────────────────
# Handlers: Pages
# ──────────────────────────────────────────────

async def index_handler(request: web.Request) -> web.Response:
    return web.FileResponse(STATIC_DIR / "index.html")


# ──────────────────────────────────────────────
# Handlers: API — Overview
# ──────────────────────────────────────────────

async def overview_handler(request: web.Request) -> web.Response:
    payload = {
        "health": _health_payload(),
        "queue": _queue_stats(),
        "active": _active_task_counts(),
        "recent_errors": _recent_errors(),
        "endpoints": [
            "/api/overview", "/api/metrics",
            "/api/queue", "/api/queue/retry", "/api/queue/cancel", "/api/queue/clear",
            "/api/titles", "/api/titles/add", "/api/titles/delete",
            "/api/settings", "/api/settings/update",
            "/api/glossary", "/api/glossary/add", "/api/glossary/delete",
            "/api/logs", "/api/ws",
        ],
    }
    return web.json_response(payload)


async def metrics_handler(request: web.Request) -> web.Response:
    return web.json_response(_health_payload().get("metrics", {}))


# ──────────────────────────────────────────────
# Handlers: API — Queue Management
# ──────────────────────────────────────────────

async def queue_list_handler(request: web.Request) -> web.Response:
    """GET /api/queue — полный список задач."""
    try:
        db = TranslationQueueDB()
        tasks = db.get_all_tasks()
        db.close()
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)
    return web.json_response({"tasks": tasks})


async def queue_retry_handler(request: web.Request) -> web.Response:
    """POST /api/queue/retry — перезапустить задачу."""
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "invalid json"}, status=400)
    manga_id = data.get("manga_id")
    chapter = data.get("chapter")
    if not manga_id or not chapter:
        return web.json_response({"error": "manga_id and chapter required"}, status=400)
    try:
        db = TranslationQueueDB()
        db.update_task_status(manga_id, chapter, "pending")
        db.close()
        await _broadcast_ws({"event": "queue_updated", "queue": _queue_stats()})
        return web.json_response({"ok": True})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


async def queue_cancel_handler(request: web.Request) -> web.Response:
    """POST /api/queue/cancel — отменить задачу."""
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "invalid json"}, status=400)
    manga_id = data.get("manga_id")
    chapter = data.get("chapter")
    if not manga_id or not chapter:
        return web.json_response({"error": "manga_id and chapter required"}, status=400)
    try:
        db = TranslationQueueDB()
        db.update_task_status(manga_id, chapter, "cancelled")
        db.close()
        await _broadcast_ws({"event": "queue_updated", "queue": _queue_stats()})
        return web.json_response({"ok": True})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


async def queue_clear_handler(request: web.Request) -> web.Response:
    """POST /api/queue/clear — очистить завершённые/ошибки."""
    try:
        data = await request.json() if request.can_read_body else {}
    except Exception:
        data = {}
    status_filter = data.get("status")  # "completed", "failed", "cancelled", or None for all done
    try:
        db = TranslationQueueDB()
        if status_filter:
            db.clear_tasks_by_status(status_filter)
        else:
            db.clear_completed_tasks()
        db.close()
        await _broadcast_ws({"event": "queue_updated", "queue": _queue_stats()})
        return web.json_response({"ok": True})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


async def queue_add_handler(request: web.Request) -> web.Response:
    """POST /api/queue/add — добавить задачу в очередь вручную."""
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "invalid json"}, status=400)
    manga_id = data.get("manga_id")
    chapter = data.get("chapter")
    source_lang = data.get("source_lang", "ko")
    if not manga_id or not chapter:
        return web.json_response({"error": "manga_id and chapter required"}, status=400)
    try:
        db = TranslationQueueDB()
        db.add_task(manga_id, chapter, source_lang)
        db.close()
        await _broadcast_ws({"event": "queue_updated", "queue": _queue_stats()})
        return web.json_response({"ok": True})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


# ──────────────────────────────────────────────
# Handlers: API — Titles Management
# ──────────────────────────────────────────────

async def titles_list_handler(request: web.Request) -> web.Response:
    """GET /api/titles — список тайтлов из конфига."""
    titles = CONFIG.get("titles", [])
    return web.json_response({"titles": titles})


async def titles_add_handler(request: web.Request) -> web.Response:
    """POST /api/titles/add — добавить тайтл."""
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "invalid json"}, status=400)
    name = data.get("name")
    mangadex_id = data.get("mangadex_id")
    source_lang = data.get("source_lang", "ko")
    if not name or not mangadex_id:
        return web.json_response({"error": "name and mangadex_id required"}, status=400)
    titles = CONFIG.get("titles", [])
    # Check for duplicates
    if any(t.get("mangadex_id") == mangadex_id for t in titles):
        return web.json_response({"error": "title already exists"}, status=409)
    new_title = {
        "name": name,
        "mangadex_id": mangadex_id,
        "source_lang": source_lang,
        "chapters_count": data.get("chapters_count", 0),
        "first_chapter": data.get("first_chapter", "1"),
        "last_chapter": data.get("last_chapter", "0"),
    }
    titles.append(new_title)
    CONFIG["titles"] = titles
    save_config()
    await _broadcast_ws({"event": "titles_updated", "titles": titles})
    return web.json_response({"ok": True, "title": new_title})


async def titles_delete_handler(request: web.Request) -> web.Response:
    """POST /api/titles/delete — удалить тайтл."""
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "invalid json"}, status=400)
    mangadex_id = data.get("mangadex_id")
    if not mangadex_id:
        return web.json_response({"error": "mangadex_id required"}, status=400)
    titles = CONFIG.get("titles", [])
    titles = [t for t in titles if t.get("mangadex_id") != mangadex_id]
    CONFIG["titles"] = titles
    save_config()
    await _broadcast_ws({"event": "titles_updated", "titles": titles})
    return web.json_response({"ok": True})


async def titles_update_handler(request: web.Request) -> web.Response:
    """POST /api/titles/update — обновить тайтл."""
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "invalid json"}, status=400)
    mangadex_id = data.get("mangadex_id")
    if not mangadex_id:
        return web.json_response({"error": "mangadex_id required"}, status=400)
    titles = CONFIG.get("titles", [])
    for t in titles:
        if t.get("mangadex_id") == mangadex_id:
            for key in ("name", "source_lang", "chapters_count", "first_chapter", "last_chapter"):
                if key in data:
                    t[key] = data[key]
            break
    else:
        return web.json_response({"error": "title not found"}, status=404)
    CONFIG["titles"] = titles
    save_config()
    await _broadcast_ws({"event": "titles_updated", "titles": titles})
    return web.json_response({"ok": True})


# ──────────────────────────────────────────────
# Handlers: API — Settings
# ──────────────────────────────────────────────

async def settings_get_handler(request: web.Request) -> web.Response:
    """GET /api/settings — текущие настройки."""
    # Return safe settings (mask secrets)
    llm_cfg = CONFIG.get("llm", {})
    safe = {
        "telegram": CONFIG.get("telegram", {}),
        "titles_count": len(CONFIG.get("titles", [])),
        "auto_check_interval_hours": CONFIG.get("auto_check_interval_hours", 6),
        "inpaint": CONFIG.get("inpaint", {}),
        "llm": {
            "provider": llm_cfg.get("provider", "groq"),
            "model": llm_cfg.get("model", ""),
        },
        "ocr": CONFIG.get("ocr", {}),
        "proxy": bool(CONFIG.get("proxy")),
        "deepl": {
            "api_key": "***" if llm_cfg.get("deepl_api_key") else "",
        },
        "openrouter": {
            "api_key": "***" if llm_cfg.get("openrouter_api_key") else "",
        },
        "groq": {
            "api_key": "***" if llm_cfg.get("groq_api_key") else "",
        },
        "gemini": {
            "api_key": "***" if llm_cfg.get("gemini_api_key") else "",
        },
    }
    return web.json_response(safe)


async def settings_update_handler(request: web.Request) -> web.Response:
    """POST /api/settings/update — обновить настройки."""
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "invalid json"}, status=400)
    # Update allowed settings
    if "auto_check_interval_hours" in data:
        CONFIG["auto_check_interval_hours"] = int(data["auto_check_interval_hours"])
    if "inpaint" in data:
        CONFIG.setdefault("inpaint", {}).update(data["inpaint"])
    if "ocr" in data:
        CONFIG.setdefault("ocr", {}).update(data["ocr"])
    if "llm" in data:
        llm_update = data["llm"]
        if "provider" in llm_update:
            CONFIG.setdefault("llm", {})["provider"] = llm_update["provider"]
        if "model" in llm_update:
            CONFIG.setdefault("llm", {})["model"] = llm_update["model"]
    if "telegram" in data and "chat_id" in data["telegram"]:
        CONFIG.setdefault("telegram", {})["chat_id"] = data["telegram"]["chat_id"]
    # API keys: empty = clear, "***" = unchanged, otherwise update
    llm_cfg = CONFIG.setdefault("llm", {})
    for provider in ("deepl", "openrouter", "groq", "gemini"):
        if provider in data:
            val = data[provider].get("api_key", "")
            if val == "":
                # Clear the key
                key_map = {
                    "deepl": "deepl_api_key",
                    "openrouter": "openrouter_api_key",
                    "groq": "groq_api_key",
                    "gemini": "gemini_api_key",
                }
                llm_cfg.pop(key_map[provider], None)
            elif not val.endswith("***"):
                # Update with new value
                key_map = {
                    "deepl": "deepl_api_key",
                    "openrouter": "openrouter_api_key",
                    "groq": "groq_api_key",
                    "gemini": "gemini_api_key",
                }
                llm_cfg[key_map[provider]] = val
    save_config()
    await _broadcast_ws({"event": "settings_updated"})
    return web.json_response({"ok": True})


# ──────────────────────────────────────────────
# Handlers: API — Glossary
# ──────────────────────────────────────────────

async def glossary_get_handler(request: web.Request) -> web.Response:
    """GET /api/glossary — получить глоссарий."""
    glossary = load_glossary()
    return web.json_response({"glossary": glossary})


async def glossary_add_handler(request: web.Request) -> web.Response:
    """POST /api/glossary/add — добавить запись в глоссарий."""
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "invalid json"}, status=400)
    term = data.get("term")
    translation = data.get("translation")
    category = data.get("category", "terms")  # "characters" or "terms"
    if not term or not translation:
        return web.json_response({"error": "term and translation required"}, status=400)
    glossary = load_glossary()
    glossary.setdefault(category, {})[term] = translation
    save_glossary(glossary)
    await _broadcast_ws({"event": "glossary_updated"})
    return web.json_response({"ok": True})


async def glossary_delete_handler(request: web.Request) -> web.Response:
    """POST /api/glossary/delete — удалить запись из глоссария."""
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "invalid json"}, status=400)
    term = data.get("term")
    category = data.get("category", "terms")
    if not term:
        return web.json_response({"error": "term required"}, status=400)
    glossary = load_glossary()
    if category in glossary and term in glossary[category]:
        del glossary[category][term]
        save_glossary(glossary)
        await _broadcast_ws({"event": "glossary_updated"})
        return web.json_response({"ok": True})
    return web.json_response({"error": "term not found"}, status=404)


async def glossary_import_handler(request: web.Request) -> web.Response:
    """POST /api/glossary/import — импортировать глоссарий из JSON."""
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "invalid json"}, status=400)
    new_entries = data.get("glossary", {})
    if not isinstance(new_entries, dict):
        return web.json_response({"error": "glossary must be object"}, status=400)
    merge = data.get("merge", True)
    glossary = load_glossary() if merge else {"characters": {}, "terms": {}}
    for category in ("characters", "terms"):
        if category in new_entries and isinstance(new_entries[category], dict):
            glossary.setdefault(category, {}).update(new_entries[category])
    save_glossary(glossary)
    await _broadcast_ws({"event": "glossary_updated"})
    return web.json_response({"ok": True})


# ──────────────────────────────────────────────
# Handlers: API — Logs
# ──────────────────────────────────────────────

async def logs_handler(request: web.Request) -> web.Response:
    """GET /api/logs?limit=50&level=ERROR — последние логи."""
    limit = int(request.query.get("limit", 50))
    level_filter = request.query.get("level")
    log_file = Path(DATA_DIR) / "logs" / "app.jsonl"
    if not log_file.exists():
        return web.json_response({"logs": []})
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
                if level_filter and entry.get("level") != level_filter:
                    continue
                lines.append(entry)
    except Exception:
        return web.json_response({"logs": []})
    return web.json_response({"logs": lines[-limit:]})


# ──────────────────────────────────────────────
# WebSocket
# ──────────────────────────────────────────────

async def ws_handler(request: web.Request) -> web.WebSocketResponse:
    """GET /api/ws — WebSocket для real-time обновлений."""
    if DASHBOARD_TOKEN:
        auth = request.headers.get("Authorization", "")
        token = auth[7:] if auth.startswith("Bearer ") else request.query.get("token", "")
        if token != DASHBOARD_TOKEN:
            raise web.HTTPUnauthorized()
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    ws_clients.add(ws)
    logger.info("WS client connected, total: %d", len(ws_clients))

    # Send initial state
    await ws.send_str(json.dumps({
        "event": "connected",
        "queue": _queue_stats(),
        "active": _active_task_counts(),
    }, ensure_ascii=False))

    async for msg in ws:
        if msg.type == WSMsgType.TEXT:
            # Handle ping/echo
            try:
                data = json.loads(msg.data)
                if data.get("type") == "ping":
                    await ws.send_str(json.dumps({"type": "pong"}))
            except Exception:
                pass
        elif msg.type in (WSMsgType.CLOSE, WSMsgType.ERROR, WSMsgType.CLOSING):
            break

    ws_clients.discard(ws)
    logger.info("WS client disconnected, total: %d", len(ws_clients))
    return ws


# ──────────────────────────────────────────────
# App Builder
# ──────────────────────────────────────────────

def _build_app() -> web.Application:
    app = web.Application(middlewares=[auth_middleware])
    # Pages
    app.router.add_get("/", index_handler)
    # API — Overview
    app.router.add_get("/api/overview", overview_handler)
    app.router.add_get("/api/metrics", metrics_handler)
    # API — Queue
    app.router.add_get("/api/queue", queue_list_handler)
    app.router.add_post("/api/queue/add", queue_add_handler)
    app.router.add_post("/api/queue/retry", queue_retry_handler)
    app.router.add_post("/api/queue/cancel", queue_cancel_handler)
    app.router.add_post("/api/queue/clear", queue_clear_handler)
    # API — Titles
    app.router.add_get("/api/titles", titles_list_handler)
    app.router.add_post("/api/titles/add", titles_add_handler)
    app.router.add_post("/api/titles/update", titles_update_handler)
    app.router.add_post("/api/titles/delete", titles_delete_handler)
    # API — Settings
    app.router.add_get("/api/settings", settings_get_handler)
    app.router.add_post("/api/settings/update", settings_update_handler)
    # API — Glossary
    app.router.add_get("/api/glossary", glossary_get_handler)
    app.router.add_post("/api/glossary/add", glossary_add_handler)
    app.router.add_post("/api/glossary/delete", glossary_delete_handler)
    app.router.add_post("/api/glossary/import", glossary_import_handler)
    # API — Logs
    app.router.add_get("/api/logs", logs_handler)
    # WebSocket
    app.router.add_get("/api/ws", ws_handler)
    # Static
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
    logger.info("Starting dashboard on http://0.0.0.0:%d", DASHBOARD_PORT)
    while True:
        await asyncio.sleep(3600)


if __name__ == "__main__":
    asyncio.run(main())

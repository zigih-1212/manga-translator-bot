"""Supervisor for Kaggle: watchdog + keepalive + Telegram alerts.

Run this as a BLOCKING cell in Kaggle Notebook.

Features:
- Restarts server.py and main.py (bot) if they crash
- Keepalive pings to prevent Kaggle idle shutdown (every 10 min)
- Telegram notifications on crash/restart
- Persists config.json across restarts (survives git reset)
- Health checks: server HTTP + bot via Telegram API
- Graceful shutdown on Ctrl+C
"""
import os
import sys
import time
import signal
import subprocess
import threading
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent.parent
SERVER_CMD = [sys.executable, str(ROOT / "kaggle" / "server.py")]
BOT_CMD = [sys.executable, str(ROOT / "main.py")]

if os.path.exists("/kaggle"):
    LOG_DIR = Path("/kaggle/working")
    PERSIST_DIR = Path("/kaggle/output")  # persistent across restarts
else:
    LOG_DIR = ROOT / "temp"
    PERSIST_DIR = ROOT / "cfg"
LOG_DIR.mkdir(parents=True, exist_ok=True)
PERSIST_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_PATH = ROOT / "cfg" / "config.json"
PERSIST_CONFIG = PERSIST_DIR / "config.json"

SERVER_LOG = LOG_DIR / "server.log"
BOT_LOG = LOG_DIR / "bot.log"
SUPERVISOR_LOG = LOG_DIR / "supervisor.log"

CHECK_INTERVAL = 15        # seconds between health checks
RESTART_DELAY = 5          # seconds before restarting a crashed process
SAVE_INTERVAL = 300        # persist config every 5 min
KEEPALIVE_INTERVAL = 600    # ping every 10 min to prevent idle shutdown
MAX_RESTARTS_PER_HOUR = 6  # circuit breaker: stop restarting if crashing too fast

# Telegram alerts
BOT_TOKEN = os.environ.get("TG_BOT_TOKEN", "")
CHAT_ID = os.environ.get("TG_CHAT_ID", "") or os.environ.get("CHAT_ID", "")


def _log(msg: str):
    """Write to supervisor log and print."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    try:
        with open(SUPERVISOR_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def _send_telegram_alert(text: str):
    """Send alert to Telegram chat."""
    if not BOT_TOKEN or not CHAT_ID:
        return
    try:
        import httpx
        httpx.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            json={"chat_id": CHAT_ID, "text": text, "parse_mode": "HTML"},
            timeout=10,
        )
    except Exception as e:
        _log(f"[alert] Telegram send failed: {e}")


def _restore_config():
    """Copy config.json back from persistent storage (survives git reset/restart)."""
    try:
        import json
        if PERSIST_CONFIG.exists() and CONFIG_PATH.exists():
            persist = json.loads(PERSIST_CONFIG.read_text(encoding="utf-8"))
            current = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            # Merge persistent titles/telegram into current config
            current["titles"] = persist.get("titles", current.get("titles", []))
            current["telegram"] = persist.get("telegram", current.get("telegram", {}))
            CONFIG_PATH.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")
            _log(f"[config] restored titles={len(current['titles'])}")
    except Exception as e:
        _log(f"[config] restore skipped: {e}")


def _save_config():
    """Persist current config.json so it survives restarts."""
    try:
        import json
        if CONFIG_PATH.exists():
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            PERSIST_CONFIG.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        _log(f"[config] save failed: {e}")


def _alive(proc: subprocess.Popen | None) -> bool:
    return proc is not None and proc.poll() is None


def _spawn(cmd, log_path, tag: str) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(log_path, "ab")
    proc = subprocess.Popen(
        cmd, stdout=f, stderr=subprocess.STDOUT, text=True, cwd=str(ROOT)
    )
    _log(f"[{tag}] started pid={proc.pid}")
    return proc


def _kill(proc: subprocess.Popen | None, tag: str):
    """Gracefully terminate a process."""
    if proc is None or proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=10)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    _log(f"[{tag}] stopped")


def _kill_old_processes():
    """Kill previously-started server.py/main.py so only one instance runs."""
    for script in ("server.py", "main.py"):
        try:
            out = subprocess.run(
                ["pgrep", "-f", script],
                capture_output=True, text=True, timeout=10,
            )
        except Exception:
            continue
        for pid in out.stdout.split():
            try:
                subprocess.run(["kill", "-9", pid], timeout=5)
                _log(f"[cleanup] killed old {script} pid={pid}")
            except Exception:
                pass


def _server_healthy() -> bool:
    """Check if server.py is responding on port 5003."""
    try:
        import httpx
        r = httpx.get("http://localhost:5003/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def _bot_healthy() -> bool:
    """Check if bot is alive by calling Telegram getMe API."""
    if not BOT_TOKEN:
        return True  # Can't check without token
    try:
        import httpx
        r = httpx.get(
            f"https://api.telegram.org/bot{BOT_TOKEN}/getMe",
            timeout=5,
        )
        return r.status_code == 200
    except Exception:
        return False


def _do_keepalive():
    """Ping keepalive endpoint to prevent Kaggle idle shutdown."""
    try:
        import httpx
        # Ping the keepalive URL if configured, or just touch a URL
        keepalive_url = os.environ.get("KEEPALIVE_URL", "")
        if keepalive_url:
            httpx.get(keepalive_url, timeout=10)
            _log(f"[keepalive] pinged {keepalive_url}")
        else:
            # Self-ping to keep network active
            httpx.get("http://localhost:5003/health", timeout=5)
            _log("[keepalive] pinged localhost")
    except Exception as e:
        _log(f"[keepalive] failed: {e}")


def supervise():
    _log("=" * 60)
    _log("Supervisor started. Press Ctrl+C to stop.")
    _log(f"ROOT: {ROOT}")
    _log(f"Check interval: {CHECK_INTERVAL}s | Keepalive: {KEEPALIVE_INTERVAL}s")
    _log("=" * 60)

    _kill_old_processes()
    _restore_config()

    server = None
    bot = None
    last_save = time.time()
    last_keepalive = time.time()

    # Track restarts for circuit breaker
    restart_times: list[float] = []

    def _restart_count_last_hour() -> int:
        now = time.time()
        cutoff = now - 3600
        # Remove old entries
        nonlocal restart_times
        restart_times = [t for t in restart_times if t > cutoff]
        return len(restart_times)

    def _record_restart():
        restart_times.append(time.time())

    # Don't duplicate an already-healthy server
    if _server_healthy():
        _log("[server] already healthy on :5003, not spawning a duplicate")
    else:
        server = _spawn(SERVER_CMD, SERVER_LOG, "server")

    # Track if bot was ever started
    bot_ever_started = False

    try:
        while True:
            time.sleep(CHECK_INTERVAL)

            now = time.time()

            # Periodically persist config
            if now - last_save > SAVE_INTERVAL:
                _save_config()
                last_save = now

            # Keepalive ping to prevent Kaggle idle shutdown
            if now - last_keepalive > KEEPALIVE_INTERVAL:
                _do_keepalive()
                last_keepalive = now

            # --- Server health ---
            server_ok = _server_healthy()
            if server is not None and not _alive(server):
                _log("[server] process died")
                _record_restart()
                if _restart_count_last_hour() > MAX_RESTARTS_PER_HOUR:
                    _log("[server] CRASH LOOP DETECTED — not restarting")
                    _send_telegram_alert("⚠️ <b>Server crash loop</b> — manual intervention needed!")
                else:
                    server = _spawn(SERVER_CMD, SERVER_LOG, "server")
                    _send_telegram_alert("🔄 <b>Server restarted</b> (was down)")
            elif server is None and not server_ok:
                _log("[server] not healthy, starting...")
                _record_restart()
                server = _spawn(SERVER_CMD, SERVER_LOG, "server")

            # --- Bot health ---
            bot_alive = _alive(bot)
            if bot is not None and not bot_alive:
                _log("[bot] process died")
                _record_restart()
                if _restart_count_last_hour() > MAX_RESTARTS_PER_HOUR:
                    _log("[bot] CRASH LOOP DETECTED — not restarting")
                    _send_telegram_alert("⚠️ <b>Bot crash loop</b> — manual intervention needed!")
                else:
                    time.sleep(RESTART_DELAY)
                    bot = _spawn(BOT_CMD, BOT_LOG, "bot")
                    _send_telegram_alert("🔄 <b>Bot restarted</b> (was down)")
            elif bot is None:
                # Start bot if not running
                _log("[bot] starting...")
                bot = _spawn(BOT_CMD, BOT_LOG, "bot")
                bot_ever_started = True

            # Additional bot check via Telegram API (only if bot was started)
            if bot_ever_started and bot is not None and _alive(bot):
                if not _bot_healthy():
                    _log("[bot] Telegram API unreachable, may be network issue")
                    # Don't restart immediately — could be Telegram outage
                    # Wait for next cycle

    except KeyboardInterrupt:
        _log("Supervisor interrupted, shutting down...")
    finally:
        _save_config()
        _kill(bot, "bot")
        _kill(server, "server")
        _log("All processes stopped.")
        _send_telegram_alert("🛑 <b>Supervisor stopped</b> — bot is offline.")


if __name__ == "__main__":
    supervise()

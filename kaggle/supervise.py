"""Supervisor for Kaggle: keeps server.py and main.py (bot) alive.

Run this as a BLOCKING cell in Kaggle. The running cell keeps the session
active (no idle shutdown), and the supervisor restarts any crashed process.
"""

import os
import sys
import time
import signal
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SERVER_CMD = [sys.executable, str(ROOT / "kaggle" / "server.py")]
BOT_CMD = [sys.executable, str(ROOT / "main.py")]

if os.path.exists("/kaggle"):
    LOG_DIR = Path("/kaggle/working")
else:
    LOG_DIR = ROOT / "temp"
LOG_DIR.mkdir(parents=True, exist_ok=True)

SERVER_LOG = LOG_DIR / "server.log"
BOT_LOG = LOG_DIR / "bot.log"

CHECK_INTERVAL = 15
RESTART_DELAY = 5


def _alive(proc: subprocess.Popen | None) -> bool:
    return proc is not None and proc.poll() is None


def _spawn(cmd, log_path, tag: str) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(log_path, "ab")
    proc = subprocess.Popen(
        cmd, stdout=f, stderr=subprocess.STDOUT, text=True, cwd=str(ROOT)
    )
    print(f"[{tag}] started pid={proc.pid} (log: {log_path})")
    return proc


def _server_healthy() -> bool:
    try:
        import httpx

        r = httpx.get("http://localhost:5003/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def supervise():
    print("Supervisor started. Press Ctrl+C or stop the cell to stop everything.")
    print(f"ROOT: {ROOT}")

    server = None
    bot = None

    # Don't duplicate an already-healthy server (e.g. started in a previous cell)
    if _server_healthy():
        print("[server] already healthy on :5003, not spawning a duplicate")
    else:
        server = _spawn(SERVER_CMD, SERVER_LOG, "server")

    while True:
        time.sleep(CHECK_INTERVAL)

        # --- server ---
        if server is not None and not _alive(server):
            print("[server] crashed, restarting...")
            server = _spawn(SERVER_CMD, SERVER_LOG, "server")
        elif server is None and not _server_healthy():
            print("[server] not healthy, starting...")
            server = _spawn(SERVER_CMD, SERVER_LOG, "server")
        elif _server_healthy() and server is not None and _alive(server):
            pass  # healthy, nothing to do

        # --- bot ---
        if bot is None or not _alive(bot):
            if bot is not None:
                print("[bot] crashed, restarting...")
            time.sleep(RESTART_DELAY)
            bot = _spawn(BOT_CMD, BOT_LOG, "bot")


if __name__ == "__main__":
    try:
        supervise()
    except KeyboardInterrupt:
        print("\nSupervisor stopped.")
        sys.exit(0)

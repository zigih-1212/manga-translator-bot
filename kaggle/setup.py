#!/usr/bin/env python3
"""
Bootstrap для бесплатного Colab/Kaggle GPU-сервера.

Сценарий:
1. Ставит зависимости.
2. Берет `GEMINI_API_KEY` из env или спрашивает его.
3. Запускает актуальный `server.py` из репозитория.
4. Проверяет `/health`.
5. Поднимает бесплатный tunnel через cloudflared.
"""

import os
import re
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SERVER_PATH = ROOT / "server.py"
REQS_PATH = ROOT / "requirements.txt"


def install_deps():
    print("[1/5] Installing packages...")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "-r", str(REQS_PATH)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for pkg in ("google-generativeai", "httpx", "cloudflared"):
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", pkg],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    print("[1/5] Done!")


def ensure_gemini_key():
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key and sys.stdin and sys.stdin.isatty():
        key = input("GEMINI_API_KEY (Enter to skip translate endpoint): ").strip()
    if key:
        os.environ["GEMINI_API_KEY"] = key
        print("[2/5] Gemini translation enabled")
    else:
        print("[2/5] GEMINI_API_KEY not set, /translate will return fallback text")


def start_server():
    print("[3/5] Starting server...")
    proc = subprocess.Popen(
        [sys.executable, str(SERVER_PATH)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    time.sleep(8)
    if proc.poll() is not None:
        out = proc.stdout.read() if proc.stdout else ""
        raise RuntimeError(f"Server failed to start:\n{out[:2000]}")
    print("[3/5] Done!")
    return proc


def wait_health(timeout_sec: int = 30):
    print("[4/5] Checking health...")
    import httpx

    deadline = time.time() + timeout_sec
    last_error = None
    while time.time() < deadline:
        try:
            r = httpx.get("http://127.0.0.1:5003/health", timeout=5)
            r.raise_for_status()
            print("[4/5] Server OK:", r.json())
            return
        except Exception as e:
            last_error = e
            time.sleep(2)
    raise RuntimeError(f"Health check failed: {last_error}")


def open_tunnel():
    print("[5/5] Creating public URL...")
    proc = subprocess.Popen(
        [sys.executable, "-m", "cloudflared", "tunnel", "--url", "http://127.0.0.1:5003"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    deadline = time.time() + 60
    url = ""
    pattern = re.compile(r"https://[^\s]+trycloudflare\.com")
    while time.time() < deadline:
        line = proc.stdout.readline() if proc.stdout else ""
        if not line:
            time.sleep(1)
            continue
        match = pattern.search(line)
        if match:
            url = match.group(0)
            break
    print("=" * 50)
    if url:
        print("YOUR SERVER URL:", url)
        print("Copy this to .env as:")
        print("COLAB_URL=" + url)
        print("REMOTE_SERVER_URL=" + url)
    else:
        print("Tunnel URL not found. Check cloudflared output above.")
    print("=" * 50)


def main():
    install_deps()
    ensure_gemini_key()
    start_server()
    wait_health()
    open_tunnel()


if __name__ == "__main__":
    main()

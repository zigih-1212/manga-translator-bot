import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SERVER_PATH = ROOT / "server.py"
REQS_PATH = ROOT / "requirements.txt"


print("[1/3] Installing packages...")
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "-r", str(REQS_PATH)],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
for pkg in ("google-generativeai", "httpx"):
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", pkg],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
print("[1/3] Done!")

print("[2/3] Starting server...")
if not os.environ.get("GEMINI_API_KEY") and sys.stdin and sys.stdin.isatty():
    key = input("GEMINI_API_KEY (Enter to skip /translate): ").strip()
    if key:
        os.environ["GEMINI_API_KEY"] = key

proc = subprocess.Popen(
    [sys.executable, str(SERVER_PATH)],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
)
time.sleep(8)

if proc.poll() is not None:
    out = proc.stdout.read() if proc.stdout else ""
    print("SERVER FAILED! Log:", out[:1500])
else:
    print("[2/3] Server started")

print("[3/3] Checking health...")
import httpx

try:
    r = httpx.get("http://127.0.0.1:5003/health", timeout=5)
    print("[3/3] OK!", r.json())
except Exception as e:
    print("[3/3] Error:", e)
    print("Check server logs above")

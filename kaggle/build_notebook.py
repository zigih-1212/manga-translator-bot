"""Regenerate kaggle/setup.ipynb from kaggle/server.py.

kaggle/server.py is the single source of truth for the Colab server code.
This script embeds its current contents into the notebook so the deployed
Colab server always matches the canonical server (and the colab_client format),
instead of a hand-copied copy that drifts out of sync.

Usage:
    python kaggle/build_notebook.py
"""

import json
from pathlib import Path

KAGGLE = Path(__file__).resolve().parent
ROOT = KAGGLE.parent

SERVER_SRC = (KAGGLE / "server.py").read_text(encoding="utf-8")

# Escape the whole server source as a Python string literal (single line).
# json.dumps handles quotes, backslashes and newlines so the outer notebook
# cell does NOT interpret any of server.py's f-strings / braces.
SERVER_LITERAL = json.dumps(SERVER_SRC)

markdown_intro = [
    "# Manga Publisher - Colab GPU Server\n",
    "\n",
    "Дополнительный GPU-ресурс: OCR (easyocr), перевод (Gemini) и инпейнт (LaMa TorchScript) на видеокарте Colab.\n",
    "Используется как второй уровень после Modal (или «страховка» по очереди).\n",
    "\n",
    "1. **Runtime -> Change runtime type -> T4 GPU**\n",
    "2. Запусти ячейку ниже (один раз)\n",
    "3. Введи `GEMINI_API_KEY` (для /translate) — можно оставить пустым, если используется только инпейнт/OCR\n",
    "4. Дождись появления публичного URL\n",
    "5. Скопируй URL в `.env` как `COLAB_URL=...` (или `REMOTE_SERVER_URL=...`)\n",
    "\n",
    "Серверный код генерируется из `kaggle/server.py` (см. `kaggle/build_notebook.py`).\n",
]

code_lines = [
    "import subprocess, sys, os, io, json, base64, asyncio, threading, time\n",
    "from pathlib import Path\n",
    "\n",
    "print('[1/5] Installing packages...')\n",
    "pkgs = ['fastapi','uvicorn[standard]','python-multipart','Pillow','numpy','opencv-python-headless',\n",
    "        'google-generativeai','easyocr','torch','torchvision']\n",
    "for p in pkgs:\n",
    "    subprocess.check_call([sys.executable,'-m','pip','install','-q',p],\n",
    "                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n",
    "print('[1/5] Done!')\n",
    "\n",
    "# Let user paste GEMINI_API_KEY (optional)\n",
    "key = os.environ.get('GEMINI_API_KEY','').strip()\n",
    "if not key:\n",
    "    key = input('GEMINI_API_KEY (Enter to skip): ').strip()\n",
    "os.environ['GEMINI_API_KEY'] = key\n",
    "\n",
    "print('[2/5] Writing server...')\n",
    "# Auto-generated from kaggle/server.py (immutable here - edit that file and rebuild)\n",
    "SERVER = " + SERVER_LITERAL + "\n",
    "with open('/content/server.py','w') as f: f.write(SERVER)\n",
    "print('[2/5] Done!')\n",
    "\n",
    "print('[3/5] Launching server...')\n",
    "def _run():\n",
    "    os.execv(sys.executable, [sys.executable, '/content/server.py'])\n",
    "threading.Thread(target=_run, daemon=True).start()\n",
    "time.sleep(6)\n",
    "\n",
    "print('[4/5] Creating public URL...')\n",
    "subprocess.check_call([sys.executable,'-m','pip','install','-q','localtunnel'],\n",
    "                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n",
    "import subprocess as sp\n",
    "lt = sp.Popen([sys.executable,'-m','localtunnel','--port','5003'],\n",
    "               stdout=sp.PIPE, stderr=sp.STDOUT, text=True)\n",
    "time.sleep(8)\n",
    "url=''\n",
    "if lt.poll() is None:\n",
    "    line = lt.stdout.readline().strip()\n",
    "    if line:\n",
    "        url = line\n",
    "    else:\n",
    "        try:\n",
    "            import requests\n",
    "            r = requests.get('http://127.0.0.1:4040/api/tunnels', timeout=5)\n",
    "            tunnels = r.json().get('tunnels', [])\n",
    "            if tunnels: url = tunnels[0]['public_url']\n",
    "        except Exception: pass\n",
    "print('')\n",
    "print('='*50)\n",
    "print('YOUR SERVER URL:', url if url else 'check output above')\n",
    "print('='*50)\n",
    "print('Copy and paste into .env as COLAB_URL='+url)\n",
    "\n",
    "print('[5/5] Setup complete')\n",
]

notebook = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "colab": {"provenance": []},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
    },
    "cells": [
        {"cell_type": "markdown", "metadata": {}, "source": markdown_intro},
        {"cell_type": "code", "metadata": {}, "source": code_lines, "execution_count": None, "outputs": []},
    ],
}

out = KAGGLE / "setup.ipynb"
out.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(f"Regenerated {out} from kaggle/server.py")
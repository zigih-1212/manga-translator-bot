"""Ops hardening: log rotation, safe config writes, temp cleanup, page cache."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import logging.handlers
import shutil
import time
from pathlib import Path

from cfg import CONFIG, CONFIG_PATH, DATA_DIR, TEMP_DIR

# ── 1. Log rotation ──────────────────────────────────────────────
# bot.log / server.log previously grew unbounded -> filled Kaggle disk.

def setup_rotating_logger(logfile: Path, level=logging.INFO) -> logging.Handler:
    logfile.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(
        logfile, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    handler.setLevel(level)
    return handler


# ── 2. Safe config writes (fixes race condition) ─────────────────
# save_config() was called from handlers + loops concurrently; concurrent
# tmp-replace could interleave and corrupt config.json.

_config_lock = asyncio.Lock()


async def asave_config():
    """Async-safe wrapper around cfg.save_config()."""
    async with _config_lock:
        _backup_config()
        await asyncio.to_thread(_save_config_sync)


def _save_config_sync():
    # import late to avoid circulars; original atomic write logic lives there
    from cfg import save_config as _orig
    _orig()


def _backup_config(keep: int = 5):
    """Rotate timestamped backups of config.json before each write."""
    if not CONFIG_PATH.exists():
        return
    bdir = Path(DATA_DIR) / "backups"
    bdir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    try:
        shutil.copy2(CONFIG_PATH, bdir / f"config-{stamp}.json")
    except Exception:
        return
    backups = sorted(bdir.glob("config-*.json"))
    for old in backups[:-keep]:
        try:
            old.unlink()
        except Exception:
            pass


# ── 3. Temp cleanup ──────────────────────────────────────────────

async def cleanup_temp(max_age_hours: float = 6.0):
    """Delete files in TEMP_DIR older than max_age_hours. Call periodically."""
    cutoff = time.time() - max_age_hours * 3600
    removed = 0
    def _clean():
        n = 0
        for p in Path(TEMP_DIR).glob("*"):
            try:
                if p.is_file() and p.stat().st_mtime < cutoff:
                    p.unlink()
                    n += 1
            except Exception:
                pass
        return n
    removed = await asyncio.to_thread(_clean)
    if removed:
        logging.getLogger("ops").info("cleanup_temp removed %d stale files", removed)
    return removed


# ── 4. Page download cache (avoid re-downloading same pages) ────

class PageCache:
    """Disk cache keyed by URL hash with TTL. Saves bandwidth & time on retries."""

    def __init__(self, root: Path | None = None, ttl_hours: float = 48.0):
        self.root = Path(root) if root else (Path(TEMP_DIR) / "pagecache")
        self.ttl = ttl_hours * 3600
        self._mem: dict[str, bytes] = {}
        self._lock = asyncio.Lock()

    def _key(self, url: str) -> str:
        return hashlib.sha1(url.encode()).hexdigest()

    def _path(self, key: str) -> Path:
        return self.root / f"{key}.bin"

    async def get(self, url: str) -> bytes | None:
        key = self._key(url)
        async with self._lock:
            if key in self._mem:
                return self._mem[key]
        path = self._path(key)
        if not path.exists():
            return None
        try:
            if time.time() - path.stat().st_mtime > self.ttl:
                path.unlink(missing_ok=True)
                return None
            data = await asyncio.to_thread(path.read_bytes)
            async with self._lock:
                self._mem[key] = data
            return data
        except Exception:
            return None

    async def put(self, url: str, data: bytes):
        key = self._key(url)
        async with self._lock:
            self._mem[key] = data
        def _write():
            self.root.mkdir(parents=True, exist_ok=True)
            tmp = self._path(key).with_suffix(".tmp")
            tmp.write_bytes(data)
            tmp.replace(self._path(key))
        await asyncio.to_thread(_write)

    async def prune(self):
        """Remove expired cache entries."""
        cutoff = time.time() - self.ttl
        def _prune():
            n = 0
            for p in self.root.glob("*.bin"):
                try:
                    if p.stat().st_mtime < cutoff:
                        p.unlink()
                        n += 1
                except Exception:
                    pass
            return n
        return await asyncio.to_thread(_prune)


PAGE_CACHE = PageCache()

#!/usr/bin/env python3
"""
Backup script: archives config, glossary, fonts, and metrics into a
dated backup folder (and optionally into Telegram if chat_id is set).

Usage:
    python backup.py            # local backup to cfg/backups/
    python backup.py --tg       # also upload the archive to Telegram chat
"""
import argparse
import io
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from cfg import DATA_DIR, TG_BOT_TOKEN, CONFIG
from translator.log import log

BACKUP_ROOT = Path(__file__).resolve().parent / "cfg" / "backups"
KEEP_LAST = 10


def _paths_to_backup() -> list[Path]:
    paths = []
    for name in ("config.json", "glossary.json", "fonts.json", "memory.json", "metrics.json", "config.db"):
        p = DATA_DIR / name
        if p.exists():
            paths.append(p)
    fonts = DATA_DIR / "fonts"
    if fonts.exists():
        paths.append(fonts)
    return paths


def make_archive() -> Path:
    BACKUP_ROOT.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    archive = BACKUP_ROOT / f"backup_{ts}.zip"
    paths = _paths_to_backup()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in paths:
            if p.is_dir():
                for f in p.rglob("*"):
                    if f.is_file():
                        zf.write(f, arcname=f.relative_to(DATA_DIR))
            else:
                zf.write(p, arcname=p.relative_to(DATA_DIR))
        manifest = {
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "files": [str(p.relative_to(DATA_DIR)) for p in paths],
        }
        zf.writestr("_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    # Rotate: keep last N backups
    backups = sorted(BACKUP_ROOT.glob("backup_*.zip"), key=lambda x: x.name, reverse=True)
    for old in backups[KEEP_LAST:]:
        old.unlink()
        log.info("Removed old backup: %s", old.name)
    log.info("Backup created: %s (%d KB)", archive.name, archive.stat().st_size // 1024)
    return archive


async def send_to_telegram(archive: Path) -> bool:
    """Upload backup archive to the configured Telegram chat via bot API."""
    chat_id = CONFIG.get("telegram", {}).get("chat_id")
    if not TG_BOT_TOKEN or not chat_id:
        log.warning("No TG_BOT_TOKEN/chat_id — skipping Telegram upload")
        return False
    import httpx
    proxy = __import__("os").environ.get("HTTP_PROXY") or __import__("os").environ.get("https_proxy")
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0), proxy=proxy) as client:
            with open(archive, "rb") as f:
                files = {"document": (archive.name, f, "application/zip")}
                r = await client.post(
                    f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendDocument",
                    data={"chat_id": chat_id, "caption": f"Backup: {archive.name}"},
                    files=files,
                )
            if r.status_code == 200:
                log.info("Backup uploaded to Telegram: %s", archive.name)
                return True
            log.warning("Telegram upload failed: %s %s", r.status_code, r.text[:200])
            return False
    except Exception as e:
        log.warning("Telegram upload error: %s", e)
        return False


def main():
    parser = argparse.ArgumentParser(description="Backup bot data")
    parser.add_argument("--tg", action="store_true", help="upload archive to Telegram")
    args = parser.parse_args()

    archive = make_archive()
    if args.tg:
        import asyncio
        asyncio.run(send_to_telegram(archive))


if __name__ == "__main__":
    main()

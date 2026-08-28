from __future__ import annotations
import asyncio
from pathlib import Path
from .checkpoint import load_json, mark_stage, is_stage_done

async def run_stage(manifest_path: Path, stage: str, fn, *args, **kwargs) -> bool:
    if is_stage_done(manifest_path, stage):
        print(f"[SKIP] {stage}")
        return False
    mark_stage(manifest_path, stage, "running")
    print(f"[START] {stage}")
    try:
        if asyncio.iscoroutinefunction(fn):
            await fn(*args, **kwargs)
        else:
            fn(*args, **kwargs)
        mark_stage(manifest_path, stage, "done")
        print(f"[DONE] {stage}")
        return True
    except Exception as e:
        mark_stage(manifest_path, stage, "failed", str(e))
        print(f"[FAILED] {stage}: {e}")
        raise

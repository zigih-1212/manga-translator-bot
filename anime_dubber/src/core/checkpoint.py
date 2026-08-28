from __future__ import annotations

import json
from pathlib import Path
from typing import Any

def save_json(path: Path | str, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def load_json(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def mark_stage(
    manifest_path: Path | str,
    stage: str,
    status: str,
    error: str | None = None,
) -> None:
    data = load_json(manifest_path)
    data.setdefault("stages", {})
    data["stages"][stage] = status
    if error:
        data.setdefault("errors", {})[stage] = error
    elif status == "done":
        data.get("errors", {}).pop(stage, None)
    # bump updated_at
    from datetime import datetime
    data["updated_at"] = datetime.now().isoformat()
    save_json(manifest_path, data)


def get_stage_status(manifest_path: Path | str, stage: str) -> str:
    data = load_json(manifest_path)
    return data.get("stages", {}).get(stage, "pending")


def is_stage_done(manifest_path: Path | str, stage: str) -> bool:
    return get_stage_status(manifest_path, stage) == "done"


def get_manifest_data(manifest_path: Path | str) -> dict:
    return load_json(manifest_path)


def update_manifest_field(manifest_path: Path | str, key: str, value: Any) -> None:
    data = load_json(manifest_path)
    data[key] = value
    save_json(manifest_path, data)
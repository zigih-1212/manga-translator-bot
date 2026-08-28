from __future__ import annotations
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

@dataclass
class Manifest:
    job_id: str
    pipeline_version: str = "0.1.0"
    status: str = "pending"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    source: dict = field(default_factory=dict)
    stages: dict = field(default_factory=dict)
    errors: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"job_id": self.job_id, "pipeline_version": self.pipeline_version, "status": self.status, "created_at": self.created_at, "updated_at": self.updated_at, "source": self.source, "stages": self.stages, "errors": self.errors}
    def save(self, path: Path) -> None:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f: json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        tmp.replace(path)
    @classmethod
    def load(cls, path: Path) -> "Manifest":
        with open(path, encoding="utf-8") as f: data = json.load(f)
        return cls(job_id=data["job_id"], pipeline_version=data.get("pipeline_version","0.1.0"), status=data.get("status","pending"), created_at=data.get("created_at", datetime.now().isoformat()), updated_at=data.get("updated_at", datetime.now().isoformat()), source=data.get("source", {}), stages=data.get("stages", {}), errors=data.get("errors", {}))

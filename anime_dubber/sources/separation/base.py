from __future__ import annotations
import abc
from pathlib import Path

class SeparationBackend:
    @property
    def name(self) -> str:
        ...

    async def separate(self, input_audio: str, output_dir: str | Path) -> tuple[str, str]:
        ...
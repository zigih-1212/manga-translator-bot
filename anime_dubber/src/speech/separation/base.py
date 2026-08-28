from __future__ import annotations
import abc
from pathlib import Path
class SeparationBackend(abc.ABC):
    @property
    @abc.abstractmethod
    def name(self) -> str: ...
    @abc.abstractmethod
    async def separate(self, input_audio: str|Path, output_dir: str|Path) -> tuple[str,str]: ...

from __future__ import annotations
from pathlib import Path

class JobPaths:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    @property
    def source(self) -> Path: return self.root / "source"
    @property
    def audio(self) -> Path: return self.root / "audio"
    @property
    def scenes(self) -> Path: return self.root / "scenes"
    @property
    def transcript(self) -> Path: return self.root / "transcript"
    @property
    def speakers(self) -> Path: return self.root / "speakers"
    @property
    def references(self) -> Path: return self.root / "references"
    @property
    def translation(self) -> Path: return self.root / "translation"
    @property
    def tts(self) -> Path: return self.root / "tts"
    @property
    def mix(self) -> Path: return self.root / "mix"
    @property
    def qc(self) -> Path: return self.root / "qc"
    @property
    def final(self) -> Path: return self.root / "final"

    def create_all(self) -> None:
        for d in (self.source, self.audio, self.scenes, self.transcript, self.speakers, self.references, self.translation, self.tts, self.mix, self.qc, self.final):
            d.mkdir(parents=True, exist_ok=True)
    def get_audio_path(self, name: str) -> Path: return self.audio / name

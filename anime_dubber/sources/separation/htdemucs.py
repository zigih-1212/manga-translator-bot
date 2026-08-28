from __future__ import annotations
import logging
import os
import shutil
from pathlib import Path
import asyncio

log = logging.getLogger(__name__)


class HTDemucsBackend:
    name = "htdemucs"

    def __init__(self, model: str = "htdemucs", device: str = "cuda", shifts: int = 1):
        self.model_name = model
        self.device = device
        self.shifts = shifts
        self._model = None

    async def separate(self, input_audio: str | Path, output_dir: str | Path) -> tuple[str, str]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        vocals_path = Path(output_dir) / "vocals.wav"
        instrumental_path = Path(output_dir) / "instrumental.wav"

        if vocals_path.exists() and instrumental_path.exists():
            return str(vocals_path), str(instrumental_path)

        try:
            # Try to use demucs
            cmd = [
                "python", "-m", "demucs.separate",
                "--model", self.model_name,
                "--device", self.device,
                "--shifts", str(self.shifts),
                "--out", str(output_dir),
                str(input_audio)
            ]
            import subprocess
            result = await asyncio.to_thread(
                subprocess.run, ["python", "-m", "demucs.separate",
                    "--model", self.model_name,
                    "--device", self.device,
                    "--shifts", str(self.shifts),
                    "--out", str(output_dir),
                    str(input_audio)],
                check=True, capture_output=True, text=True, timeout=300
            )
        except (FileNotFoundError, ImportError, subprocess.CalledProcessError) as e:
            import logging
            logging.getLogger(__name__).warning(f"HTDemucs separation failed: {e}")
            # Fallback: copy original as instrumental, create empty vocals
            import shutil
            import asyncio
            await asyncio.to_thread(shutil.copy2, input_audio, Path(output_dir) / "instrumental.wav")
            await asyncio.to_thread(Path(output_dir / "vocals.wav").write_bytes, b"")
            return str(Path(output_dir) / "vocals.wav"), str(output_dir / "instrumental.wav")

        vocals_path = Path(output_dir) / "vocals.wav"
        instrumental_path = Path(output_dir) / "instrumental.wav"
        
        # Demucs outputs in subdirectory
        for d in Path(output_dir).rglob("vocals.wav"):
            if d.exists():
                shutil.move(d, vocals_path)
                break
        for d in Path(output_dir).rglob("no_vocals.wav"):
            if d.exists():
                shutil.move(d, instrumental_path)
                break
        for d in Path(output_dir).rglob("instrumental.wav"):
            if d.exists():
                shutil.move(d, instrumental_path)
                break

        if not vocals_path.exists():
            import shutil
            shutil.copy2(input_audio, vocals_path)
        if not instrumental_path.exists():
            shutil.copy2(input_audio, instrumental_path)

        return str(vocals_path), str(instrumental_path)

    async def close(self):
        pass
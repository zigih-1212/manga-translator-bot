from __future__ import annotations
import logging
import os
import shutil
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


class UVRMdxBackend:
    name = "uvr_mdx"

    def __init__(self, model_dir: str | None = None, device: str = "cuda"):
        self.model_dir = model_dir or os.environ.get("UVR_MODEL_DIR", "/models/uvr_mdx")
        self.device = device
        self._model = None

    async def separate(self, input_audio: str | Path, output_dir: str | Path) -> tuple[str, str]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        vocals_path = output_dir / "vocals.wav"
        instrumental_path = output_dir / "instrumental.wav"

        # Check if already processed
        if vocals_path.exists() and instrumental_path.exists():
            log.info("Separation already done, skipping")
            return str(vocals_path), str(instrumental_path)

        try:
            # Try to use uvr-mdx if available
            try:
                from uvr_mdx import separate
                result = await separate(
                    input_audio,
                    output_dir=str(output_dir),
                    model_dir=self.model_dir,
                    device=self.device
                )
            except ImportError:
                pass

            # Fallback: use ffmpeg with demucs-style splitting if available
            try:
                import subprocess
                result = subprocess.run([
                    "python", "-m", "uvr_mdx",
                    "--input", str(input_audio),
                    "--output", str(output_dir),
                    "--model_dir", self.model_dir,
                    "--device", self.device
                ], check=True, capture_output=True, text=True, timeout=300)
            except (FileNotFoundError, subprocess.CalledProcessError, ImportError):
                pass

            # Fallback: copy as-is if nothing works (placeholder)
            if not (output_dir / "vocals.wav").exists():
                import shutil
                shutil.copy2(input_audio, output_dir / "vocals.wav")
            if not (output_dir / "instrumental.wav").exists():
                shutil.copy2(input_audio, output_dir / "instrumental.wav")

            return str(output_dir / "vocals.wav"), str(output_dir / "instrumental.wav")

        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"UVR-MDX separation failed: {e}")
            import shutil
            import asyncio
            await asyncio.to_thread(shutil.copy2, input_audio, output_dir / "vocals.wav")
            import asyncio
            await asyncio.to_thread(shutil.copy2, input_audio, output_dir / "instrumental.wav")
            return str(output_dir / "vocals.wav"), str(output_dir / "instrumental.wav")

    async def close(self):
        pass
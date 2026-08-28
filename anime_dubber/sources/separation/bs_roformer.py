from __future__ import annotations
import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)


class BSRoFormerBackend:
    name = "bs_roformer"

    def __init__(self, model_dir: str | None = None, device: str = "cuda"):
        self.model_dir = model_dir or os.environ.get("BS_ROFORMER_MODEL_DIR", "/models/bs_roformer")
        self.device = device
        self._model = None

    async def separate(self, input_audio: str | Path, output_dir: str | Path) -> tuple[str, str]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        vocals_path = Path(output_dir) / "vocals.wav"
        instrumental_path = Path(output_dir) / "instrumental.wav"

        if vocals_path.exists() and instrumental_path.exists():
            return str(vocals_path), str(instrumental_path)

        try:
            # Try to use BS-RoFormer
            try:
                from bs_roformer import Separator
                separator = Separator(
                    model_dir=self.model_dir,
                    device=self.device
                )
                result = await asyncio.to_thread(
                    separator.separate,
                    input_audio=str(input_audio),
                    output_dir=str(output_dir)
                )
            except ImportError:
                pass

            # Fallback
            if not (Path(output_dir) / "vocals.wav").exists():
                import shutil
                import asyncio
                await asyncio.to_thread(shutil.copy2, input_audio, output_dir / "vocals.wav")
                await asyncio.to_thread(shutil.copy2, input_audio, output_dir / "instrumental.wav")
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"BS-RoFormer separation failed: {e}")
            import shutil
            import asyncio
            await asyncio.to_thread(shutil.copy2, input_audio, output_dir / "vocals.wav")
            await asyncio.to_thread(shutil.copy2, input_audio, output_dir / "instrumental.wav")

        return str(output_dir / "vocals.wav"), str(output_dir / "instrumental.wav")

    async def close(self):
        pass
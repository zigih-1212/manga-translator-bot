from __future__ import annotations
from pathlib import Path
import shutil, logging
log=logging.getLogger(__name__)
class HTDemucsBackend:
    name="htdemucs"
    def __init__(self, model="htdemucs", device="cuda", shifts=1): self.model_name=model; self.device=device
    async def separate(self, input_audio, output_dir):
        output_dir=Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
        log.warning("HTDemucs stub")
        shutil.copy2(input_audio, output_dir/"vocals.wav") if Path(input_audio).exists() else Path(output_dir/"vocals.wav").write_bytes(b"")
        shutil.copy2(input_audio, output_dir/"instrumental.wav") if Path(input_audio).exists() else Path(output_dir/"instrumental.wav").write_bytes(b"")
        return str(output_dir/"vocals.wav"), str(output_dir/"instrumental.wav")
    async def close(self): pass

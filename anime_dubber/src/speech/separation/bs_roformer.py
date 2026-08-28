from __future__ import annotations
from pathlib import Path
import shutil, logging
log=logging.getLogger(__name__)
class BSRoFormerBackend:
    name="bs_roformer"
    def __init__(self, model_dir=None, device="cuda"): self.model_dir=model_dir
    async def separate(self, input_audio, output_dir):
        output_dir=Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
        log.warning("BS-RoFormer stub")
        shutil.copy2(input_audio, output_dir/"vocals.wav") if Path(input_audio).exists() else Path(output_dir/"vocals.wav").write_bytes(b"")
        shutil.copy2(input_audio, output_dir/"instrumental.wav") if Path(input_audio).exists() else Path(output_dir/"instrumental.wav").write_bytes(b"")
        return str(output_dir/"vocals.wav"), str(output_dir/"instrumental.wav")
    async def close(self): pass

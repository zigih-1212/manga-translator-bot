from __future__ import annotations
import logging
from pathlib import Path
log = logging.getLogger(__name__)

class CosyVoice3Backend:
    def __init__(self, model_dir: str = "models/cosyvoice3", device: str = "cuda"):
        self.model_dir = model_dir; self.device = device
    def synthesize(self, text: str, reference_audio: str | Path, reference_text: str | None, output_path: str | Path, style: str | None = None) -> str:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        # stub: write 1sec silence 24kHz mono
        try:
            import soundfile as sf, numpy as np
            sr=24000; dur=max(0.8, len(text)*0.06)
            sf.write(str(output_path), np.zeros(int(sr*dur), dtype=np.float32), sr)
        except Exception:
            Path(output_path).write_bytes(b"RIFF....WAVE")
        log.warning("CosyVoice3 stub: %s", text[:30])
        return str(output_path)

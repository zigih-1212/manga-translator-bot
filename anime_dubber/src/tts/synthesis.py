from __future__ import annotations
from pathlib import Path
from .cosyvoice3 import CosyVoice3Backend
from .fallback import FallbackTTS
import logging
log=logging.getLogger(__name__)
class TTSOrchestrator:
    def __init__(self, primary_backend: str = "cosyvoice3", model_dir: str = "models/cosyvoice3", device: str = "cuda"):
        self.primary = CosyVoice3Backend(model_dir, device); self.fallback = FallbackTTS()
    async def synthesize(self, text: str, reference_audio, reference_text, output_path, style=None) -> str:
        try: return self.primary.synthesize(text, reference_audio, reference_text, output_path, style)
        except Exception as e:
            log.warning("primary failed %s fallback", e)
            return self.fallback.synthesize(text, reference_audio, reference_text, output_path)

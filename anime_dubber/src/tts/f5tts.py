"""F5-TTS TTS backend (verify weights license before production use).
Reference: https://github.com/SWivid/F5-TTS
"""
from __future__ import annotations
import logging
from pathlib import Path
log = logging.getLogger(__name__)

class F5TTSBackend:
    """F5-TTS adapter with optional Russian checkpoint support.

    Russian checkpoint: hotstone228/F5-TTS-Russian (verify license on HF before use).
    """
    name = "f5tts"

    def __init__(
        self,
        model_dir: str = "models/f5tts",
        device: str = "cuda",
        variant: str = "base",
    ):
        self.model_dir = model_dir
        self.device = device
        self.variant = variant
        self._model = None

    def load(self) -> None:
        try:
            from f5_tts.api import F5TTS
            self._model = F5TTS(model_type=self.variant, ckpt_path=self.model_dir, device=self.device)
            log.info("F5-TTS loaded from %s", self.model_dir)
        except Exception as e:
            log.warning("F5-TTS load failed (stub mode): %s", e)
            self._model = None

    def synthesize(
        self,
        text: str,
        reference_audio: str,
        output_path: str,
        **kwargs,
    ) -> None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        if self._model is None:
            self._stub_silence(text, output_path)
            return
        try:
            self._model.infer(
                ref_file=reference_audio,
                ref_text=kwargs.get("ref_text", ""),
                gen_text=text,
                file_wav=output_path,
            )
        except Exception as e:
            log.warning("F5-TTS synth failed, stub: %s", e)
            self._stub_silence(text, output_path)

    def _stub_silence(self, text: str, output_path: str) -> None:
        import soundfile as sf, numpy as np
        sr = 24000
        dur = max(0.8, len(text) * 0.06)
        sf.write(output_path, np.zeros(int(sr * dur), dtype=np.float32), sr)

    def unload(self) -> None:
        self._model = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

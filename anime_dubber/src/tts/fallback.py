from pathlib import Path
import logging
log=logging.getLogger(__name__)
class FallbackTTS:
    def synthesize(self, text: str, reference_audio, reference_text, output_path, **kw):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"")
        return str(output_path)

from pathlib import Path
def synthesize_room_tone(reference_audio, duration: float, output_path, sample_rate=48000): Path(output_path).parent.mkdir(parents=True, exist_ok=True); Path(output_path).write_bytes(b"")
def extract_room_tone(*a, **kw): pass

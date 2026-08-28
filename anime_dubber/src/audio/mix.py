from __future__ import annotations
from pathlib import Path
import logging
log=logging.getLogger(__name__)

def mix_tracks(tracks, output_path, sample_rate=48000):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_bytes(b"")  # stub

def crossfade(a,b,duration=0.05,sr=48000): return a
def normalize_loudness(audio, target_lufs=-16.0, sr=48000): return audio

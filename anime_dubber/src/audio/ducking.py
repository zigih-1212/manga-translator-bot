"""Audio ducking: suppress original vocals while preserving BGM/SFX."""
from __future__ import annotations
import numpy as np
import logging

log = logging.getLogger(__name__)


def apply_ducking_simple(
    original: np.ndarray,
    segments: list[tuple[float, float]],
    sr: int = 48000,
    duck_db: float = -15.0,
    attack_ms: float = 50,
    release_ms: float = 200,
) -> np.ndarray:
    """Apply ducking to original audio during speech segments.

    This is a simplified version that doesn't separate vocals.
    For proper vocal separation, use Demucs (separate_vocals_demucs).

    Args:
        original: Audio array (mono or stereo)
        segments: List of (start_sec, end_sec) to duck
        sr: Sample rate
        duck_db: How much to attenuate (negative dB)
        attack_ms: Fade-in time in ms
        release_ms: Fade-out time in ms

    Returns:
        Ducked audio array
    """
    output = original.copy().astype(np.float64)
    if output.ndim == 1:
        output = output.reshape(-1, 1)

    duck_factor = 10 ** (duck_db / 20)
    attack_samples = int(attack_ms * sr / 1000)
    release_samples = int(release_ms * sr / 1000)

    for start_sec, end_sec in segments:
        start_sample = int(start_sec * sr)
        end_sample = int(end_sec * sr)
        start_sample = max(0, start_sample)
        end_sample = min(len(output), end_sample)

        if start_sample >= end_sample:
            continue

        # Create envelope
        envelope = np.ones(end_sample - start_sample)

        # Attack (fade down)
        if attack_samples > 0 and len(envelope) > attack_samples:
            envelope[:attack_samples] = np.linspace(1.0, duck_factor, attack_samples)

        # Release (fade up)
        if release_samples > 0 and len(envelope) > release_samples:
            envelope[-release_samples:] = np.linspace(duck_factor, 1.0, release_samples)

        # Sustain
        envelope[attack_samples:-release_samples] = duck_factor

        # Apply
        output[start_sample:end_sample] *= envelope[:, np.newaxis]

    return output


def separate_vocals_demucs(audio_path: str, output_dir: str) -> tuple[str, str]:
    """Separate vocals from background using Demucs.

    Returns:
        Tuple of (vocals_path, background_path)
    """
    import subprocess
    import os

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run demucs
    cmd = [
        "python", "-m", "demucs",
        "--two-stems", "vocals",
        "-o", str(output_dir),
        str(audio_path),
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=300)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        log.error(f"Demucs failed: {e}")
        return None, None

    # Find output files
    audio_name = Path(audio_path).stem
    vocals_path = output_dir / "htdemucs" / audio_name / "vocals.wav"
    background_path = output_dir / "htdemucs" / audio_name / "no_vocals.wav"

    if vocals_path.exists() and background_path.exists():
        return str(vocals_path), str(background_path)

    return None, None


from pathlib import Path

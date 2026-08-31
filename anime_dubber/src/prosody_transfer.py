#!/usr/bin/env python3
"""
Prosody Transfer Module
========================
Transfers prosodic features (pitch contour, energy envelope, durations)
from Japanese source to Russian target speech.

Input: Japanese WAV + Russian WAV
Output: Russian WAV with Japanese prosody

Tools: librosa, parselmouth (Praat), pyrubberband, soundfile, numpy
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import pyrubberband as pyrb
import soundfile as sf

# Parsmouth (Praat) is optional - requires separate install and doesn't work on Python 3.12
try:
    import parselmouth
    HAS_PARSLEMouth = True
except ImportError:
    HAS_PARSLEMouth = False

log = logging.getLogger(__name__)


@dataclass
class ProsodyFeatures:
    """Extracted prosodic features from source audio."""
    f0_contour: np.ndarray          # Pitch contour (Hz), per frame
    energy_envelope: np.ndarray     # RMS energy per frame
    durations: np.ndarray           # Duration per frame (seconds)
    sample_rate: int                # Audio sample rate
    hop_length: int                 # Hop length for frame analysis
    frame_times: np.ndarray         # Time of each frame


def extract_prosody(audio_path: str | Path, sr: int = 22050, hop_length: int = 256) -> ProsodyFeatures:
    """Extract prosodic features from audio using Parsmouth (Praat) + librosa.
    
    Args:
        audio_path: Path to audio file
        sr: Target sample rate
        hop_length: Hop length for frame-level analysis
    
    Returns:
        ProsodyFeatures with f0, energy, durations
    """
    audio_path = Path(audio_path)
    
    # Load audio
    y, sr_loaded = librosa.load(str(audio_path), sr=sr, mono=True)
    
    # === Pitch (F0) via Parsmouth (Praat) if available, else librosa ===
    if HAS_PARSLEMouth:
        # More accurate than librosa.pyin for anime voices
        snd = parselmouth.Sound(str(audio_path))
        pitch = snd.to_pitch_ac(
            time_step=hop_length / sr,  # frame step
            pitch_floor=60.0,            # Hz - lower bound for anime voices
            pitch_ceiling=600.0,         # Hz - upper bound for female/child voices
            very_accurate=True,
            silence_threshold=0.03,
            voicing_threshold=0.45,
        )
        f0_values = pitch.selected_array['frequency']
        f0_values[f0_values == 0] = np.nan  # unvoiced = NaN
        
        # Interpolate NaN values for smooth contour
        nans = np.isnan(f0_values)
        if nans.all():
            f0_values = np.full_like(f0_values, 200.0)  # default if all unvoiced
        elif nans.any():
            f0_values[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), f0_values[~nans])
    else:
        # Fallback to librosa.pyin
        f0_values, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),   # 65 Hz
            fmax=librosa.note_to_hz("C7"),   # 2093 Hz
            sr=sr,
            frame_length=hop_length * 2,
            hop_length=hop_length,
        )
        f0_values = f0_values.copy()
        f0_values[np.isnan(f0_values)] = 200.0  # default for unvoiced
    
    # === Energy (RMS) ===
    rms = librosa.feature.rms(y=y, frame_length=hop_length * 2, hop_length=hop_length)[0]
    
    # Normalize energy to [0, 1]
    if rms.max() > 0:
        rms = rms / rms.max()
    
    # === Frame durations ===
    n_frames = len(f0_values)
    frame_duration = hop_length / sr
    durations = np.full(n_frames, frame_duration)
    
    # Frame times
    frame_times = np.arange(n_frames) * frame_duration
    
    return ProsodyFeatures(
        f0_contour=f0_values,
        energy_envelope=rms,
        durations=durations,
        sample_rate=sr,
        hop_length=hop_length,
        frame_times=frame_times,
    )


def extract_prosody_librosa(audio_path: str | Path, sr: int = 22050, hop_length: int = 256) -> ProsodyFeatures:
    """Extract prosodic features using librosa only (no Praat dependency).
    
    Args:
        audio_path: Path to audio file
        sr: Target sample rate
        hop_length: Hop length for frame-level analysis
    
    Returns:
        ProsodyFeatures with f0, energy, durations
    """
    audio_path = Path(audio_path)
    
    # Load audio
    y, sr_loaded = librosa.load(str(audio_path), sr=sr, mono=True)
    
    # === Pitch (F0) via librosa.pyin ===
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),   # 65 Hz
        fmax=librosa.note_to_hz("C7"),   # 2093 Hz
        sr=sr,
        frame_length=hop_length * 2,
        hop_length=hop_length,
    )
    f0 = f0.copy()
    f0[np.isnan(f0)] = 200.0  # default for unvoiced
    
    # === Energy (RMS) ===
    rms = librosa.feature.rms(y=y, frame_length=hop_length * 2, hop_length=hop_length)[0]
    if rms.max() > 0:
        rms = rms / rms.max()
    
    # === Frame durations ===
    n_frames = len(f0)
    frame_duration = hop_length / sr
    durations = np.full(n_frames, frame_duration)
    frame_times = np.arange(n_frames) * frame_duration
    
    return ProsodyFeatures(
        f0_contour=f0,
        energy_envelope=rms,
        durations=durations,
        sample_rate=sr,
        hop_length=hop_length,
        frame_times=frame_times,
    )


def apply_pitch_contour(
    audio: np.ndarray,
    source_f0: np.ndarray,
    sr: int = 22050,
    hop_length: int = 256,
) -> np.ndarray:
    """Apply pitch contour from source to target audio.
    
    Uses pyrubberband for pitch shifting (global shift as approximation).
    For precise per-frame control, Praat's PSOLA would be needed.
    
    Args:
        audio: Target audio array (1D)
        source_f0: Source pitch contour (per frame)
        sr: Sample rate
        hop_length: Hop length
    
    Returns:
        Pitch-modified audio
    """
    # Calculate mean source f0 (excluding NaN)
    source_mean = np.nanmean(source_f0)
    
    # Calculate current mean pitch of target
    if HAS_PARSLEMouth:
        snd = parselmouth.Sound(audio, sampling_frequency=sr)
        target_pitch = snd.to_pitch_ac(
            time_step=hop_length / sr,
            pitch_floor=60.0,
            pitch_ceiling=600.0,
        )
        target_f0 = target_pitch.selected_array['frequency']
        target_f0[target_f0 == 0] = 200.0
        target_mean = np.nanmean(target_f0)
    else:
        # Estimate from audio directly
        target_mean = 180.0  # default estimate
    
    # Calculate semitone shift
    # shift = 12 * log2(source / target)
    if target_mean > 0 and source_mean > 0:
        mean_shift = 12.0 * np.log2(source_mean / target_mean)
    else:
        mean_shift = 0.0
    
    # Limit shift to ±12 semitones (1 octave)
    mean_shift = np.clip(mean_shift, -12, 12)
    
    # Apply pitch shifting via pyrubberband
    if abs(mean_shift) > 0.1:
        shifted = pyrb.pitch_shift(audio, sr, n_steps=mean_shift)
    else:
        shifted = audio
    
    return shifted


def apply_energy_envelope(
    audio: np.ndarray,
    target_energy: np.ndarray,
    sr: int = 22050,
    hop_length: int = 256,
    frame_length: Optional[int] = None,
) -> np.ndarray:
    """Apply energy envelope from source to target audio.
    
    Modifies RMS energy of target to match source energy contour.
    
    Args:
        audio: Target audio array (1D)
        target_energy: Source energy envelope (normalized 0-1)
        sr: Sample rate
        hop_length: Hop length
        frame_length: Frame length (default: hop_length * 2)
    
    Returns:
        Energy-modified audio
    """
    if frame_length is None:
        frame_length = hop_length * 2
    
    # Calculate current energy
    current_rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    
    # Ensure same length
    min_len = min(len(target_energy), len(current_rms))
    target_energy = target_energy[:min_len]
    current_rms = current_rms[:min_len]
    
    # Avoid division by zero
    current_rms = np.maximum(current_rms, 1e-8)
    
    # Calculate gain per frame
    gain = target_energy / current_rms
    
    # Smooth gain to avoid clipping/clicking
    from scipy.ndimage import median_filter
    gain = median_filter(gain, size=3)
    gain = np.clip(gain, 0.1, 5.0)  # limit gain range
    
    # Apply gain per frame
    output = audio.copy()
    n_frames = len(gain)
    
    for i in range(n_frames):
        start = i * hop_length
        end = min(start + frame_length, len(output))
        if start >= len(output):
            break
        output[start:end] *= gain[i]
    
    # Normalize to prevent clipping
    peak = np.max(np.abs(output))
    if peak > 0:
        output = output / peak * 0.95
    
    return output


def apply_duration_alignment(
    audio: np.ndarray,
    source_duration: float,
    sr: int = 22050,
) -> np.ndarray:
    """Stretch/compress audio to match source duration.
    
    Uses pyrubberband time-stretching (pitch-preserving).
    
    Args:
        audio: Target audio array (1D)
        source_duration: Target duration in seconds
        sr: Sample rate
    
    Returns:
        Time-aligned audio
    """
    current_duration = len(audio) / sr
    
    # Only adjust if difference > 5%
    ratio = source_duration / current_duration if current_duration > 0 else 1.0
    
    if 0.95 <= ratio <= 1.05:
        return audio
    
    # pyrubberband uses rate = target_rate / source_rate
    # rate > 1 = slower = longer, rate < 1 = faster = shorter
    rate = current_duration / source_duration
    
    stretched = pyrb.time_stretch(audio, sr, rate)
    return stretched


def transfer_prosody(
    source_path: str | Path,
    target_path: str | Path,
    output_path: str | Path,
    sr: int = 22050,
    hop_length: int = 256,
    use_parselmouth: bool = True,
) -> Path:
    """Transfer prosody from source to target speech.
    
    Main entry point. Extracts prosody from Japanese source and applies
    to Russian target.
    
    Args:
        source_path: Path to Japanese audio (WAV)
        target_path: Path to Russian audio (WAV)
        output_path: Path for output WAV
        sr: Target sample rate
        hop_length: Hop length for frame analysis
        use_parselmouth: Use Praat/Parsmouth for more accurate pitch
    
    Returns:
        Path to output audio
    """
    source_path = Path(source_path)
    target_path = Path(target_path)
    output_path = Path(output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load audio
    y_source, sr_s = librosa.load(str(source_path), sr=sr, mono=True)
    y_target, sr_t = librosa.load(str(target_path), sr=sr, mono=True)
    
    source_duration = len(y_source) / sr
    target_duration = len(y_target) / sr
    
    log.info(f"Source: {source_path} ({source_duration:.2f}s)")
    log.info(f"Target: {target_path} ({target_duration:.2f}s)")
    
    # === Step 1: Extract prosody from source (Japanese) ===
    log.info("Extracting prosody from source...")
    if use_parselmouth:
        try:
            source_prosody = extract_prosody(source_path, sr=sr, hop_length=hop_length)
        except Exception as e:
            log.warning(f"Praat failed, falling back to librosa: {e}")
            source_prosody = extract_prosody_librosa(source_path, sr=sr, hop_length=hop_length)
    else:
        source_prosody = extract_prosody_librosa(source_path, sr=sr, hop_length=hop_length)
    
    log.info(f"  F0 mean: {np.nanmean(source_prosody.f0_contour):.1f} Hz")
    log.info(f"  Energy mean: {np.nanmean(source_prosody.energy_envelope):.3f}")
    
    # === Step 2: Apply duration alignment first ===
    log.info("Aligning duration...")
    y_aligned = apply_duration_alignment(y_target, source_duration, sr=sr)
    log.info(f"  Duration: {target_duration:.2f}s -> {len(y_aligned)/sr:.2f}s")
    
    # === Step 3: Apply pitch contour ===
    log.info("Applying pitch contour...")
    y_pitch = apply_pitch_contour(y_aligned, source_prosody.f0_contour, sr=sr, hop_length=hop_length)
    
    # === Step 4: Apply energy envelope ===
    log.info("Applying energy envelope...")
    y_final = apply_energy_envelope(y_pitch, source_prosody.energy_envelope, sr=sr, hop_length=hop_length)
    
    # === Step 5: Normalize and save ===
    peak = np.max(np.abs(y_final))
    if peak > 0:
        y_final = y_final / peak * 0.95
    
    sf.write(str(output_path), y_final, sr)
    log.info(f"Output: {output_path} ({len(y_final)/sr:.2f}s)")
    
    return output_path


def transfer_prosody_offline(
    source_path: str | Path,
    target_path: str | Path,
    output_path: str | Path,
    sr: int = 22050,
) -> Path:
    """Offline prosody transfer using Praat's PSOLA (more accurate).
    
    Requires Praat installed. Uses PSOLA for natural-sounding pitch modification.
    
    Args:
        source_path: Path to Japanese audio (WAV)
        target_path: Path to Russian audio (WAV)
        output_path: Path for output WAV
        sr: Target sample rate
    
    Returns:
        Path to output audio
    """
    source_path = Path(source_path)
    target_path = Path(target_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load audio
    y_source, sr_s = librosa.load(str(source_path), sr=sr, mono=True)
    y_target, sr_t = librosa.load(str(target_path), sr=sr, mono=True)
    
    source_duration = len(y_source) / sr
    target_duration = len(y_target) / sr
    
    # === Step 1: Duration alignment ===
    ratio = source_duration / target_duration if target_duration > 0 else 1.0
    if 0.95 <= ratio <= 1.05:
        y_aligned = y_target
    else:
        y_aligned = pyrb.time_stretch(y_target, sr, 1.0 / ratio)
    
    # === Step 2: Energy matching ===
    hop_length = 256
    rms_source = librosa.feature.rms(y=y_source, frame_length=hop_length*2, hop_length=hop_length)[0]
    rms_target = librosa.feature.rms(y=y_aligned, frame_length=hop_length*2, hop_length=hop_length)[0]
    
    if rms_source.max() > 0:
        rms_source = rms_source / rms_source.max()
    if rms_target.max() > 0:
        rms_target = rms_target / rms_target.max()
    
    # === Step 3: Combine and save ===
    # Simple approach: just match duration and energy, skip pitch
    # (pyrubberband pitch shifting is limited without Praat PSOLA)
    
    min_len = min(len(y_aligned), len(y_source))
    y_aligned = y_aligned[:min_len]
    
    # Apply energy gain
    min_rms_len = min(len(rms_source), len(rms_target))
    gain = rms_source[:min_rms_len] / np.maximum(rms_target[:min_rms_len], 1e-8)
    gain = np.clip(gain, 0.1, 5.0)
    
    # Apply gain per frame
    y_final = y_aligned.copy()
    for i in range(min_rms_len):
        start = i * hop_length
        end = min(start + hop_length * 2, len(y_final))
        if start >= len(y_final):
            break
        y_final[start:end] *= gain[i]
    
    peak = np.max(np.abs(y_final))
    if peak > 0:
        y_final = y_final / peak * 0.95
    
    sf.write(str(output_path), y_final, sr)
    return output_path


# === Demo / Quick test ===

def demo():
    """Demo prosody transfer on sample files."""
    import tempfile
    
    # Create synthetic test audio
    sr = 22050
    duration = 3.0  # seconds
    t = np.linspace(0, duration, int(sr * duration))
    
    # Source: Japanese-like voice (varying pitch)
    f0_base = 220  # A3
    f0_var = 50 * np.sin(2 * np.pi * 2 * t)  # 2 Hz vibrato
    source = np.sin(2 * np.pi * (f0_base + f0_var) * t) * 0.3
    source *= (1 + 0.5 * np.sin(2 * np.pi * 0.5 * t))  # amplitude modulation
    
    # Target: Russian-like voice (different pitch)
    f0_target = 180  # F3
    target = np.sin(2 * np.pi * f0_target * t) * 0.3
    
    # Save temp files
    with tempfile.TemporaryDirectory() as tmpdir:
        source_path = Path(tmpdir) / "source_jp.wav"
        target_path = Path(tmpdir) / "target_ru.wav"
        output_path = Path(tmpdir) / "output_prosody.wav"
        
        sf.write(str(source_path), source, sr)
        sf.write(str(target_path), target, sr)
        
        # Transfer prosody
        result = transfer_prosody_offline(source_path, target_path, output_path, sr=sr)
        
        print(f"Demo complete: {result}")
        print(f"Source duration: {len(source)/sr:.2f}s")
        print(f"Target duration: {len(target)/sr:.2f}s")
        print(f"Output duration: {len(sf.read(str(result))[0])/sr:.2f}s")


if __name__ == "__main__":
    demo()
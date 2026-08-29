# COMPARISON_MATRIX.md — Open-Source Anime Dubbing Projects

| Project | License | Status | RU | Cloning | Emotion | Diarization | Separation | Timing | Cache | Resume | Kaggle | VRAM |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **voice-pro** (abus-aikorea) | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED |
| **OmniVoice Studio** (Eirias) | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED |
| **video-dubbing-system** (sun-kic) | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED |
| **ZastTranslate** (zast57) | UNVERIFIED | ACTIVE (2026-08-25) | YES | YES | UNVERIFIED | UNVERIFIED | YES (Demucs) | YES (global gap-aware) | YES (per-segment) | UNVERIFIED | needs-work (Windows-only) | 4 GB min, 16 GB rec |
| **WeeaBlind** (FlorianEagox) | UNVERIFIED | STALE (2024-11-21) | NO | YES (Coqui) | NO | YES (pyannote 3.2.0) | YES (Spleeter) | YES (audiotsm) | partial (wav output cache) | NO | needs-work (wxPython GUI) | UNVERIFIED |
| **TachiDUBB** (tachikomared) | MIT (verified) | ACTIVE (2026-05-20) | YES (built-in RU glossaries) | YES (VoxCPM2 + F5-TTS) | partial (VoxCPM2 picks up emotion from ref) | YES (pyannote 3.1) | YES (Demucs htdemucs_ft) | YES (atempo + gap-aware) | YES (per-stage checkpoints) | YES (multi-stage checkpoints) | UNVERIFIED (no Dockerfile) | 8 GB min, 12 GB rec |
| **Violin** (shang-zhu) | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED |
| **LangSwap** (langswap-app) | AGPL (UNVERIFIED confirmed by user) | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED |

## Final Recommendation (based on confirmed audits only)

| Role | Pick | License | Reason |
|---|---|---|---|
| **Base architecture** | **TachiDUBB** | MIT (verified at master/LICENSE) | Only MIT, ACTIVE, mature, multi-stage checkpoints + per-segment regen; can borrow assembler.py + time_sync.py |
| **TTS PRIMARY** | **CosyVoice3** | Apache-2.0 code (verify weights) | Most documented, multi-lingual, cross-lingual cloning, active repo |
| **TTS SECONDARY** | **Qwen3-TTS** | Apache-2.0 (verify weights) | Alibaba's recent TTS, good Russian support |
| **TTS EXPERIMENTAL** | **F5-TTS-Russian** (hotstone228) | UNVERIFIED — verify before use | Russian checkpoint availability |
| **SKIP** | WeeaBlind, ZastTranslate, Violin, LangSwap | Various | Stale / Windows-only / AGPL / UNVERIFIED |
| **WeeaBlind specific issue** | hardcoded HF token in diarize.py | UNVERIFIED | Remove before any reuse/fork |

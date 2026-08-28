# BENCHMARK_REPORT.md — TTS Benchmark Results

## Setup

- Date: 2026-08-28
- GPU: not available (CPU stub mode)
- Reference: 5 sec silence placeholder (replace before real benchmark)
- Prompts: 10 (neutral, happy, sad, angry, shout, whisper, laugh, gasp, cry)
- Mode: stub (all 4 backends fell back to silence because dependencies not installed in CPU env)
- Reference benchmark output: `benchmark/last_run.log`, `benchmark/results/benchmark_results.json`

## Auto-Metrics (CPU stub run, NOT real quality numbers)

| Backend   | Success | Fail | RTF (mean) | VRAM peak (GB) | Status |
|-----------|---------|------|------------|----------------|--------|
| cosyvoice3 | 10/10  | 0    | 0.054      | 0.0            | stub   |
| omnivoice  | 10/10  | 0    | 0.074      | 0.0            | stub   |
| qwen3      | 10/10  | 0    | 0.051      | 0.0            | stub   |
| f5tts      | 10/10  | 0    | 0.064      | 0.0            | stub   |

RTF computed on stub output (silence) — NOT real numbers. All four models produced identical 0.5s silence files.

## Open-Source Project Audit Summary

| Project | License | Status | Best For | License Risk |
|---|---|---|---|---|
| **WeeaBlind** (FlorianEagox) | UNVERIFIED (no LICENSE file) | STALE (last commit 2024-11-21) | Anime-specific wxPython GUI with Coqui TTS | hardcoded HF token in `diarize.py` must be removed before any reuse |
| **TachiDUBB** (tachikomared) | **MIT (verified at master/LICENSE)** | ACTIVE (last commit 2026-05-20) | Reference implementation: end-to-end pipeline with `pipeline/assembler.py` (time-align + atempo + loudness), `modules/time_sync.py` (global gap-aware timing), `step6_regenerate_segment` (per-line regen) | safe to read and adapt algorithms; cannot reuse TachikomaRed/smolemaru brand names |
| **ZastTranslate** (zast57) | UNVERIFIED (only README claim) | ACTIVE (last commit 2026-08-25) | `modules/time_sync.py` global gap-aware timing, `step6_regenerate_segment` per-line regen without full rebuild | README explicitly says "Tested on Windows only", no Dockerfile, bitsandbytes pinned (no macOS) |

## Final Recommendation

**Base project: TachiDUBB** (MIT, ACTIVE, mature, GPU-supported)
- borrow `pipeline/assembler.py` for time-align + atempo stretch + loudness norm
- borrow `modules/time_sync.py` for global gap-aware timing
- borrow `modules/reformulator.py` for batched LLM translate+fit pattern
- borrow per-segment regeneration pattern (`step6_regenerate_segment`)

**Reusable modules (architectural reference, AGPL-free):**
- TachiDUBB `modules/transcriber.py` (WhisperX wrapper)
- TachiDUBB `modules/diarizer.py` (pyannote + reference mining)
- TachiDUBB `pipeline/assembler.py` (audio reconstruction)

**TTS candidates to test on Kaggle T4:**
- **PRIMARY:** CosyVoice3 0.5B (Apache-2.0 code; verify weights license on HF)
- **SECONDARY:** Qwen3-TTS 0.6B/1.7B
- **EXPERIMENTAL:** F5-TTS-Russian (hotstone228 — verify license before use)

**Skip:**
- WeeaBlind (STALE 9 months, no LICENSE, hardcoded HF token)
- ZastTranslate (Windows-only, single contributor)
- LangSwap (AGPL)
- Violin (UNVERIFIED)

**Build order:**
1. Run TTS benchmark with real models + real `reference.wav` on Kaggle T4
2. Pick winner by human score on `cry_01` / `shout_01` / `whisper_01`
3. Borrow TachiDUBB `modules/transcriber.py` + `modules/diarizer.py` patterns
4. Integrate into our existing `anime_dubber/` checkpoint pipeline
5. Use our existing `local_ducking` (src/audio/ducking.py)
6. Single-segment regeneration via our `run_stage` checkpoint system

## STOP CRITERIA Results (real bench needed)

| Criterion | cosyvoice3 | omnivoice | qwen3 | f5tts |
|---|---|---|---|---|
| Russian naturalness >= acceptable | NOT TESTED | NOT TESTED | NOT TESTED | NOT TESTED |
| Speaker similarity >= acceptable | NOT TESTED | NOT TESTED | NOT TESTED | NOT TESTED |
| Common speech >= acceptable | NOT TESTED | NOT TESTED | NOT TESTED | NOT TESTED |

If after real benchmark no model meets these criteria, STOP and explore other TTS options.

## Notes on this run

- Ran locally on CPU with no GPU. All 4 backends produced stub output (silence) because dependencies (`cosyvoice`, `transformers`, `f5_tts`, `omnivoice`) are not installed in this env.
- Reference.wav is a 5 sec silence placeholder. Real voice needed before any human scoring.
- Real benchmark must run on Kaggle T4 with weights downloaded to `/kaggle/input/anime-dubber-models/` via the `bootstrap.sh` workflow.

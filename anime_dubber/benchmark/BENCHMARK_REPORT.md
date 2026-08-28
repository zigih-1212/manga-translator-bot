# BENCHMARK_REPORT.md — TTS Benchmark Results

## Setup

- **Date:** 2026-08-28
- **GPU:** not available (CPU stub mode)
- **Reference:** `benchmark/reference.wav` — synthetic 5 sec placeholder (formant tones); replace with real voice before real benchmark
- **Prompts:** 10 (neutral_01, neutral_02, happy_01, sad_01, angry_01, shout_01, whisper_01, laugh_01, gasp_01, cry_01)
- **Backends tested:** 4 (cosyvoice3, omnivoice, qwen3, f5tts)
- **Mode:** stub (all 4 backends fell back to silence because model dependencies not installed in CPU env)

## Auto-Metrics Summary (CPU stub run, NOT real quality)

| Backend    | Success | Fail | RTF (mean) | VRAM peak (GB) | Status      | Note                                |
|------------|---------|------|------------|----------------|-------------|-------------------------------------|
| cosyvoice3 | 10/10   | 0    | null       | 0.0            | stub        | `cosyvoice` not installed           |
| omnivoice  | 10/10   | 0    | null       | 0.0            | stub        | `omnivoice` not installed           |
| qwen3      | 10/10   | 0    | null       | 0.0            | stub        | `transformers` not installed        |
| f5tts      | 10/10   | 0    | null       | 0.0            | stub        | `f5_tts` not installed             |

RTF is `null` because `total_generation_seconds=0.0` (stub returns instantly) and `avg_rtf` is computed only if `total_generation > 0`. All output wav files are stub silence of duration proportional to text length (~1.4 sec for short lines, 2.0 for long). Cache check: if `output_path` exists, status is `cached` and no re-generation occurs.

## Open-Source Project Audit Summary

3 projects audited in parallel via `task()` agents, with strict requirement to use UNVERIFIED if not confirmable from code:

| Project | License | Status | Best For | Risk |
|---|---|---|---|---|
| **WeeaBlind** (FlorianEagox) | UNVERIFIED (no LICENSE file in repo) | STALE (last commit 2024-11-21) | anime-specific wxPython GUI with Coqui TTS pipeline | hardcoded HF token in `diarize.py` must be removed before any reuse; no LICENSE file |
| **TachiDUBB** (tachikomared) | **MIT (verified at master/LICENSE)** | ACTIVE (last commit 2026-05-20) | reference implementation for full pipeline: `pipeline/assembler.py` (time-align + atempo + loudness), `modules/time_sync.py` (global gap-aware timing), `step6_regenerate_segment` (per-line regen without full rebuild) | safe to read & adapt algorithms; cannot reuse TachikomaRed/smolemaru brand names; README says "Tested on Windows only" |
| **ZastTranslate** (zast57) | UNVERIFIED (only README claim) | ACTIVE (last commit 2026-08-25) | `modules/time_sync.py` global gap-aware timing; `step6_regenerate_segment` per-line regen | README explicitly says "Tested on Windows only", no Dockerfile, bitsandbytes pinned (no macOS) |

4 projects NOT audited in this run (voice-pro, OmniVoice Studio, video-dubbing-system, Violin, LangSwap) — keep as low-priority for next round.

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
- **SKIP for now:** OmniVoice (license UNVERIFIED)

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
- Reference.wav is a 5 sec synthetic placeholder (3 formants 180/800/2400Hz). Real voice needed before any human scoring.
- Real benchmark must run on Kaggle T4 with weights downloaded to `/kaggle/input/anime-dubber-models/` via the `bootstrap.sh` workflow.
- HF download of `hotstone228/F5-TTS-Russian` failed (timeout); weights license UNVERIFIED in any case.
- Cache check works: if wav already exists (from previous run), pipeline skips re-generation. To force re-run, use `--force` flag.

## What to do next

1. `pip install cosyvoice transformers f5-tts qwen-tts` on Kaggle T4 GPU.
2. Replace `benchmark/reference.wav` with 3-10 sec of real voice (your own recording, public-domain, or licensed).
3. `python benchmark/run_benchmark.py --backends cosyvoice3,qwen3,f5tts --force`.
4. Listen to each `benchmark/results/<backend>/<id>.wav`.
5. Fill `HUMAN_SCORE.md` with 0-4 ratings per category per model.
6. Update STOP CRITERIA table in this file.
7. Re-run with `--style` flag once on `shout_01` / `cry_01` to see emotion control.
8. If quality >= acceptable for 2+ models, pick primary for production. Otherwise stop and explore alternatives.

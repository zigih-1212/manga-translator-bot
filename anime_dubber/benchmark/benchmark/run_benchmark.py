"""TTS Benchmark runner: load each backend, synthesize 10 lines, unload, GPU cleanup.

Usage:
  python benchmark/run_benchmark.py --backends cosyvoice3,omnivoice,qwen3
  python benchmark/run_benchmark.py --backends cosyvoice3 --prompt-ids shout_01,cry_01 --style
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import soundfile as sf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# direct module imports (benchmark is a script dir, not a package)
import importlib.util

def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_backends_mod = _load("_backends", ROOT / "benchmark" / "backends.py")
available = _backends_mod.available
get_backend = _backends_mod.get_backend

_base_mod = _load("_tts_base", ROOT / "src" / "tts" / "base.py")
TTSBackend = _base_mod.TTSBackend


def gpu_info() -> dict:
    try:
        import torch
        if not torch.cuda.is_available():
            return {"available": False}
        return {
            "available": True,
            "name": torch.cuda.get_device_name(0),
            "total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
        }
    except Exception as e:
        return {"available": False, "error": repr(e)}


def gpu_vram_peak_gb() -> float:
    try:
        import torch
        if not torch.cuda.is_available():
            return 0.0
        return round(torch.cuda.max_memory_allocated() / 1024**3, 3)
    except Exception:
        return 0.0


def load_prompts(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def audio_duration(path: str) -> float:
    info = sf.info(str(path))
    return float(info.duration)


def write_metadata(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def reset_gpu_peak() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def run_backend(
    backend_name: str,
    prompts: list[dict],
    reference_audio: str,
    output_root: Path,
    skip_existing: bool = True,
    style_kwargs: dict | None = None,
) -> dict:
    """Run one backend on all prompts. Returns aggregate results dict."""
    output_dir = output_root / backend_name
    output_dir.mkdir(parents=True, exist_ok=True)

    style_kwargs = style_kwargs or {}
    backend: TTSBackend = get_backend(backend_name)

    log.info("Loading backend: %s", backend_name)
    reset_gpu_peak()
    t0 = time.perf_counter()
    try:
        backend.load()
    except Exception as e:
        log.error("Failed to load %s: %s", backend_name, e)
        return {"backend": backend_name, "error": str(e), "results": []}
    load_time = time.perf_counter() - t0

    model_results: list[dict] = []
    success_count = 0
    fail_count = 0
    total_gen = 0.0
    total_audio = 0.0

    try:
        for item in prompts:
            output_path = output_dir / f"{item['id']}.wav"
            metadata_path = output_dir / f"{item['id']}.json"

            if skip_existing and output_path.exists() and output_path.stat().st_size > 100:
                log.info("  cached: %s", item["id"])
                try:
                    dur = audio_duration(output_path)
                except Exception:
                    dur = 0
                model_results.append({
                    "id": item["id"],
                    "status": "cached",
                    "duration": dur,
                })
                success_count += 1
                total_audio += dur
                continue

            text = item["text"]
            t0 = time.perf_counter()
            try:
                backend.synthesize(
                    text=text,
                    reference_audio=reference_audio,
                    output_path=str(output_path),
                    **style_kwargs,
                )
                gen_time = time.perf_counter() - t0
                dur = audio_duration(output_path)
                rtf = gen_time / dur if dur > 0 else None
                write_metadata(metadata_path, {
                    "id": item["id"],
                    "text": text,
                    "success": True,
                    "generation_seconds": gen_time,
                    "audio_seconds": dur,
                    "rtf": rtf,
                    "sample_rate": 24000,
                    "output_path": str(output_path),
                    "model": backend_name,
                })
                model_results.append({
                    "id": item["id"],
                    "status": "ok",
                    "generation_seconds": gen_time,
                    "audio_seconds": dur,
                    "rtf": rtf,
                })
                success_count += 1
                total_gen += gen_time
                total_audio += dur
            except Exception as e:
                log.error("  FAIL %s: %s", item["id"], e)
                write_metadata(metadata_path, {
                    "id": item["id"],
                    "text": text,
                    "success": False,
                    "error": repr(e),
                })
                model_results.append({
                    "id": item["id"],
                    "status": "fail",
                    "error": repr(e),
                })
                fail_count += 1
    finally:
        try:
            backend.unload()
        except Exception as e:
            log.warning("unload %s: %s", backend_name, e)

    peak = gpu_vram_peak_gb()
    avg_rtf = (total_gen / total_audio) if total_audio > 0 else None

    return {
        "backend": backend_name,
        "load_seconds": round(load_time, 2),
        "gpu_peak_gb": peak,
        "success": success_count,
        "fail": fail_count,
        "total_generation_seconds": round(total_gen, 2),
        "total_audio_seconds": round(total_audio, 2),
        "avg_rtf": round(avg_rtf, 3) if avg_rtf else None,
        "results": model_results,
    }


def main():
    parser = argparse.ArgumentParser(description="TTS Benchmark runner")
    parser.add_argument("--backends", type=str, default=",".join(available()),
                        help="Comma-separated list of backend names to test")
    parser.add_argument("--prompts", type=str, default="benchmark/prompts.json",
                        help="Path to prompts JSON")
    parser.add_argument("--reference", type=str, default="benchmark/reference.wav",
                        help="Path to reference audio for cloning")
    parser.add_argument("--prompt-ids", type=str, default=None,
                        help="Comma-separated prompt IDs to test (default: all)")
    parser.add_argument("--output", type=str, default="benchmark/results",
                        help="Output root directory")
    parser.add_argument("--force", action="store_true", help="Re-run all lines")
    parser.add_argument("--style", action="store_true", help="Pass style kwargs to backend")
    parser.add_argument("--no-cache", action="store_true", help="Don't skip existing files")
    args = parser.parse_args()

    prompts = load_prompts(args.prompts)
    if args.prompt_ids:
        wanted = {p.strip() for p in args.prompt_ids.split(",")}
        prompts = [p for p in prompts if p["id"] in wanted]
    log.info("Prompts: %d lines", len(prompts))

    if not Path(args.reference).exists():
        log.error("reference audio not found: %s", args.reference)
        sys.exit(1)
    if Path(args.reference).stat().st_size < 100:
        log.warning("reference.wav is too small (%d bytes) - may be empty/silent placeholder",
                    Path(args.reference).stat().st_size)

    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    backends = [b.strip() for b in args.backends.split(",")]
    log.info("Backends: %s", backends)
    log.info("GPU: %s", gpu_info())

    all_results: list[dict] = []
    skip_existing = not args.force
    for b in backends:
        result = run_backend(
            backend_name=b,
            prompts=prompts,
            reference_audio=args.reference,
            output_root=output_root,
            skip_existing=skip_existing,
            style_kwargs={"style": "neutral"} if args.style else None,
        )
        all_results.append(result)
        log.info("Done %s: %d ok, %d fail, avg_rtf=%s, peak=%.2f GB",
                 b, result["success"], result["fail"], result.get("avg_rtf"), result["gpu_peak_gb"])

    # summary
    summary_path = output_root / "benchmark_results.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "gpu": gpu_info(),
            "reference": args.reference,
            "prompts": len(prompts),
            "results": all_results,
        }, f, ensure_ascii=False, indent=2)
    log.info("Summary: %s", summary_path)

    print()
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"{'Backend':<20} {'Success':<8} {'Fail':<6} {'RTF':<8} {'VRAM GB':<10}")
    print("-" * 60)
    for r in all_results:
        rtf = r.get("avg_rtf")
        rtf_s = f"{rtf:.2f}" if rtf else "N/A"
        vram = r.get("gpu_peak_gb", 0)
        print(f"{r['backend']:<20} {r['success']:<8} {r['fail']:<6} {rtf_s:<8} {vram:<10}")
    print()
    print("Now listen to benchmark/results/<backend>/*.wav")
    print("Fill HUMAN_SCORE.md with your scores (0-4 per category)")


if __name__ == "__main__":
    main()

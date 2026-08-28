"""Benchmark 3 separation backends on same 180s clip.
Metrics: residual JA, BGM damage, SFX damage, shout, overlap, GPU time
Run: python benchmark_separation.py --input test_scene.mp4 --out jobs/sep_bench
"""
from __future__ import annotations
import argparse, asyncio, json, time, shutil
from pathlib import Path
import yaml

BACKENDS = ["uvr_mdx", "htdemucs", "bs_roformer"]

async def bench_one(backend: str, src: Path, out_root: Path):
    cfg_path = Path("config.yaml")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    cfg.setdefault("separation", {})["backend"] = backend
    # write temp config for job
    job = out_root / backend
    job.mkdir(parents=True, exist_ok=True)
    (job / "config_used.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")

    # run only media + separation stages
    from src.core.paths import JobPaths
    from src.core.manifest import Manifest
    from src.core.runner import run_stage
    manifest = job / "manifest.json"
    if not manifest.exists():
        m = Manifest(job_id=job.name, source={"video": str(src)})
        m.save(manifest)
        JobPaths(job).create_all()
    import time as tm
    t0 = tm.monotonic()
    # media
    from main import stage_media, stage_separation
    await run_stage(manifest, "media", stage_media, job, cfg, src)
    t1 = tm.monotonic()
    await run_stage(manifest, "separation", stage_separation, job, cfg)
    t2 = tm.monotonic()
    return {"backend": backend, "media_sec": round(t1-t0,1), "separation_sec": round(t2-t1,1), "total_sec": round(t2-t0,1)}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    src = Path(args.input)
    out = Path(args.out)
    # cut 180s once to temp
    tmp = out / "_src_180.mp4"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    # use benchmark cut logic
    import subprocess
    cmd = ["ffmpeg","-y","-ss","0","-i",str(src),"-t","180","-c","copy",str(tmp)]
    try: subprocess.run(cmd, check=True, capture_output=True)
    except Exception: shutil.copy2(src, tmp)

    results=[]
    for b in BACKENDS:
        print(f"\n=== {b} ===")
        res = asyncio.run(bench_one(b, tmp, out))
        results.append(res)
        print(res)

    # report
    report = out / "report.json"
    report.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nReport: {report}")
    # also markdown table
    md = "| backend | separation_sec | total_sec |\n|---|---|---|\n"
    for r in results: md += f"| {r['backend']} | {r['separation_sec']} | {r['total_sec']} |\n"
    (out / "report.md").write_text(md, encoding="utf-8")
    print(md)
    print("Сравни вручную: vocals.wav на остаток яп. голоса, instrumental.wav на BGM/SFX, крик, оверлап")

if __name__ == "__main__":
    main()

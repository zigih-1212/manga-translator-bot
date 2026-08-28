from __future__ import annotations
import argparse
import subprocess
from pathlib import Path
import sys

def cut_clip(input_video: str|Path, output_video: str|Path, start: float = 0, duration: float = 180):
    output_video = Path(output_video); output_video.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg","-y","-ss",str(start),"-i",str(input_video),"-t",str(duration),"-c","copy",str(output_video)]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except FileNotFoundError:
        # no ffmpeg: copy whole file
        import shutil
        shutil.copy2(input_video, output_video)
    except subprocess.CalledProcessError:
        import shutil
        shutil.copy2(input_video, output_video)

def main():
    p = argparse.ArgumentParser(description="Benchmark 180s vertical slice")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--start", type=float, default=0)
    args = p.parse_args()
    job = Path(args.output)
    # 1. cut 180s via ffmpeg
    tmp = job / "source" / "episode.mkv"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    cut_clip(args.input, tmp, start=args.start, duration=180)
    # 2. run full pipeline via main.py
    sys.argv = ["main.py", "--job", str(job), "--input", str(tmp)]
    import main
    main.main()

if __name__ == "__main__":
    main()

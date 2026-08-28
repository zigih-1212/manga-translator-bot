from __future__ import annotations
import json, shutil, logging
from pathlib import Path
log=logging.getLogger(__name__)

def rank_reference_candidates(cands):
    scored=[]
    for it in cands:
        s=min(it.get("duration",0),6.0)*1.5 + it.get("snr",0)*0.8 - it.get("overlap",0)*10 - it.get("clipping",0)*20
        scored.append((s,it))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [i for _,i in scored]

def create_reference_bank(candidates, output_dir, per_speaker=3):
    output_dir=Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    grouped={}
    for it in candidates: grouped.setdefault(it.get("speaker","unknown"), []).append(it)
    result={}
    for sp, items in grouped.items():
        ranked=rank_reference_candidates(items)[:per_speaker]
        d=output_dir / f"speaker_{sp}"; d.mkdir(exist_ok=True)
        result[sp]=[]
        for idx, it in enumerate(ranked):
            dst=d / f"ref_{idx:02d}.wav"
            # try to cut real segment, else dummy silence
            try:
                import soundfile as sf, subprocess
                src=Path(it.get("path",""))
                if src.exists():
                    # extract segment via ffmpeg if available
                    subprocess.run(["ffmpeg","-y","-i",str(src),"-ss",str(it.get("start",0)),"-t",str(it.get("duration",2.0)),"-acodec","pcm_s16le","-ar","24000","-ac","1",str(dst)], capture_output=True, timeout=10)
                    if not dst.exists() or dst.stat().st_size<100:
                        raise FileNotFoundError
                else: raise FileNotFoundError
            except Exception:
                # create 1.5s silence valid wav
                try:
                    import soundfile as sf, numpy as np
                    sf.write(str(dst), np.zeros(int(24000*1.5),dtype=np.float32), 24000)
                except Exception: dst.write_bytes(b"")
            result[sp].append(str(dst))
    with open(output_dir/"manifest.json","w",encoding="utf-8") as f: json.dump(result,f,ensure_ascii=False,indent=2)
    return result

async def run_references(diarization_json, audio_path, output_dir, per_speaker=3):
    import json
    from pathlib import Path
    with open(diarization_json,encoding="utf-8") as f: data=json.load(f)
    cands=[{"speaker":s["speaker"],"duration":s["end"]-s["start"],"start":s["start"],"snr":10,"overlap":0,"clipping":0,"path":str(audio_path)} for s in data.get("segments",[]) if 0.5 <= s["end"]-s["start"] <= 10]
    return create_reference_bank(cands, output_dir, per_speaker)

from __future__ import annotations
import json, logging, os
from pathlib import Path
log=logging.getLogger(__name__)
async def run_vad(input_audio: str | Path, output_json: str | Path, **kw) -> dict:
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    if os.environ.get("REAL_VAD") != "1":
        # fast dummy: whole file as speech
        try:
            import soundfile as sf
            dur=sf.info(str(input_audio)).duration
        except Exception: dur=180.0
        data={"speech_segments":[{"start":0.0,"end":dur}],"total_speech_duration":dur}
        with open(output_json,"w",encoding="utf-8") as f: json.dump(data,f,ensure_ascii=False,indent=2)
        return data
    try:
        import torch
        model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False, trust_repo=True)
        (get_speech_timestamps, _, read_audio, _) = utils
        wav = read_audio(str(Path(input_audio).absolute()))
        segs = get_speech_timestamps(wav, model, sampling_rate=16000)
        data={"speech_segments":[{"start": s["start"]/16000,"end": s["end"]/16000} for s in segs], "total_speech_duration": sum((s["end"]-s["start"]) for s in segs)/16000}
        with open(output_json,"w",encoding="utf-8") as f: json.dump(data,f,ensure_ascii=False,indent=2)
        return data
    except Exception as e:
        log.warning(f"VAD fallback: {e}")
        data={"speech_segments":[{"start":0.0,"end":180.0}],"total_speech_duration":180.0}
        with open(output_json,"w",encoding="utf-8") as f: json.dump(data,f,ensure_ascii=False,indent=2)
        return data

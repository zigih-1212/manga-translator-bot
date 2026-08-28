#!/usr/bin/env bash
set -e
echo "=== SYSTEM ==="
nvidia-smi || true
python --version
ffmpeg -version | head -n1 || echo "ffmpeg not found"

# Kaggle dataset mount
MODEL_ROOT="/kaggle/input/anime-dubber-models"
if [ -d "$MODEL_ROOT" ]; then
  echo "=== Using cached models from $MODEL_ROOT ==="
  export UVR_MODEL_DIR="$MODEL_ROOT/uvr_mdx"
  export BS_ROFORMER_MODEL_DIR="$MODEL_ROOT/bs_roformer"
  export HF_HOME="$MODEL_ROOT/hf_cache"
  mkdir -p models
  ln -sf "$MODEL_ROOT/uvr_mdx" models/uvr_mdx 2>/dev/null || true
  ln -sf "$MODEL_ROOT/cosyvoice3" models/cosyvoice3 2>/dev/null || true
else
  echo "=== No dataset mount, models will be downloaded ==="
fi

echo "=== pip ==="
pip install -q -U pip
# use wheel cache if present
pip install -q torch torchaudio torchcodec numpy scipy soundfile librosa pydub faster-whisper whisperx pyannote.audio transformers sentencepiece accelerate huggingface_hub pyyaml tqdm rich opencv-python scikit-learn ffmpeg-python pyloudnorm || \
pip install -q -r requirements.txt

if [ ! -d "CosyVoice" ]; then
  echo "=== CosyVoice ==="
  git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git || true
  pip install -q -r CosyVoice/requirements.txt || true
fi

echo "=== DONE ==="
echo "Run: python benchmark.py --input /kaggle/input/test_scene.mp4 --output jobs/bench_001"
echo "Or compare separation: python benchmark_separation.py --input /kaggle/input/test_scene.mp4 --out jobs/sep_bench"

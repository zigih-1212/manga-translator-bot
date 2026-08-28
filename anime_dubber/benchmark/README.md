# TTS Benchmark — anime_dubber

## Запуск

```bash
# Stub mode (CPU, быстрый sanity check):
python benchmark/run_benchmark.py

# Полный бенчмарк с реальными моделями на Kaggle T4:
python benchmark/run_benchmark.py --backends cosyvoice3,omnivoice,qwen3,f5tts --force
```

## Что внутри

| Файл | Что делает |
|---|---|
| `prompts.json` | 10 реплик: neutral/happy/sad/angry/shout/whisper/laugh/gasp/cry |
| `reference.wav` | 5 сек placeholder (синтетика: 3 форманта 180/800/2400 Hz). **Замени на реальный голос перед бенчмарком!** |
| `HUMAN_SCORE.md` | Шаблон ручной оценки 0-4 по 10 категориям |
| `backends.py` | Registry 4 бэкендов: cosyvoice3, omnivoice, qwen3, f5tts |
| `run_benchmark.py` | load→synthesize→unload, RTF, VRAM peak, cache |
| `results/<backend>/<id>.wav + .json` | Audio + метаданные каждой генерации |
| `BENCHMARK_REPORT.md` | Полный отчёт с аудитом open-source проектов |
| `COMPARISON_MATRIX.md` | Матрица 8 open-source dubbing проектов |

## Результат локального теста (2026-08-28)

| Backend   | Success | Fail | Status |
|-----------|---------|------|--------|
| cosyvoice3 | 10/10  | 0    | stub (без cosyvoice) |
| omnivoice  | 10/10  | 0    | stub (без omnivoice) |
| qwen3      | 10/10  | 0    | stub (без transformers) |
| f5tts      | 10/10  | 0    | stub (без f5_tts) |

CPU-only окружение. На Kaggle T4 с `bootstrap.sh` (cosyvoice, f5-tts, transformers, qwen-tts) будет реальный benchmark.

## Структура `job/` (manifest + checkpoint)

После прогона — на каждом этапе `status.json`:
- `media/`, `asr/`, `diar/`, `references/`, `translation/`, `tts/`, `mix/`, `qc/`
- `manifest.json` со `stages: {media: done, asr: done, ...}`
- `--force` перезапускает всё; без --force пропускает done-этапы

## Что нужно сделать

1. `pip install cosyvoice transformers f5-tts qwen-tts` (на Kaggle T4)
2. Заменить `reference.wav` на 3-10 сек реального голоса
3. Запустить `python benchmark/run_benchmark.py --backends cosyvoice3,qwen3,f5tts --force`
4. Послушать каждый wav в `results/<backend>/` и заполнить `HUMAN_SCORE.md`
5. Обновить `BENCHMARK_REPORT.md` STOP CRITERIA на реальные оценки
6. Если хотя бы 1 модель показывает `>= acceptable` — двигаемся к Phase 2 (TachiDUBB интеграция)
7. Если ни одна — STOP и ищем другие TTS

## Реальный бенчмарк (Kaggle T4)

```bash
# В Kaggle ноутбуке:
%cd /kaggle/working/manga-translator-bot/anime_dubber
!pip install -q cosyvoice qwen-tts f5-tts transformers accelerate

# Reference.wav уже 5 сек placeholder - замените на свой голос
!ls /kaggle/input/

!python benchmark/run_benchmark.py --backends cosyvoice3,omnivoice,qwen3,f5tts --force
# затем послушать results/<backend>/<id>.wav
```

## F5-TTS с русским чекпоинтом

```python
from src.tts.f5tts import F5TTSBackend
backend = F5TTSBackend(
    model_dir="models/f5tts/checkpoint",  # hotstone228/F5-TTS-Russian weights
    device="cuda",
    variant="base",
)
backend.load()
backend.synthesize("Привет!", "reference.wav", "out.wav")
```

Перед production — проверить лицензию `hotstone228/F5-TTS-Russian` на HF. Если CC-BY-NC — только для личных бенчмарков.

## Конфигурация бэкенда

```yaml
# config.yaml
separation:
  backend: uvr_mdx  # htdemucs | bs_roformer
asr:
  model: large-v3-turbo
tts:
  backend: cosyvoice3  # cosyvoice3 | omnivoice | qwen3 | f5tts
```

Переменные окружения:
- `MANGA_PROXY` — прокси для доступа к источникам манги
- `HF_TOKEN` — для pyannote
- `KAGGLE_USERNAME/KAGGLE_KEY` — для kagglehub

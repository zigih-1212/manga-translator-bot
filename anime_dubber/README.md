# anime_dubber — checkpointed vertical slice (180s benchmark)

Структура утвержденной архитектуры Phase 0-5 с локальным ducking и pluggable separation (uvr_mdx/htdemucs/bs_roformer).

## Запуск benchmark (180с вертикальный срез)
```bash
# 1. Kaggle: подключи Dataset anime-dubber-models в /kaggle/input/anime-dubber-models
bash bootstrap.sh

python benchmark.py --input test_scene.mp4 --output jobs/benchmark_001
# упал на TTS 43% -> повтор продолжит:
python main.py --job jobs/benchmark_001 --input jobs/benchmark_001/source/scene.mp4
```
Каждый этап пишет `manifest.json:stages`. `done` не перегенерируется.

## Сравнение separation бэкендов (обязательно перед выбором)
```bash
python benchmark_separation.py --input test_scene.mp4 --out jobs/sep_bench
# результаты: jobs/sep_bench/report.json + report.md
# сравни вручную: остаток яп. голоса / BGM / SFX / крик / оверлап / GPU sec
cat jobs/sep_bench/report.md
```
Выбери победителя и зафиксируй в `config.yaml` `separation.backend`.

## Конфиг
`config.yaml`:
```yaml
separation:
  backend: uvr_mdx # htdemucs | bs_roformer
```

## Тесты качества (A-E + дубль персонажа)
Срез из 20 реплик: 4 обычные, 4 эмоции, 3 шепот, 3 крик, 2 смех, 2 gasp, 1 плач, 1 оверлап + тест консистентности (scene_03 char A == scene_27 char A, episode 2 char A)
См. `jobs/benchmark_001/qc/report.json` + слепой A/B/C (WATCH/NOTICEABLE/STOP).

## Принцип локального ducking
Не `amix` (наложение). `speech_mask -> duck 9dB (attack 10ms) -> background-preserved -> + TTS -> loudness -16 LUFS`.

## Принцип
DO NOT BUILD WHOLE DUBBER FIRST — build measurable benchmark. Every stage consume files, produce files, write done, be resumable.

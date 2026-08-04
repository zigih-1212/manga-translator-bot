# Manga Translator Bot

Автоматический Telegram-бот для перевода манги на русский язык. Источники: MangaDex и Naver Webtoon. Бесплатно работает на Railway (CPU) с опциональным GPU-инпейнтом через Modal.

## Возможности

- Загрузка глав с MangaDex и Naver Webtoon (единый роутер источников)
- OCR текста (ocr.space через Kaggle/Colab + локальный Manga-OCR ONNX для японского)
- Перевод через мульти-провайдерную цепочку:
  - Remote server (Kaggle/Colab, основной, rate-limited)
  - Groq (Llama 70B)
  - Gemini 2.0 Flash
  - OpenRouter (free/paid)
  - deep-translator (fallback)
- Контекст истории, глоссарий персонажей/терминов, RAG-память прошлых глав
- SFX-детектор (звукоподражания сохраняются в оригинале)
- Умная маска бабблов + инпейнт (Modal GPU или LaMa CPU)
- Пост-обработка инпейнта (выравнивание яркости по краю баббла)
- Продвинутый рендер: кириллические шрифты, вертикальный японский текст, warp-наклон по полигону OCR
- Пост-валидатор переводов + авто-исправление
- JSON-логирование, health-сервер `/health` `/metrics`, Telegram-алерты
- Web Translation Editor — правка переведённого текста в браузере с перерисовкой страниц
- Web Dashboard — метрики, очередь, активные задачи и ошибки в браузере
- Webhooks — HTTP-уведомления о готовности/ошибках глав
- Graceful shutdown, авто-ретраи

## Фазы разработки

| Фаза | Статус | Содержание |
|------|--------|------------|
| 0 | ✅ | Стабильность: JSON-логи, health-сервер, TG-алерты, graceful shutdown |
| 1 | ✅ | OCR: manga-ocr ONNX, SFX-детектор, deskew/deshake/sauvola, batch OCR, confidence+auto-retry |
| 2 | ✅ | Качество перевода: RAG-память, glossary pre-replace, self-correction |
| 3 | ✅ | Инпейнт v2: smart mask, Modal mask passthrough, postprocess feather |
| 4 | ✅ | Рендер: вертикальный JP, warp-текст, 9 кириллических TTF |
| 5 | ✅ | Пост-валидатор, метрики JSON/TG, авто-ретрай страницы |
| 6 | 🔄 | Прод: Docker, CI, README, backup |

## Быстрый старт

### Локально

```bash
pip install -r requirements.txt
# настроить cfg/.env и cfg/config.json
python main.py
```

### Переменные окружения (`cfg/.env` или `.env`)

```
TG_BOT_TOKEN=...
TG_API_ID=...
TG_API_HASH=...
GROQ_API_KEY=...        # необязательно
GEMINI_API_KEY=...      # необязательно
OPENROUTER_API_KEY=...  # необязательно
REMOTE_SERVER_URL=...   # URL Kaggle/Colab-сервера OCR/перевода
DATA_DIR=/app/data/config  # Railway Volume для конфигов
HEALTH_PORT=8080
EDITOR_PORT=8090        # запускает Web Translation Editor при старте бота
DASHBOARD_PORT=8091     # запускает Web Dashboard при старте бота
WEBHOOK_URLS=           # через запятую URL для HTTP-уведомлений о готовности глав
METRICS_REPORT_INTERVAL=3600
```

### Конфигурация (`cfg/config.json`)

- `telegram.chat_id` — ID чата для алертов/метрик
- `llm.provider` — основной провайдер перевода
- `fonts` — шрифты для диалогов/SFX/нарратива
- `glossary.json` — персонажи и термины

### Docker

```bash
docker build -t manga-translator-bot .
docker run -d --env-file .env -p 8080:8080 manga-translator-bot
```

### Railway

1. Задеплойте репозиторий (push в `main` пересобирает).
2. Добавьте Volume в `/app/data/config` и `DATA_DIR=/app/data/config`.
3. Укажите переменные окружения из таблицы выше.

### Modal (GPU инпейнт)

```bash
cd modal_inpaint
modal deploy app.py
```

Бот автоматически использует Modal для инпейнта, если он доступен, иначе падает на LaMa (CPU).

### Web Translation Editor

Позволяет править переведённый текст пузырей прямо в браузере и перерисовывать страницы.

Авто-запуск вместе с ботом (требуется `EDITOR_PORT` в окружении), либо вручную:

```bash
python -m editor.server
# открыть http://localhost:8090/
```

Данные редактора сохраняются во `temp/editor/{manga_id}/{chapter}/` в момент перевода главы:
`NNN.src.png` — оригинал, `NNN.out.png` — результат, `NNN.json` — пузыри (bbox/текст/шрифт/угол).

### Web Dashboard

Показывает статус бота: uptime, метрики (главы/страницы/LLM/OCR), очередь переводов,
активные задачи и последние ошибки. Авто-обновление каждые 5 секунд.

Авто-запуск вместе с ботом (требуется `DASHBOARD_PORT` в окружении), либо вручную:

```bash
python -m dashboard.server
# открыть http://localhost:8091/
```

### Webhooks

При завершении или ошибке перевода главы бот отправляет HTTP POST с JSON на каждый
URL из `cfg/config.json` (`webhooks.urls`) или из переменной `WEBHOOK_URLS` (через запятую).
Формат:

```json
{
  "event": "chapter_done",
  "manga": "Название",
  "chapter": "7",
  "zip_url": "",
  "status": "done"
}
```

Для ошибок — `event: "chapter_failed"` и поле `error`. Доставка с ретраями (3 попытки).

## CI

GitHub Actions (`.github/workflows/ci.yml`) прогоняет компиляцию, импорт-проверку, sanity-тесты и сборку Docker-образа на каждый push в `main`.

## Лицензия

Код — MIT. Шрифты и модели — согласно их собственным лицензиям.

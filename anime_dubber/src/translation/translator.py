from __future__ import annotations
import json
import logging
import re
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


# JP -> RU glossary for common anime terms
GLOSSARY = {
    "お前": "Ты",
    "俺": "Я",
    "私": "Я",
    "僕": "Я",
    "彼女": "Она",
    "他": "Он",
    "彼女": "Она",
    "先生": "Учитель",
    "先輩": "Сенпай",
    "後輩": "Кохай",
    "馬鹿": "Идиот",
    "バカ": "Идиот",
    "ありがとう": "Спасибо",
    "すみません": "Извините",
    "ごめんなさい": "Прости",
    "おはよう": "Доброе утро",
    "こんにちは": "Привет",
    "こんばんは": "Добрый вечер",
    "さようなら": "До свидания",
    "はい": "Да",
    "いいえ": "Нет",
    "大丈夫": "В порядке",
    "大変": "Тяжело / Серьезно",
    "待って": "Подожди",
    "行く": "Иду",
    "来る": "Прихожу",
    "分かる": "Понимаю",
    "分からない": "Не понимаю",
    "知らない": "Не знаю",
    "好き": "Нравится",
    "好きです": "Нравится",
    "大好き": "Очень нравится / Люблю",
    "嫌い": "Не нравится",
    "大嫌い": "Ненавижу",
    "戦う": "Буду сражаться",
    "守る": "Буду защищать",
    "約束": "Обещание",
    "約束する": "Обещаю",
    "絶対": "Абсолютно / Обязательно",
    "絶対に": "Обязательно",
    "絶対に負けない": "Никогда не проиграю",
    "守る": "Защищу",
    "信じる": "Верю",
    "信じて": "Поверь",
    "信じられない": "Не могу поверить",
    "嘘": "Ложь",
    "嘘だ": "Это ложь",
    "本当": "Правда",
    "本当だ": "Это правда",
    "嘘じゃない": "Не лгу",
    "約束は守る": "Сдержу обещание",
    "必ず": "Обязательно",
    "絶対に守る": "Обязательно защищу",
    "守ってあげる": "Защищу",
    "助ける": "Помогу / Спасу",
    "助けて": "Помоги",
    "待って": "Подожди",
    "待ってて": "Подожди меня",
    "行って": "Иди",
    "行くよ": "Пойду",
    "来て": "Иди",
    "来ない": "Не приди",
    "来ないで": "Не иди",
    "止まれ": "Стой",
    "止めて": "Прекрати",
    "やめろ": "Прекрати",
    "やめろよ": "Прекрати уже",
    "やめて": "Прекрати",
    "やめてよ": "Прекрати уже",
    "ダメ": "Нельзя",
    "ダメだ": "Нельзя",
    "ダメだよ": "Так нельзя",
    "いけない": "Нельзя",
    "いけないよ": "Так нельзя",
    "良くない": "Нехорошо",
    "良くないよ": "Это нехорошо",
    "悪い": "Плохо / Моя вина",
    "悪いよ": "Плохо / Извини",
    "悪いな": "Моя вина",
    "ごめんね": "Извини",
    "ごめんなさい": "Извини",
    "ごめん": "Извини",
    "ごめんよ": "Извини",
    "申し訳ない": "Мне жаль / Извините",
    "申し訳ありません": "Извините",
    "許して": "Прости",
    "許してよ": "Прости меня",
    "許してよ": "Прости",
    "許してよ": "Прости",
    "許せ": "Прости",
    "許せない": "Не могу простить",
    "許さない": "Не прощу",
    "許さない": "Не прощу",
    "許さん": "Не прощу",
    "許せん": "Не прощу",
}

def translate_ja_ru(text: str) -> str:
    """Basic JA->RU translation with glossary."""
    for ja, ru in sorted(GLOSSARY.items(), key=lambda x: -len(x[0])):
        if ja in text:
            text = text.replace(ja, ru)
    return text + " [RU]"


def adapt_translation(ja_text: str, ru_text: str, max_chars: int = None) -> str:
    """Adapt translation for lip-sync: shorten if needed, keep meaning."""
    if max_chars and len(ru_text) > max_chars:
        # Try to compress - remove polite forms, shorten
        ru = ru_text
        ru = ru.replace("です", "").replace("ます", "")
        ru = ru.replace("です。", "。").replace("ます。", "。")
        if len(ru) <= max_chars:
            return ru
        # If still too long, truncate
        return ru[:max_chars]
    return ru_text


def format_timing_constraint(
    source_start: float,
    source_end: float,
    target_duration: float,
) -> dict:
    source_duration = source_end - source_start
    if source_duration <= 0:
        return {"ratio": 1.0, "status": "error"}
    ratio = target_duration / source_duration
    return {
        "ratio": ratio,
        "needs_compression": ratio > 1.15,
        "needs_expansion": ratio < 0.85,
    }


if __name__ == "__main__":
    ja = "お前、本当に来たのか？"
    ru = translate_ja_ru(ja)
    print(f"JA: {ja}")
    print(f"RU: {ru}")
    print(f"Adapted: {adapt_translation(ja, ru, 20)}")
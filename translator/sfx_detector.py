import re

# --- Text-based SFX heuristics ---------------------------------------------

# Katakana-only SFX often consists solely of katakana (no particles/verbs).
_KATAKANA_RE = re.compile(r'^[ァ-ヶー・゛゜ﾟﾞ\s]{1,12}$')
# Hiragana onomatopoeia without grammatical particles.
_HIRAGANA_ONO = re.compile(r'^(?:きゃ|きゅ|きょ|ぎゃ|ぎゅ|ぎょ|しゃ|しゅ|しょ|じゃ|じゅ|じょ|ちゃ|ちゅ|ちょ|ぱ|ぴ|ぷ|ぺ|ぽ|ば|び|ぶ|べ|ぼ|だ|で|ど|が|ぎ|ぐ|げ|ご|ざ|ず|ぜ|ぞ|つ|っ|ん|あ|い|う|え|お|か|き|く|け|こ|さ|し|す|せ|そ|た|ち|て|と|な|に|ぬ|ね|の|は|ひ|ふ|へ|ほ|ま|み|む|め|も|や|ゆ|よ|ら|り|る|れ|ろ|わ|を){1,6}$')
# Katakana character repetition like ドドド, バキバキ, グオオオ
_KATAKANA_REPEAT = re.compile(r'^[ァ-ヶ]{1,4}(ー)?([ァ-ヶ]{1,3})(\1)+\1?$')
# Latin SFX: all-caps short "words" that are not real English words.
_LATIN_SFX = re.compile(r'^[A-Z]{2,12}[!！]?$')
# Korean onomatopoeia patterns (repeated syllables).
_KOREAN_ONO = re.compile(r'^(?:쾅|쿵|탕|펑|팡|윙|휙|삐|삐|빠|빵|뿅|꽝|퍽|우르|부르|콰|드르|두두|뚜두|와|헉|하하|히히|호호|으악|으윽|헐|컥|아하|어이|오오|우우|아아|에에|뿌|삐뽀|짝짝|쾅쾅|쿵쾅|드르르){1,6}$')

_KNOWN_LATIN_SFX = {
    "BOOM", "BANG", "CRASH", "WHOOSH", "WHAM", "SLAM", "SPLAT", "THUD", "CLANG",
    "ZAP", "POW", "SMASH", "BLAM", "CRACK", "POP", "SNAP", "ROAR", "RUMBLE",
    "VROOM", "SCREECH", "BEEP", "TICK", "TOCK", "DING", "DONG", "CLINK", "PLOP",
    "GULP", "HMM", "HAHA", "HEHE", "AHEM", "COUGH", "SNIFF", "SHUSH", "SHHH",
    "GASP", "SIGH", "MOAN", "GROAN", "STOMP", "STEP", "THUMP", "BUMP", "CLICK",
    "KISS", "MUAH", "PECK", "CRUNCH", "MUNCH", "CHOMP", "SLURP", "SIP", "SHLIK",
    "SWISH", "WOOSH", "WHIRR", "HUM", "BUZZ", "FIZZ", "SPLASH", "DRIP", "TAP",
    "KNOCK", "RING", "TING", "DING", "CHIME", "BOING", "SPROING", "TWANG",
}

_SFX_CHARS = re.compile(r'[〜ー・∞☆★♪♫♬=≡≠≠≪≫!！?？…]')

# Katakana/hiragana particles that indicate a sentence, not SFX.
_PARTICLES = set("はがをにでとやものかよねのへからよりてもしばあぞずぜこそつ")


def is_sfx_text(text: str, source_lang: str = "ja") -> bool:
    """Heuristic SFX detection from OCR text alone."""
    if not text:
        return False
    t = text.strip()
    if len(t) > 14:
        return False

    if t.upper() in _KNOWN_LATIN_SFX:
        return True

    # Latin all-caps that OCR picked up as a real word should not be SFX if too long
    if _LATIN_SFX.match(t):
        if len(t) >= 3 and t.lower() in {"the", "and", "you", "why", "yes", "no", "hey", "oh", "ok", "it", "is", "are"}:
            return False
        return True

    if _SFX_CHARS.search(t):
        return True

    if source_lang == "ja":
        if _KATAKANA_RE.match(t):
            # Exclude katakana sentences: real katakana words that are nouns
            # (e.g., コンビニ, エレベーター) — these are usually too long or in bubbles.
            if len(t) >= 4 and _HIRAGANA_ONO.match(t):
                return True
            if _KATAKANA_REPEAT.match(t) or _has_repetition(t):
                return True
            # Very short katakana only = almost always SFX
            if len(t) <= 2:
                return True
        if _HIRAGANA_ONO.match(t):
            return True
    elif source_lang == "ko":
        if _KOREAN_ONO.match(t):
            return True
        if _has_repetition(t) and len(t) <= 8:
            return True
    elif source_lang == "en":
        return t.upper() in _KNOWN_LATIN_SFX or bool(_LATIN_SFX.match(t))

    return False

def _has_repetition(text: str) -> bool:
    """True if a char/syllable repeats 3+ times or in pairs (バキバキ, ドドド)."""
    if not text:
        return False
    for ch in set(text):
        if ch in "ー・":
            continue
        if text.count(ch) >= 3:
            return True
    # pair repetition: ABAB
    for i in range(len(text) - 3):
        if text[i:i + 2] == text[i + 2:i + 4]:
            return True
    return False


def annotate_sfx(ocr_texts: list[dict], source_lang: str = "ja") -> list[dict]:
    """Mark OCR regions as SFX in-place; returns the same list."""
    for r in ocr_texts:
        text = r.get("text", "")
        if is_sfx_text(text, source_lang):
            r["sfx"] = True
        else:
            r.setdefault("sfx", False)
    return ocr_texts

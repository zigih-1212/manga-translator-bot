import re


_KO_RE = re.compile(r'[\uac00-\ud7af\u1100-\u11ff\u3130-\u318f]')
_JA_RE = re.compile(r'[\u3040-\u30ff\u4e00-\u9fff]')
_CJK_LIKE_RE = re.compile(r'[\uac00-\ud7af\u1100-\u11ff\u3130-\u318f\u3040-\u30ff\u4e00-\u9fff\u3400-\u4dbf]')

# Russian vowels/consonants ratio sanity range for natural Cyrillic text
_RU_VOWELS = set('аеёиоуыэюяАЕЁИОУЫЭЮЯ')
_RU_CONSONANTS = set('бвгджзклмнпрстфхцчшщБВГДЖЗКЛМНПРСТФХЦЧШЩ')
_RU_MARK = set('ьъЬЪ')


def _contains_untranslated(text: str) -> bool:
    """True if text still has Korean/Japanese/Chinese characters."""
    return bool(_CJK_LIKE_RE.search(text))


def _cyrillic_ratio(text: str) -> float:
    if not text:
        return 0.0
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for ch in letters if _RU_VOWELS.union(_RU_CONSONANTS).union(_RU_MARK).__contains__(ch)) / len(letters)


def _looks_garbled(text: str) -> bool:
    """Detect repeated-letter / keyboard-layout garbage (e.g. 'ччч', 'ghbdtn' mixes)."""
    if not text:
        return False
    # Repeating same letter 4+ times in a row
    if re.search(r'(.)\1{3,}', text):
        return True
    # Mixed scripts in a word where cyrillic ratio is low and CJK-like present
    return False


def validate_translation(source_text: str, translated: str, source_lang: str = "ko") -> dict:
    """
    Validate a translated string. Returns dict:
      {ok: bool, issues: list[str], fixed: str|None}
    """
    issues = []
    fixed = None

    if not translated or not translated.strip():
        issues.append("empty")
    else:
        t = translated.strip()

        # Untranslated source script remaining
        if _contains_untranslated(t):
            issues.append("untranslated-script")

        # Source-lang specific checks
        if source_lang == "ko" and _KO_RE.search(t) and _cyrillic_ratio(t) < 0.5:
            issues.append("korean-left")

        # For Russian target, if the text is supposed to be Cyrillic but looks garbled
        if _cyrillic_ratio(t) > 0.4 and _looks_garbled(t):
            issues.append("garbled")

        # Hallucinated placeholder braces like [X] or {X}
        if re.search(r'\[[^\]]{1,12}\]', t):
            issues.append("placeholder-brackets")

    return {"ok": len(issues) == 0, "issues": issues, "fixed": fixed}


def fix_translation(source_text: str, translated: str, source_lang: str = "ko") -> str | None:
    """
    Attempt automatic fixes; returns cleaned string or None if unfixable.
    """
    if not translated:
        return None
    t = translated.strip()
    if not t:
        return None

    changed = False
    # Strip surrounding brackets that look like accidental placeholders
    new_t = re.sub(r'^\[([^\]]{1,12})\]$', r'\1', t.strip())
    if new_t != t:
        t, changed = new_t, True

    # Collapse excessive repeated whitespace
    new_t = re.sub(r'\s{2,}', ' ', t)
    if new_t != t:
        t, changed = new_t, True

    return t if changed else None

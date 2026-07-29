from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import json
import re
import os
import platform
from config import FONTS, FONTS_PATH


FONT_EXTENSIONS = (".ttf", ".otf")

FALLBACK_FONTS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    "arial.ttf",
    "Arial.ttf",
]

if platform.system() == "Windows":
    _CYRILLIC_FALLBACK = os.path.join(os.environ.get("SystemRoot", "C:\\Windows"), "Fonts", "arial.ttf")
else:
    _CYRILLIC_FALLBACK = next((p for p in FALLBACK_FONTS if os.path.exists(p)), "DejaVuSans.ttf")


def _scan_all_fonts() -> list[dict]:
    """Scan fonts/ directory recursively, return list of {path, name, keywords}."""
    fonts_dir = Path(__file__).resolve().parent.parent / "fonts"
    results = []
    for f in fonts_dir.rglob("*"):
        if f.suffix.lower() in FONT_EXTENSIONS:
            name = f.stem.lower()
            keywords = re.split(r"[\s_\-\.]+", name)
            results.append({
                "path": str(f.relative_to(fonts_dir.parent)),
                "name": f.stem,
                "keywords": keywords,
            })
    return results


FONT_CATEGORIES = {
    "dialogue": [
        "anime", "manga", "comic", "sans", "clean", "simple", "regular",
        "picsue", "modern", "midstar", "outlinerz", "bigbelow", "secline",
        "electro", "hustle", "gocake", "bitrank", "colinoosh", "miracle",
        "candy", "peach", "rainy", "salmon", "original", "classic", "say_hello",
        "sugar", "wait", "wister",
    ],
    "sfx": [
        "bold", "heavy", "black", "impact", "action", "beat", "warfare",
        "gorilla", "tarmiles", "bring", "super", "jungle", "hullgaria",
        "laughter", "maron", "world", "bandit", "stone", "vengeance",
        "stroke", "shizuoka", "cyberpunk",
    ],
    "narration": [
        "light", "thin", "script", "handwrit", "italic", "serif", "byliner",
        "darling", "laugh", "mooligat", "mitshuka", "privilege",
    ],
}


class TextRenderer:
    def __init__(self):
        with open(FONTS_PATH, "r", encoding="utf-8") as f:
            self.font_config = json.load(f)
        self.fonts_dir = Path(__file__).resolve().parent.parent / "fonts"
        self._scanned_fonts = _scan_all_fonts()
        self._font_cache: dict[str, str] = {}

    def _categorize_fonts(self) -> dict[str, list[str]]:
        categorized = {"dialogue": [], "sfx": [], "narration": [], "other": []}
        for font_info in self._scanned_fonts:
            keywords = " ".join(font_info["keywords"])
            placed = False
            for category, patterns in FONT_CATEGORIES.items():
                for pattern in patterns:
                    if pattern in keywords:
                        categorized[category].append(font_info["path"])
                        placed = True
                        break
                if placed:
                    break
            if not placed:
                categorized["other"].append(font_info["path"])
        return categorized

    @staticmethod
    def _contains_cyrillic(text: str) -> bool:
        return bool(re.search(r'[А-Яа-яЁё]', text))

    @staticmethod
    def _supports_cyrillic(font_path: str) -> bool:
        try:
            font = ImageFont.truetype(font_path, 24)
            sizes = set()
            for cp in ["\u0410", "\u0416", "\u042f", "\u0428"]:
                mask = font.getmask(cp)
                sizes.add(mask.size)
            return len(sizes) >= 2
        except Exception:
            return False

    def _get_font_path(self, font_type: str = "dialogue", text: str = "") -> str:
        needs_cyrillic = text and self._contains_cyrillic(text)
        cache_key = f"cyr_{font_type}" if needs_cyrillic else font_type

        if cache_key in self._font_cache:
            cached = self.fonts_dir.parent / self._font_cache[cache_key]
            if cached.exists():
                return str(cached)

        cfg = self.font_config.get(font_type, {})
        default = cfg.get("default", "")
        if default:
            path = self.fonts_dir.parent / default
            if path.exists():
                result = str(path)
                if needs_cyrillic and not self._supports_cyrillic(result):
                    return self._get_cyrillic_fallback()
                self._font_cache[cache_key] = default
                return result

        categorized = self._categorize_fonts()
        candidates = categorized.get(font_type, []) + categorized.get("other", [])
        if candidates:
            path = str(self.fonts_dir.parent / candidates[0])
            if needs_cyrillic and not self._supports_cyrillic(path):
                return self._get_cyrillic_fallback()
            self._font_cache[cache_key] = candidates[0]
            return path

        return _CYRILLIC_FALLBACK

    def _get_cyrillic_fallback(self) -> str:
        bundled = self.fonts_dir / "DejaVuSans.ttf"
        if bundled.exists():
            return str(bundled)
        for font_info in self._scanned_fonts:
            path = str(self.fonts_dir.parent / font_info["path"])
            if self._supports_cyrillic(path):
                return path
        if os.path.exists(_CYRILLIC_FALLBACK):
            return _CYRILLIC_FALLBACK
        return str(bundled) if bundled.name else ""

    def _fit_text(
        self,
        draw: ImageDraw.Draw,
        text: str,
        max_width: int,
        max_height: int,
        font_path: str,
        min_size: int = 10,
        max_size: int = 36,
    ) -> ImageFont.FreeTypeFont:
        for size in range(max_size, min_size - 1, -1):
            try:
                font = ImageFont.truetype(font_path, size)
            except Exception:
                font = ImageFont.load_default()
            words = text.split()
            lines = []
            cur = ""
            for w in words:
                test = f"{cur} {w}".strip()
                if font.getbbox(test)[2] - font.getbbox(test)[0] <= max_width:
                    cur = test
                else:
                    if cur:
                        lines.append(cur)
                    cur = w
            if cur:
                lines.append(cur)
            lh = font.getbbox("Аg")[3] - font.getbbox("Аg")[1] + 2
            total_h = lh * len(lines)
            if total_h <= max_height:
                return font
        try:
            return ImageFont.truetype(font_path, min_size)
        except Exception:
            return ImageFont.load_default()

    def _wrap_text(self, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
        words = text.split()
        lines = []
        current = ""
        for word in words:
            test = f"{current} {word}".strip()
            bbox = font.getbbox(test)
            if bbox[2] - bbox[0] <= max_width:
                current = test
            else:
                if current:
                    lines.append(current)
                wb = font.getbbox(word)
                if wb[2] - wb[0] > max_width:
                    for ch in word:
                        test2 = current + ch if current else ch
                        b2 = font.getbbox(test2)
                        if b2[2] - b2[0] <= max_width:
                            current = test2
                        else:
                            if current:
                                lines.append(current)
                            current = ch
                else:
                    current = word
        if current:
            lines.append(current)
        return lines

    def render_bubble_text(
        self,
        img: Image.Image,
        bbox: tuple[int, int, int, int],
        text: str,
        font_type: str = "dialogue",
        font_color: str = "white",
        outline_color: str = "black",
        outline_width: int = 2,
    ) -> Image.Image:
        img = img.copy()
        draw = ImageDraw.Draw(img)
        x1, y1, x2, y2 = bbox
        bw = x2 - x1
        bh = y2 - y1
        pad = 10
        font_path = self._get_font_path(font_type, text)
        font = self._fit_text(draw, text, bw - pad * 2, bh - pad * 2, font_path)
        lines = self._wrap_text(text, font, bw - pad * 2)
        line_height = font.getbbox("Аg")[3] - font.getbbox("Аg")[1] + 2
        total_height = line_height * len(lines)
        start_y = y1 + (bh - total_height) // 2
        for i, line in enumerate(lines):
            bbox_line = draw.textbbox((0, 0), line, font=font)
            lw = bbox_line[2] - bbox_line[0]
            lx = x1 + (bw - lw) // 2
            ly = start_y + i * line_height
            for dx in range(-outline_width, outline_width + 1):
                for dy in range(-outline_width, outline_width + 1):
                    if dx != 0 or dy != 0:
                        draw.text((lx + dx, ly + dy), line, font=font, fill=outline_color)
            draw.text((lx, ly), line, font=font, fill=font_color)
        return img

    def render_sfx(
        self,
        img: Image.Image,
        bbox: tuple[int, int, int, int],
        text: str,
        font_color: str = "white",
        outline_color: str = "black",
        outline_width: int = 3,
    ) -> Image.Image:
        img = img.copy()
        draw = ImageDraw.Draw(img)
        x1, y1, x2, y2 = bbox
        bw = x2 - x1
        bh = y2 - y1
        font_path = self._get_font_path("sfx", text)
        font = self._fit_text(draw, text, bw, bh, font_path, min_size=10, max_size=48)
        bbox_text = draw.textbbox((0, 0), text, font=font)
        tw = bbox_text[2] - bbox_text[0]
        th = bbox_text[3] - bbox_text[1]
        tx = x1 + (bw - tw) // 2
        ty = y1 + (bh - th) // 2
        for dx in range(-outline_width, outline_width + 1):
            for dy in range(-outline_width, outline_width + 1):
                if dx != 0 or dy != 0:
                    draw.text((tx + dx, ty + dy), text, font=font, fill=outline_color)
        draw.text((tx, ty), text, font=font, fill=font_color)
        return img

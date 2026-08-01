from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import json
import re
import os
import platform
import numpy as np
import cv2
from cfg import FONTS, FONTS_PATH


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


def _extract_text_color(
    original_img: Image.Image,
    bubble_mask: Image.Image,
    bbox: tuple[int, int, int, int],
) -> tuple[int, int, int]:
    """
    Extract actual text color from bubble using k-means on edge pixels.
    Returns RGB tuple (r, g, b).
    """
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        return (0, 0, 0)

    # Crop bubble region
    orig_crop = np.array(original_img.crop((x1, y1, x2, y2)), dtype=np.float32)
    mask_crop = np.array(bubble_mask.crop((x1, y1, x2, y2)).convert("L"), dtype=np.uint8)

    # Find edge pixels (text is typically near mask boundary)
    edge_pixels = []
    for i in range(bh):
        for j in range(bw):
            if mask_crop[i, j] > 128:
                # Check if this is near boundary (text pixels)
                is_edge = False
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        ni, nj = i + di, j + dj
                        if 0 <= ni < bh and 0 <= nj < bw and mask_crop[ni, nj] <= 128:
                            is_edge = True
                            break
                    if is_edge:
                        break
                if is_edge:
                    edge_pixels.append(orig_crop[i, j])

    if not edge_pixels:
        return (0, 0, 0)

    edge_pixels = np.array(edge_pixels)

    # Simple 2-cluster k-means (text color vs background)
    # Initialize with darkest and brightest pixels
    brightness = np.mean(edge_pixels, axis=1)
    idx_dark = np.argmin(brightness)
    idx_bright = np.argmax(brightness)
    centers = edge_pixels[[idx_dark, idx_bright]].copy().astype(np.float32)

    for _ in range(10):  # max iterations
        dists = np.sum((edge_pixels[:, None] - centers[None]) ** 2, axis=2)
        labels = np.argmin(dists, axis=1)
        new_centers = np.array([
            edge_pixels[labels == 0].mean(axis=0) if np.any(labels == 0) else centers[0],
            edge_pixels[labels == 1].mean(axis=0) if np.any(labels == 1) else centers[1],
        ])
        if np.allclose(centers, new_centers):
            break
        centers = new_centers

    # Choose the cluster that represents text (usually smaller, darker cluster)
    counts = np.bincount(labels, minlength=2)
    text_cluster = 0 if counts[0] < counts[1] else 1
    text_color = centers[text_cluster].astype(np.uint8)

    return tuple(int(c) for c in text_color)


def _classify_font_style(
    original_img: Image.Image,
    bubble_mask: Image.Image,
    bbox: tuple[int, int, int, int],
) -> str:
    """
    Classify text region style for font matching.
    Returns "narration" (handwritten/italic) or "dialogue" (clean/print).
    Uses edge roughness & line straightness of the text mask inside the bubble.
    """
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        return "dialogue"

    try:
        mask_crop = np.array(bubble_mask.crop((x1, y1, x2, y2)).convert("L"), dtype=np.uint8)
    except Exception:
        return "dialogue"

    binary = (mask_crop > 128).astype(np.uint8)
    if binary.sum() < 50:
        return "dialogue"

    # Edge roughness: ratio of perimeter to area. Handwritten text has ragged edges.
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return "dialogue"
    total_perim = sum(cv2.arcLength(c, True) for c in contours)
    total_area = float(binary.sum())
    if total_area <= 0:
        return "dialogue"
    roughness = total_perim / total_area

    # Aspect: handwritten is often narrow/tall strokes; captions are wide & flat
    xs, ys = np.nonzero(binary)
    h_span = int(xs.max() - xs.min()) + 1
    w_span = int(ys.max() - ys.min()) + 1
    aspect = (w_span / max(h_span, 1)) if h_span > 0 else 1.0

    # Handwritten/italic text: higher roughness + moderate aspect
    if roughness > 0.35 and aspect < 4.5:
        return "narration"
    return "dialogue"


def _is_caption_region(
    cv_img: np.ndarray,
    text_bbox: tuple[int, int, int, int],
) -> bool:
    """
    Detect narration captions: text in a rectangular box without a round bubble.
    Returns True for rectangular captions (narration), False for speech bubbles.
    """
    from translator.bubbles import _find_bubble_contour
    bubble = _find_bubble_contour(cv_img, text_bbox)
    if bubble:
        # It's a bubble — but could still be a rounded rectangle caption.
        bx1, by1, bx2, by2 = bubble
        bw = bx2 - bx1
        bh = by2 - by1
        if bw <= 0 or bh <= 0:
            return False
        rect_area = bw * bh
        # Rounded-rectangle captions are fairly rectangular
        return rect_area > 0 and (bw / max(bh, 1)) > 1.6
    # No bubble contour: if text is wide & flat, treat as narration caption
    tx1, ty1, tx2, ty2 = text_bbox
    tw = tx2 - tx1
    th = ty2 - ty1
    if tw > 0 and th > 0 and tw / max(th, 1) > 1.8:
        return True
    return False


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
        if not words:
            return []
        lines = self._smart_break(text, font, max_width)
        if len(lines) > 1:
            balanced = self._balance_lines(lines, font, max_width)
            if balanced:
                return balanced
        return lines

    @staticmethod
    def _smart_break(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
        import re
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

    @staticmethod
    def _balance_lines(lines: list[str], font: ImageFont.FreeTypeFont, max_width: int) -> list[str] | None:
        if len(lines) < 2:
            return None
        last_w = font.getbbox(lines[-1])[2] - font.getbbox(lines[-1])[0]
        if last_w > max_width * 0.3:
            return None
        for i in range(len(lines) - 1):
            merged = f"{lines[i]} {lines[i+1]}"
            mw = font.getbbox(merged)[2] - font.getbbox(merged)[0]
            if mw <= max_width:
                new_lines = lines[:i] + [merged] + lines[i+2:]
                return new_lines
        return None

    @staticmethod
    def _estimate_brightness(img: Image.Image, bbox: tuple) -> float:
        x1, y1, x2, y2 = bbox
        region = img.crop((x1, y1, x2, y2))
        gray = region.convert("L")
        pixels = list(gray.getdata())
        return sum(pixels) / len(pixels) if pixels else 255

    @staticmethod
    def _get_bubble_mask_region(img_w: int, img_h: int, bbox: tuple) -> Image.Image:
        x1, y1, x2, y2 = bbox
        mask = Image.new("L", (img_w, img_h), 0)
        draw = ImageDraw.Draw(mask)
        pad = 5
        draw.rounded_rectangle([x1 - pad, y1 - pad, x2 + pad, y2 + pad], radius=15, fill=255)
        return mask

    @staticmethod
    def _colorize_inpainted(original_img: Image.Image, inpainted_img: Image.Image, bbox: tuple, fallback_color: tuple = None) -> Image.Image:
        x1, y1, x2, y2 = bbox
        orig_crop = original_img.crop((x1, y1, x2, y2))
        inp_crop = inpainted_img.crop((x1, y1, x2, y2))
        orig_arr = np.array(orig_crop, dtype=np.float32)
        inp_arr = np.array(inp_crop, dtype=np.float32)
        diff = orig_arr - inp_arr
        gray_diff = np.mean(np.abs(diff), axis=2)
        is_gray = gray_diff.mean() < 10
        if is_gray and fallback_color:
            result = np.where(orig_arr.mean(axis=2, keepdims=True) < 15, inp_arr, orig_arr)
        else:
            result = inp_arr * 0.3 + orig_arr * 0.7
        result = result.clip(0, 255).astype(np.uint8)
        return Image.fromarray(result)

    def render_bubble_text(
        self,
        img: Image.Image,
        bbox: tuple[int, int, int, int],
        text: str,
        font_type: str = "dialogue",
        outline_width: int = 2,
        is_bubble: bool = True,
        original_img: Image.Image | None = None,
        angle: float = 0.0,
    ) -> Image.Image:
        img = img.copy()
        # Extract actual text color from original
        if original_img is not None and is_bubble:
            bubble_mask = self._get_bubble_mask_region(img.width, img.height, bbox)
            text_color = _extract_text_color(original_img, bubble_mask, bbox)
            # Determine if dark or light text based on brightness
            avg_brightness = sum(text_color) / 3
            if avg_brightness < 128:
                font_color = text_color
                outline_color = tuple(min(255, c + 60) for c in text_color)
            else:
                font_color = tuple(max(0, c - 60) for c in text_color)
                outline_color = text_color
        else:
            # Fallback to brightness-based
            brightness = self._estimate_brightness(img, bbox)
            if brightness < 128:
                font_color, outline_color = "white", "black"
            else:
                font_color, outline_color = "black", "white"

        if not is_bubble:
            white_bg = Image.new("RGB", (bbox[2] - bbox[0], bbox[3] - bbox[1]), (255, 255, 255))
            img.paste(white_bg, (bbox[0], bbox[1]))
        pad = 10
        font_path = self._get_font_path(font_type, text)
        font = self._fit_text(ImageDraw.Draw(img), text, bbox[2] - bbox[0] - pad * 2, bbox[3] - bbox[1] - pad * 2, font_path)
        lines = self._wrap_text(text, font, bbox[2] - bbox[0] - pad * 2)
        line_height = font.getbbox("Аg")[3] - font.getbbox("Аg")[1] + 2
        total_height = line_height * len(lines)
        start_y = bbox[1] + (bbox[3] - bbox[1] - total_height) // 2
        text_layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        text_draw = ImageDraw.Draw(text_layer)
        for i, line in enumerate(lines):
            bbox_line = text_draw.textbbox((0, 0), line, font=font)
            lw = bbox_line[2] - bbox_line[0]
            lx = bbox[0] + (bbox[2] - bbox[0] - lw) // 2
            ly = start_y + i * line_height
            for dx in range(-outline_width, outline_width + 1):
                for dy in range(-outline_width, outline_width + 1):
                    if dx != 0 or dy != 0:
                        text_draw.text((lx + dx, ly + dy), line, font=font, fill=outline_color)
            text_draw.text((lx, ly), line, font=font, fill=font_color)

        if angle:
            cx = (bbox[0] + bbox[2]) // 2
            cy = (bbox[1] + bbox[3]) // 2
            text_layer = text_layer.rotate(angle, resample=Image.Resampling.BICUBIC, center=(cx, cy), expand=False)
            img = Image.alpha_composite(img.convert("RGBA"), text_layer).convert("RGB")
        else:
            img = Image.alpha_composite(img.convert("RGBA"), text_layer).convert("RGB")
        return img

    def render_text_in_bubble_shape(
        self,
        img: Image.Image,
        bbox: tuple[int, int, int, int],
        text: str,
        font_type: str = "dialogue",
        outline_width: int = 2,
        is_bubble: bool = True,
        original_img: Image.Image | None = None,
    ) -> Image.Image:
        img = img.copy()
        x1, y1, x2, y2 = bbox
        bw = x2 - x1
        bh = y2 - y1
        if is_bubble:
            bubble_mask = self._get_bubble_mask_region(img.width, img.height, bbox)
        else:
            bubble_mask = None

        # Extract actual text color from original
        if original_img is not None and is_bubble and bubble_mask:
            text_color = _extract_text_color(original_img, bubble_mask, bbox)
            avg_brightness = sum(text_color) / 3
            if avg_brightness < 128:
                font_color = text_color
                outline_color = tuple(min(255, c + 60) for c in text_color)
            else:
                font_color = tuple(max(0, c - 60) for c in text_color)
                outline_color = text_color
        else:
            # Fallback to brightness-based
            brightness = self._estimate_brightness(img, bbox)
            if brightness < 128:
                font_color, outline_color = "white", "black"
            else:
                font_color, outline_color = "black", "white"

        if not is_bubble:
            white_bg = Image.new("RGB", (bw, bh), (255, 255, 255))
            img.paste(white_bg, (x1, y1))
        pad = 10
        font_path = self._get_font_path(font_type, text)
        font = self._fit_text(ImageDraw.Draw(img), text, bw - pad * 2, bh - pad * 2, font_path)
        lines = self._wrap_text(text, font, bw - pad * 2)
        line_height = font.getbbox("Аg")[3] - font.getbbox("Аg")[1] + 2
        total_height = line_height * len(lines)
        start_y = y1 + (bh - total_height) // 2
        text_layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        text_draw = ImageDraw.Draw(text_layer)
        for i, line in enumerate(lines):
            bbox_line = text_draw.textbbox((0, 0), line, font=font)
            lw = bbox_line[2] - bbox_line[0]
            lx = x1 + (bw - lw) // 2
            ly = start_y + i * line_height
            for dx in range(-outline_width, outline_width + 1):
                for dy in range(-outline_width, outline_width + 1):
                    if dx != 0 or dy != 0:
                        text_draw.text((lx + dx, ly + dy), line, font=font, fill=outline_color + (255,))
            text_draw.text((lx, ly), line, font=font, fill=font_color + (255,))
        if bubble_mask:
            img = img.convert("RGBA")
            img.paste(Image.composite(text_layer, Image.new("RGBA", img.size, (0, 0, 0, 0)), bubble_mask.convert("L").point(lambda x: 255 if x > 128 else 0)), (0, 0))
            img = img.convert("RGB")
        else:
            img = Image.alpha_composite(img.convert("RGBA"), text_layer).convert("RGB")
        if original_img is not None:
            img = self._colorize_inpainted(original_img, img, bbox, fallback_color=(240, 240, 240))
        return img

    def render_vertical_text(
        self,
        img: Image.Image,
        bbox: tuple[int, int, int, int],
        text: str,
        font_type: str = "dialogue",
        outline_width: int = 2,
        original_img: Image.Image | None = None,
    ) -> Image.Image:
        """Render Japanese text vertically (one character per line, top-to-bottom)."""
        img = img.copy()
        x1, y1, x2, y2 = bbox
        bw = x2 - x1
        bh = y2 - y1

        font_path = self._get_font_path(font_type, text)
        # Fit font size: width determines char size for vertical layout
        font = None
        pad = 6
        for size in range(48, 8, -1):
            try:
                f = ImageFont.truetype(font_path, size)
            except Exception:
                f = ImageFont.load_default()
            ch_bbox = f.getbbox("ア")
            ch_w = ch_bbox[2] - ch_bbox[0]
            ch_h = ch_bbox[3] - ch_bbox[1]
            total_h = ch_h * len(text)
            if ch_w <= bw - pad * 2 and total_h <= bh - pad * 2:
                font = f
                break
        if font is None:
            font = ImageFont.load_default()

        # Text color extraction (reuse bubble logic)
        brightness = self._estimate_brightness(img, bbox)
        if brightness < 128:
            font_color, outline_color = "white", "black"
        else:
            font_color, outline_color = "black", "white"

        text_layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        text_draw = ImageDraw.Draw(text_layer)
        chars = [ch for ch in text if ch.strip() or ch == " "]
        ch_h = font.getbbox("ア")[3] - font.getbbox("ア")[1] + 2
        total_h = ch_h * len(chars)
        start_y = y1 + (bh - total_h) // 2
        lx = x1 + (bw - (font.getbbox("ア")[2] - font.getbbox("ア")[0])) // 2
        for i, ch in enumerate(chars):
            ly = start_y + i * ch_h
            for dx in range(-outline_width, outline_width + 1):
                for dy in range(-outline_width, outline_width + 1):
                    if dx != 0 or dy != 0:
                        text_draw.text((lx + dx, ly + dy), ch, font=font, fill=outline_color + (255,))
            text_draw.text((lx, ly), ch, font=font, fill=font_color + (255,))

        img = img.convert("RGBA")
        img = Image.alpha_composite(img, text_layer).convert("RGB")
        if original_img is not None:
            img = self._colorize_inpainted(original_img, img, bbox, fallback_color=(240, 240, 240))
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


def _polygon_angle(poly: list) -> float:
    """
    Estimate the skew (rotation) angle of a text polygon in degrees.

    Computes the minimum-area rectangle of the polygon and returns the
    rotation angle of its long axis (clamped to [-45, 45]).
    """
    import numpy as np
    pts = np.array([(float(p[0]), float(p[1])) for p in poly], dtype=np.float32)
    if len(pts) < 3:
        return 0.0
    try:
        rect = cv2.minAreaRect(pts)
        angle = rect[2]
        # minAreaRect angle semantics: for near-square it flips; normalize to long axis
        w, h = rect[1]
        if w < h:
            angle += 90.0
        if angle > 45.0:
            angle -= 90.0
        if angle < -45.0:
            angle += 90.0
        return float(angle)
    except Exception:
        return 0.0


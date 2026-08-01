import numpy as np
import cv2


def deskew(img: np.ndarray) -> np.ndarray:
    """Correct skew of a scanned page image via minAreaRect of text contour."""
    if img is None:
        return img
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        # Invert for text detection (white text on black)
        inv = 255 - gray
        thresh = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        coords = cv2.findNonZero(thresh)
        if coords is None:
            return img
        rect = cv2.minAreaRect(coords)
        angle = rect[-1]
        h, w = img.shape[:2]

        # Normalize angle: keep within -45..45
        if angle > 45:
            angle -= 90
        if abs(angle) < 0.3:
            return img

        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            img, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return rotated
    except Exception:
        return img


def _estimate_vignette_bg(img_gray: np.ndarray, kernel: int = 51) -> np.ndarray:
    """Approximate page background via large median blur (removes gradients)."""
    k = kernel if kernel % 2 == 1 else kernel + 1
    return cv2.medianBlur(img_gray, k)


def deshake(img: np.ndarray, bg_kernel: int = 51) -> np.ndarray:
    """Remove background gradient/lighting for manga pages (flatten illumination)."""
    if img is None:
        return img
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        bg = _estimate_vignette_bg(gray, bg_kernel)
        # Avoid division by zero
        bg = cv2.max(bg, 1)
        flat = (gray.astype(np.float32) / bg.astype(np.float32)) * 255.0
        flat = np.clip(flat, 0, 255).astype(np.uint8)
        if img.ndim == 3:
            flat_rgb = cv2.cvtColor(flat, cv2.COLOR_GRAY2BGR)
            # Apply the same flattening to each channel using per-channel bg
            for c in range(3):
                ch = img[:, :, c]
                bg_c = _estimate_vignette_bg(ch, bg_kernel)
                bg_c = cv2.max(bg_c, 1)
                flat_rgb[:, :, c] = np.clip((ch.astype(np.float32) / bg_c.astype(np.float32)) * 255.0, 0, 255).astype(np.uint8)
            return flat_rgb
        return flat
    except Exception:
        return img


def sauvola(img: np.ndarray, window: int = 25, k: float = 0.2, r: float = 128.0) -> np.ndarray:
    """Adaptive Sauvola binarization. Returns a binary mask (0/255)."""
    if img is None:
        return img
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        gray_f = gray.astype(np.float32)

        win = window if window % 2 == 1 else window + 1
        # Integral images for local mean and local variance
        integral = cv2.integral(gray_f)
        integral_sq = cv2.integral(gray_f ** 2)
        h, w = gray_f.shape

        half = win // 2
        # Pad by replicating borders
        padded = cv2.copyMakeBorder(gray_f, half, half, half, half, cv2.BORDER_REPLICATE)
        int_pad = cv2.integral(padded)
        int_sq_pad = cv2.integral(padded ** 2)

        # Sliding window sums
        x = np.arange(w) + 2 * half
        y = np.arange(h) + 2 * half
        A = int_pad[y - win + 1, x - win + 1]
        B = int_pad[y - win + 1, x + 1]
        C = int_pad[y + 1, x - win + 1]
        D = int_pad[y + 1, x + 1]
        sum_win = A + D - B - C

        A = int_sq_pad[y - win + 1, x - win + 1]
        B = int_sq_pad[y - win + 1, x + 1]
        C = int_sq_pad[y + 1, x - win + 1]
        D = int_sq_pad[y + 1, x + 1]
        sum_sq_win = A + D - B - C

        n = win * win
        mean = sum_win / n
        variance = (sum_sq_win / n) - mean ** 2
        variance = np.maximum(variance, 0)
        std = np.sqrt(variance)

        threshold = mean * (1.0 + k * ((std / r) - 1.0))
        binary = (gray_f > threshold).astype(np.uint8) * 255
        return binary
    except Exception:
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]


def preprocess_page(img: np.ndarray, use_sauvola: bool = False) -> np.ndarray:
    """Full preprocessing chain for OCR: deskew -> deshake (optional) -> normalize."""
    if img is None:
        return img
    img = deskew(img)
    img = deshake(img)
    if use_sauvola:
        binary = sauvola(img)
        img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    return img

import cv2
import numpy as np


MIN_BUBBLE_W = 80
MIN_BUBBLE_H = 30


def _find_bubble_contour(img_bgr, text_bbox):
    tx1, ty1, tx2, ty2 = text_bbox
    cx = (tx1 + tx2) // 2
    cy = (ty1 + ty2) // 2
    h, w = img_bgr.shape[:2]
    expand = int(max(tx2 - tx1, ty2 - ty1) * 2.0)
    expand = max(expand, 120)
    rx1 = max(0, tx1 - expand)
    ry1 = max(0, ty1 - expand)
    rx2 = min(w, tx2 + expand)
    ry2 = min(h, ty2 + expand)
    region = img_bgr[ry1:ry2, rx1:rx2]
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

    masks = []
    for thresh_val in [200, 220, 240]:
        _, binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
        kernel_close = np.ones((9, 9), np.uint8)
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close, iterations=2)
        kernel_open = np.ones((5, 5), np.uint8)
        cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open, iterations=1)
        masks.append(cleaned)

    merged = cv2.bitwise_or(masks[0], cv2.bitwise_or(masks[1], masks[2]))

    kernel_diag = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    merged = cv2.dilate(merged, kernel_diag, iterations=1)

    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_score = 0
    lcx, lcy = cx - rx1, cy - ry1
    for cnt in contours:
        if cv2.pointPolygonTest(cnt, (float(lcx), float(lcy)), False) < 0:
            continue
        area = cv2.contourArea(cnt)
        if area < 500:
            continue
        x, y, cw, ch = cv2.boundingRect(cnt)
        if cw < MIN_BUBBLE_W * 0.5 or ch < MIN_BUBBLE_H * 0.5:
            continue
        aspect = cw / max(ch, 1)
        if aspect > 15 or aspect < 0.1:
            continue
        dist_to_center = abs(x + cw // 2 - lcx) + abs(y + ch // 2 - lcy)
        score = area / (1 + dist_to_center * 0.001)
        if score > best_score:
            best_score = score
            best = cnt

    if best is not None:
        bx, by, bw, bh = cv2.boundingRect(best)
        x1 = max(0, rx1 + bx - 12)
        y1 = max(0, ry1 + by - 12)
        x2 = min(w, rx1 + bx + bw + 12)
        y2 = min(h, ry1 + by + bh + 12)
        return (x1, y1, x2, y2)
    return None


def expand_bbox(bbox, img_w, img_h, factor=0.6):
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    px = max(int(bw * factor), 30)
    py = max(int(bh * factor), 15)
    x1 = max(0, x1 - px)
    y1 = max(0, y1 - py)
    x2 = min(img_w, x2 + px)
    y2 = min(img_h, y2 + py)
    if x2 - x1 < MIN_BUBBLE_W:
        cx = (x1 + x2) // 2
        x1 = max(0, cx - MIN_BUBBLE_W // 2)
        x2 = min(img_w, cx + MIN_BUBBLE_W // 2)
    if y2 - y1 < MIN_BUBBLE_H:
        cy = (y1 + y2) // 2
        y1 = max(0, cy - MIN_BUBBLE_H // 2)
        y2 = min(img_h, cy + MIN_BUBBLE_H // 2)
    return (x1, y1, x2, y2)


def get_bubble_bounds(img_bgr, text_bbox, img_w, img_h):
    bubble = _find_bubble_contour(img_bgr, text_bbox)
    if bubble:
        return bubble, True
    return expand_bbox(text_bbox, img_w, img_h), False


def build_mask(img_h, img_w, all_bubble_bboxes, pad=10):
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    for x1, y1, x2, y2 in all_bubble_bboxes:
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(img_w, x2 + pad)
        y2 = min(img_h, y2 + pad)
        mask[y1:y2, x1:x2] = 255
    kernel = np.ones((7, 7), np.uint8)
    return cv2.dilate(mask, kernel, iterations=3)

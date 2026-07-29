import cv2
import numpy as np


MIN_BUBBLE_W = 100
MIN_BUBBLE_H = 40


def _find_bubble_contour(img_bgr, text_bbox):
    tx1, ty1, tx2, ty2 = text_bbox
    cx = (tx1 + tx2) // 2
    cy = (ty1 + ty2) // 2
    h, w = img_bgr.shape[:2]
    expand = int(max(tx2 - tx1, ty2 - ty1) * 1.5)
    expand = max(expand, 80)
    rx1 = max(0, tx1 - expand)
    ry1 = max(0, ty1 - expand)
    rx2 = min(w, tx2 + expand)
    ry2 = min(h, ty2 + expand)
    region = img_bgr[ry1:ry2, rx1:rx2]
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 220, 255, cv2.THRESH_BINARY)
    kernel = np.ones((7, 7), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_area = 0
    lcx, lcy = cx - rx1, cy - ry1
    for cnt in contours:
        if cv2.pointPolygonTest(cnt, (float(lcx), float(lcy)), False) >= 0:
            area = cv2.contourArea(cnt)
            if area > best_area:
                best_area = area
                best = cnt
    if best is not None:
        bx, by, bw, bh = cv2.boundingRect(best)
        x1 = max(0, rx1 + bx - 8)
        y1 = max(0, ry1 + by - 8)
        x2 = min(w, rx1 + bx + bw + 8)
        y2 = min(h, ry1 + by + bh + 8)
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
    return cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=2)

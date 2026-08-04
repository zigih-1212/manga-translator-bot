import asyncio
import io
import json
import os
import base64
import logging
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, Query, Request
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import cv2

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("kaggle_server")

app = FastAPI()
ocr_reader = None
gemini_model = None
lama_model = None

MODELS_DIR = Path(os.environ.get("MODELS_DIR", "/content/models"))
MODEL_URL = "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt"
MODEL_MD5 = "e3aa4aaa15225a33ec84f9f4bc47e500"
MODEL_PATH = MODELS_DIR / "big-lama.pt"


def get_ocr():
    global ocr_reader
    if ocr_reader is None:
        import easyocr
        try:
            import torch
            use_gpu = torch.cuda.is_available()
        except Exception:
            use_gpu = False
        ocr_reader = easyocr.Reader(["ko", "en"], gpu=use_gpu, verbose=False)
    return ocr_reader


def get_gemini():
    global gemini_model
    if gemini_model is None:
        import google.generativeai as genai
        key = os.environ.get("GEMINI_API_KEY", "")
        if not key:
            raise RuntimeError("GEMINI_API_KEY not set on server")
        genai.configure(api_key=key)
        gemini_model = genai.GenerativeModel("gemini-2.0-flash")
    return gemini_model


def _download_lama():
    import hashlib
    import urllib.request
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.exists():
        md5 = hashlib.md5(MODEL_PATH.read_bytes()).hexdigest()
        if md5 == MODEL_MD5:
            return
        MODEL_PATH.unlink()
    log.info("Downloading LaMa model (%s)...", MODEL_URL)
    req = urllib.request.Request(MODEL_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=600) as src:
        data = src.read()
    if hashlib.md5(data).hexdigest() != MODEL_MD5:
        raise RuntimeError("LaMa model MD5 mismatch")
    MODEL_PATH.write_bytes(data)
    log.info("LaMa model downloaded")


def get_lama():
    """Load TorchScript LaMa for GPU inpainting (lazy, only on first inpaint)."""
    global lama_model
    if lama_model is None:
        import torch
        if not MODEL_PATH.exists():
            try:
                _download_lama()
            except Exception as e:
                log.warning("LaMa model unavailable, inpaint will fallback: %s", e)
                return None
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            lama_model = torch.jit.load(str(MODEL_PATH), map_location=device)
            lama_model.eval()
            if device == "cuda":
                lama_model = lama_model.cuda()
            log.info("LaMa loaded on %s", device)
        except Exception as e:
            log.error("Failed to load LaMa: %s", e)
            return None
    return lama_model


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/translate")
async def translate_endpoint(req: dict = None):
    b = req or {}
    texts = b.get("korean_texts", [])
    if not texts:
        return JSONResponse({"translations": []})
    ctx = b.get("context", {})
    g = ctx.get("glossary", {})
    prev = ctx.get("previous_pages", [])
    gb = ""
    for k, v in g.get("characters", {}).items():
        gb += f"  {k} -> {v}\n"
    for k, v in g.get("terms", {}).items():
        gb += f"  {k} -> {v}\n"
    if not gb:
        gb = "  (empty)"
    cb = ""
    if prev:
        for i, p in enumerate(prev[-5:]):
            cb += f"  Page {i+1}:\n{p}\n"
    else:
        cb = "  (first page)"
    kb = "\n".join(f"  [{i+1}] {t}" for i, t in enumerate(texts))
    en = b.get("english_texts", [])
    eb = ""
    if en:
        eb = "\nEnglish reference:\n" + "\n".join(
            f"  [{i+1}] {t}" for i, t in enumerate(en)
        )
    prompt = (
        f"Translate Korean manga text to Russian.\n"
        f"GLOSSARY:\n{gb}\nCONTEXT:\n{cb}\n"
        f"KOREAN TEXT (page {b.get('page_number', 1)}):\n{kb}\n{eb}\n"
        f"Rules: use glossary for names, be concise, casual Russian.\n"
        f'Reply ONLY JSON array: [{{"id": 1, "ru": "translation"}}]'
    )
    try:
        m = get_gemini()
        r = await asyncio.to_thread(
            m.generate_content,
            prompt,
            generation_config={
                "temperature": b.get("temperature", 0.3),
                "max_output_tokens": 2000,
            },
        )
        raw = r.text.strip()
        raw = (
            raw.removeprefix("```json")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )
        return JSONResponse({"translations": json.loads(raw)})
    except Exception as e:
        return JSONResponse(
            {
                "translations": [
                    {"id": i + 1, "ru": t} for i, t in enumerate(texts)
                ],
                "error": str(e),
            }
        )


@app.post("/ocr")
async def ocr_endpoint(pages: list[UploadFile] = File(...)):
    reader = get_ocr()
    results = []
    for u in pages:
        img = Image.open(io.BytesIO(await u.read())).convert("RGB")
        dets = reader.readtext(np.array(img))
        results.append(
            [
                {
                    "bbox": [[int(p[0]), int(p[1])] for p in bb],
                    "text": t,
                    "confidence": float(c),
                    "type": "text",
                }
                for bb, t, c in dets
                if c > 0.3
            ]
        )
    return JSONResponse({"results": results})


def _inpaint_np(img_np: np.ndarray, mask_np: np.ndarray) -> np.ndarray:
    """Real LaMa inpaint on GPU; fallback to cv2 if model unavailable."""
    model = get_lama()
    if model is not None:
        try:
            import torch
            h, w = img_np.shape[:2]
            ph = (8 - h % 8) % 8
            pw = (8 - w % 8) % 8
            if ph or pw:
                img = cv2.copyMakeBorder(img_np, 0, ph, 0, pw, cv2.BORDER_REFLECT)
                msk = cv2.copyMakeBorder(mask_np, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=0)
            else:
                img, msk = img_np, mask_np
            img_t = torch.from_numpy(img.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0)
            msk_t = (torch.from_numpy(msk) > 127).float().unsqueeze(0).unsqueeze(0)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            with torch.no_grad():
                out = model(img_t.to(device), msk_t.to(device)).cpu()
            out = out[0].permute(1, 2, 0).numpy()
            out = ((out + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
            if ph or pw:
                out = out[:h, :w]
            return out
        except Exception as e:
            log.warning("LaMa GPU inpaint failed, using cv2: %s", e)
    import cv2
    m = cv2.dilate(mask_np, np.ones((5, 5), np.uint8), iterations=3)
    return cv2.inpaint(img_np, m, 10, cv2.INPAINT_NS)


async def _read_mask_file(upload: UploadFile | None, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    if upload is None:
        return mask
    try:
        raw = await upload.read()
        if not raw:
            return mask
        arr = np.frombuffer(raw, np.uint8)
        decoded = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if decoded is None:
            return mask
        if decoded.shape[:2] != shape:
            decoded = cv2.resize(
                decoded,
                (shape[1], shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        return np.where(decoded > 127, 255, 0).astype(np.uint8)
    except Exception as e:
        log.warning("Failed to decode uploaded mask: %s", e)
        return mask


def _apply_bbox_payload(mask: np.ndarray, payload: list[dict] | None) -> np.ndarray:
    if not payload:
        return mask
    h, w = mask.shape[:2]
    for item in payload:
        bb = item.get("bbox", [])
        if len(bb) != 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in bb]
        mask[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = 255
    return mask


def _parse_masks_payload(raw: str | None):
    if not raw:
        return []
    try:
        data = json.loads(raw)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _resolve_masks_data(request: Request, masks_data: str | None) -> list:
    raw = masks_data
    if not raw:
        raw = request.query_params.get("masks_data") or request.query_params.get("masks")
    return _parse_masks_payload(raw)


@app.post("/inpaint")
async def inpaint_endpoint(
    request: Request,
    page: UploadFile = File(...),
    mask: UploadFile | None = File(None),
    masks_data: str | None = Form(None),
):
    img = Image.open(io.BytesIO(await page.read()))
    a = np.array(img.convert("RGB"))
    mask_np = await _read_mask_file(mask, a.shape[:2])
    bbox_payload = _resolve_masks_data(request, masks_data)
    if bbox_payload and bbox_payload and isinstance(bbox_payload[0], list):
        bbox_payload = bbox_payload[0]
    mask_np = _apply_bbox_payload(mask_np, bbox_payload)
    out = _inpaint_np(a, mask_np)
    buf = io.BytesIO()
    Image.fromarray(out).save(buf, format="PNG")
    return JSONResponse({"image_b64": base64.b64encode(buf.getvalue()).decode()})


@app.post("/inpaint_batch")
async def inpaint_batch_endpoint(
    request: Request,
    pages: list[UploadFile] = File(...),
    masks: list[UploadFile] | None = File(None),
    masks_data: str | None = Form(None),
):
    all_m = _resolve_masks_data(request, masks_data)
    res = []
    for i, u in enumerate(pages):
        img = Image.open(io.BytesIO(await u.read())).convert("RGB")
        a = np.array(img)
        mask_np = await _read_mask_file(masks[i] if masks and i < len(masks) else None, a.shape[:2])
        page_masks = all_m[i] if i < len(all_m) and isinstance(all_m[i], list) else []
        mask_np = _apply_bbox_payload(mask_np, page_masks)
        out = _inpaint_np(a, mask_np)
        buf = io.BytesIO()
        Image.fromarray(out).save(buf, format="PNG")
        res.append(base64.b64encode(buf.getvalue()).decode())
    return JSONResponse({"clean_pages_b64": res})


@app.post("/process")
async def process_endpoint(pages: list[UploadFile] = File(...)):
    reader = get_ocr()
    ocr_res, clean = [], []
    for u in pages:
        img = Image.open(io.BytesIO(await u.read())).convert("RGB")
        a = np.array(img)
        dets = reader.readtext(a)
        po = [
            {
                "bbox": [[int(p[0]), int(p[1])] for p in bb],
                "text": t,
                "confidence": float(c),
                "type": "text",
            }
            for bb, t, c in dets
            if c > 0.3
        ]
        ocr_res.append(po)
        mask = np.zeros(a.shape[:2], dtype=np.uint8)
        for d in po:
            bb = d["bbox"]
            if len(bb) == 4:
                x1 = max(0, int(min(p[0] for p in bb)))
                y1 = max(0, int(min(p[1] for p in bb)))
                x2 = min(a.shape[1], int(max(p[0] for p in bb)))
                y2 = min(a.shape[0], int(max(p[1] for p in bb)))
                mask[y1:y2, x1:x2] = 255
        out = _inpaint_np(a, mask)
        buf = io.BytesIO()
        Image.fromarray(out).save(buf, format="PNG")
        clean.append(base64.b64encode(buf.getvalue()).decode())
    return JSONResponse({"ocr_results": ocr_res, "clean_pages_b64": clean})


@app.post("/render")
async def render_endpoint(
    page: UploadFile = File(...), translations: str = Query("[]")
):
    img = Image.open(io.BytesIO(await page.read())).convert("RGB")
    draw = __import__("PIL.ImageDraw", fromlist=["ImageDraw"]).Draw(img)
    for item in json.loads(translations):
        bb, txt = item.get("bbox", []), item.get("ru", "")
        if txt and len(bb) == 4:
            try:
                f = __import__(
                    "PIL.ImageFont", fromlist=["ImageFont"]
                ).truetype("arial.ttf", 14)
                x1, y1 = [int(v) for v in bb[:2]]
                draw.text((x1 + 2, y1 + 2), txt, font=f, fill="black")
                draw.text((x1, y1), txt, font=f, fill="white")
            except Exception:
                pass
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return JSONResponse({"image_b64": base64.b64encode(buf.getvalue()).decode()})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5003)

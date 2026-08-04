import io
import base64
import hashlib
import logging
import numpy as np
import cv2
from pathlib import Path
import modal
from modal import fastapi_endpoint
from pydantic import BaseModel

app = modal.App("manga-inpaint")

log = logging.getLogger("modal-inpaint")

# Volume for model persistence across container restarts
models_volume = modal.Volume.from_name("manga-inpaint-models", create_if_missing=True)

# LaMa big model (TorchScript), same source as official IOPaint.
# ONNX-версия (big-lama.onnx) больше не распространяется на GitHub.
LAMA_MODEL_URL = (
    "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt"
)
LAMA_MODEL_MD5 = "e3aa4aaa15225a33ec84f9f4bc47e500"
MODEL_PATH = Path("/models/big-lama.pt")

image = (
    modal.Image.debian_slim()
    .apt_install("libgl1")
    .pip_install(
        "torch>=2.1.0",
        "opencv-python-headless>=4.9.0",
        "numpy>=1.24.0",
        "Pillow>=10.0.0",
        "pydantic>=2.0.0",
        "fastapi>=0.110.0",
        "onnx",
    )
    .run_commands("mkdir -p /models")
)


def _check_md5(path: Path, expected: str) -> bool:
    if not path.exists():
        return False
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest() == expected


@app.function(gpu="any", image=image, timeout=900, scaledown_window=60, volumes={"/models": models_volume})
def inpaint_batch(images_b64: list[str], dilation: int = 5, radius: int = 10, masks_b64: list[str] | None = None) -> list[str]:
    import torch

    # Модель — big-lama TorchScript (.pt), как в официальном IOPaint.
    if not (MODEL_PATH.exists() and _check_md5(MODEL_PATH, LAMA_MODEL_MD5)):
        # Скачиваем если нет / битая
        import urllib.request
        tmp = Path("/models/big-lama.pt.tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(LAMA_MODEL_URL, tmp)
        if not _check_md5(tmp, LAMA_MODEL_MD5):
            raise RuntimeError("big-lama.pt MD5 mismatch")
        tmp.replace(MODEL_PATH)

    # Загружаем TorchScript JIT-модель (произвольный размер, кратность 8)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.jit.load(str(MODEL_PATH), map_location=device).eval()

    def _inpaint_lama(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        ph = (8 - h % 8) % 8
        pw = (8 - w % 8) % 8
        if ph or pw:
            img = cv2.copyMakeBorder(image, 0, ph, 0, pw, cv2.BORDER_REFLECT)
            msk = cv2.copyMakeBorder(mask, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=0)
        else:
            img, msk = image, mask
        # Нормализация как в официальном IOPaint: /255 (0..1), канал первый (NCHW)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32).transpose(2, 0, 1) / 255.0
        inp = torch.from_numpy(rgb).unsqueeze(0).to(device)
        msk_in = torch.from_numpy((msk > 127).astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(inp, msk_in)
        out = out[0].permute(1, 2, 0).detach().cpu().numpy()
        out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        if ph or pw:
            out = out[:h, :w]
        return out

    results = []
    for i, img_b64 in enumerate(images_b64):
        img_bytes = base64.b64decode(img_b64)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        h, w = img.shape[:2]

        if masks_b64 and i < len(masks_b64) and masks_b64[i]:
            mask_bytes = base64.b64decode(masks_b64[i])
            m_np = np.frombuffer(mask_bytes, np.uint8)
            mask = cv2.imdecode(m_np, cv2.IMREAD_GRAYSCALE)
            if mask is None or mask.shape[:2] != (h, w):
                mask = None
        else:
            mask = None

        if mask is None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
            kernel = np.ones((3, 3), np.uint8)
            dilated = cv2.dilate(binary, kernel, iterations=dilation)
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            mask = np.zeros((h, w), dtype=np.uint8)
            for cnt in contours:
                x, y, cw, ch = cv2.boundingRect(cnt)
                if cw > 60 and ch > 40 and cw < w * 0.8 and ch < h * 0.8:
                    cv2.drawContours(mask, [cnt], -1, 255, -1)
            mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=2)

        inpainted = _inpaint_lama(img, mask)

        _, buf = cv2.imencode(".png", inpainted)
        results.append(base64.b64encode(buf).decode())
    return results


@app.local_entrypoint()
def test():
    test_img = np.ones((512, 512, 3), dtype=np.uint8) * 200
    _, buf = cv2.imencode(".png", test_img)
    b64 = base64.b64encode(buf).decode()
    result = inpaint_batch.remote([b64])
    print(f"OK: {len(result)} images, {len(result[0])} bytes")


class InpaintRequest(BaseModel):
    images_b64: list[str]
    dilation: int = 5
    radius: int = 10
    masks_b64: list[str] | None = None


@app.function(image=image)
@fastapi_endpoint(method="POST")
def inpaint_api(request: InpaintRequest):
    result = inpaint_batch.local(request.images_b64, request.dilation, request.radius, request.masks_b64)
    return result
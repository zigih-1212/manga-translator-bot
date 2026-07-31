import io
import base64
import numpy as np
import cv2
from pathlib import Path
from typing import Optional
import modal
from modal import fastapi_endpoint
from pydantic import BaseModel

app = modal.App("manga-inpaint")

# Volume for model persistence across container restarts
models_volume = modal.Volume.from_name("manga-inpaint-models", create_if_missing=True)

image = (
    modal.Image.debian_slim()
    .apt_install("libgl1")
    .pip_install(
        "opencv-python-headless>=4.9.0",
        "numpy>=1.24.0",
        "Pillow>=10.0.0",
        "onnxruntime>=1.17.0",
        "pydantic>=2.0.0",
        "fastapi>=0.110.0",
    )
    .run_commands(
        "mkdir -p /models",
        'pip install onnx',
    )
)


@app.function(gpu="any", image=image, timeout=600, scaledown_window=60, volumes={"/models": models_volume})
def inpaint_batch(images_b64: list[str], dilation: int = 5, radius: int = 10) -> list[str]:
    import asyncio
    import onnxruntime
    import aiohttp

    async def _download_model():
        model_path = Path("/models/lama.onnx")
        if model_path.exists():
            return
        url = "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.onnx"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=300)) as resp:
                resp.raise_for_status()
                data = await resp.read()
                model_path.write_bytes(data)

    asyncio.run(_download_model())

    sess = onnxruntime.InferenceSession(
        "/models/lama.onnx", providers=["CPUExecutionProvider"]
    )
    results = []
    for img_b64 in images_b64:
        img_bytes = base64.b64decode(img_b64)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        h, w = img.shape[:2]

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
        inpainted = _inpaint_lama(sess, img, mask)

        _, buf = cv2.imencode(".png", inpainted)
        results.append(base64.b64encode(buf).decode())
    return results


def _inpaint_lama(sess, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    ph = (8 - h % 8) % 8
    pw = (8 - w % 8) % 8
    if ph or pw:
        img = cv2.copyMakeBorder(image, 0, ph, 0, pw, cv2.BORDER_REFLECT)
        msk = cv2.copyMakeBorder(mask, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=0)
    else:
        img, msk = image, mask
    inp = img.astype(np.float32) / 127.5 - 1.0
    inp = inp.transpose(2, 0, 1)[np.newaxis, ...]
    msk_in = (msk > 127).astype(np.float32)[np.newaxis, np.newaxis, ...]
    out = sess.run(None, {sess.get_inputs()[0].name: inp, sess.get_inputs()[1].name: msk_in})[0]
    out = out[0].transpose(1, 2, 0)
    out = ((out + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    if ph or pw:
        out = out[:h, :w]
    return out


@app.local_entrypoint()
def test():
    test_img = np.ones((256, 256, 3), dtype=np.uint8) * 200
    _, buf = cv2.imencode(".png", test_img)
    b64 = base64.b64encode(buf).decode()
    result = inpaint_batch.remote([b64])
    print(f"OK: {len(result)} images, {len(result[0])} bytes")


class InpaintRequest(BaseModel):
    images_b64: list[str]
    dilation: int = 5
    radius: int = 10


@app.function(image=image)
@fastapi_endpoint(method="POST")
def inpaint_api(request: InpaintRequest):
    result = inpaint_batch.local(request.images_b64, request.dilation, request.radius)
    return result

import os
import base64
import modal

MODAL_TOKEN_ID = os.getenv("MODAL_TOKEN_ID")
MODAL_TOKEN_SECRET = os.getenv("MODAL_TOKEN_SECRET")
MODAL_AVAILABLE = bool(MODAL_TOKEN_ID and MODAL_TOKEN_SECRET)


def inpaint_batch_sync(images: list[bytes], dilation: int = 5, radius: int = 10) -> list[bytes] | None:
    if not MODAL_AVAILABLE:
        return None
    try:
        f = modal.Function.lookup("manga-inpaint", "inpaint_batch")
        images_b64 = [base64.b64encode(img).decode() for img in images]
        results_b64 = f.remote(images_b64, dilation=dilation, radius=radius)
        return [base64.b64decode(r) for r in results_b64]
    except Exception as e:
        print(f"[Modal] Error: {e}")
        return None

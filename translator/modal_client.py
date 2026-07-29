import os
import base64

MODAL_TOKEN_ID = os.getenv("MODAL_TOKEN_ID")
MODAL_TOKEN_SECRET = os.getenv("MODAL_TOKEN_SECRET")
MODAL_APP_URL = os.getenv("MODAL_APP_URL", "")
MODAL_AVAILABLE = bool(MODAL_TOKEN_ID and MODAL_TOKEN_SECRET and MODAL_APP_URL)


def inpaint_batch_sync(images: list[bytes], dilation: int = 5, radius: int = 10) -> list[bytes] | None:
    return None


async def inpaint_batch(images: list[bytes], dilation: int = 5, radius: int = 10) -> list[bytes] | None:
    return None
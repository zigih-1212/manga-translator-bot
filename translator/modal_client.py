import os
import base64
import logging

log = logging.getLogger("manga_translator")

MODAL_TOKEN_ID = os.getenv("MODAL_TOKEN_ID")
MODAL_TOKEN_SECRET = os.getenv("MODAL_TOKEN_SECRET")
MODAL_APP_ID = os.getenv("MODAL_APP_ID", "dvybornyh332/manga-inpaint")
MODAL_APP_URL = os.getenv("MODAL_APP_URL", "")

if not MODAL_APP_URL and MODAL_APP_ID and "/" in MODAL_APP_ID:
    workspace, app_name = MODAL_APP_ID.split("/", 1)
    MODAL_APP_URL = f"https://{workspace}--{app_name}-inpaint-api.modal.run"

_MODAL_SDK_AVAILABLE = False
try:
    import modal
    _MODAL_SDK_AVAILABLE = True
except ImportError:
    log.warning("Modal SDK not installed, using REST API fallback")


def inpaint_batch_sync(images: list[bytes], dilation: int = 5, radius: int = 10) -> list[bytes] | None:
    if not MODAL_TOKEN_ID or not MODAL_TOKEN_SECRET:
        log.warning("Modal not configured (missing MODAL_TOKEN_ID/SECRET)")
        return None

    images_b64 = [base64.b64encode(img).decode() for img in images]

    # SDK path
    if _MODAL_SDK_AVAILABLE:
        try:
            import modal
            app_name = MODAL_APP_ID.split("/")[-1] if "/" in MODAL_APP_ID else "manga-inpaint"
            f = modal.Function.lookup(app_name, "inpaint_batch")
            result_b64 = f.remote(images_b64, dilation=dilation, radius=radius)
            log.info("Modal GPU inpaint OK (%d images via SDK)", len(images))
            return [base64.b64decode(b64) for b64 in result_b64]
        except Exception as e:
            log.warning("Modal SDK call failed: %s, trying REST API", e)

    # REST API path (no SDK needed)
    try:
        import httpx
        payload = {"images_b64": images_b64, "dilation": dilation, "radius": radius}
        with httpx.Client(timeout=600) as client:
            resp = client.post(
                MODAL_APP_URL,
                json=payload,
                headers={
                    "X-Modal-Token-Id": MODAL_TOKEN_ID,
                    "X-Modal-Token-Secret": MODAL_TOKEN_SECRET,
                },
            )
            resp.raise_for_status()
            result_b64 = resp.json()
            log.info("Modal GPU inpaint OK (%d images via REST)", len(images))
            return [base64.b64decode(b64) for b64 in result_b64]
    except Exception as e:
        log.error("Modal REST API failed: %s", e)
        return None


async def inpaint_batch(images: list[bytes], dilation: int = 5, radius: int = 10) -> list[bytes] | None:
    return inpaint_batch_sync(images, dilation, radius)

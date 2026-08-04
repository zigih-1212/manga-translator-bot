import asyncio
import json
import os
from pathlib import Path
from aiohttp import web

from editor import store, engine

EDITOR_PORT = int(os.environ.get("EDITOR_PORT", "8090"))
STATIC_DIR = Path(__file__).parent / "static"

async def index_handler(request):
    return web.FileResponse(STATIC_DIR / "index.html")

async def list_chapters_handler(request):
    chapters = store.list_chapters()
    return web.json_response(chapters)

async def get_page_meta_handler(request):
    manga_id = request.query.get("manga")
    chapter = request.query.get("chapter")
    page = int(request.query.get("page"))
    if not all([manga_id, chapter, page is not None]):
        raise web.HTTPBadRequest(reason="Missing manga, chapter, or page query parameters")
    
    page_meta = store.load_page(manga_id, chapter, page)
    if not page_meta:
        raise web.HTTPNotFound(reason="Page metadata not found")
    return web.json_response(page_meta)

async def get_page_image_handler(request):
    manga_id = request.query.get("manga")
    chapter = request.query.get("chapter")
    page = int(request.query.get("page"))
    kind = request.query.get("kind", "out") # src or out
    if not all([manga_id, chapter, page is not None]):
        raise web.HTTPBadRequest(reason="Missing manga, chapter, or page query parameters")

    page_paths = store._paths(manga_id, chapter, page)
    img_path = page_paths.get(kind)

    if not img_path or not img_path.exists():
        raise web.HTTPNotFound(reason=f"{kind.capitalize()} image for page not found")
    
    return web.FileResponse(img_path)

async def edit_page_handler(request):
    data = await request.json()
    manga_id = data.get("manga_id")
    chapter = data.get("chapter")
    page = data.get("page")
    bubble_id = data.get("bubble_id")
    ru_text = data.get("ru")
    if not all([manga_id, chapter, page is not None, bubble_id is not None, ru_text is not None]):
        raise web.HTTPBadRequest(reason="Missing data for edit")
    
    store.save_edits(manga_id, chapter, page, bubble_id, ru_text)
    return web.json_response({"status": "ok"})

async def rerender_page_handler(request):
    data = await request.json()
    manga_id = data.get("manga_id")
    chapter = data.get("chapter")
    page = data.get("page")
    edits = data.get("edits", [])
    if not all([manga_id, chapter, page is not None]):
        raise web.HTTPBadRequest(reason="Missing manga, chapter, or page data")
    
    try:
        new_out_data = await engine.rerender_page(manga_id, chapter, page, edits)
        # Optionally, save new_out_data to store, but engine.rerender_page already does it
        return web.Response(body=new_out_data, content_type="image/png")
    except ValueError as e:
        raise web.HTTPNotFound(reason=str(e))
    except Exception as e:
        raise web.HTTPInternalServerError(reason=f"Error rerendering page: {e}")

async def download_zip_handler(request):
    manga_id = request.query.get("manga")
    chapter = request.query.get("chapter")
    if not all([manga_id, chapter]):
        raise web.HTTPBadRequest(reason="Missing manga or chapter query parameters")
    
    zip_data = store.build_zip(manga_id, chapter)
    response = web.Response(body=zip_data, content_type="application/zip")
    response.headers["Content-Disposition"] = f"attachment; filename=\"{manga_id}_{chapter}.zip\""
    return response

async def _build_app():
    app = web.Application()
    app.router.add_get("/", index_handler)
    app.router.add_get("/api/chapters", list_chapters_handler)
    app.router.add_get("/api/page", get_page_meta_handler)
    app.router.add_get("/img", get_page_image_handler)
    app.router.add_post("/api/page/edit", edit_page_handler)
    app.router.add_post("/api/page/render", rerender_page_handler)
    app.router.add_get("/api/zip", download_zip_handler)
    app.router.add_static("/static", STATIC_DIR)
    return app


async def start_editor_server(port: int = EDITOR_PORT):
    """Запустить веб-редактор в фоне (для вызова из bot.main)."""
    app = await _build_app()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    return runner


async def main():
    runner = await start_editor_server()
    print(f"Starting editor server on http://0.0.0.0:{EDITOR_PORT}")
    # Keep the server running indefinitely
    while True:
        await asyncio.sleep(3600)

if __name__ == "__main__":
    asyncio.run(main())

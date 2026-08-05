# Lazy imports to avoid aiogram load time at module import
start_router = None
titles_router = None
translate_router = None
status_router = None
manga_info_router = None
manga_translate_router = None


def _load_routers():
    global start_router, titles_router, translate_router, status_router, manga_info_router, manga_translate_router
    if start_router is not None:
        return
    from .start import router as start_router
    from .titles import router as titles_router
    from .translate import router as translate_router
    from .status import router as status_router
    from .manga_info import router as manga_info_router
    from .manga_translate import router as manga_translate_router


def get_routers():
    """Load and return all routers. Call once in main()."""
    _load_routers()
    return [
        start_router, titles_router, translate_router,
        status_router, manga_info_router, manga_translate_router
    ]


__all__ = ["get_routers"]

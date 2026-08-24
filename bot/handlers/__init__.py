# Lazy imports to avoid aiogram load time at module import
_routers: list | None = None


def _load_routers():
    global _routers
    if _routers is not None:
        return
    from .start import router as start_router
    from .titles import router as titles_router
    from .translate import router as translate_router
    from .status import router as status_router
    from .manga_info import router as manga_info_router
    from .manga_translate import router as manga_translate_router
    from .controls import router as controls_router

    _routers = [
        start_router, titles_router, translate_router,
        status_router, manga_info_router, manga_translate_router,
        controls_router,  # /cancel /retry /stats
    ]


def get_routers():
    """Load and return all routers. Call once in main()."""
    _load_routers()
    return _routers


__all__ = ["get_routers"]

from .start import router as start_router
from .titles import router as titles_router
from .translate import router as translate_router
from .status import router as status_router

__all__ = ["start_router", "titles_router", "translate_router", "status_router"]

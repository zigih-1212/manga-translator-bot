from aiogram import BaseMiddleware
from aiogram.types import Message
from aiogram.fsm.context import FSMContext
from typing import Callable, Awaitable, Any
import logging

log = logging.getLogger("middleware")


class CommandResetState(BaseMiddleware):
    async def __call__(
        self,
        handler: Callable[[Message, dict[str, Any]], Awaitable[Any]],
        event: Message,
        data: dict[str, Any],
    ) -> Any:
        try:
            if event.text and event.text.startswith("/"):
                state: FSMContext | None = data.get("state")
                if state is not None:
                    current_state = await state.get_state()
                    if current_state is not None:
                        await state.clear()
        except Exception as e:
            log.warning("CommandResetState error: %s", e)
        return await handler(event, data)
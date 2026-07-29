from aiogram import Router
from aiogram.types import Message
from aiogram.filters import Command

router = Router()


@router.message(Command("status"))
async def cmd_status(message: Message):
    from bot.handlers.translate import active_tasks
    user_id = message.from_user.id

    if user_id in active_tasks and not active_tasks[user_id].done():
        await message.answer("Перевод в процессе...")
    else:
        await message.answer("Нет активных задач.")

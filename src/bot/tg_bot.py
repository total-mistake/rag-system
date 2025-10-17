from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler,
    filters, ContextTypes, CallbackQueryHandler
)
from .bot_rag_adapter import RAGAdapter

# Сохраняем контекст сессии для пользователя
user_context = {}
adapter = RAGAdapter()

# --- Команда /start ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_context[update.effective_user.id] = {"last_question": None}
    await update.message.reply_text(
        "Привет! Я RAG-бот. Напиши мне вопрос, и я постараюсь ответить.\n"
    )

# Обработка вопроса
async def handle_question(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    question = update.message.text

    await update.message.reply_text("Думаю над ответом...")

    answer = adapter.answer_question(question)

    user_context[user_id]["last_question"] = question
    user_context[user_id]["last_answer"] = answer
    
    keyboard = [
        [InlineKeyboardButton("Больше информации об ответе", callback_data="more_info")]
    ]
    await update.message.reply_text(answer, reply_markup=InlineKeyboardMarkup(keyboard))

async def show_modules(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    # кнопки с выбором модулей
    keyboard = [
        [InlineKeyboardButton(name, callback_data=f"module_{key}")]
        for key, name in adapter.MODULE_NAMES.items()
    ]
    keyboard.append([InlineKeyboardButton("Назад к ответу", callback_data="back_to_answer")])

    await query.edit_message_text(
        "Выбери модуль, информацию о котором хочешь посмотреть:",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

async def show_params(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    module_key = query.data.split("_")[1]
    params = adapter.PARAM_NAMES.get(module_key, {})

    keyboard = [
        [InlineKeyboardButton(name, callback_data=f"param_{module_key}_{param_key}")]
        for param_key, name in params.items()
    ]
    keyboard.append([InlineKeyboardButton("Назад", callback_data="more_info")])
    keyboard.append([InlineKeyboardButton("Назад к ответу", callback_data="back_to_answer")])

    await query.edit_message_text(
        f"{adapter.MODULE_NAMES[module_key]} — выбери параметр:",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

async def show_param_value(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    _, module_key, param_key = query.data.split("_", 2)
    text = adapter.format_debug_info(module_key, param_key)

    keyboard = [[InlineKeyboardButton("Назад", callback_data=f"module_{module_key}")]]
    keyboard.append([InlineKeyboardButton("Назад к ответу", callback_data="back_to_answer")])
    await query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(keyboard))

async def back_to_answer(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    user_id = update.effective_user.id
    answer = user_context[user_id].get("last_answer", "Ответ не найден.")
    await query.edit_message_text(text=answer)
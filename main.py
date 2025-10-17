from src.bot.tg_bot import *
from src.config.settings import settings

def main():
    app = ApplicationBuilder().token(settings.telegram_token).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_question))
    app.add_handler(CallbackQueryHandler(show_modules, pattern="^more_info$"))
    app.add_handler(CallbackQueryHandler(show_params, pattern="^module_"))
    app.add_handler(CallbackQueryHandler(show_param_value, pattern="^param_"))
    app.add_handler(CallbackQueryHandler(back_to_answer, pattern="^back_to_answer$"))

    print("Бот запущен.")
    app.run_polling()

if __name__ == "__main__":
    main()
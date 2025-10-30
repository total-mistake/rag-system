from src.bot.tg_bot import *
from src.config.settings import settings
from datetime import datetime
import logging

# Настройка корневого логгера
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/program_log_{datetime.now().strftime("%Y-%m-%d_%H:%M")}.txt', mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# Отключаем логи телеграм бота
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('telegram').setLevel(logging.WARNING)
logging.getLogger('telegram.ext').setLevel(logging.WARNING)

# Создаем логгер для main
logger = logging.getLogger(__name__)

def main():
    
    app = ApplicationBuilder().token(settings.telegram_token).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_question))
    app.add_handler(CallbackQueryHandler(show_modules, pattern="^more_info$"))
    app.add_handler(CallbackQueryHandler(show_params, pattern="^module_"))
    app.add_handler(CallbackQueryHandler(show_param_value, pattern="^param_"))
    app.add_handler(CallbackQueryHandler(back_to_answer, pattern="^back_to_answer$"))

    logger.info("Бот запущен и готов к работе")
    app.run_polling()

if __name__ == "__main__":
    main()
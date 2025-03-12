from digital_assistant_first.telegram_system.telegram_rag import EnhancedRAGSystem
from digital_assistant_first.telegram_system.telegram_data_initializer import update_telegram_messages
from digital_assistant_first.telegram_system.telegram_data_initializer import TelegramManager

    
async def fetch_telegram_data(user_input, rag_system, k):  
    """Асинхронная функция для получения данных из Telegram."""
    # Предполагаем, что метод query может быть тяжелым, но он не использует I/O,
    # поэтому мы его не переделываем в асинхронный, а просто вызываем
    telegram_results, context = rag_system.query(user_input, k=k)

    telegram_context = "\n\n".join([
        f"Категория: {result['category']}\n"
        f"Источник: {result['metadata']['channel']}\n"
        f"Дата: {result['metadata']['date']}\n"
        f"Текст: {result['text']}\n"
        f"Ссылка: {result['metadata']['link']}"
        for result in telegram_results
    ])

    return telegram_context
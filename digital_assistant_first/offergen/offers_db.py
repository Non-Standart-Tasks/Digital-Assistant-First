# Создайте этот файл для тестов
class OffersDatabase:
    def __init__(self, config):
        self.config = config
        # Заглушка для тестирования - не инициализируем реальное соединение
        
    def get_offers_by_category(self, category):
        # Мок-имплементация для тестов
        if category == "Рестораны":
            return [{"id": "offer1", "name": "Тестовое предложение 1", "category": "Рестораны", "city": "Москва"}]
        elif category == "Шопинг":
            return [{"id": "offer2", "name": "Тестовое предложение 2", "category": "Шопинг", "city": "Санкт-Петербург"}]
        return []
        
    def search_offers(self, query, city=None):
        # Мок-имплементация для тестов
        results = []
        
        if "ресторан" in query.lower():
            results.append({"id": "offer1", "name": "Тестовое предложение 1", "description": "Описание", "category": "Рестораны", "city": "Москва"})
        
        if "магазин" in query.lower() or "шопинг" in query.lower():
            results.append({"id": "offer2", "name": "Тестовое предложение 2", "description": "Описание", "category": "Шопинг", "city": "Санкт-Петербург"})
            
        if not results:
            results = [
                {"id": "offer1", "name": "Тестовое предложение 1", "description": "Описание", "category": "Рестораны", "city": "Москва"},
                {"id": "offer2", "name": "Тестовое предложение 2", "description": "Описание", "category": "Шопинг", "city": "Санкт-Петербург"}
            ]
            
        if city:
            return [offer for offer in results if offer["city"] == city]
        return results 
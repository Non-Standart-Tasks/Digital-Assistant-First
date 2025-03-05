import json
from pathlib import Path

def analyze_restaurant_request(user_input, model):
    """
    Отправляет запрос к LLM, который анализирует, требуется ли в запросе
    рекомендация ресторана и определяет категорию.
    Возвращает словарь, например:
      {"restaurant_recommendation": "true", "category": "русская кухня"}
      или {"restaurant_recommendation": "false"}
    """
    # Пример простого промпта. Его можно детализировать под конкретные нужды.
    prompt = (
        "Анализируя запрос пользователя ниже, ответь в формате JSON. "
        "Если пользователь запрашивает рекомендацию ресторана, поставь 'restaurant_recommendation' равным 'true' "
        "и укажи наиболее релевантную категорию ресторана - вот доступные: ('русская кухня', 'итальянская кухня', 'рыба и морепродукты',"
        "'завтраки', 'веранда', 'мясные рестораны', 'японская кухня', 'винотека', 'можно с животными', 'живая музыка', 'паркинг', 'chef's table')"
        "Если запрос не касается ресторанов, верни 'restaurant_recommendation' со значением 'false'.\n\n"
        "Пример выходных данных:\n"
        '{"restaurant_recommendation": "true", "category": "русская кухня"}\n\n'
        "Запрос: " + user_input
    )

    # Отправляем запрос к LLM. В данном примере используется model.invoke(),
    # если у вас другая обёртка – адаптируйте этот метод.
    messages = [
        {
            "role": "system",
            "content": "Ты – помощник, который анализирует запросы на предмет рекомендаций ресторанов.",
        },
        {"role": "user", "content": prompt},
    ]
    response = model.invoke(messages, stream=False)

    # Извлекаем текст ответа
    if hasattr(response, "content"):
        content = response.content
    elif hasattr(response, "message"):
        content = response.message.content
    else:
        content = str(response)

    # Если ответ обёрнут в markdown-блоки, убираем их
    content = content.strip()
    if content.startswith("```json"):
        content = content[7:]
    if content.endswith("```"):
        content = content[:-3]
    try:
        analysis = json.loads(content.strip())
    except Exception as e:
        # Если получить корректный JSON не удалось – по умолчанию считаем, что ресторанов не запрашивают
        analysis = {"restaurant_recommendation": "false"}
    return analysis


def get_restaurants_by_category(category, json_path=None):
    """
    Загружает локальный JSON с ресторанами и возвращает отфильтрованный список
    по введенной категории.
    """
    if json_path is None:
        json_path = Path(__file__).parent.parent.parent / "content" / "restaurants.json"
    
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        restaurants = data.get("restaurants", [])
    except Exception as e:
        print("No yandex-restaurants file in system", e)
        restaurants = []
    
    # Пример фильтра: если категория входит в список ресторанных категорий (без учёта регистра)
    filtered = []
    for restaurant in restaurants:
        categories = restaurant.get("categories", [])
        for cat in categories:
            if category.lower() in cat.lower():
                filtered.append(restaurant)
                break  # если нашли совпадение, переходим к следующему ресторану

    return filtered

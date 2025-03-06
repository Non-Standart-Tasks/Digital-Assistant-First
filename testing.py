from pydantic import BaseModel
from typing import Any, Optional
import streamlit as st

# Импорты функций для получения ключей и поиска
from digital_assistant_first.utils.check_serp_response import APIKeyManager
from digital_assistant_first.internet_search import search_shopping, search_places

# Инициализация менеджера ключей
serpapi_key_manager = APIKeyManager(path_to_file="api_keys_status.csv")

# Модель входных данных пользователя
class UserInput(BaseModel):
    content: str

# Модель для собранной информации
class CollectedInfo(BaseModel):
    user_input: str
    shopping_res: Any
    internet_res: Any
    links: Optional[Any] = None

def collect_information(user_input: UserInput) -> CollectedInfo:
    """
    Функция собирает информацию через агента:
      - Получает API-ключ через serpapi_key_manager.
      - Выполняет поиск для shopping и интернет-результатов.
      - Возвращает объект CollectedInfo с исходным запросом и результатами.
    """
    # Получаем лучший API-ключ
    _, serpapi_key = serpapi_key_manager.get_best_api_key()
    
    # Вызываем функции поиска
    shopping_res = search_shopping(user_input.content, serpapi_key)
    internet_res, links, _ = search_places(user_input.content, serpapi_key)
    
    # Возвращаем результаты в виде модели CollectedInfo
    return CollectedInfo(
        user_input=user_input.content,
        shopping_res=shopping_res,
        internet_res=internet_res,
        links=links
    )

def main():
    # Извлекаем последний запрос пользователя из состояния Streamlit
    if "messages" in st.session_state and st.session_state["messages"]:
        last_message = st.session_state["messages"][-1]["content"]
        user_input = UserInput(content=last_message)
        
        # Сбор информации через агента
        info = collect_information(user_input)
        
        # Отображаем результат (в виде словаря)
        st.write("Собранная информация:", info.dict())
    else:
        st.warning("Сообщения пользователя не найдены в session_state.")

if __name__ == "__main__":
    main()
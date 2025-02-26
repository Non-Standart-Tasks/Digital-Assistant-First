from streamlit.testing.v1 import AppTest
import re

timeout = 45


# Функция для поиска и обработки ключей ID и WIDGET_ID
def find_id_widget_keys(text, patterns):
    id_dict = {}
    for pattern in patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            if ": " in match:
                key, value = match.split(": ", 1)
                id_dict[key] = None  # Добавляем ключ в словарь с пустым значением
    return id_dict


# Проверка опций selectbox
def test_selectbox_options():
    at = AppTest.from_file("streamlit_app.py").run()
    assert set(at.selectbox[0].options).issuperset(
        ["gpt-4o", "gpt-4o-mini"]
    ), "Опции моделей неправильны"

    # Проверяем первый selectbox "Выберите тип системы"
    assert at.selectbox[1].options == [
        "default",
        "RAG",
        "File",
    ], "Опции первого selectbox не совпадают"

    # Проверяем второй selectbox "Выберите тип цепочки"
    assert at.selectbox[2].options == [
        "refine",
        "map_reduce",
        "stuff",
    ], "Опции второго selectbox не совпадают"

    # Проверяем третий selectbox "Выберите модель эмбеддингов"
    assert at.selectbox[3].options == [
        "BAAI/bge-m3",
        "sentence-transformers/all-MiniLM-L6-v2",
        "OpenAIEmbeddings",
    ], "Опции третьего selectbox не совпадают"


# Тест для выбора конфигурации
def test_config_selection():
    at = AppTest.from_file("streamlit_app.py").run()

    # Устанавливаем значения в selectbox
    selectbox_values = ["gpt-4o", "RAG", "refine", "OpenAIEmbeddings"]
    for i, value in enumerate(selectbox_values):
        at.selectbox[i].set_value(value).run()

    # Нажимаем на первую кнопку и запускаем приложение заново
    assert at.button[0].click().run(timeout=timeout)


# Общая логика для поиска ключей и их обработки
def key_processing(at):
    # Преобразуем session_state в строку
    text = str(at.session_state)

    # Паттерны для поиска ключей ID и WIDGET_ID
    patterns = [r"\$\$WIDGET_ID-[\w\d-]+-None: [^,}]+", r"\$\$ID-[\w\d-]+-None: [^,}]+"]

    # Находим ключи и значения
    id_dict = find_id_widget_keys(text, patterns)

    # Нажимаем на первую кнопку и запускаем приложение заново
    at.button[0].click().run(timeout=timeout)

    # Присваиваем значения ключам из id_dict
    for id_key, val in id_dict.items():
        at.session_state[id_key] = val


# Тест для RAG функции
def test_RAG_function():
    at = AppTest.from_file("streamlit_app.py").run()

    # Устанавливаем значения в selectbox
    selectbox_values = ["gpt-4o-mini", "RAG", "refine", "OpenAIEmbeddings"]
    for i, value in enumerate(selectbox_values):
        at.selectbox[i].set_value(value).run()

    # Обрабатываем ключи
    key_processing(at)

    # Устанавливаем значение в чат для ввода и запускаем
    assert at.chat_input[0].set_value("Выведи все сокращения").run(timeout=timeout)

    # Проверяем результат
    result = at.chat_input[0].set_value("Выведи все сокращения").run(timeout=timeout)

    # Получаем все сообщения ассистента
    assistant_messages = [
        message["content"]
        for message in result.session_state.messages
        if message["role"] == "assistant"
    ]

    # Список слов для проверки
    words_to_check = ["ДККиФМ", "АПД", "ДБ", "ЗНИ"]

    # Проверяем, что все слова из списка содержатся в одном из сообщений ассистента
    assert any(
        all(word in message for word in words_to_check)
        for message in assistant_messages
    ), "Не все слова найдены в одном сообщении ассистента"


# Тест для default функции
def test_default_function():
    at = AppTest.from_file("streamlit_app.py").run()

    # Устанавливаем значения в selectbox
    selectbox_values = ["gpt-4o-mini", "default", "refine", "OpenAIEmbeddings"]
    for i, value in enumerate(selectbox_values):
        at.selectbox[i].set_value(value).run()

    # Обрабатываем ключи
    key_processing(at)

    # Устанавливаем значение в чат для ввода и запускаем
    assert at.chat_input[0].set_value("Выведи все сокращения").run(timeout=timeout)

    # Проверяем результат
    result = at.chat_input[0].set_value("Выведи все сокращения").run(timeout=timeout)

    # Получаем все сообщения ассистента
    assistant_messages = [
        message["content"]
        for message in result.session_state.messages
        if message["role"] == "assistant"
    ]

    # Список слов для проверки
    words_to_check = ["ДККиФМ", "АПД", "ДБ", "ЗНИ"]

    # Проверяем, что ни одно сообщение ассистента не содержит всех слов из списка
    assert not any(
        all(word in message for word in words_to_check)
        for message in assistant_messages
    ), "Некоторое сообщение содержит все слова"

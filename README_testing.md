# Тестирование цифрового ассистента

В этом проекте реализованы тесты для следующих компонентов:

1. Основные функции приложения Streamlit (`test_streamlit_app.py`)
2. Функции базы данных (`test_database.py`)
3. Интерфейсные функции (`test_interface.py`)
4. Интеграционные тесты Streamlit (`test_streamlit_integration.py`)
5. Docker контейнеризация (`test_docker.py` и `test_docker_integration.py`)

## Структура тестов

- `conftest.py` - общие фикстуры и настройки для всех тестов
- `test_config.yaml` - конфигурационный файл для тестирования
- `test_streamlit_app.py` - модульные тесты функций из streamlit_app.py
- `test_database.py` - тесты функций работы с базой данных
- `test_interface.py` - тесты интерфейсных функций 
- `test_streamlit_integration.py` - интеграционные тесты Streamlit приложения
- `test_docker.py` - тесты для Docker контейнеризации
- `test_docker_integration.py` - интеграционные тесты запущенного приложения в Docker

## Запуск тестов

### Установка зависимостей для тестирования

```bash
pip install pytest pytest-mock pytest-asyncio requests
```

### Запуск всех тестов

```bash
python -m pytest
```

### Запуск конкретного модуля тестов

```bash
python -m pytest test_streamlit_app.py
python -m pytest test_database.py
python -m pytest test_interface.py
python -m pytest test_streamlit_integration.py
```

### Запуск тестов Docker

```bash
# Только тесты сборки и запуска Docker с подробным выводом
python -m pytest test_docker.py -v

# Интеграционные тесты с запущенным приложением
python -m pytest test_docker_integration.py -m integration -v

# Запуск теста для CI-окружения
CI=true python -m pytest test_docker.py::TestDockerCompose::test_docker_image_contents -v
```

### Запуск с подробным выводом

```bash
python -m pytest -v
```

### Запуск с покрытием кода тестами

```bash
pip install pytest-cov
python -m pytest --cov=streamlit_app --cov=digital_assistant_first
```

## Автоматическая очистка Docker ресурсов

Тесты Docker настроены на автоматическую очистку всех созданных ресурсов (контейнеров и образов) после завершения. Это обеспечивается с помощью:

1. Fixture `docker_cleanup` в `test_docker.py`
2. Fixture `ensure_cleanup` в `test_docker_integration.py`

Обе фикстуры имеют атрибут `autouse=True`, что гарантирует их запуск даже без явного указания в тестах. Процесс очистки включает:

- Остановку всех контейнеров с `docker compose down`
- Определение созданных образов через фильтр `*digital-assistant*`
- Удаление образов (сначала стандартное, затем принудительное)
- Проверку, что все образы были успешно удалены

Если по какой-то причине вам нужно вручную очистить образы, используйте:

```bash
# Остановить все контейнеры
docker compose down

# Удалить образы, созданные для проекта
docker images --filter "reference=*digital-assistant*" --format "{{.Repository}}:{{.Tag}}" | xargs -r docker rmi -f
```

## Рекомендации по разработке тестов

1. **Модульные тесты** - тестируйте отдельные функции в isolation, используя моки для внешних зависимостей.
2. **Интеграционные тесты** - проверяйте взаимодействие между компонентами.
3. **Стандарт именования тестов** - используйте формат `test_что_тестируется_при_каких_условиях`.

## Streamlit Testing Framework

Для интеграционного тестирования используется Streamlit Testing Framework:

```python
from streamlit.testing.v1 import AppTest

# Инициализация приложения
at = AppTest.from_file("streamlit_app.py")

# Запуск приложения
at.run()

# Проверка взаимодействия
at.chat_input[0].set_value("Привет").run()
```

### Проверка компонентов интерфейса

```python
# Проверка заголовка
assert "Цифровой Помощник AMA" in at.title[0].value

# Проверка боковой панели
assert len(at.sidebar.radio) > 0
```

## Тесты Docker

Тесты Docker проверяют:

1. **Сборка контейнера** - правильность выполнения `docker compose build`
2. **Запуск контейнера** - успешный запуск с помощью `docker compose up`
3. **Доступность приложения** - возможность отправки HTTP-запросов к запущенному приложению
4. **Проверка логов** - отсутствие ошибок в логах контейнера
5. **Проверка ресурсов** - корректное использование ресурсов контейнером

### Требования к Docker тестам

Для запуска тестов Docker:

1. Установленный Docker и docker compose
2. Файл `docker-compose.yml` в корне проекта
3. Доступный порт 8501 (стандартный порт Streamlit)
4. Права на запуск Docker команд от имени текущего пользователя 
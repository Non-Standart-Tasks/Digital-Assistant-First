FROM python:3.10.14-slim

# Установка системных зависимостей
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Установка Poetry
ENV POETRY_VERSION=2.1.1
RUN pip install "poetry==$POETRY_VERSION"

# Рабочая директория
WORKDIR /app

# Копируем зависимости
COPY pyproject.toml poetry.lock ./

COPY . .

# Установка зависимостей
RUN poetry lock && poetry env use python && poetry install --no-interaction

# Команда запуска
CMD ["poetry", "run", "streamlit", "run", "streamlit_app.py", "--server.port=8501"]
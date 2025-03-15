# Stage 1: Build
FROM python:3.11.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install "poetry==2.1.1"

WORKDIR /app
COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-root

# Stage 2: Runtime
FROM python:3.11.11-slim

# Устанавливаем только базовые необходимые пакеты (если нужны)
RUN apt-get update && apt-get install -y --no-install-recommends \
    procps \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY . .

CMD poetry run streamlit run streamlit_app.py --server.port $STREAMLIT_SERVER_PORT --server.address 0.0.0.0
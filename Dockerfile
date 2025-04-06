# Stage 1: Build
FROM python:3.11.11-slim AS builder

ENV UV_LINK_MODE=copy
ENV UV_CACHE_DIR=/root/.cache/uv
VOLUME /root/.cache/uv

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-pip \
    && pip install uv \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync

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

CMD uv run streamlit run streamlit_app.py --server.port $STREAMLIT_SERVER_PORT --server.address 0.0.0.0
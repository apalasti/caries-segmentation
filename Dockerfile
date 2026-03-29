FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

COPY pyproject.toml /app/pyproject.toml
COPY uv.lock /app/uv.lock
RUN uv sync --no-dev

COPY dataset.tar.gz /app/data/dataset.tar.gz
RUN tar -xzvf /app/data/dataset.tar.gz -C /app/data/ && rm -f /app/data/dataset.tar.gz

COPY config.toml /app/config.toml

COPY scripts/ /app/scripts
COPY src/ /app/src

ENV PYTHONPATH=/app

CMD ["uv", "run", "python", "-m", "src.train"]
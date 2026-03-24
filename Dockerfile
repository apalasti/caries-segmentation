FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

COPY pyproject.toml .
RUN uv sync --no-dev

ADD dataset.tar.gz /app/data/
COPY config.toml /app/config.toml

COPY scripts/ /app/scripts
COPY src/ /app/src

ENV PYTHONPATH=/app

CMD ["uv", "run", "python", "-m", "src.train"]
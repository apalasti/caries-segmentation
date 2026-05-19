FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# OpenCV runtime dependencies for slim Debian images.
RUN apt-get update \
	&& apt-get install -y --no-install-recommends \
		libgl1 \
		libglib2.0-0 \
		libgomp1 \
		libsm6 \
		libxext6 \
		libxrender1 \
	&& rm -rf /var/lib/apt/lists/*

COPY pyproject.toml /app/pyproject.toml
# Resolve inside the container to avoid local lock/platform issues.
RUN uv sync --no-dev

# Ensure only the headless OpenCV variant is present (transitive deps may
# pull in the GUI version despite the pyproject override).
RUN uv pip uninstall --python /app/.venv/bin/python opencv-python 2>/dev/null || true \
	&& uv pip install --python /app/.venv/bin/python opencv-python-headless==4.10.0.84

# Fail image build early if OpenCV import is broken.
RUN uv run python -c "import cv2, albumentations; print(cv2.__version__)"

COPY dataset.tar.gz /app/data/dataset.tar.gz
RUN tar -xzvf /app/data/dataset.tar.gz -C /app/data/ && rm -f /app/data/dataset.tar.gz

COPY config.toml /app/config.toml

COPY scripts/ /app/scripts
COPY src/ /app/src

ENV PYTHONPATH=/app

CMD ["uv", "run", "python", "-m", "src.train"]
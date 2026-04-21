# ── build stage: install dependencies with uv ────────────────────────────────
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock ./

# Install production deps into a virtual env inside /app/.venv
RUN uv sync --frozen --no-dev --no-install-project

# ── runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

# PyQt6 wheels bundle Qt, but still need these at runtime if Qt ever loads
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libdbus-1-3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy venv from builder, then the full project source
COPY --from=builder /app/.venv /app/.venv
COPY . .

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

# Default: run the FastAPI server.
# Override CMD in docker-compose to run the RQ worker instead.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

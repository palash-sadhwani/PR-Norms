FROM python:3.12-slim
WORKDIR /app
ARG INSTALL_PHASE2=true
COPY pyproject.toml .
RUN pip install --no-cache-dir ".[dev]"
# CPU-only torch keeps the image small; Phase 2 mining and PR analysis need it.
RUN if [ "$INSTALL_PHASE2" = "true" ]; then \
      pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
      && pip install --no-cache-dir ".[phase2]"; \
    fi
COPY app app
COPY alembic.ini .
COPY alembic alembic
COPY scripts scripts
COPY tests tests
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000"]

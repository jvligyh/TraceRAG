FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TRACERAG_DATA_DIR=/app/data/runtime HF_HOME=/app/data/models
WORKDIR /app
COPY pyproject.toml README.md constraints.txt ./
COPY src ./src
RUN pip install --no-cache-dir torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -c constraints.txt '.[local-translation]'
RUN useradd --create-home --uid 10001 tracerag && mkdir -p data/runtime data/models && chown -R tracerag:tracerag /app
USER tracerag
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=3)"
CMD ["python", "-m", "tracerag.cli", "serve", "--host", "0.0.0.0"]

# syntax=docker/dockerfile:1
# Prediction API image. Build after `dvc repro` so models/ holds the trained model.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
COPY requirements/serve.txt requirements/serve.txt
# Optional secret `pip_ca`: a CA bundle for builds behind a TLS-inspecting proxy
# (docker build --secret id=pip_ca,src=/path/to/ca.crt ...). Not stored in the image.
RUN --mount=type=secret,id=pip_ca,required=false \
    if [ -f /run/secrets/pip_ca ]; then export PIP_CERT=/run/secrets/pip_ca; fi; \
    pip install -r requirements/serve.txt

# --no-deps: runtime deps come from the pins above (xgboost-cpu, not xgboost).
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps . && rm -rf src
# xgboost-cpu bundles its own OpenMP, so no system packages are needed.

COPY models/cascade.joblib models/model_info.json models/

RUN useradd --system --uid 10001 app
USER app

ENV MODEL_PATH=/app/models/cascade.joblib
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "milk_adulteration.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

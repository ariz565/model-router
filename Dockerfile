# Production gateway image. Build once; select gateway/control/worker at
# runtime by overriding CMD in the orchestrator.
FROM python:3.12-slim AS build

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src
# Deliberately excludes heavyweight synthetic ML extras. Production service
# dependencies remain explicit and reproducible at image-build time.
RUN python -m pip install --upgrade pip && \
    python -m pip wheel --wheel-dir /wheels ".[server,contracts,redis,postgres,kafka,aws,crypto,sso]"

FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MODELROUTER_STORAGE=postgres \
    MODELROUTER_REQUIRE_MIGRATIONS=true
WORKDIR /app

RUN groupadd --gid 10001 modelrouter && \
    useradd --uid 10001 --gid modelrouter --create-home --home-dir /app --shell /usr/sbin/nologin modelrouter
COPY --from=build /wheels /wheels
RUN python -m pip install --no-cache-dir /wheels/* && rm -rf /wheels
COPY --chown=modelrouter:modelrouter src ./src

USER modelrouter
EXPOSE 8000
# urllib is in the stdlib, unlike curl/wget in minimal images. Readiness is
# intentionally used here: liveness alone must not advertise an unmigrated app.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1:8000/ready', timeout=3)"

CMD ["python", "-m", "uvicorn", "modelrouter.api.gateway:app", "--host", "0.0.0.0", "--port", "8000"]

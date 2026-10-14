# 공개 데모 이미지: AWS Lambda(arm64) + Lambda Web Adapter. docs/deploy-plan.md 5단계.
# 빌드는 scripts/deploy.sh가 한다(docker buildx build --platform linux/arm64 --provenance=false).
# 키는 이미지에 넣지 않는다. 실행할 때 SSM Parameter Store에서 읽는다(ANTHROPIC_API_KEY_SSM_PARAM).

FROM public.ecr.aws/docker/library/python:3.12-slim-bookworm AS build
COPY --from=ghcr.io/astral-sh/uv:0.12.21 /uv /bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
# GGML_NATIVE=OFF: 빌드한 맥의 CPU 기능에 맞추지 않는다(Lambda의 Graviton에서 돌아야 한다).
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never CMAKE_ARGS="-DGGML_NATIVE=OFF"
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --extra gguf-embed --extra aws --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev --extra gguf-embed --extra aws
COPY scripts/fetch_gguf.py ./scripts/
RUN .venv/bin/python scripts/fetch_gguf.py /opt/models

FROM public.ecr.aws/docker/library/python:3.12-slim-bookworm
# llama.cpp가 OpenMP를 쓴다
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:1.1.0 /lambda-adapter /opt/extensions/lambda-adapter
COPY --from=build /app /app
COPY --from=build /opt/models /opt/models
WORKDIR /app
COPY data/clean ./data/clean
COPY data/entities.json ./data/entities.json
COPY data/index/local__mykor__KURE-v1-gguf__KURE-v1-Q8_0.gguf ./data/index/local__mykor__KURE-v1-gguf__KURE-v1-Q8_0.gguf
ENV PATH="/app/.venv/bin:$PATH" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 \
    AWS_LWA_PORT=8080 AWS_LWA_READINESS_CHECK_PATH=/api/config \
    PUBLIC_DEMO=1 LLM_PROVIDER=anthropic EMBED_PROVIDER=local RETRIEVE_K=20 GGUF_MODELS_DIR=/opt/models
CMD ["sfqa", "web", "--port", "8080"]

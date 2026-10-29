# Default: small CPU image that runs `make test` / `make smoke` (what CI checks).
# GPU image for the full run (needs the NVIDIA container toolkit):
#   docker build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu126 -t repro-cifar10:gpu .
#   docker run --gpus all -v "$PWD/results:/app/results" repro-cifar10:gpu make repro
FROM python:3.11-slim

ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 PY=/app/.venv/bin/python

RUN apt-get update \
    && apt-get install -y --no-install-recommends make \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Heavy layer first so code edits don't re-download torch.
RUN python -m venv .venv \
    && .venv/bin/pip install --upgrade pip \
    && .venv/bin/pip install torch torchvision --index-url "${TORCH_INDEX}"

COPY . .
RUN .venv/bin/pip install -e ".[dev]"

CMD ["make", "test"]

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    YOLO_CONFIG_DIR=/tmp/ultralytics

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY embeddedexample ./embeddedexample
# Install CPU-only PyTorch explicitly. This avoids pulling several gigabytes of
# CUDA libraries into the default simulation image.
RUN python -m pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.10.0 torchvision==0.25.0 \
    && python -m pip install --no-cache-dir .

RUN mkdir -p /app/runs /app/datasets /app/.flwr /tmp/matplotlib /tmp/ultralytics

COPY docker/config.toml /app/.flwr/config.toml
COPY docker/submit.py /app/docker/submit.py
COPY config.yaml /app/config.yaml

ENV FLWR_HOME=/app/.flwr

CMD ["flower-superlink", "--insecure"]

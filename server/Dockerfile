# OpenOne gate service: vLLM 0.31 + the app's shared core. Model package is mounted read-only at /model.
ARG BASE=vllm/vllm-openai@sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3
FROM ${BASE}
WORKDIR /app
COPY core /app/core
COPY server/gate_server.py /app/gate_server.py
ENV PYTHONPATH=/app/core:/app PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 OPENONE_MODEL_DIR=/model VLLM_NO_USAGE_STATS=1
EXPOSE 8130
ENTRYPOINT ["python3", "-m", "uvicorn", "gate_server:app", "--host", "0.0.0.0", "--port", "8130", "--app-dir", "/app"]

#!/usr/bin/env bash
# Build and (re)start the OpenOne gate service container.
#   ./run_gate.sh PACKAGE_DIR [GPU_INDEX] [PORT]
# The API key lives in ~/openone-gate/gate.key (created on first run, mode 600); it is never printed.
set -euo pipefail
PKG=$(readlink -f "$1"); GPU=${2:-0}; PORT=${3:-8130}; NAME=$(basename "$PKG")
SRC=$(cd "$(dirname "$0")/.." && pwd)
ROOT=${OPENONE_GATE_HOME:-$HOME/openone-gate}
mkdir -p "$ROOT/cache"
if [ ! -s "$ROOT/gate.key" ]; then (umask 077; head -c 32 /dev/urandom | base64 | tr -d '/+=\n' > "$ROOT/gate.key"); fi
docker build -q -t openone-gate:0.1 -f "$SRC/server/Dockerfile" "$SRC" >/dev/null
docker rm -f openone-gate >/dev/null 2>&1 || true
docker run -d --name openone-gate --restart unless-stopped --gpus "device=$GPU" -p "0.0.0.0:$PORT:8130" \
  -v "$PKG:/models/$NAME:ro" -e OPENONE_MODEL_DIR="/models/$NAME" -v "$ROOT/cache:/cache" -v "$ROOT/gate.key:/run/gate.key:ro" \
  -e OPENONE_GATE_KEY_FILE=/run/gate.key -e OPENONE_VLLM_MEM=${OPENONE_VLLM_MEM:-0.45} ${OPENONE_SETTINGS_FILE:+-v "$OPENONE_SETTINGS_FILE:/run/settings.json:ro" -e OPENONE_SETTINGS_FILE=/run/settings.json} \
  ${OPENONE_MODE:+-e OPENONE_MODE=$OPENONE_MODE} openone-gate:0.1 >/dev/null
echo "started openone-gate on port $PORT (GPU $GPU); waiting for /health ..."
for i in $(seq 1 180); do
  if curl -fsS "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok":true'; then echo ready; exit 0; fi
  sleep 2
done
echo "not ready after 6 min; see: docker logs openone-gate" >&2; exit 1

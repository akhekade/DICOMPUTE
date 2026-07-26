#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export DICO_AUTH_DISABLED=true
export DICO_DATA_DIR="${DICO_DATA_DIR:-$ROOT/.mesh-data}"
mkdir -p "$DICO_DATA_DIR"

cleanup() {
  jobs -p | xargs -r kill 2>/dev/null || true
}
trap cleanup EXIT

echo "Starting coordinator on :7400 ..."
dico coordinator --port 7400 --auth-disabled --data-dir "$DICO_DATA_DIR" &
sleep 1

echo "Starting WS provider ..."
dico join --coordinator http://127.0.0.1:7400 --transport websocket &
sleep 1

echo "Starting HTTP provider on :7402 ..."
dico join --coordinator http://127.0.0.1:7400 --port 7402 --transport http &
sleep 1

echo "Training round ..."
dico train --coordinator http://127.0.0.1:7400 --epochs 2 --samples 64

echo "Inference ..."
dico infer --coordinator http://127.0.0.1:7400 \
  --features 0.1,-0.2,0.3,0.4,-0.5,0.6,0.1,-0.3 --routing cheapest

echo "Status:"
dico status --coordinator http://127.0.0.1:7400

echo "Mesh demo complete. Coordinator still running — Ctrl+C to stop."
wait

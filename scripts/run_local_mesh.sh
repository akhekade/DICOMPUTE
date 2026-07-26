#!/usr/bin/env bash
# Spin up a coordinator + two providers on localhost for a quick demo.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export PATH="${HOME}/.local/bin:${PATH}"
if [[ -d .venv ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

COORD_PORT="${COORD_PORT:-7400}"
P1_PORT="${P1_PORT:-7401}"
P2_PORT="${P2_PORT:-7402}"
COORD_URL="http://127.0.0.1:${COORD_PORT}"

cleanup() {
  [[ -n "${COORD_PID:-}" ]] && kill "$COORD_PID" 2>/dev/null || true
  [[ -n "${P1_PID:-}" ]] && kill "$P1_PID" 2>/dev/null || true
  [[ -n "${P2_PID:-}" ]] && kill "$P2_PID" 2>/dev/null || true
}
trap cleanup EXIT

PYTHON="${PYTHON:-python3}"
"$PYTHON" -m dico.cli coordinator --port "$COORD_PORT" &
COORD_PID=$!
sleep 1
"$PYTHON" -m dico.cli join --coordinator "$COORD_URL" --port "$P1_PORT" --node-id mini-a &
P1_PID=$!
"$PYTHON" -m dico.cli join --coordinator "$COORD_URL" --port "$P2_PORT" --node-id mini-b &
P2_PID=$!
sleep 1

echo "== status =="
"$PYTHON" -m dico.cli status --coordinator "$COORD_URL"
echo "== train =="
"$PYTHON" -m dico.cli train --coordinator "$COORD_URL" --epochs 3 --samples 128
echo "== infer =="
"$PYTHON" -m dico.cli infer --coordinator "$COORD_URL" \
  --features 0.1,-0.2,0.3,0.4,-0.5,0.6,0.1,-0.3
echo
echo "Dashboard: ${COORD_URL}/  (Ctrl+C to stop)"
wait

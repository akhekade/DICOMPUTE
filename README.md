# DICO — Distributed Compute Mesh

Production-oriented distributed computing infrastructure for **Mac Minis and laptops**. Machines join as providers, participate in federated training, and serve **collective AI inference** through a Darkbloom-inspired control plane.

White paper: [`docs/WHITEPAPER.md`](docs/WHITEPAPER.md) · Architecture: [`docs/architecture/overview.md`](docs/architecture/overview.md)

## What you get (v0.2)

- **Coordinator control plane** — auth, registry, cost-aware scheduling, FedAvg, checkpoints
- **Outbound WebSocket providers** — workers dial out (NAT-friendly); HTTP LAN mode still available
- **Prepaid billing** — micro-USD reserve → settle ledger with refunds
- **OpenAI-compatible API** — `POST /v1/chat/completions`, `/v1/models`, `/v1/balance`
- **Persistence** — SQLite store for keys, ledger, usage, model checkpoints
- **Observability** — Prometheus-style `/metrics`, structured outcomes (client/provider/billing)
- **Dashboard** — live cluster view at the coordinator URL

The default model is still a small NumPy MLP so the mesh runs without PyTorch/MLX. The protocol, scheduler, and store are ready for heavier backends.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Terminal 1 — coordinator (local demo: auth off)
dico coordinator --port 7400 --auth-disabled

# Terminal 2 — provider (outbound WebSocket — no inbound port)
dico join --coordinator http://127.0.0.1:7400 --transport websocket

# Train + infer + OpenAI chat
dico train --coordinator http://127.0.0.1:7400 --epochs 5
dico infer --coordinator http://127.0.0.1:7400 \
  --features 0.1,-0.2,0.3,0.4,-0.5,0.6,0.1,-0.3 --routing cheapest
dico chat --coordinator http://127.0.0.1:7400 --message "classify this signal"
dico status --coordinator http://127.0.0.1:7400
```

Dashboard: `http://127.0.0.1:7400/` · Metrics: `http://127.0.0.1:7400/metrics`

```bash
dico demo   # full multi-terminal recipe
```

## Production mode

```bash
export DICO_AUTH_DISABLED=false
export DICO_BOOTSTRAP_API_KEY="sk-dico-your-admin-secret"
export DICO_DATA_DIR=/var/lib/dico
export DICO_LOG_JSON=true

dico coordinator --host 0.0.0.0 --port 7400 --data-dir /var/lib/dico

# Providers (any network path to the coordinator)
dico join --coordinator https://coord.example.com --transport websocket

# Consumers
export DICO_API_KEY=sk-dico-...
dico infer --coordinator https://coord.example.com \
  --features 0.1,-0.2,0.3,0.4,-0.5,0.6,0.1,-0.3 --api-key "$DICO_API_KEY"
```

Create additional keys:

```bash
dico create-key --coordinator https://coord.example.com \
  --api-key "$DICO_BOOTSTRAP_API_KEY" --name team-a --role consumer
```

## Mac Mini LAN setup

1. Coordinator: `dico coordinator --port 7400 --auth-disabled`
2. Note LAN IP (`ipconfig getifaddr en0` on macOS)
3. On each Mac Mini / laptop:

```bash
dico join --coordinator http://192.168.x.x:7400 --transport websocket
```

Apple Silicon nodes advertise `has_metal=true` for future MLX backends.

## Architecture

```
Client / OpenAI SDK
        │  API key + reserve/settle
        ▼
┌───────────────────┐     outbound WS / HTTP
│   Coordinator     │◄──────────────────────┐
│ auth·bill·sched   │                       │
│ FedAvg·checkpoint │                       │
└─────────┬─────────┘                       │
          │ dispatch                        │
          ▼                                 │
   ┌────────────┐  ┌────────────┐  ┌────────────┐
   │ Mac Mini A │  │ Mac Mini B │  │  Laptop C  │
   └────────────┘  └────────────┘  └────────────┘
```

Inspired by Darkbloom’s split between control plane and outbound workers — see [`docs/architecture/overview.md`](docs/architecture/overview.md).

## HTTP API (coordinator)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/status` | Cluster inventory |
| `GET` | `/metrics` | Prometheus text metrics |
| `POST` | `/nodes/register` | HTTP provider join |
| `POST` | `/nodes/heartbeat` | Liveness + capacity |
| `WS` | `/ws/provider` | Outbound provider protocol |
| `POST` | `/infer` | Collective inference |
| `POST` | `/train/round` | Federated round |
| `GET` | `/model` | Global weights + checksum |
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat |
| `GET` | `/v1/models` | Model list |
| `GET` | `/v1/balance` | Prepaid balance |
| `POST` | `/admin/keys` | Create API key (admin) |
| `POST` | `/admin/deposit` | Credit an account (admin) |

## Configuration

All settings accept `DICO_` environment variables (see `src/dico/config.py`):

| Variable | Default | Meaning |
|----------|---------|---------|
| `DICO_AUTH_DISABLED` | `false` | Skip API keys (demos) |
| `DICO_BOOTSTRAP_API_KEY` | — | Seed admin key |
| `DICO_DATA_DIR` | `./data` | SQLite + checkpoints |
| `DICO_STORE_BACKEND` | `sqlite` | `sqlite` or `memory` |
| `DICO_BILLING_ENABLED` | `true` | Reserve/settle ledger |
| `DICO_PRICE_INFER_MICRO_USD` | `100` | Cost per inference unit |
| `DICO_RATE_LIMIT_RPM` | `120` | Per-key rate limit |
| `DICO_HEARTBEAT_TTL_S` | `30` | Stale provider eviction |

## Docker

```bash
docker build -t dicompute -f deploy/Dockerfile .
docker run --rm -p 7400:7400 -e DICO_AUTH_DISABLED=true dicompute
```

## Development

```bash
pip install -e ".[dev]"
pytest -q
```

## Roadmap

- MLX / llama.cpp backends for real LLM shards on Apple Silicon
- Provider attestation ladder (`self_signed` / `hardware`)
- Hop-by-hop request sealing
- mDNS auto-discovery
- Token incentives / XDC settlement (white paper)
- Stripe deposits + provider payouts

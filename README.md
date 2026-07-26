# DICO — Distributed Compute Mesh

Basic distributed computing infrastructure for **Mac Minis and laptops**. Any machine can join as an infrastructure provider, participate in federated training, and serve **collective AI inference**.

White paper: [`docs/WHITEPAPER.md`](docs/WHITEPAPER.md)

## What you get

- **Coordinator** — bootstrap node that tracks providers, routes ensemble inference, and runs FedAvg
- **Provider nodes** — Mac Minis / laptops that join the mesh, train locally, and answer inference
- **Collective intelligence** — after a training round, the global model is averaged across contributors and queries can ensemble live nodes
- **Dashboard** — live cluster view at the coordinator URL

This MVP uses a small NumPy MLP so it runs without PyTorch/MLX. The mesh protocol is ready to swap in heavier backends later.

## Quick start (one Mac or several)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# Terminal 1 — coordinator
dico coordinator --port 7400

# Terminal 2 — provider (same machine or another Mac Mini on the LAN)
dico join --coordinator http://<coordinator-lan-ip>:7400 --port 7401

# Federated training + collective inference
dico train --coordinator http://<coordinator-lan-ip>:7400 --epochs 5
dico infer --coordinator http://<coordinator-lan-ip>:7400 \
  --features 0.1,-0.2,0.3,0.4,-0.5,0.6,0.1,-0.3
dico status --coordinator http://<coordinator-lan-ip>:7400
```

Open the dashboard: `http://<coordinator-lan-ip>:7400/`

Print the full local multi-node recipe:

```bash
dico demo
```

## Mac Mini LAN setup

1. Pick one Mac Mini as coordinator: `dico coordinator --port 7400`
2. Note its LAN IP (System Settings → Network, or `ipconfig getifaddr en0`)
3. On every other Mac Mini / laptop:

```bash
dico join --coordinator http://192.168.x.x:7400 --port 7401
```

4. Allow incoming connections for Python on the first run (macOS firewall prompt), or open TCP `7400` / `7401+` on your LAN.
5. Run `dico train` then `dico infer` against the coordinator.

Apple Silicon nodes advertise `has_metal=true` in capabilities for future MLX/Metal backends.

## Architecture

```
Laptop / API client
        │
        ▼
┌───────────────────┐      heartbeat / train / infer
│   Coordinator     │◄──────────────────────────────┐
│  registry+FedAvg  │                               │
│  ensemble infer   │                               │
└─────────┬─────────┘                               │
          │ sync weights / route work               │
          ▼                                         │
   ┌────────────┐  ┌────────────┐  ┌────────────┐   │
   │ Mac Mini A │  │ Mac Mini B │  │  Laptop C  │───┘
   │  provider  │  │  provider  │  │  provider  │
   └────────────┘  └────────────┘  └────────────┘
```

| Piece | Role |
|-------|------|
| `dico coordinator` | Registry, dashboard, FedAvg, collective `/infer` |
| `dico join` | Provider: register, heartbeat, local `/train` + `/infer` |
| `CollectiveMLP` | Tiny shared model (JSON-serializable weights) |
| FedAvg | Weighted average of local weight updates |
| Ensemble inference | Mean of probability vectors across live nodes |

## HTTP API (coordinator)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/status` | Cluster + provider inventory |
| `POST` | `/nodes/register` | Provider join |
| `POST` | `/nodes/heartbeat` | Liveness / load |
| `POST` | `/infer` | Collective inference |
| `POST` | `/train/round` | Kick federated round on all nodes |
| `GET` | `/model` | Current global weights |

Providers expose `/infer`, `/train`, `/model/sync`, `/health`.

## Development

```bash
pip install -e ".[dev]"
pytest -q
```

## Roadmap (beyond this MVP)

- MLX / llama.cpp backends for real LLM shards on Apple Silicon
- mDNS auto-discovery (Bonjour) so LAN joins need no IP
- Token incentives / XDC settlement (see white paper)
- Model / pipeline parallelism for larger nets

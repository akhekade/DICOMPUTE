# DICOMPUTE Architecture Overview

DICOMPUTE is a distributed compute mesh for Mac Minis and laptops. The control
plane is modeled after production private-inference networks (notably
[Darkbloom / Layr-Labs d-inference](https://github.com/Layr-Labs/d-inference)):
a coordinator owns auth, scheduling, billing, and registry; providers connect
**outbound** and advertise live capacity.

## System diagram

```
Consumer (CLI / OpenAI SDK / Dashboard)
        │  HTTPS + API key
        ▼
┌──────────────────────────────────────────┐
│  Coordinator (control plane)             │
│  • API key auth + rate limits            │
│  • Prepaid reserve → settle ledger       │
│  • Provider registry + cost scheduler    │
│  • FedAvg + model checkpoints (SQLite)   │
│  • OpenAI-compatible /v1/*               │
│  • WebSocket hub /ws/provider            │
└───────────────┬──────────────────────────┘
                │ outbound WS (preferred) or HTTP
     ┌──────────┼──────────┐
     ▼          ▼          ▼
 Provider A  Provider B  Provider C
 (Mac Mini)  (laptop)    (Mac Mini)
```

## Components

| Component | Path | Role |
|-----------|------|------|
| Coordinator | `src/dico/coordinator.py` | Control plane |
| Provider | `src/dico/node.py` | Worker (WS or HTTP) |
| Registry / scheduler | `src/dico/registry.py` | Capacity + cost selection |
| Store | `src/dico/store/` | Memory + SQLite persistence |
| Billing | `src/dico/billing.py` | Micro-USD reserve/settle |
| Auth | `src/dico/auth.py` | `sk-dico-…` API keys |
| OpenAI façade | `src/dico/api/openai_compat.py` | `/v1/chat/completions` |
| Config | `src/dico/config.py` | `DICO_*` env settings |
| Telemetry | `src/dico/telemetry.py` | Metrics + structured logs |

## Request path (inference)

1. Consumer authenticates (`Authorization: Bearer sk-dico-…`).
2. Rate limiter checks per-key RPM.
3. Billing **reserves** worst-case micro-USD cost.
4. Scheduler selects eligible providers (capacity, cooldown, warm models).
5. Coordinator dispatches via WebSocket (preferred) or HTTP.
6. Results are ensembled (mean probabilities) or single-provider.
7. Billing **settles** actual cost and refunds the remainder.
8. Metrics record client / provider / billing outcomes separately.

## Provider transports

| Mode | How | When |
|------|-----|------|
| **WebSocket** (default) | Provider dials `ws://coord/ws/provider` | NAT / production — no inbound ports |
| **HTTP** | Provider exposes `/infer` `/train`; coordinator dials in | Simple LAN demos |

## Persistence

- SQLite by default under `DICO_DATA_DIR` (`./data/dico.db`)
- Tables: accounts, api_keys, ledger, reservations, usage, checkpoints
- Global model also written to `data/global_model.json`
- `MemoryStore` for unit tests (`DICO_STORE_BACKEND=memory`)

## Trust model (current)

Providers advertise `trust_level` (`none` \| `self_signed` \| `hardware`).
This release accepts `none` for open LAN meshes. Hardware attestation
(Secure Enclave / MDM) is intentionally out of scope for the Python MVP;
the field is reserved for a future Darkbloom-style ladder.

## What we adapted from Darkbloom

- Outbound worker connections + typed protocol envelopes
- Store interface with memory/SQLite backends
- Reserve → settle prepaid ledger
- Cost-based scheduling with capacity heartbeats
- Health ejection / error cooldowns
- OpenAI-compatible consumer API
- Namespaced config (`DICO_*`) with validation
- Separated client / provider / billing outcomes in metrics

## What remains product-specific

- Inference backend is still a NumPy `CollectiveMLP` (swap-ready for MLX/llama.cpp)
- No hop-by-hop NaCl sealing or Confidential VM coordinator yet
- No Stripe Connect payouts (ledger is local prepaid credits)

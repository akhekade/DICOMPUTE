# DICOMPUTE Threat Model (STRIDE summary)

Scoped to the Python control plane in this repository.

| Threat | Asset | Mitigation (current) | Gap / future |
|--------|-------|----------------------|--------------|
| Spoofed provider | Registry | Optional API keys; protocol version gate | Provider auth tokens + attestation |
| Spoofed consumer | API | `sk-dico-` hashed keys, rate limits | mTLS, spend caps per route |
| Tampered weights | Model | SHA-256 checksum on sync/checkpoint | Signed model manifests / CDN |
| Unpaid work | Ledger | Reserve before dispatch, settle after | Stripe deposits, provider payouts |
| Noisy neighbor | Scheduler | Capacity heartbeats, cooldowns | Per-tenant isolation |
| Stale / zombie nodes | Registry | Heartbeat TTL eviction | Challenge-response liveness |
| Prompt leakage | Inference | TLS recommended in prod | Hop-by-hop seal (Darkbloom) |
| Coordinator compromise | All plaintext at control plane | Minimize logging of features | Run coordinator in CVM/TEE |
| Model poisoning | FedAvg | Checksums only | Byzantine-robust aggregation, DP |
| DoS | API | RPM limiter | Edge WAF, admission control |

## Production checklist

- [ ] Set `DICO_AUTH_DISABLED=false`
- [ ] Set `DICO_BOOTSTRAP_API_KEY` or rotate the printed bootstrap key
- [ ] Put TLS terminator (Caddy/nginx) in front of the coordinator
- [ ] Persist `DICO_DATA_DIR` on durable disk
- [ ] Prefer `--transport websocket` for providers
- [ ] Scrape `/metrics` into Prometheus/Datadog
- [ ] Restrict `/admin/*` to private networks

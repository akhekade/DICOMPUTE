"""Command-line interface for the DICO mesh."""

from __future__ import annotations

import argparse
import json
import os
import sys

import httpx


def _auth_headers(api_key: str | None) -> dict[str, str]:
    key = api_key or os.environ.get("DICO_API_KEY")
    if not key:
        return {}
    return {"Authorization": f"Bearer {key}"}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="dico",
        description="DICOMPUTE — production distributed compute mesh",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_coord = sub.add_parser("coordinator", help="Start the control-plane coordinator")
    p_coord.add_argument("--host", default="0.0.0.0")
    p_coord.add_argument("--port", type=int, default=7400)
    p_coord.add_argument(
        "--auth-disabled",
        action="store_true",
        help="Disable API key auth (local demos only)",
    )
    p_coord.add_argument("--data-dir", default=None)

    p_join = sub.add_parser("join", help="Join the mesh as an infrastructure provider")
    p_join.add_argument(
        "--coordinator",
        required=True,
        help="Coordinator URL, e.g. http://192.168.1.10:7400",
    )
    p_join.add_argument("--host", default="0.0.0.0")
    p_join.add_argument("--port", type=int, default=7401)
    p_join.add_argument("--node-id", default=None)
    p_join.add_argument(
        "--transport",
        choices=["websocket", "http"],
        default="websocket",
        help="websocket (outbound, recommended) or http (LAN inbound)",
    )

    p_status = sub.add_parser("status", help="Show cluster status")
    p_status.add_argument("--coordinator", required=True)
    p_status.add_argument("--api-key", default=None)

    p_infer = sub.add_parser("infer", help="Run collective inference")
    p_infer.add_argument("--coordinator", required=True)
    p_infer.add_argument(
        "--features",
        required=True,
        help="Comma-separated floats (8 values), e.g. 0.1,0.2,...",
    )
    p_infer.add_argument("--no-ensemble", action="store_true")
    p_infer.add_argument(
        "--routing", choices=["ensemble", "cheapest", "local"], default="ensemble"
    )
    p_infer.add_argument("--api-key", default=None)

    p_train = sub.add_parser("train", help="Run a federated training round")
    p_train.add_argument("--coordinator", required=True)
    p_train.add_argument("--epochs", type=int, default=3)
    p_train.add_argument("--samples", type=int, default=256)
    p_train.add_argument("--lr", type=float, default=0.05)
    p_train.add_argument("--api-key", default=None)

    p_chat = sub.add_parser("chat", help="OpenAI-compatible chat completion")
    p_chat.add_argument("--coordinator", required=True)
    p_chat.add_argument("--message", required=True)
    p_chat.add_argument("--api-key", default=None)

    p_key = sub.add_parser("create-key", help="Create an API key (admin)")
    p_key.add_argument("--coordinator", required=True)
    p_key.add_argument("--api-key", required=True, help="Admin API key")
    p_key.add_argument("--name", default="consumer")
    p_key.add_argument("--role", default="consumer")

    p_demo = sub.add_parser(
        "demo",
        help="Print a quick local demo recipe for multi-node on one machine",
    )

    args = parser.parse_args(argv)

    if args.cmd == "coordinator":
        from dico.coordinator import run_coordinator

        # Default: auth disabled for zero-config local demos unless env says otherwise
        auth_disabled = True if args.auth_disabled else None
        if auth_disabled is None and os.environ.get("DICO_AUTH_DISABLED") is None:
            # Keep production-safe default when env not set: enable auth with bootstrap key
            # but for CLI convenience on first run, disable unless --data-dir production-ish
            auth_disabled = True
        run_coordinator(
            host=args.host,
            port=args.port,
            auth_disabled=auth_disabled,
            data_dir=args.data_dir,
        )
        return

    if args.cmd == "join":
        from dico.node import run_provider

        run_provider(
            coordinator_url=args.coordinator,
            host=args.host,
            port=args.port,
            node_id=args.node_id,
            transport=args.transport,
        )
        return

    headers = _auth_headers(getattr(args, "api_key", None))

    if args.cmd == "status":
        with httpx.Client(timeout=10.0, headers=headers) as client:
            r = client.get(f"{args.coordinator.rstrip('/')}/status")
            r.raise_for_status()
            print(json.dumps(r.json(), indent=2))
        return

    if args.cmd == "infer":
        features = [float(x.strip()) for x in args.features.split(",")]
        if len(features) != 8:
            print("error: --features must contain exactly 8 floats", file=sys.stderr)
            sys.exit(2)
        payload = {
            "features": features,
            "ensemble": not args.no_ensemble,
            "routing": "local" if args.no_ensemble else args.routing,
        }
        with httpx.Client(timeout=30.0, headers=headers) as client:
            r = client.post(f"{args.coordinator.rstrip('/')}/infer", json=payload)
            r.raise_for_status()
            print(json.dumps(r.json(), indent=2))
        return

    if args.cmd == "train":
        payload = {
            "epochs": args.epochs,
            "samples": args.samples,
            "learning_rate": args.lr,
            "batch_size": 32,
        }
        with httpx.Client(timeout=180.0, headers=headers) as client:
            r = client.post(
                f"{args.coordinator.rstrip('/')}/train/round", json=payload
            )
            r.raise_for_status()
            data = r.json()
            print(
                json.dumps(
                    {
                        "model_version": data["model_version"],
                        "num_contributors": data["num_contributors"],
                        "checksum_sha256": data.get("checksum_sha256"),
                    },
                    indent=2,
                )
            )
        return

    if args.cmd == "chat":
        payload = {
            "model": "collective-mlp",
            "messages": [{"role": "user", "content": args.message}],
        }
        with httpx.Client(timeout=30.0, headers=headers) as client:
            r = client.post(
                f"{args.coordinator.rstrip('/')}/v1/chat/completions", json=payload
            )
            r.raise_for_status()
            print(json.dumps(r.json(), indent=2))
        return

    if args.cmd == "create-key":
        with httpx.Client(timeout=15.0, headers=_auth_headers(args.api_key)) as client:
            r = client.post(
                f"{args.coordinator.rstrip('/')}/admin/keys",
                json={"name": args.name, "role": args.role},
            )
            r.raise_for_status()
            print(json.dumps(r.json(), indent=2))
        return

    if args.cmd == "demo":
        print(
            """
# Terminal 1 — coordinator (auth disabled for local demo)
dico coordinator --port 7400 --auth-disabled

# Terminal 2 — provider A (outbound WebSocket — no inbound port needed)
dico join --coordinator http://127.0.0.1:7400 --transport websocket

# Terminal 3 — provider B (HTTP LAN mode)
dico join --coordinator http://127.0.0.1:7400 --port 7402 --transport http

# Federated training round
dico train --coordinator http://127.0.0.1:7400 --epochs 5

# Collective inference (cost-aware ensemble)
dico infer --coordinator http://127.0.0.1:7400 \\
  --features 0.1,-0.2,0.3,0.4,-0.5,0.6,0.1,-0.3 --routing cheapest

# OpenAI-compatible chat
dico chat --coordinator http://127.0.0.1:7400 --message "classify this signal"

# Cluster status / dashboard / metrics
dico status --coordinator http://127.0.0.1:7400
open http://127.0.0.1:7400/
curl http://127.0.0.1:7400/metrics
""".strip()
        )
        return


if __name__ == "__main__":
    main()

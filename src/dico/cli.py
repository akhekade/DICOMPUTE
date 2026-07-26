"""Command-line interface for the DICO mesh."""

from __future__ import annotations

import argparse
import json
import sys

import httpx


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="dico",
        description="DICOMPUTE — distributed compute mesh for Mac Minis & laptops",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_coord = sub.add_parser("coordinator", help="Start a coordinator (bootstrap node)")
    p_coord.add_argument("--host", default="0.0.0.0")
    p_coord.add_argument("--port", type=int, default=7400)

    p_join = sub.add_parser("join", help="Join the mesh as an infrastructure provider")
    p_join.add_argument(
        "--coordinator",
        required=True,
        help="Coordinator URL, e.g. http://192.168.1.10:7400",
    )
    p_join.add_argument("--host", default="0.0.0.0")
    p_join.add_argument("--port", type=int, default=7401)
    p_join.add_argument("--node-id", default=None)

    p_status = sub.add_parser("status", help="Show cluster status")
    p_status.add_argument("--coordinator", required=True)

    p_infer = sub.add_parser("infer", help="Run collective inference")
    p_infer.add_argument("--coordinator", required=True)
    p_infer.add_argument(
        "--features",
        required=True,
        help="Comma-separated floats (8 values), e.g. 0.1,0.2,...",
    )
    p_infer.add_argument("--no-ensemble", action="store_true")

    p_train = sub.add_parser("train", help="Run a federated training round")
    p_train.add_argument("--coordinator", required=True)
    p_train.add_argument("--epochs", type=int, default=3)
    p_train.add_argument("--samples", type=int, default=256)
    p_train.add_argument("--lr", type=float, default=0.05)

    p_demo = sub.add_parser(
        "demo",
        help="Print a quick local demo recipe for multi-node on one machine",
    )

    args = parser.parse_args(argv)

    if args.cmd == "coordinator":
        from dico.coordinator import run_coordinator

        run_coordinator(host=args.host, port=args.port)
        return

    if args.cmd == "join":
        from dico.node import run_provider

        run_provider(
            coordinator_url=args.coordinator,
            host=args.host,
            port=args.port,
            node_id=args.node_id,
        )
        return

    if args.cmd == "status":
        with httpx.Client(timeout=10.0) as client:
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
        }
        with httpx.Client(timeout=30.0) as client:
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
        with httpx.Client(timeout=180.0) as client:
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
                    },
                    indent=2,
                )
            )
        return

    if args.cmd == "demo":
        print(
            """
# Terminal 1 — coordinator (any Mac Mini / laptop)
dico coordinator --port 7400

# Terminal 2 — provider A
dico join --coordinator http://127.0.0.1:7400 --port 7401

# Terminal 3 — provider B
dico join --coordinator http://127.0.0.1:7400 --port 7402

# Federated training round (all nodes train locally, FedAvg merges)
dico train --coordinator http://127.0.0.1:7400 --epochs 5

# Collective inference (ensemble across online providers)
dico infer --coordinator http://127.0.0.1:7400 \\
  --features 0.1,-0.2,0.3,0.4,-0.5,0.6,0.1,-0.3

# Cluster status / dashboard
dico status --coordinator http://127.0.0.1:7400
open http://127.0.0.1:7400/
""".strip()
        )
        return


if __name__ == "__main__":
    main()

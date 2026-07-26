"""Coordinator: node registry, job routing, FedAvg, collective inference."""

from __future__ import annotations

import asyncio
import time
import uuid
from pathlib import Path

import httpx
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from dico.hardware import local_ip, probe_capabilities
from dico.model import CollectiveMLP, federated_average
from dico.protocol import (
    AggregateRequest,
    AggregateResponse,
    ClusterStatus,
    CollectiveInferenceResponse,
    Heartbeat,
    InferenceRequest,
    InferenceResult,
    NodeInfo,
    NodeRole,
    RegisterRequest,
    TrainLocalRequest,
    WeightUpdate,
)

HEARTBEAT_TTL_S = 30.0
WEB_DIR = Path(__file__).resolve().parents[2] / "web"


class CoordinatorState:
    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self.node_id = f"coord-{uuid.uuid4().hex[:8]}"
        self.nodes: dict[str, NodeInfo] = {}
        self.model = CollectiveMLP()
        self.pending_updates: list[WeightUpdate] = []
        self.lock = asyncio.Lock()
        caps = probe_capabilities()
        self.self_info = NodeInfo(
            node_id=self.node_id,
            role=NodeRole.COORDINATOR,
            host=host,
            port=port,
            capabilities=caps,
            status="online",
            model_version=self.model.version,
            last_heartbeat=time.time(),
        )

    def prune_stale(self) -> None:
        now = time.time()
        stale = [
            nid
            for nid, n in self.nodes.items()
            if now - n.last_heartbeat > HEARTBEAT_TTL_S
        ]
        for nid in stale:
            del self.nodes[nid]

    def cluster_status(self) -> ClusterStatus:
        self.prune_stale()
        providers = list(self.nodes.values())
        return ClusterStatus(
            coordinator_id=self.node_id,
            model_version=self.model.version,
            nodes=[self.self_info, *providers],
            total_cpu_cores=sum(n.capabilities.cpu_cores for n in providers),
            total_memory_gb=round(
                sum(n.capabilities.memory_gb for n in providers), 2
            ),
            online_providers=len(providers),
        )


def create_coordinator_app(state: CoordinatorState) -> FastAPI:
    app = FastAPI(title="DICO Coordinator", version="0.1.0")

    @app.get("/health")
    async def health() -> dict:
        return {"ok": True, "role": "coordinator", "node_id": state.node_id}

    @app.get("/status", response_model=ClusterStatus)
    async def status() -> ClusterStatus:
        return state.cluster_status()

    @app.post("/nodes/register")
    async def register(req: RegisterRequest) -> dict:
        async with state.lock:
            node = req.node
            node.last_heartbeat = time.time()
            node.status = "online"
            # Push current global weights on join
            state.nodes[node.node_id] = node
        return {
            "ok": True,
            "model_version": state.model.version,
            "weights": state.model.export_weights(),
            "coordinator_id": state.node_id,
        }

    @app.post("/nodes/heartbeat")
    async def heartbeat(hb: Heartbeat) -> dict:
        async with state.lock:
            node = state.nodes.get(hb.node_id)
            if not node:
                raise HTTPException(404, "unknown node; register first")
            node.last_heartbeat = time.time()
            node.load = hb.load
            node.model_version = hb.model_version
            node.status = hb.status
        return {"ok": True, "model_version": state.model.version}

    @app.delete("/nodes/{node_id}")
    async def leave(node_id: str) -> dict:
        async with state.lock:
            state.nodes.pop(node_id, None)
        return {"ok": True}

    @app.get("/model")
    async def get_model() -> dict:
        return {
            "model_version": state.model.version,
            "weights": state.model.export_weights(),
        }

    @app.post("/infer", response_model=CollectiveInferenceResponse)
    async def infer(req: InferenceRequest) -> CollectiveInferenceResponse:
        state.prune_stale()
        request_id = req.request_id or uuid.uuid4().hex
        contributors: list[InferenceResult] = []

        # Always include coordinator-local inference as a baseline.
        t0 = time.perf_counter()
        pred, probs = state.model.predict(req.features)
        contributors.append(
            InferenceResult(
                request_id=request_id,
                node_id=state.node_id,
                prediction=pred,
                probabilities=probs,
                model_version=state.model.version,
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
        )

        providers = list(state.nodes.values())
        if req.ensemble and providers:
            async with httpx.AsyncClient(timeout=req.timeout_s) as client:
                tasks = [
                    client.post(
                        f"{n.endpoint}/infer",
                        json={
                            "features": req.features,
                            "request_id": request_id,
                            "ensemble": False,
                        },
                    )
                    for n in providers
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)
            for node, result in zip(providers, results):
                if isinstance(result, Exception):
                    continue
                if result.status_code != 200:
                    continue
                contributors.append(InferenceResult(**result.json()))

        # Weighted average of probability vectors (equal weight per live node).
        mats = [np.asarray(c.probabilities, dtype=np.float64) for c in contributors]
        avg = np.mean(np.stack(mats, axis=0), axis=0)
        prediction = float(np.argmax(avg))
        return CollectiveInferenceResponse(
            request_id=request_id,
            prediction=prediction,
            probabilities=[float(v) for v in avg],
            model_version=state.model.version,
            contributors=contributors,
            strategy="ensemble_mean" if len(contributors) > 1 else "local",
        )

    @app.post("/train/round")
    async def train_round(req: TrainLocalRequest) -> AggregateResponse:
        """Ask all providers (and self) to train locally, then FedAvg."""
        state.prune_stale()
        updates: list[WeightUpdate] = []

        # Local training on coordinator
        local = await _local_train(state, req)
        updates.append(local)

        providers = list(state.nodes.values())
        if providers:
            payload = req.model_dump()
            async with httpx.AsyncClient(timeout=120.0) as client:
                tasks = [
                    client.post(f"{n.endpoint}/train", json=payload)
                    for n in providers
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception) or result.status_code != 200:
                    continue
                updates.append(WeightUpdate(**result.json()))

        if not updates:
            raise HTTPException(503, "no successful training updates")

        averaged = federated_average(
            [(u.weights, u.num_samples) for u in updates]
        )
        state.model.load_weights(averaged)
        state.model.version = int(averaged["version"][0])
        state.self_info.model_version = state.model.version

        # Push new global model to providers
        async with httpx.AsyncClient(timeout=30.0) as client:
            push_tasks = [
                client.post(
                    f"{n.endpoint}/model/sync",
                    json={"model_version": state.model.version, "weights": averaged},
                )
                for n in providers
            ]
            await asyncio.gather(*push_tasks, return_exceptions=True)

        return AggregateResponse(
            model_version=state.model.version,
            num_contributors=len(updates),
            weights=averaged,
        )

    @app.post("/train/submit", response_model=AggregateResponse)
    async def submit_update(update: WeightUpdate) -> AggregateResponse:
        """Accept a single provider update and fold it into the global model."""
        async with state.lock:
            state.pending_updates.append(update)
            averaged = federated_average(
                [
                    (state.model.export_weights(), 1),
                    (update.weights, update.num_samples),
                ]
            )
            state.model.load_weights(averaged)
            state.model.version = int(averaged["version"][0])
            state.self_info.model_version = state.model.version
            version = state.model.version
            weights = state.model.export_weights()
        return AggregateResponse(
            model_version=version,
            num_contributors=1,
            weights=weights,
        )

    @app.post("/aggregate", response_model=AggregateResponse)
    async def aggregate(req: AggregateRequest) -> AggregateResponse:
        if not req.updates:
            raise HTTPException(400, "updates required")
        averaged = federated_average(
            [(u.weights, u.num_samples) for u in req.updates]
        )
        state.model.load_weights(averaged)
        state.model.version = int(averaged["version"][0])
        state.self_info.model_version = state.model.version
        return AggregateResponse(
            model_version=state.model.version,
            num_contributors=len(req.updates),
            weights=averaged,
        )

    # Dashboard
    index = WEB_DIR / "index.html"
    if WEB_DIR.is_dir():
        static = WEB_DIR / "static"
        if static.is_dir():
            app.mount("/static", StaticFiles(directory=str(static)), name="static")

        @app.get("/")
        async def dashboard() -> FileResponse:
            return FileResponse(index)

    return app


async def _local_train(state: CoordinatorState, req: TrainLocalRequest) -> WeightUpdate:
    from dico.model import make_synthetic_dataset

    x, y = make_synthetic_dataset(
        n=req.samples,
        input_dim=state.model.input_dim,
        output_dim=state.model.output_dim,
        seed=int(time.time()) % 10_000,
    )
    loss = state.model.train_batch(
        x, y, lr=req.learning_rate, epochs=req.epochs, batch_size=req.batch_size
    )
    return WeightUpdate(
        node_id=state.node_id,
        model_version=state.model.version,
        num_samples=req.samples,
        weights=state.model.export_weights(),
        loss=loss,
    )


def run_coordinator(host: str = "0.0.0.0", port: int = 7400) -> None:
    import uvicorn

    bind_host = host if host != "0.0.0.0" else local_ip()
    state = CoordinatorState(host=bind_host, port=port)
    app = create_coordinator_app(state)
    print(f"DICO coordinator {state.node_id} listening on http://{bind_host}:{port}")
    print(f"Dashboard: http://{bind_host}:{port}/")
    uvicorn.run(app, host=host, port=port, log_level="info")

"""Provider node: joins the mesh, trains locally, serves inference."""

from __future__ import annotations

import asyncio
import time
import uuid
from contextlib import asynccontextmanager

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException

from dico.hardware import local_ip, probe_capabilities
from dico.model import CollectiveMLP, make_synthetic_dataset
from dico.protocol import (
    Heartbeat,
    InferenceRequest,
    InferenceResult,
    NodeInfo,
    NodeRole,
    RegisterRequest,
    TrainLocalRequest,
    WeightUpdate,
)


class ProviderState:
    def __init__(
        self,
        coordinator_url: str,
        host: str,
        port: int,
        node_id: str | None = None,
    ) -> None:
        self.coordinator_url = coordinator_url.rstrip("/")
        self.host = host
        self.port = port
        self.node_id = node_id or f"node-{uuid.uuid4().hex[:8]}"
        self.model = CollectiveMLP()
        self.capabilities = probe_capabilities()
        self.active_tasks = 0
        self._stop = asyncio.Event()

    @property
    def info(self) -> NodeInfo:
        load = self.active_tasks / max(1, self.capabilities.max_concurrent_tasks)
        return NodeInfo(
            node_id=self.node_id,
            role=NodeRole.PROVIDER,
            host=self.host,
            port=self.port,
            capabilities=self.capabilities,
            status="online",
            model_version=self.model.version,
            load=load,
            last_heartbeat=time.time(),
        )

    async def register(self) -> None:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                f"{self.coordinator_url}/nodes/register",
                json=RegisterRequest(node=self.info).model_dump(),
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("weights"):
                self.model.load_weights(data["weights"])
                self.model.version = int(data.get("model_version", 0))
            print(
                f"Joined mesh as {self.node_id} "
                f"(model v{self.model.version}) via {self.coordinator_url}"
            )

    async def heartbeat_loop(self) -> None:
        while not self._stop.is_set():
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    await client.post(
                        f"{self.coordinator_url}/nodes/heartbeat",
                        json=Heartbeat(
                            node_id=self.node_id,
                            load=self.info.load,
                            model_version=self.model.version,
                            status="online",
                        ).model_dump(),
                    )
            except Exception as exc:
                print(f"heartbeat failed: {exc}")
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=10.0)
            except asyncio.TimeoutError:
                pass

    async def leave(self) -> None:
        self._stop.set()
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                await client.delete(f"{self.coordinator_url}/nodes/{self.node_id}")
        except Exception:
            pass


def create_provider_app(state: ProviderState) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await state.register()
        hb_task = asyncio.create_task(state.heartbeat_loop())
        yield
        await state.leave()
        hb_task.cancel()
        try:
            await hb_task
        except asyncio.CancelledError:
            pass

    app = FastAPI(title="DICO Provider", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict:
        return {
            "ok": True,
            "role": "provider",
            "node_id": state.node_id,
            "model_version": state.model.version,
            "load": state.info.load,
            "capabilities": state.capabilities.model_dump(),
        }

    @app.post("/infer", response_model=InferenceResult)
    async def infer(req: InferenceRequest) -> InferenceResult:
        state.active_tasks += 1
        try:
            t0 = time.perf_counter()
            pred, probs = state.model.predict(req.features)
            return InferenceResult(
                request_id=req.request_id or uuid.uuid4().hex,
                node_id=state.node_id,
                prediction=pred,
                probabilities=probs,
                model_version=state.model.version,
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
        finally:
            state.active_tasks -= 1

    @app.post("/train", response_model=WeightUpdate)
    async def train(req: TrainLocalRequest) -> WeightUpdate:
        state.active_tasks += 1
        try:
            x, y = make_synthetic_dataset(
                n=req.samples,
                input_dim=state.model.input_dim,
                output_dim=state.model.output_dim,
                seed=(hash(state.node_id) ^ int(time.time())) % 10_000,
            )
            loss = state.model.train_batch(
                x,
                y,
                lr=req.learning_rate,
                epochs=req.epochs,
                batch_size=req.batch_size,
            )
            return WeightUpdate(
                node_id=state.node_id,
                model_version=state.model.version,
                num_samples=req.samples,
                weights=state.model.export_weights(),
                loss=float(loss),
            )
        finally:
            state.active_tasks -= 1

    @app.post("/model/sync")
    async def sync_model(payload: dict) -> dict:
        weights = payload.get("weights")
        if not weights:
            raise HTTPException(400, "weights required")
        state.model.load_weights(weights)
        state.model.version = int(payload.get("model_version", state.model.version))
        return {"ok": True, "model_version": state.model.version}

    return app


def run_provider(
    coordinator_url: str,
    host: str = "0.0.0.0",
    port: int = 7401,
    node_id: str | None = None,
) -> None:
    advertise = host if host != "0.0.0.0" else local_ip()
    state = ProviderState(
        coordinator_url=coordinator_url,
        host=advertise,
        port=port,
        node_id=node_id,
    )
    app = create_provider_app(state)
    print(f"DICO provider {state.node_id} on http://{advertise}:{port}")
    print(f"Joining coordinator at {coordinator_url}")
    uvicorn.run(app, host=host, port=port, log_level="info")

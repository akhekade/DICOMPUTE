"""Provider node: outbound WebSocket (preferred) or HTTP join to the mesh."""

from __future__ import annotations

import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException

from dico.checksum import weights_checksum
from dico.config import AppConfig, load_config
from dico.hardware import local_ip, probe_capabilities
from dico.model import CollectiveMLP, make_synthetic_dataset
from dico.protocol import (
    PROTOCOL_VERSION,
    BackendCapacity,
    Heartbeat,
    InferenceRequest,
    InferenceResult,
    NodeInfo,
    NodeRole,
    RegisterRequest,
    TrainLocalRequest,
    WeightUpdate,
    WsEnvelope,
)
from dico.telemetry import get_logger, setup_logging

log = get_logger("dico.provider")


class ProviderState:
    def __init__(
        self,
        coordinator_url: str,
        host: str,
        port: int,
        node_id: str | None = None,
        *,
        transport: str = "websocket",
        cfg: AppConfig | None = None,
    ) -> None:
        self.cfg = cfg or load_config()
        self.coordinator_url = coordinator_url.rstrip("/")
        self.host = host
        self.port = port
        self.node_id = node_id or f"node-{uuid.uuid4().hex[:8]}"
        self.model = CollectiveMLP()
        self.capabilities = probe_capabilities()
        self.capabilities.software_version = "0.2.0"
        self.active_tasks = 0
        self._stop = asyncio.Event()
        self.transport = transport  # websocket | http
        self._ws: Any = None

    def capacity(self) -> BackendCapacity:
        max_slots = self.capabilities.max_concurrent_tasks
        free = max(0, max_slots - self.active_tasks)
        return BackendCapacity(
            max_slots=max_slots,
            free_slots=free,
            queued=0,
            warm_models=["collective-mlp"],
            memory_pressure=min(1.0, self.active_tasks / max(1, max_slots)),
            cpu_util=min(1.0, self.active_tasks / max(1, max_slots)),
        )

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
            transport="websocket" if self.transport == "websocket" else "http",
            capacity=self.capacity(),
            protocol_version=PROTOCOL_VERSION,
        )

    def apply_weights(self, weights: dict, version: int | None = None) -> None:
        self.model.load_weights(weights)
        if version is not None:
            self.model.version = int(version)

    async def register_http(self) -> None:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                f"{self.coordinator_url}/nodes/register",
                json=RegisterRequest(node=self.info).model_dump(),
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("weights"):
                self.apply_weights(data["weights"], data.get("model_version"))
            log.info(
                "Joined mesh as %s (model v%s) via %s",
                self.node_id,
                self.model.version,
                self.coordinator_url,
            )

    async def heartbeat_loop_http(self) -> None:
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
                            capacity=self.capacity(),
                        ).model_dump(),
                    )
            except Exception as exc:
                log.warning("heartbeat failed: %s", exc)
            try:
                await asyncio.wait_for(
                    self._stop.wait(), timeout=self.cfg.heartbeat_interval_s
                )
            except asyncio.TimeoutError:
                pass

    async def leave_http(self) -> None:
        self._stop.set()
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                await client.delete(f"{self.coordinator_url}/nodes/{self.node_id}")
        except Exception:
            pass

    def _ws_url(self) -> str:
        base = self.coordinator_url
        if base.startswith("https://"):
            return "wss://" + base[len("https://") :] + "/ws/provider"
        if base.startswith("http://"):
            return "ws://" + base[len("http://") :] + "/ws/provider"
        return base.rstrip("/") + "/ws/provider"

    async def run_websocket_loop(self) -> None:
        """Outbound WS client — no inbound ports required on the provider."""
        try:
            import websockets
        except ImportError as exc:
            raise RuntimeError(
                "websockets package required for WS transport; pip install websockets"
            ) from exc

        url = self._ws_url()
        backoff = 1.0
        while not self._stop.is_set():
            try:
                async with websockets.connect(url, ping_interval=20) as ws:
                    self._ws = ws
                    backoff = 1.0
                    await ws.send(
                        WsEnvelope(
                            type="register",
                            payload={"node": self.info.model_dump()},
                        ).model_dump_json()
                    )
                    # pull initial model over HTTP once
                    try:
                        async with httpx.AsyncClient(timeout=15.0) as client:
                            resp = await client.get(f"{self.coordinator_url}/model")
                            if resp.status_code == 200:
                                data = resp.json()
                                if data.get("weights"):
                                    self.apply_weights(
                                        data["weights"], data.get("model_version")
                                    )
                    except Exception as exc:
                        log.warning("initial model fetch failed: %s", exc)

                    log.info(
                        "WS connected as %s → %s (model v%s)",
                        self.node_id,
                        url,
                        self.model.version,
                    )

                    hb_task = asyncio.create_task(self._ws_heartbeat(ws))
                    try:
                        async for raw in ws:
                            if self._stop.is_set():
                                break
                            await self._handle_ws_message(ws, raw)
                    finally:
                        hb_task.cancel()
                        try:
                            await hb_task
                        except asyncio.CancelledError:
                            pass
            except Exception as exc:
                if self._stop.is_set():
                    break
                log.warning("ws reconnect in %.1fs: %s", backoff, exc)
                await asyncio.sleep(backoff)
                backoff = min(30.0, backoff * 2)

    async def _ws_heartbeat(self, ws: Any) -> None:
        while not self._stop.is_set():
            await ws.send(
                WsEnvelope(
                    type="heartbeat",
                    payload=Heartbeat(
                        node_id=self.node_id,
                        load=self.info.load,
                        model_version=self.model.version,
                        status="online",
                        capacity=self.capacity(),
                    ).model_dump(),
                ).model_dump_json()
            )
            try:
                await asyncio.wait_for(
                    self._stop.wait(), timeout=self.cfg.heartbeat_interval_s
                )
            except asyncio.TimeoutError:
                pass

    async def _handle_ws_message(self, ws: Any, raw: str | bytes) -> None:
        import json

        data = json.loads(raw)
        env = WsEnvelope.model_validate(data)
        if env.type == "ack":
            return
        if env.type == "model_sync":
            weights = env.payload.get("weights")
            if weights:
                self.apply_weights(weights, env.payload.get("model_version"))
                log.info("model synced to v%s", self.model.version)
            return
        if env.type == "inference_request":
            self.active_tasks += 1
            try:
                t0 = time.perf_counter()
                features = env.payload["features"]
                pred, probs = self.model.predict(features)
                result = InferenceResult(
                    request_id=env.request_id or env.payload.get("request_id") or "",
                    node_id=self.node_id,
                    prediction=pred,
                    probabilities=probs,
                    model_version=self.model.version,
                    latency_ms=(time.perf_counter() - t0) * 1000,
                )
                await ws.send(
                    WsEnvelope(
                        type="inference_complete",
                        request_id=env.request_id,
                        payload=result.model_dump(),
                    ).model_dump_json()
                )
            except Exception as exc:
                await ws.send(
                    WsEnvelope(
                        type="inference_error",
                        request_id=env.request_id,
                        payload={"error": str(exc)},
                    ).model_dump_json()
                )
            finally:
                self.active_tasks -= 1
            return
        if env.type == "train_request":
            self.active_tasks += 1
            try:
                req = TrainLocalRequest.model_validate(env.payload)
                update = await self._train(req)
                await ws.send(
                    WsEnvelope(
                        type="train_complete",
                        request_id=env.request_id,
                        payload=update.model_dump(),
                    ).model_dump_json()
                )
            except Exception as exc:
                await ws.send(
                    WsEnvelope(
                        type="inference_error",
                        request_id=env.request_id,
                        payload={"error": str(exc)},
                    ).model_dump_json()
                )
            finally:
                self.active_tasks -= 1
            return
        if env.type == "cancel":
            return

    async def _train(self, req: TrainLocalRequest) -> WeightUpdate:
        x, y = make_synthetic_dataset(
            n=req.samples,
            input_dim=self.model.input_dim,
            output_dim=self.model.output_dim,
            seed=(hash(self.node_id) ^ int(time.time())) % 10_000,
        )
        loss = self.model.train_batch(
            x,
            y,
            lr=req.learning_rate,
            epochs=req.epochs,
            batch_size=req.batch_size,
        )
        weights = self.model.export_weights()
        return WeightUpdate(
            node_id=self.node_id,
            model_version=self.model.version,
            num_samples=req.samples,
            weights=weights,
            loss=float(loss),
            checksum_sha256=weights_checksum(weights),
        )


def create_provider_app(state: ProviderState) -> FastAPI:
    """HTTP provider app (LAN mode). WS-only providers skip this server."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await state.register_http()
        hb_task = asyncio.create_task(state.heartbeat_loop_http())
        yield
        await state.leave_http()
        hb_task.cancel()
        try:
            await hb_task
        except asyncio.CancelledError:
            pass

    app = FastAPI(title="DICO Provider", version="0.2.0", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict:
        return {
            "ok": True,
            "role": "provider",
            "node_id": state.node_id,
            "model_version": state.model.version,
            "load": state.info.load,
            "capabilities": state.capabilities.model_dump(),
            "capacity": state.capacity().model_dump(),
            "protocol_version": PROTOCOL_VERSION,
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
            return await state._train(req)
        finally:
            state.active_tasks -= 1

    @app.post("/model/sync")
    async def sync_model(payload: dict) -> dict:
        weights = payload.get("weights")
        if not weights:
            raise HTTPException(400, "weights required")
        expected = payload.get("checksum_sha256")
        if expected and weights_checksum(weights) != expected:
            raise HTTPException(400, "checksum mismatch")
        state.apply_weights(weights, payload.get("model_version", state.model.version))
        return {"ok": True, "model_version": state.model.version}

    return app


def run_provider(
    coordinator_url: str,
    host: str = "0.0.0.0",
    port: int = 7401,
    node_id: str | None = None,
    *,
    transport: str = "websocket",
) -> None:
    cfg = load_config()
    setup_logging(cfg.log_level, cfg.log_json)
    advertise = host if host != "0.0.0.0" else local_ip()
    state = ProviderState(
        coordinator_url=coordinator_url,
        host=advertise,
        port=port,
        node_id=node_id,
        transport=transport,
        cfg=cfg,
    )

    if transport == "websocket":
        print(f"DICO provider {state.node_id} connecting outbound WS to {coordinator_url}")
        print("(no inbound ports required)")

        async def _main() -> None:
            try:
                await state.run_websocket_loop()
            except KeyboardInterrupt:
                state._stop.set()

        try:
            asyncio.run(_main())
        except KeyboardInterrupt:
            state._stop.set()
        return

    app = create_provider_app(state)
    print(f"DICO provider {state.node_id} on http://{advertise}:{port}")
    print(f"Joining coordinator at {coordinator_url}")
    uvicorn.run(app, host=host, port=port, log_level=cfg.log_level.lower())

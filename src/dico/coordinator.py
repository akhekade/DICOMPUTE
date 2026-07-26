"""Coordinator control plane: registry, scheduling, billing, FedAvg, WS hub."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, WebSocket
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from dico.auth import (
    AuthContext,
    RateLimiter,
    bootstrap_store,
    generate_api_key,
    make_auth_dependency,
    require_admin,
)
from dico.billing import BillingService
from dico.checksum import weights_checksum
from dico.config import AppConfig, load_config
from dico.hardware import local_ip, probe_capabilities
from dico.model import CollectiveMLP, federated_average, make_synthetic_dataset
from dico.protocol import (
    PROTOCOL_VERSION,
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
from dico.registry import ProviderRegistry
from dico.store import Store, open_store
from dico.store.base import AccountRecord, ApiKeyRecord, ModelCheckpoint
from dico.telemetry import METRICS, RequestTimer, get_logger, setup_logging
from dico.ws_hub import WebSocketHub

log = get_logger("dico.coordinator")
WEB_DIR = Path(__file__).resolve().parents[2] / "web"


class CoordinatorState:
    def __init__(self, cfg: AppConfig, store: Store) -> None:
        self.cfg = cfg
        self.store = store
        host = cfg.advertise_host or (
            local_ip() if cfg.host == "0.0.0.0" else cfg.host
        )
        self.host = host
        self.port = cfg.port
        self.node_id = f"coord-{uuid.uuid4().hex[:8]}"
        self.registry = ProviderRegistry(cfg)
        self.hub = WebSocketHub(self.registry)
        self.billing = BillingService(store, cfg)
        self.limiter = RateLimiter(cfg.rate_limit_rpm, cfg.rate_limit_burst)
        self.model = CollectiveMLP()
        self.pending_updates: list[WeightUpdate] = []
        self.lock = asyncio.Lock()
        self._load_checkpoint()
        caps = probe_capabilities()
        caps.software_version = "0.2.0"
        self.self_info = NodeInfo(
            node_id=self.node_id,
            role=NodeRole.COORDINATOR,
            host=host,
            port=self.port,
            capabilities=caps,
            status="online",
            model_version=self.model.version,
            last_heartbeat=time.time(),
            transport="http",
        )
        self.bootstrap_key_plaintext: str | None = None

    def _load_checkpoint(self) -> None:
        cp = self.store.load_checkpoint("collective-mlp")
        if cp:
            try:
                weights = json.loads(cp.weights_json)
                self.model.load_weights(weights)
                self.model.version = cp.version
                log.info("loaded checkpoint v%s checksum=%s", cp.version, cp.checksum_sha256[:12])
                return
            except Exception as exc:
                log.warning("checkpoint load failed: %s", exc)
        path = self.cfg.model_checkpoint_path()
        if path.exists():
            try:
                self.model = CollectiveMLP.load(path)
                log.info("loaded file checkpoint %s v%s", path, self.model.version)
            except Exception as exc:
                log.warning("file checkpoint load failed: %s", exc)

    def persist_model(self) -> None:
        weights = self.model.export_weights()
        checksum = weights_checksum(weights)
        payload = json.dumps(weights)
        self.store.save_checkpoint(
            ModelCheckpoint(
                model_id="collective-mlp",
                version=self.model.version,
                weights_json=payload,
                checksum_sha256=checksum,
                updated_at=time.time(),
            )
        )
        if self.cfg.checkpoint_model:
            self.cfg.model_checkpoint_path().write_text(payload, encoding="utf-8")

    def cluster_status(self) -> ClusterStatus:
        providers = self.registry.online_providers()
        self.self_info.model_version = self.model.version
        self.self_info.last_heartbeat = time.time()
        return ClusterStatus(
            coordinator_id=self.node_id,
            model_version=self.model.version,
            nodes=[self.self_info, *providers],
            total_cpu_cores=sum(n.capabilities.cpu_cores for n in providers),
            total_memory_gb=round(
                sum(n.capabilities.memory_gb for n in providers), 2
            ),
            online_providers=len(providers),
            protocol_version=PROTOCOL_VERSION,
            auth_required=not self.cfg.auth_disabled,
            billing_enabled=self.cfg.billing_enabled,
            ws_providers=self.registry.ws_count(),
        )


async def _dispatch_infer_http(
    node: NodeInfo, req: InferenceRequest
) -> InferenceResult:
    async with httpx.AsyncClient(timeout=req.timeout_s) as client:
        resp = await client.post(
            f"{node.endpoint}/infer",
            json={
                "features": req.features,
                "request_id": req.request_id,
                "ensemble": False,
                "model": req.model,
            },
        )
        resp.raise_for_status()
        return InferenceResult(**resp.json())


async def _dispatch_train_http(
    node: NodeInfo, req: TrainLocalRequest
) -> WeightUpdate:
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(f"{node.endpoint}/train", json=req.model_dump())
        resp.raise_for_status()
        return WeightUpdate(**resp.json())


def create_coordinator_app(
    state: CoordinatorState | None = None,
    *,
    cfg: AppConfig | None = None,
    store: Store | None = None,
) -> FastAPI:
    if state is None:
        cfg = cfg or load_config()
        store = store or open_store(cfg.resolved_database_url())
        state = CoordinatorState(cfg, store)
        state.bootstrap_key_plaintext = bootstrap_store(store, cfg)

    cfg = state.cfg
    auth_dep = make_auth_dependency(state.store, cfg, state.limiter)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if state.bootstrap_key_plaintext:
            log.warning(
                "BOOTSTRAP API KEY (store securely, shown once): %s",
                state.bootstrap_key_plaintext,
            )
        yield

    app = FastAPI(
        title="DICO Coordinator",
        version="0.2.0",
        description="Production control plane for the DICOMPUTE mesh",
        lifespan=lifespan,
    )
    app.state.coordinator = state

    @app.get("/health")
    async def health() -> dict:
        return {
            "ok": True,
            "role": "coordinator",
            "node_id": state.node_id,
            "protocol_version": PROTOCOL_VERSION,
            "model_version": state.model.version,
            "capacity": state.registry.capacity_summary(),
        }

    @app.get("/metrics")
    async def metrics() -> PlainTextResponse:
        if not cfg.metrics_enabled:
            raise HTTPException(404, "metrics disabled")
        return PlainTextResponse(METRICS.prometheus_text(), media_type="text/plain")

    @app.get("/status", response_model=ClusterStatus)
    async def status(_auth: AuthContext | None = Depends(auth_dep)) -> ClusterStatus:
        return state.cluster_status()

    @app.post("/nodes/register")
    async def register(req: RegisterRequest) -> dict:
        if req.protocol_version < 1:
            raise HTTPException(400, "unsupported protocol version")
        node = req.node
        node.transport = "http"
        state.registry.register(node)
        return {
            "ok": True,
            "model_version": state.model.version,
            "weights": state.model.export_weights(),
            "checksum_sha256": weights_checksum(state.model.export_weights()),
            "coordinator_id": state.node_id,
            "protocol_version": PROTOCOL_VERSION,
        }

    @app.post("/nodes/heartbeat")
    async def heartbeat(hb: Heartbeat) -> dict:
        node = state.registry.heartbeat(hb)
        if not node:
            raise HTTPException(404, "unknown node; register first")
        return {"ok": True, "model_version": state.model.version}

    @app.delete("/nodes/{node_id}")
    async def leave(node_id: str) -> dict:
        state.registry.leave(node_id)
        return {"ok": True}

    @app.websocket("/ws/provider")
    async def provider_ws(websocket: WebSocket) -> None:
        if not cfg.ws_providers_enabled:
            await websocket.close(code=1008)
            return
        await state.hub.handle(websocket)

    @app.get("/model")
    async def get_model(_auth: AuthContext | None = Depends(auth_dep)) -> dict:
        weights = state.model.export_weights()
        return {
            "model_version": state.model.version,
            "weights": weights,
            "checksum_sha256": weights_checksum(weights),
            "model_id": "collective-mlp",
        }

    async def run_inference(
        req: InferenceRequest, account_id: str | None
    ) -> CollectiveInferenceResponse:
        request_id = req.request_id or uuid.uuid4().hex
        contributors: list[InferenceResult] = []

        with RequestTimer("infer"):
            if cfg.include_coordinator_in_ensemble or req.routing == "local":
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

            if req.routing != "local" and req.ensemble:
                if req.routing == "cheapest":
                    chosen = state.registry.select(
                        model=req.model, k=1, strategy="cheapest"
                    )
                else:
                    chosen = state.registry.select(
                        model=req.model, k=99, strategy="ensemble"
                    )

                async def call_one(node: NodeInfo) -> InferenceResult | None:
                    try:
                        if node.transport == "websocket" and state.hub.has(node.node_id):
                            data = await state.hub.infer(
                                node.node_id,
                                {
                                    "features": req.features,
                                    "request_id": request_id,
                                    "model": req.model,
                                },
                                timeout=req.timeout_s,
                            )
                            result = InferenceResult(**data)
                        else:
                            result = await _dispatch_infer_http(node, req)
                        state.registry.record_success(node.node_id)
                        return result
                    except Exception as exc:
                        log.warning("infer failed on %s: %s", node.node_id, exc)
                        state.registry.record_error(node.node_id, shape="infer")
                        METRICS.outcome("provider", "infer_fail")
                        return None

                if chosen:
                    results = await asyncio.gather(
                        *[call_one(c.node) for c in chosen]
                    )
                    contributors.extend([r for r in results if r is not None])

            if not contributors:
                METRICS.outcome("client", "no_capacity")
                raise HTTPException(503, "no providers available for inference")

            mats = [np.asarray(c.probabilities, dtype=np.float64) for c in contributors]
            avg = np.mean(np.stack(mats, axis=0), axis=0)
            prediction = float(np.argmax(avg))
            strategy = (
                "ensemble_mean"
                if len(contributors) > 1
                else ("local" if contributors[0].node_id == state.node_id else "single")
            )
            provider_ids = [
                c.node_id for c in contributors if c.node_id != state.node_id
            ]
            cost = state.billing.estimate_infer_cost(max(1, len(contributors)))
            METRICS.outcome("client", "infer_ok")
            return CollectiveInferenceResponse(
                request_id=request_id,
                prediction=prediction,
                probabilities=[float(v) for v in avg],
                model_version=state.model.version,
                contributors=contributors,
                strategy=strategy,
                cost_micro_usd=cost,
                provider_ids=provider_ids,
            )

    @app.post("/infer", response_model=CollectiveInferenceResponse)
    async def infer(
        req: InferenceRequest,
        auth: AuthContext | None = Depends(auth_dep),
    ) -> CollectiveInferenceResponse:
        account_id = auth.account_id if auth else "acct-default"
        request_id = req.request_id or uuid.uuid4().hex
        req.request_id = request_id
        reserved = state.billing.reserve(
            account_id,
            request_id,
            state.billing.estimate_infer_cost(max(1, state.registry.ws_count() + 1)),
        )
        t0 = time.perf_counter()
        try:
            result = await run_inference(req, account_id)
        except HTTPException:
            state.billing.settle(
                reserved,
                actual_cost=0,
                kind="infer",
                units=0,
                provider_id=None,
                latency_ms=0,
            )
            raise
        state.billing.settle(
            reserved,
            actual_cost=result.cost_micro_usd,
            kind="infer",
            units=1,
            provider_id=result.provider_ids[0] if result.provider_ids else state.node_id,
            latency_ms=(time.perf_counter() - t0) * 1000,
            meta={"strategy": result.strategy},
        )
        return result

    @app.post("/train/round")
    async def train_round(
        req: TrainLocalRequest,
        auth: AuthContext | None = Depends(auth_dep),
    ) -> AggregateResponse:
        account_id = auth.account_id if auth else "acct-default"
        request_id = f"train-{uuid.uuid4().hex[:12]}"
        reserved = state.billing.reserve(
            account_id,
            request_id,
            state.billing.estimate_train_cost(req.samples),
        )
        updates: list[WeightUpdate] = []
        t0 = time.perf_counter()

        local = await _local_train(state, req)
        updates.append(local)

        providers = state.registry.online_providers()

        async def train_one(node: NodeInfo) -> WeightUpdate | None:
            try:
                if node.transport == "websocket" and state.hub.has(node.node_id):
                    data = await state.hub.train(
                        node.node_id, req.model_dump(), timeout=120.0
                    )
                    update = WeightUpdate(**data)
                else:
                    update = await _dispatch_train_http(node, req)
                state.registry.record_success(node.node_id)
                return update
            except Exception as exc:
                log.warning("train failed on %s: %s", node.node_id, exc)
                state.registry.record_error(node.node_id, shape="train")
                return None

        if providers:
            results = await asyncio.gather(*[train_one(n) for n in providers])
            updates.extend([u for u in results if u is not None])

        if not updates:
            state.billing.settle(
                reserved, 0, kind="train", units=0, provider_id=None, latency_ms=0
            )
            raise HTTPException(503, "no successful training updates")

        averaged = federated_average([(u.weights, u.num_samples) for u in updates])
        state.model.load_weights(averaged)
        state.model.version = int(averaged["version"][0])
        state.self_info.model_version = state.model.version
        checksum = weights_checksum(averaged)
        state.persist_model()

        # Push sync via WS and HTTP
        async def sync_one(node: NodeInfo) -> None:
            payload = {
                "model_version": state.model.version,
                "weights": averaged,
                "checksum_sha256": checksum,
            }
            try:
                if node.transport == "websocket" and state.hub.has(node.node_id):
                    await state.hub.sync_model(node.node_id, payload)
                else:
                    async with httpx.AsyncClient(timeout=30.0) as client:
                        await client.post(f"{node.endpoint}/model/sync", json=payload)
            except Exception as exc:
                log.warning("model sync failed %s: %s", node.node_id, exc)

        await asyncio.gather(*[sync_one(n) for n in providers], return_exceptions=True)

        state.billing.settle(
            reserved,
            actual_cost=state.billing.estimate_train_cost(req.samples),
            kind="train",
            units=req.samples,
            provider_id=state.node_id,
            latency_ms=(time.perf_counter() - t0) * 1000,
            meta={"contributors": len(updates)},
        )
        return AggregateResponse(
            model_version=state.model.version,
            num_contributors=len(updates),
            weights=averaged,
            checksum_sha256=checksum,
        )

    @app.post("/train/submit", response_model=AggregateResponse)
    async def submit_update(
        update: WeightUpdate,
        auth: AuthContext | None = Depends(auth_dep),
    ) -> AggregateResponse:
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
            state.persist_model()
            version = state.model.version
            weights = state.model.export_weights()
        return AggregateResponse(
            model_version=version,
            num_contributors=1,
            weights=weights,
            checksum_sha256=weights_checksum(weights),
        )

    @app.post("/aggregate", response_model=AggregateResponse)
    async def aggregate(
        req: AggregateRequest,
        auth: AuthContext | None = Depends(auth_dep),
    ) -> AggregateResponse:
        if not req.updates:
            raise HTTPException(400, "updates required")
        averaged = federated_average(
            [(u.weights, u.num_samples) for u in req.updates]
        )
        state.model.load_weights(averaged)
        state.model.version = int(averaged["version"][0])
        state.self_info.model_version = state.model.version
        state.persist_model()
        return AggregateResponse(
            model_version=state.model.version,
            num_contributors=len(req.updates),
            weights=averaged,
            checksum_sha256=weights_checksum(averaged),
        )

    # --- Admin / billing ---
    @app.post("/admin/keys")
    async def create_key(
        body: dict[str, Any],
        auth: AuthContext | None = Depends(auth_dep),
    ) -> dict:
        require_admin(auth, cfg)
        name = str(body.get("name", "key"))
        role = str(body.get("role", "consumer"))
        account_id = str(body.get("account_id", "acct-default"))
        if not state.store.get_account(account_id):
            state.store.create_account(
                AccountRecord(
                    account_id=account_id,
                    name=account_id,
                    balance_micro_usd=int(body.get("credits_micro_usd", 0)),
                    created_at=time.time(),
                )
            )
        raw, h, prefix = generate_api_key()
        kid = f"key-{uuid.uuid4().hex[:10]}"
        state.store.create_api_key(
            ApiKeyRecord(
                key_id=kid,
                key_hash=h,
                name=name,
                role=role,
                created_at=time.time(),
                key_prefix=prefix,
            ),
            account_id=account_id,
        )
        return {"key_id": kid, "api_key": raw, "account_id": account_id, "role": role}

    @app.post("/admin/deposit")
    async def deposit(
        body: dict[str, Any],
        auth: AuthContext | None = Depends(auth_dep),
    ) -> dict:
        require_admin(auth, cfg)
        account_id = str(body.get("account_id", "acct-default"))
        amount = int(body["amount_micro_usd"])
        bal = state.billing.deposit(account_id, amount)
        return {"account_id": account_id, "balance_micro_usd": bal}

    @app.get("/v1/payments/balance")
    async def payments_balance(
        auth: AuthContext | None = Depends(auth_dep),
    ) -> dict:
        account_id = auth.account_id if auth else "acct-default"
        acc = state.store.get_account(account_id)
        if not acc:
            raise HTTPException(404, "account not found")
        return {
            "account_id": account_id,
            "balance_micro_usd": acc.balance_micro_usd,
            "balance_usd": acc.balance_micro_usd / 1_000_000,
        }

    if cfg.openai_compat_enabled:
        from dico.api.openai_compat import create_openai_router

        app.include_router(
            create_openai_router(
                cfg=cfg,
                billing=state.billing,
                infer_fn=run_inference,
                auth_dep=auth_dep,
            )
        )

    # Dashboard
    if WEB_DIR.is_dir():
        static = WEB_DIR / "static"
        if static.is_dir():
            app.mount("/static", StaticFiles(directory=str(static)), name="static")

        @app.get("/")
        async def dashboard() -> FileResponse:
            return FileResponse(WEB_DIR / "index.html")

    return app


async def _local_train(state: CoordinatorState, req: TrainLocalRequest) -> WeightUpdate:
    x, y = make_synthetic_dataset(
        n=req.samples,
        input_dim=state.model.input_dim,
        output_dim=state.model.output_dim,
        seed=int(time.time()) % 10_000,
    )
    loss = state.model.train_batch(
        x, y, lr=req.learning_rate, epochs=req.epochs, batch_size=req.batch_size
    )
    weights = state.model.export_weights()
    return WeightUpdate(
        node_id=state.node_id,
        model_version=state.model.version,
        num_samples=req.samples,
        weights=weights,
        loss=loss,
        checksum_sha256=weights_checksum(weights),
    )


def run_coordinator(
    host: str = "0.0.0.0",
    port: int = 7400,
    *,
    auth_disabled: bool | None = None,
    data_dir: str | None = None,
) -> None:
    import uvicorn

    overrides: dict[str, Any] = {"host": host, "port": port}
    if auth_disabled is not None:
        overrides["auth_disabled"] = auth_disabled
    if data_dir is not None:
        overrides["data_dir"] = Path(data_dir)
    cfg = load_config(**overrides)
    setup_logging(cfg.log_level, cfg.log_json)
    store = open_store(cfg.resolved_database_url())
    state = CoordinatorState(cfg, store)
    state.bootstrap_key_plaintext = bootstrap_store(store, cfg)
    app = create_coordinator_app(state)
    bind_host = cfg.advertise_host or (
        local_ip() if host == "0.0.0.0" else host
    )
    print(f"DICO coordinator {state.node_id} listening on http://{bind_host}:{port}")
    print(f"Dashboard: http://{bind_host}:{port}/")
    print(f"OpenAI API: http://{bind_host}:{port}/v1/chat/completions")
    print(f"Provider WS: ws://{bind_host}:{port}/ws/provider")
    if state.bootstrap_key_plaintext:
        print(f"BOOTSTRAP API KEY: {state.bootstrap_key_plaintext}")
    elif cfg.auth_disabled:
        print("Auth disabled (DICO_AUTH_DISABLED=true) — local demo mode")
    uvicorn.run(app, host=host, port=port, log_level=cfg.log_level.lower())

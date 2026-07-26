"""WebSocket hub: providers connect outbound to the coordinator (Darkbloom pattern)."""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, Callable, Awaitable

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from dico.protocol import (
    BackendCapacity,
    Heartbeat,
    NodeInfo,
    NodeRole,
    WsEnvelope,
)
from dico.registry import ProviderRegistry
from dico.telemetry import METRICS, get_logger

log = get_logger("dico.ws")

InferHandler = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]
TrainHandler = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]


class ProviderConnection:
    def __init__(self, node_id: str, ws: WebSocket) -> None:
        self.node_id = node_id
        self.ws = ws
        self._pending: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._lock = asyncio.Lock()

    async def send(self, envelope: WsEnvelope) -> None:
        await self.ws.send_json(envelope.model_dump())

    async def request(
        self, msg_type: str, payload: dict[str, Any], timeout: float
    ) -> dict[str, Any]:
        request_id = uuid.uuid4().hex
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._pending[request_id] = fut
        try:
            await self.send(
                WsEnvelope(type=msg_type, request_id=request_id, payload=payload)  # type: ignore[arg-type]
            )
            return await asyncio.wait_for(fut, timeout=timeout)
        finally:
            self._pending.pop(request_id, None)

    def resolve(self, request_id: str | None, payload: dict[str, Any], error: bool = False) -> None:
        if not request_id:
            return
        fut = self._pending.get(request_id)
        if not fut or fut.done():
            return
        if error:
            fut.set_exception(RuntimeError(payload.get("error", "provider error")))
        else:
            fut.set_result(payload)


class WebSocketHub:
    def __init__(self, registry: ProviderRegistry) -> None:
        self.registry = registry
        self.connections: dict[str, ProviderConnection] = {}

    async def handle(self, websocket: WebSocket) -> None:
        await websocket.accept()
        conn: ProviderConnection | None = None
        node_id: str | None = None
        try:
            while True:
                raw = await websocket.receive_json()
                try:
                    env = WsEnvelope.model_validate(raw)
                except ValidationError as exc:
                    await websocket.send_json(
                        WsEnvelope(
                            type="error",
                            payload={"error": f"invalid envelope: {exc}"},
                        ).model_dump()
                    )
                    continue

                if env.type == "register":
                    node = NodeInfo.model_validate(env.payload.get("node", env.payload))
                    node.transport = "websocket"
                    node.last_heartbeat = time.time()
                    if "capacity" in env.payload and not node.capacity:
                        node.capacity = BackendCapacity.model_validate(
                            env.payload["capacity"]
                        )
                    self.registry.register(node)
                    self.registry.mark_ws(node.node_id, True)
                    conn = ProviderConnection(node.node_id, websocket)
                    self.connections[node.node_id] = conn
                    node_id = node.node_id
                    await conn.send(
                        WsEnvelope(
                            type="ack",
                            request_id=env.request_id,
                            payload={"ok": True, "node_id": node.node_id},
                        )
                    )
                    METRICS.incr("ws_register")
                    continue

                if not conn or not node_id:
                    await websocket.send_json(
                        WsEnvelope(
                            type="error", payload={"error": "register first"}
                        ).model_dump()
                    )
                    continue

                if env.type == "heartbeat":
                    hb = Heartbeat.model_validate(
                        {**env.payload, "node_id": node_id}
                    )
                    self.registry.heartbeat(hb)
                    await conn.send(
                        WsEnvelope(type="ack", request_id=env.request_id, payload={"ok": True})
                    )
                    continue

                if env.type in {"inference_complete", "train_complete"}:
                    conn.resolve(env.request_id, env.payload, error=False)
                    continue

                if env.type == "inference_error":
                    conn.resolve(env.request_id, env.payload, error=True)
                    self.registry.record_error(node_id)
                    continue

                # ignore unknown for forward-compat but count
                METRICS.incr("ws_unknown_type", type=env.type)

        except WebSocketDisconnect:
            log.info("ws disconnect node=%s", node_id)
        except Exception as exc:
            log.exception("ws handler error: %s", exc)
        finally:
            if node_id:
                self.connections.pop(node_id, None)
                self.registry.mark_ws(node_id, False)
                # Keep registry entry until TTL; mark status
                node = self.registry.nodes.get(node_id)
                if node:
                    node.status = "offline"

    async def infer(
        self, node_id: str, payload: dict[str, Any], timeout: float
    ) -> dict[str, Any]:
        conn = self.connections.get(node_id)
        if not conn:
            raise RuntimeError("provider not connected via websocket")
        return await conn.request("inference_request", payload, timeout)

    async def train(
        self, node_id: str, payload: dict[str, Any], timeout: float
    ) -> dict[str, Any]:
        conn = self.connections.get(node_id)
        if not conn:
            raise RuntimeError("provider not connected via websocket")
        return await conn.request("train_request", payload, timeout)

    async def sync_model(self, node_id: str, payload: dict[str, Any]) -> None:
        conn = self.connections.get(node_id)
        if not conn:
            return
        await conn.send(WsEnvelope(type="model_sync", payload=payload))

    def has(self, node_id: str) -> bool:
        return node_id in self.connections

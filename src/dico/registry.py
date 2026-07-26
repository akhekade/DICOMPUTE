"""In-memory provider registry with health ejection and capacity tracking."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from dico.config import AppConfig
from dico.protocol import BackendCapacity, Heartbeat, NodeInfo
from dico.telemetry import METRICS, get_logger

log = get_logger("dico.registry")


@dataclass
class ScheduleCandidate:
    node: NodeInfo
    cost_ms: float
    reason: str = "ok"


class ProviderRegistry:
    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self.nodes: dict[str, NodeInfo] = {}
        self._ws_connected: set[str] = set()

    def register(self, node: NodeInfo) -> None:
        node.last_heartbeat = time.time()
        node.status = "online"
        existing = self.nodes.get(node.node_id)
        if existing:
            node.consecutive_errors = existing.consecutive_errors
            node.cooldown_until = existing.cooldown_until
        self.nodes[node.node_id] = node
        METRICS.incr("provider_register")
        log.info("registered provider %s transport=%s", node.node_id, node.transport)

    def mark_ws(self, node_id: str, connected: bool) -> None:
        if connected:
            self._ws_connected.add(node_id)
            node = self.nodes.get(node_id)
            if node:
                node.transport = "websocket"
        else:
            self._ws_connected.discard(node_id)

    def ws_count(self) -> int:
        return len(self._ws_connected)

    def heartbeat(self, hb: Heartbeat) -> NodeInfo | None:
        node = self.nodes.get(hb.node_id)
        if not node:
            return None
        node.last_heartbeat = time.time()
        node.load = hb.load
        node.model_version = hb.model_version
        node.status = hb.status
        if hb.capacity:
            node.capacity = hb.capacity
        return node

    def leave(self, node_id: str) -> None:
        self.nodes.pop(node_id, None)
        self._ws_connected.discard(node_id)

    def prune_stale(self) -> list[str]:
        now = time.time()
        stale = [
            nid
            for nid, n in self.nodes.items()
            if now - n.last_heartbeat > self.cfg.heartbeat_ttl_s
        ]
        for nid in stale:
            log.info("evicting stale provider %s", nid)
            self.leave(nid)
            METRICS.incr("provider_stale_evict")
        return stale

    def record_success(self, node_id: str) -> None:
        node = self.nodes.get(node_id)
        if node:
            node.consecutive_errors = 0

    def record_error(self, node_id: str, shape: str = "default") -> None:
        node = self.nodes.get(node_id)
        if not node:
            return
        node.consecutive_errors += 1
        METRICS.outcome("provider", f"error:{shape}")
        if node.consecutive_errors >= self.cfg.max_consecutive_errors:
            node.cooldown_until = time.time() + self.cfg.error_cooldown_s
            node.status = "cooldown"
            log.warning(
                "provider %s entered cooldown after %s errors",
                node_id,
                node.consecutive_errors,
            )
            METRICS.incr("provider_cooldown")

    def online_providers(self) -> list[NodeInfo]:
        self.prune_stale()
        now = time.time()
        out: list[NodeInfo] = []
        for n in self.nodes.values():
            if n.status in {"offline", "untrusted"}:
                continue
            if n.cooldown_until > now:
                continue
            if n.status == "cooldown" and n.cooldown_until <= now:
                n.status = "online"
                n.consecutive_errors = 0
            out.append(n)
        return out

    def eligible(
        self, node: NodeInfo, *, model: str = "collective-mlp", need_slots: int = 1
    ) -> tuple[bool, str]:
        now = time.time()
        if node.cooldown_until > now:
            return False, "cooldown"
        if node.status in {"offline", "untrusted"}:
            return False, f"status:{node.status}"
        if model not in (node.capacity.warm_models or ["collective-mlp"]):
            return False, "model_not_warm"
        if node.capacity.free_slots < need_slots:
            return False, "no_capacity"
        if node.protocol_version < 1:
            return False, "protocol"
        return True, "ok"

    def score(self, node: NodeInfo, prompt_units: int = 8) -> float:
        """Lower is better — cost in estimated milliseconds."""
        cap = node.capacity
        queue_ms = cap.queued * 3000.0
        load_ms = node.load * 2000.0
        pressure_ms = cap.memory_pressure * 5000.0
        cpu_ms = cap.cpu_util * 1500.0
        # Prefer websocket (NAT-friendly, lower connect overhead)
        transport_ms = 0.0 if node.transport == "websocket" else 50.0
        # Prefer lower load / more free slots
        slot_ms = max(0, cap.max_slots - cap.free_slots) * 500.0
        work_ms = prompt_units * 2.0
        return queue_ms + load_ms + pressure_ms + cpu_ms + transport_ms + slot_ms + work_ms

    def select(
        self,
        *,
        model: str = "collective-mlp",
        k: int = 1,
        strategy: str = "cheapest",
    ) -> list[ScheduleCandidate]:
        providers = self.online_providers()
        candidates: list[ScheduleCandidate] = []
        for n in providers:
            ok, reason = self.eligible(n, model=model)
            if not ok:
                continue
            cost = self.score(n)
            candidates.append(ScheduleCandidate(node=n, cost_ms=cost, reason=reason))

        if strategy == "ensemble":
            candidates.sort(key=lambda c: c.cost_ms)
            return candidates  # caller takes all / fan-out
        candidates.sort(key=lambda c: c.cost_ms)
        return candidates[: max(1, k)]

    def capacity_summary(self) -> dict[str, Any]:
        nodes = self.online_providers()
        return {
            "online": len(nodes),
            "free_slots": sum(n.capacity.free_slots for n in nodes),
            "max_slots": sum(n.capacity.max_slots for n in nodes),
            "ws_connected": self.ws_count(),
        }

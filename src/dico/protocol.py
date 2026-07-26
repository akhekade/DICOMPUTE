"""Shared wire protocol for the DICO mesh (HTTP + WebSocket)."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


PROTOCOL_VERSION = 1


class NodeRole(str, Enum):
    COORDINATOR = "coordinator"
    PROVIDER = "provider"


class NodeCapabilities(BaseModel):
    cpu_cores: int = 1
    memory_gb: float = 1.0
    has_metal: bool = False
    has_cuda: bool = False
    platform: str = "unknown"
    hostname: str = "unknown"
    max_concurrent_tasks: int = 2
    software_version: str = "0.2.0"


class BackendCapacity(BaseModel):
    """Live capacity advertised on heartbeats (Darkbloom-style)."""

    max_slots: int = 2
    free_slots: int = 2
    queued: int = 0
    warm_models: list[str] = Field(default_factory=lambda: ["collective-mlp"])
    memory_pressure: float = 0.0  # 0..1
    cpu_util: float = 0.0  # 0..1


class NodeInfo(BaseModel):
    node_id: str
    role: NodeRole = NodeRole.PROVIDER
    host: str
    port: int
    capabilities: NodeCapabilities
    status: str = "online"
    model_version: int = 0
    load: float = 0.0
    last_heartbeat: float = 0.0
    transport: Literal["http", "websocket"] = "http"
    trust_level: Literal["none", "self_signed", "hardware"] = "none"
    capacity: BackendCapacity = Field(default_factory=BackendCapacity)
    consecutive_errors: int = 0
    cooldown_until: float = 0.0
    protocol_version: int = PROTOCOL_VERSION

    @property
    def endpoint(self) -> str:
        return f"http://{self.host}:{self.port}"


class Heartbeat(BaseModel):
    node_id: str
    load: float = 0.0
    model_version: int = 0
    status: str = "online"
    capacity: BackendCapacity | None = None
    protocol_version: int = PROTOCOL_VERSION


class RegisterRequest(BaseModel):
    node: NodeInfo
    protocol_version: int = PROTOCOL_VERSION
    auth_token: str | None = None


class InferenceRequest(BaseModel):
    features: list[float]
    request_id: str | None = None
    ensemble: bool = True
    timeout_s: float = 10.0
    model: str = "collective-mlp"
    routing: Literal["ensemble", "cheapest", "local"] = "ensemble"


class InferenceResult(BaseModel):
    request_id: str
    node_id: str
    prediction: float
    probabilities: list[float] = Field(default_factory=list)
    model_version: int = 0
    latency_ms: float = 0.0


class CollectiveInferenceResponse(BaseModel):
    request_id: str
    prediction: float
    probabilities: list[float]
    model_version: int
    contributors: list[InferenceResult]
    strategy: str = "weighted_average"
    cost_micro_usd: int = 0
    provider_ids: list[str] = Field(default_factory=list)


class TrainLocalRequest(BaseModel):
    epochs: int = 3
    batch_size: int = 32
    learning_rate: float = 0.05
    samples: int = 256


class WeightUpdate(BaseModel):
    node_id: str
    model_version: int
    num_samples: int
    weights: dict[str, list[Any]]
    loss: float = 0.0
    checksum_sha256: str | None = None


class AggregateRequest(BaseModel):
    updates: list[WeightUpdate] = Field(default_factory=list)


class AggregateResponse(BaseModel):
    model_version: int
    num_contributors: int
    weights: dict[str, list[Any]]
    checksum_sha256: str | None = None


class ClusterStatus(BaseModel):
    coordinator_id: str
    model_version: int
    nodes: list[NodeInfo]
    total_cpu_cores: int = 0
    total_memory_gb: float = 0.0
    online_providers: int = 0
    protocol_version: int = PROTOCOL_VERSION
    auth_required: bool = False
    billing_enabled: bool = True
    ws_providers: int = 0


class TaskRecord(BaseModel):
    task_id: str
    kind: str
    status: str
    created_at: float
    detail: dict[str, Any] = Field(default_factory=dict)


# --- WebSocket provider protocol (mirrored types) ---

WsMessageType = Literal[
    "register",
    "heartbeat",
    "inference_request",
    "inference_complete",
    "inference_error",
    "train_request",
    "train_complete",
    "model_sync",
    "cancel",
    "ack",
    "error",
]


class WsEnvelope(BaseModel):
    type: WsMessageType
    protocol_version: int = PROTOCOL_VERSION
    request_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)

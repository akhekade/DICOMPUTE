"""Shared wire protocol for the DICO mesh."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


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

    @property
    def endpoint(self) -> str:
        return f"http://{self.host}:{self.port}"


class Heartbeat(BaseModel):
    node_id: str
    load: float = 0.0
    model_version: int = 0
    status: str = "online"


class RegisterRequest(BaseModel):
    node: NodeInfo


class InferenceRequest(BaseModel):
    features: list[float]
    request_id: str | None = None
    ensemble: bool = True
    timeout_s: float = 10.0


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


class AggregateRequest(BaseModel):
    updates: list[WeightUpdate] = Field(default_factory=list)


class AggregateResponse(BaseModel):
    model_version: int
    num_contributors: int
    weights: dict[str, list[Any]]


class ClusterStatus(BaseModel):
    coordinator_id: str
    model_version: int
    nodes: list[NodeInfo]
    total_cpu_cores: int = 0
    total_memory_gb: float = 0.0
    online_providers: int = 0


class TaskRecord(BaseModel):
    task_id: str
    kind: str
    status: str
    created_at: float
    detail: dict[str, Any] = Field(default_factory=dict)

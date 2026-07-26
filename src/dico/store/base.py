"""Store interface composed of domain repositories (Darkbloom pattern)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ApiKeyRecord:
    key_id: str
    key_hash: str
    name: str
    role: str  # consumer | admin | provider
    created_at: float
    disabled: bool = False
    rate_limit_rpm: int | None = None
    model_allowlist: list[str] = field(default_factory=list)
    # Prefix shown once at creation for UX (never stored hashed form recovery)
    key_prefix: str = ""


@dataclass
class AccountRecord:
    account_id: str
    name: str
    balance_micro_usd: int = 0
    created_at: float = 0.0


@dataclass
class LedgerEntry:
    entry_id: str
    account_id: str
    kind: str  # deposit | reserve | settle | refund | charge
    amount_micro_usd: int
    request_id: str | None
    created_at: float
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Reservation:
    reservation_id: str
    account_id: str
    request_id: str
    reserved_micro_usd: int
    created_at: float
    settled: bool = False


@dataclass
class UsageRecord:
    usage_id: str
    account_id: str
    request_id: str
    kind: str  # infer | train | chat
    units: int
    cost_micro_usd: int
    provider_id: str | None
    latency_ms: float
    created_at: float
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelCheckpoint:
    model_id: str
    version: int
    weights_json: str
    checksum_sha256: str
    updated_at: float


class Store(ABC):
    @abstractmethod
    def close(self) -> None: ...

    # --- API keys ---
    @abstractmethod
    def create_api_key(self, record: ApiKeyRecord, account_id: str) -> None: ...

    @abstractmethod
    def get_api_key_by_hash(self, key_hash: str) -> ApiKeyRecord | None: ...

    @abstractmethod
    def list_api_keys(self) -> list[ApiKeyRecord]: ...

    @abstractmethod
    def disable_api_key(self, key_id: str) -> None: ...

    @abstractmethod
    def account_for_key(self, key_id: str) -> str | None: ...

    # --- Accounts / ledger ---
    @abstractmethod
    def create_account(self, account: AccountRecord) -> None: ...

    @abstractmethod
    def get_account(self, account_id: str) -> AccountRecord | None: ...

    @abstractmethod
    def set_balance(self, account_id: str, balance_micro_usd: int) -> None: ...

    @abstractmethod
    def add_ledger_entry(self, entry: LedgerEntry) -> None: ...

    @abstractmethod
    def list_ledger(self, account_id: str, limit: int = 100) -> list[LedgerEntry]: ...

    @abstractmethod
    def put_reservation(self, reservation: Reservation) -> None: ...

    @abstractmethod
    def get_reservation(self, reservation_id: str) -> Reservation | None: ...

    @abstractmethod
    def mark_reservation_settled(self, reservation_id: str) -> None: ...

    # --- Usage ---
    @abstractmethod
    def add_usage(self, usage: UsageRecord) -> None: ...

    @abstractmethod
    def list_usage(self, account_id: str, limit: int = 100) -> list[UsageRecord]: ...

    # --- Model checkpoint ---
    @abstractmethod
    def save_checkpoint(self, checkpoint: ModelCheckpoint) -> None: ...

    @abstractmethod
    def load_checkpoint(self, model_id: str = "collective-mlp") -> ModelCheckpoint | None: ...

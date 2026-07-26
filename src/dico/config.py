"""Validated application configuration (Darkbloom-inspired DICO_* env namespace)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DICO_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Identity / network
    host: str = "0.0.0.0"
    port: int = 7400
    advertise_host: str | None = None
    protocol_version: int = 1
    min_provider_version: str = "0.2.0"

    # Auth — disabled for local demos; enable for production
    auth_disabled: bool = False
    admin_api_key: str | None = None
    bootstrap_api_key: str | None = None
    bootstrap_api_key_name: str = "bootstrap"
    bootstrap_credits_micro_usd: int = 10_000_000  # $10

    # Persistence
    data_dir: Path = Field(default=Path("./data"))
    store_backend: Literal["memory", "sqlite"] = "sqlite"
    database_url: str | None = None  # default: sqlite:///<data_dir>/dico.db
    checkpoint_model: bool = True

    # Registry / scheduling
    heartbeat_ttl_s: float = 30.0
    heartbeat_interval_s: float = 10.0
    error_cooldown_s: float = 60.0
    max_consecutive_errors: int = 3
    prefer_websocket: bool = True
    default_infer_timeout_s: float = 15.0
    include_coordinator_in_ensemble: bool = True

    # Billing (micro-USD: 1 USD = 1_000_000)
    billing_enabled: bool = True
    price_infer_micro_usd: int = 100  # $0.0001 per inference
    price_train_micro_usd: int = 1_000  # $0.001 per train sample-equivalent unit
    reserve_overage_cap_x: float = 2.0
    min_charge_micro_usd: int = 10

    # Rate limits
    rate_limit_rpm: int = 120
    rate_limit_burst: int = 30

    # Observability
    log_level: str = "INFO"
    log_json: bool = False
    metrics_enabled: bool = True

    # Feature flags
    openai_compat_enabled: bool = True
    ws_providers_enabled: bool = True

    @field_validator("data_dir", mode="before")
    @classmethod
    def _path(cls, v: object) -> Path:
        return Path(v) if not isinstance(v, Path) else v

    def resolved_database_url(self) -> str:
        if self.database_url:
            return self.database_url
        if self.store_backend == "memory":
            return "memory://"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{(self.data_dir / 'dico.db').resolve()}"

    def model_checkpoint_path(self) -> Path:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        return self.data_dir / "global_model.json"

    def check(self) -> None:
        if self.port < 1 or self.port > 65535:
            raise ValueError("DICO_PORT out of range")
        if not self.auth_disabled and not (
            self.admin_api_key or self.bootstrap_api_key
        ):
            # Production should set keys; we allow empty at import and warn at boot.
            pass
        if self.heartbeat_ttl_s <= self.heartbeat_interval_s:
            raise ValueError("heartbeat_ttl_s must exceed heartbeat_interval_s")


def load_config(**overrides: object) -> AppConfig:
    cfg = AppConfig(**overrides)  # type: ignore[arg-type]
    cfg.check()
    return cfg

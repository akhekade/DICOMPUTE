"""API key authentication and rate limiting."""

from __future__ import annotations

import hashlib
import hmac
import secrets
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass
from threading import Lock
from typing import Callable

from fastapi import Depends, Header, HTTPException, Request

from dico.config import AppConfig
from dico.store.base import AccountRecord, ApiKeyRecord, LedgerEntry, Store
from dico.telemetry import METRICS


KEY_PREFIX = "sk-dico-"


@dataclass
class AuthContext:
    key: ApiKeyRecord
    account_id: str
    raw_key: str | None = None


def hash_api_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def generate_api_key() -> tuple[str, str, str]:
    """Returns (raw_key, key_hash, key_prefix)."""
    secret = secrets.token_urlsafe(24)
    raw = f"{KEY_PREFIX}{secret}"
    return raw, hash_api_key(raw), raw[:16]


def bootstrap_store(store: Store, cfg: AppConfig) -> str | None:
    """Ensure a default account + optional bootstrap key exist. Returns raw key if created."""
    account_id = "acct-default"
    if not store.get_account(account_id):
        store.create_account(
            AccountRecord(
                account_id=account_id,
                name="default",
                balance_micro_usd=cfg.bootstrap_credits_micro_usd,
                created_at=time.time(),
            )
        )
        store.add_ledger_entry(
            LedgerEntry(
                entry_id=f"led-{uuid.uuid4().hex[:12]}",
                account_id=account_id,
                kind="deposit",
                amount_micro_usd=cfg.bootstrap_credits_micro_usd,
                request_id=None,
                created_at=time.time(),
                meta={"source": "bootstrap"},
            )
        )

    created_raw: str | None = None
    if cfg.bootstrap_api_key:
        raw = cfg.bootstrap_api_key
        if not raw.startswith(KEY_PREFIX):
            raw = f"{KEY_PREFIX}{raw}"
        h = hash_api_key(raw)
        if not store.get_api_key_by_hash(h):
            kid = f"key-{uuid.uuid4().hex[:10]}"
            store.create_api_key(
                ApiKeyRecord(
                    key_id=kid,
                    key_hash=h,
                    name=cfg.bootstrap_api_key_name,
                    role="admin",
                    created_at=time.time(),
                    key_prefix=raw[:16],
                ),
                account_id=account_id,
            )
    elif not store.list_api_keys() and not cfg.auth_disabled:
        raw, h, prefix = generate_api_key()
        kid = f"key-{uuid.uuid4().hex[:10]}"
        store.create_api_key(
            ApiKeyRecord(
                key_id=kid,
                key_hash=h,
                name=cfg.bootstrap_api_key_name,
                role="admin",
                created_at=time.time(),
                key_prefix=prefix,
            ),
            account_id=account_id,
        )
        created_raw = raw
    return created_raw


class RateLimiter:
    def __init__(self, rpm: int, burst: int) -> None:
        self.rpm = rpm
        self.burst = burst
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def check(self, key: str) -> None:
        now = time.time()
        window = 60.0
        with self._lock:
            q = self._events[key]
            while q and now - q[0] > window:
                q.popleft()
            if len(q) >= max(self.rpm, self.burst):
                METRICS.incr("rate_limited")
                raise HTTPException(429, "rate limit exceeded")
            q.append(now)


def extract_bearer(authorization: str | None, x_api_key: str | None) -> str | None:
    if x_api_key:
        return x_api_key.strip()
    if authorization and authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return None


def make_auth_dependency(
    store: Store, cfg: AppConfig, limiter: RateLimiter
) -> Callable:
    async def require_auth(
        request: Request,
        authorization: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    ) -> AuthContext | None:
        if cfg.auth_disabled:
            # Anonymous local context
            acc = store.get_account("acct-default")
            if not acc:
                bootstrap_store(store, cfg)
            request.state.auth = None
            return None

        raw = extract_bearer(authorization, x_api_key)
        if not raw:
            raise HTTPException(401, "missing API key")
        rec = store.get_api_key_by_hash(hash_api_key(raw))
        if not rec or rec.disabled:
            METRICS.outcome("client", "auth_failed")
            raise HTTPException(401, "invalid API key")
        account_id = store.account_for_key(rec.key_id)
        if not account_id:
            raise HTTPException(401, "key has no account")
        rpm = rec.rate_limit_rpm or cfg.rate_limit_rpm
        # temporarily override limiter rpm per key via burst check on key_id
        limiter.rpm = rpm
        limiter.check(rec.key_id)
        ctx = AuthContext(key=rec, account_id=account_id, raw_key=raw)
        request.state.auth = ctx
        return ctx

    return require_auth


def require_admin(ctx: AuthContext | None, cfg: AppConfig) -> None:
    if cfg.auth_disabled:
        return
    if ctx is None:
        raise HTTPException(403, "admin role required")
    if ctx.key.role == "admin":
        return
    if (
        cfg.admin_api_key
        and ctx.raw_key
        and hmac.compare_digest(ctx.raw_key, cfg.admin_api_key)
    ):
        return
    raise HTTPException(403, "admin role required")

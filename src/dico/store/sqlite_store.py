"""SQLite-backed store for production persistence."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from threading import Lock

from dico.store.base import (
    AccountRecord,
    ApiKeyRecord,
    LedgerEntry,
    ModelCheckpoint,
    Reservation,
    Store,
    UsageRecord,
)


class SQLiteStore(Store):
    def __init__(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._lock = Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._migrate()

    def _migrate(self) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.executescript(
                """
                CREATE TABLE IF NOT EXISTS accounts (
                  account_id TEXT PRIMARY KEY,
                  name TEXT NOT NULL,
                  balance_micro_usd INTEGER NOT NULL,
                  created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS api_keys (
                  key_id TEXT PRIMARY KEY,
                  key_hash TEXT UNIQUE NOT NULL,
                  name TEXT NOT NULL,
                  role TEXT NOT NULL,
                  created_at REAL NOT NULL,
                  disabled INTEGER NOT NULL DEFAULT 0,
                  rate_limit_rpm INTEGER,
                  model_allowlist TEXT NOT NULL DEFAULT '[]',
                  key_prefix TEXT NOT NULL DEFAULT '',
                  account_id TEXT NOT NULL REFERENCES accounts(account_id)
                );
                CREATE TABLE IF NOT EXISTS ledger (
                  entry_id TEXT PRIMARY KEY,
                  account_id TEXT NOT NULL,
                  kind TEXT NOT NULL,
                  amount_micro_usd INTEGER NOT NULL,
                  request_id TEXT,
                  created_at REAL NOT NULL,
                  meta TEXT NOT NULL DEFAULT '{}'
                );
                CREATE TABLE IF NOT EXISTS reservations (
                  reservation_id TEXT PRIMARY KEY,
                  account_id TEXT NOT NULL,
                  request_id TEXT NOT NULL,
                  reserved_micro_usd INTEGER NOT NULL,
                  created_at REAL NOT NULL,
                  settled INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS usage (
                  usage_id TEXT PRIMARY KEY,
                  account_id TEXT NOT NULL,
                  request_id TEXT NOT NULL,
                  kind TEXT NOT NULL,
                  units INTEGER NOT NULL,
                  cost_micro_usd INTEGER NOT NULL,
                  provider_id TEXT,
                  latency_ms REAL NOT NULL,
                  created_at REAL NOT NULL,
                  meta TEXT NOT NULL DEFAULT '{}'
                );
                CREATE TABLE IF NOT EXISTS checkpoints (
                  model_id TEXT PRIMARY KEY,
                  version INTEGER NOT NULL,
                  weights_json TEXT NOT NULL,
                  checksum_sha256 TEXT NOT NULL,
                  updated_at REAL NOT NULL
                );
                """
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def create_api_key(self, record: ApiKeyRecord, account_id: str) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO api_keys
                (key_id, key_hash, name, role, created_at, disabled, rate_limit_rpm,
                 model_allowlist, key_prefix, account_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.key_id,
                    record.key_hash,
                    record.name,
                    record.role,
                    record.created_at,
                    int(record.disabled),
                    record.rate_limit_rpm,
                    json.dumps(record.model_allowlist),
                    record.key_prefix,
                    account_id,
                ),
            )
            self._conn.commit()

    def get_api_key_by_hash(self, key_hash: str) -> ApiKeyRecord | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM api_keys WHERE key_hash = ?", (key_hash,)
            ).fetchone()
        return _row_to_key(row) if row else None

    def list_api_keys(self) -> list[ApiKeyRecord]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM api_keys").fetchall()
        return [_row_to_key(r) for r in rows]

    def disable_api_key(self, key_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE api_keys SET disabled = 1 WHERE key_id = ?", (key_id,)
            )
            self._conn.commit()

    def account_for_key(self, key_id: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT account_id FROM api_keys WHERE key_id = ?", (key_id,)
            ).fetchone()
        return str(row["account_id"]) if row else None

    def create_account(self, account: AccountRecord) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO accounts (account_id, name, balance_micro_usd, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    account.account_id,
                    account.name,
                    account.balance_micro_usd,
                    account.created_at,
                ),
            )
            self._conn.commit()

    def get_account(self, account_id: str) -> AccountRecord | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM accounts WHERE account_id = ?", (account_id,)
            ).fetchone()
        if not row:
            return None
        return AccountRecord(
            account_id=row["account_id"],
            name=row["name"],
            balance_micro_usd=int(row["balance_micro_usd"]),
            created_at=float(row["created_at"]),
        )

    def set_balance(self, account_id: str, balance_micro_usd: int) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE accounts SET balance_micro_usd = ? WHERE account_id = ?",
                (balance_micro_usd, account_id),
            )
            self._conn.commit()

    def add_ledger_entry(self, entry: LedgerEntry) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO ledger
                (entry_id, account_id, kind, amount_micro_usd, request_id, created_at, meta)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.entry_id,
                    entry.account_id,
                    entry.kind,
                    entry.amount_micro_usd,
                    entry.request_id,
                    entry.created_at,
                    json.dumps(entry.meta),
                ),
            )
            self._conn.commit()

    def list_ledger(self, account_id: str, limit: int = 100) -> list[LedgerEntry]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM ledger WHERE account_id = ?
                ORDER BY created_at DESC LIMIT ?
                """,
                (account_id, limit),
            ).fetchall()
        return [
            LedgerEntry(
                entry_id=r["entry_id"],
                account_id=r["account_id"],
                kind=r["kind"],
                amount_micro_usd=int(r["amount_micro_usd"]),
                request_id=r["request_id"],
                created_at=float(r["created_at"]),
                meta=json.loads(r["meta"] or "{}"),
            )
            for r in rows
        ]

    def put_reservation(self, reservation: Reservation) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO reservations
                (reservation_id, account_id, request_id, reserved_micro_usd, created_at, settled)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    reservation.reservation_id,
                    reservation.account_id,
                    reservation.request_id,
                    reservation.reserved_micro_usd,
                    reservation.created_at,
                    int(reservation.settled),
                ),
            )
            self._conn.commit()

    def get_reservation(self, reservation_id: str) -> Reservation | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM reservations WHERE reservation_id = ?",
                (reservation_id,),
            ).fetchone()
        if not row:
            return None
        return Reservation(
            reservation_id=row["reservation_id"],
            account_id=row["account_id"],
            request_id=row["request_id"],
            reserved_micro_usd=int(row["reserved_micro_usd"]),
            created_at=float(row["created_at"]),
            settled=bool(row["settled"]),
        )

    def mark_reservation_settled(self, reservation_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE reservations SET settled = 1 WHERE reservation_id = ?",
                (reservation_id,),
            )
            self._conn.commit()

    def add_usage(self, usage: UsageRecord) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO usage
                (usage_id, account_id, request_id, kind, units, cost_micro_usd,
                 provider_id, latency_ms, created_at, meta)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    usage.usage_id,
                    usage.account_id,
                    usage.request_id,
                    usage.kind,
                    usage.units,
                    usage.cost_micro_usd,
                    usage.provider_id,
                    usage.latency_ms,
                    usage.created_at,
                    json.dumps(usage.meta),
                ),
            )
            self._conn.commit()

    def list_usage(self, account_id: str, limit: int = 100) -> list[UsageRecord]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM usage WHERE account_id = ?
                ORDER BY created_at DESC LIMIT ?
                """,
                (account_id, limit),
            ).fetchall()
        return [
            UsageRecord(
                usage_id=r["usage_id"],
                account_id=r["account_id"],
                request_id=r["request_id"],
                kind=r["kind"],
                units=int(r["units"]),
                cost_micro_usd=int(r["cost_micro_usd"]),
                provider_id=r["provider_id"],
                latency_ms=float(r["latency_ms"]),
                created_at=float(r["created_at"]),
                meta=json.loads(r["meta"] or "{}"),
            )
            for r in rows
        ]

    def save_checkpoint(self, checkpoint: ModelCheckpoint) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO checkpoints
                (model_id, version, weights_json, checksum_sha256, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(model_id) DO UPDATE SET
                  version=excluded.version,
                  weights_json=excluded.weights_json,
                  checksum_sha256=excluded.checksum_sha256,
                  updated_at=excluded.updated_at
                """,
                (
                    checkpoint.model_id,
                    checkpoint.version,
                    checkpoint.weights_json,
                    checkpoint.checksum_sha256,
                    checkpoint.updated_at,
                ),
            )
            self._conn.commit()

    def load_checkpoint(self, model_id: str = "collective-mlp") -> ModelCheckpoint | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM checkpoints WHERE model_id = ?", (model_id,)
            ).fetchone()
        if not row:
            return None
        return ModelCheckpoint(
            model_id=row["model_id"],
            version=int(row["version"]),
            weights_json=row["weights_json"],
            checksum_sha256=row["checksum_sha256"],
            updated_at=float(row["updated_at"]),
        )


def _row_to_key(row: sqlite3.Row) -> ApiKeyRecord:
    return ApiKeyRecord(
        key_id=row["key_id"],
        key_hash=row["key_hash"],
        name=row["name"],
        role=row["role"],
        created_at=float(row["created_at"]),
        disabled=bool(row["disabled"]),
        rate_limit_rpm=row["rate_limit_rpm"],
        model_allowlist=json.loads(row["model_allowlist"] or "[]"),
        key_prefix=row["key_prefix"] or "",
    )

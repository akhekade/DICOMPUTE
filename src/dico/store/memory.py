"""In-memory store for unit tests and ephemeral demos."""

from __future__ import annotations

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


class MemoryStore(Store):
    def __init__(self) -> None:
        self._lock = Lock()
        self.keys: dict[str, ApiKeyRecord] = {}  # key_hash -> record
        self.keys_by_id: dict[str, ApiKeyRecord] = {}
        self.key_accounts: dict[str, str] = {}  # key_id -> account_id
        self.accounts: dict[str, AccountRecord] = {}
        self.ledger: list[LedgerEntry] = []
        self.reservations: dict[str, Reservation] = {}
        self.usage: list[UsageRecord] = []
        self.checkpoints: dict[str, ModelCheckpoint] = {}

    def close(self) -> None:
        return None

    def create_api_key(self, record: ApiKeyRecord, account_id: str) -> None:
        with self._lock:
            self.keys[record.key_hash] = record
            self.keys_by_id[record.key_id] = record
            self.key_accounts[record.key_id] = account_id

    def get_api_key_by_hash(self, key_hash: str) -> ApiKeyRecord | None:
        with self._lock:
            return self.keys.get(key_hash)

    def list_api_keys(self) -> list[ApiKeyRecord]:
        with self._lock:
            return list(self.keys_by_id.values())

    def disable_api_key(self, key_id: str) -> None:
        with self._lock:
            rec = self.keys_by_id.get(key_id)
            if rec:
                rec.disabled = True

    def account_for_key(self, key_id: str) -> str | None:
        with self._lock:
            return self.key_accounts.get(key_id)

    def create_account(self, account: AccountRecord) -> None:
        with self._lock:
            self.accounts[account.account_id] = account

    def get_account(self, account_id: str) -> AccountRecord | None:
        with self._lock:
            return self.accounts.get(account_id)

    def set_balance(self, account_id: str, balance_micro_usd: int) -> None:
        with self._lock:
            acc = self.accounts[account_id]
            acc.balance_micro_usd = balance_micro_usd

    def add_ledger_entry(self, entry: LedgerEntry) -> None:
        with self._lock:
            self.ledger.append(entry)

    def list_ledger(self, account_id: str, limit: int = 100) -> list[LedgerEntry]:
        with self._lock:
            rows = [e for e in self.ledger if e.account_id == account_id]
            return list(reversed(rows[-limit:]))

    def put_reservation(self, reservation: Reservation) -> None:
        with self._lock:
            self.reservations[reservation.reservation_id] = reservation

    def get_reservation(self, reservation_id: str) -> Reservation | None:
        with self._lock:
            return self.reservations.get(reservation_id)

    def mark_reservation_settled(self, reservation_id: str) -> None:
        with self._lock:
            r = self.reservations.get(reservation_id)
            if r:
                r.settled = True

    def add_usage(self, usage: UsageRecord) -> None:
        with self._lock:
            self.usage.append(usage)

    def list_usage(self, account_id: str, limit: int = 100) -> list[UsageRecord]:
        with self._lock:
            rows = [u for u in self.usage if u.account_id == account_id]
            return list(reversed(rows[-limit:]))

    def save_checkpoint(self, checkpoint: ModelCheckpoint) -> None:
        with self._lock:
            self.checkpoints[checkpoint.model_id] = checkpoint

    def load_checkpoint(self, model_id: str = "collective-mlp") -> ModelCheckpoint | None:
        with self._lock:
            return self.checkpoints.get(model_id)

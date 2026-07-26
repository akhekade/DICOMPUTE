"""Prepaid reserve → settle ledger (micro-USD), Darkbloom-inspired."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass

from fastapi import HTTPException

from dico.config import AppConfig
from dico.store.base import LedgerEntry, Reservation, Store, UsageRecord
from dico.telemetry import METRICS, get_logger

log = get_logger("dico.billing")


@dataclass
class SettleResult:
    charged_micro_usd: int
    refunded_micro_usd: int
    reservation_id: str


class BillingService:
    def __init__(self, store: Store, cfg: AppConfig) -> None:
        self.store = store
        self.cfg = cfg

    def estimate_infer_cost(self, ensemble_size: int = 1) -> int:
        return max(
            self.cfg.min_charge_micro_usd,
            self.cfg.price_infer_micro_usd * max(1, ensemble_size),
        )

    def estimate_train_cost(self, samples: int) -> int:
        units = max(1, samples // 32)
        return max(
            self.cfg.min_charge_micro_usd,
            self.cfg.price_train_micro_usd * units,
        )

    def reserve(
        self, account_id: str, request_id: str, amount_micro_usd: int
    ) -> Reservation:
        if not self.cfg.billing_enabled:
            return Reservation(
                reservation_id=f"rsv-noop-{uuid.uuid4().hex[:8]}",
                account_id=account_id,
                request_id=request_id,
                reserved_micro_usd=0,
                created_at=time.time(),
            )

        acc = self.store.get_account(account_id)
        if not acc:
            METRICS.outcome("billing", "account_missing")
            raise HTTPException(402, "account not found")
        if acc.balance_micro_usd < amount_micro_usd:
            METRICS.outcome("billing", "insufficient_funds")
            raise HTTPException(
                402,
                f"insufficient balance: have {acc.balance_micro_usd}, need {amount_micro_usd}",
            )

        new_bal = acc.balance_micro_usd - amount_micro_usd
        self.store.set_balance(account_id, new_bal)
        reservation = Reservation(
            reservation_id=f"rsv-{uuid.uuid4().hex[:12]}",
            account_id=account_id,
            request_id=request_id,
            reserved_micro_usd=amount_micro_usd,
            created_at=time.time(),
        )
        self.store.put_reservation(reservation)
        self.store.add_ledger_entry(
            LedgerEntry(
                entry_id=f"led-{uuid.uuid4().hex[:12]}",
                account_id=account_id,
                kind="reserve",
                amount_micro_usd=-amount_micro_usd,
                request_id=request_id,
                created_at=time.time(),
                meta={"reservation_id": reservation.reservation_id},
            )
        )
        METRICS.incr("billing_reserve", amount=amount_micro_usd)
        return reservation

    def settle(
        self,
        reservation: Reservation,
        actual_cost: int,
        *,
        kind: str,
        units: int,
        provider_id: str | None,
        latency_ms: float,
        meta: dict | None = None,
    ) -> SettleResult:
        if not self.cfg.billing_enabled or reservation.reserved_micro_usd == 0:
            return SettleResult(0, 0, reservation.reservation_id)

        cap = int(reservation.reserved_micro_usd * self.cfg.reserve_overage_cap_x)
        charged = min(max(self.cfg.min_charge_micro_usd, actual_cost), cap)
        refund = max(0, reservation.reserved_micro_usd - charged)

        if refund:
            acc = self.store.get_account(reservation.account_id)
            if acc:
                self.store.set_balance(
                    reservation.account_id, acc.balance_micro_usd + refund
                )
            self.store.add_ledger_entry(
                LedgerEntry(
                    entry_id=f"led-{uuid.uuid4().hex[:12]}",
                    account_id=reservation.account_id,
                    kind="refund",
                    amount_micro_usd=refund,
                    request_id=reservation.request_id,
                    created_at=time.time(),
                    meta={"reservation_id": reservation.reservation_id},
                )
            )

        self.store.add_ledger_entry(
            LedgerEntry(
                entry_id=f"led-{uuid.uuid4().hex[:12]}",
                account_id=reservation.account_id,
                kind="settle",
                amount_micro_usd=-charged,
                request_id=reservation.request_id,
                created_at=time.time(),
                meta={
                    "reservation_id": reservation.reservation_id,
                    "actual_cost": actual_cost,
                },
            )
        )
        self.store.mark_reservation_settled(reservation.reservation_id)
        self.store.add_usage(
            UsageRecord(
                usage_id=f"use-{uuid.uuid4().hex[:12]}",
                account_id=reservation.account_id,
                request_id=reservation.request_id,
                kind=kind,
                units=units,
                cost_micro_usd=charged,
                provider_id=provider_id,
                latency_ms=latency_ms,
                created_at=time.time(),
                meta=meta or {},
            )
        )
        METRICS.outcome("billing", "settled")
        METRICS.incr("billing_charged", amount=charged)
        return SettleResult(charged, refund, reservation.reservation_id)

    def deposit(self, account_id: str, amount_micro_usd: int, source: str = "admin") -> int:
        acc = self.store.get_account(account_id)
        if not acc:
            raise HTTPException(404, "account not found")
        new_bal = acc.balance_micro_usd + amount_micro_usd
        self.store.set_balance(account_id, new_bal)
        self.store.add_ledger_entry(
            LedgerEntry(
                entry_id=f"led-{uuid.uuid4().hex[:12]}",
                account_id=account_id,
                kind="deposit",
                amount_micro_usd=amount_micro_usd,
                request_id=None,
                created_at=time.time(),
                meta={"source": source},
            )
        )
        return new_bal

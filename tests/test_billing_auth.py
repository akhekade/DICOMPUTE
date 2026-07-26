import time

import pytest
from fastapi.testclient import TestClient

from dico.auth import bootstrap_store, generate_api_key, hash_api_key
from dico.billing import BillingService
from dico.config import load_config
from dico.coordinator import CoordinatorState, create_coordinator_app
from dico.store import MemoryStore
from dico.store.base import AccountRecord, ApiKeyRecord


def test_reserve_settle_refund():
    cfg = load_config(
        auth_disabled=True,
        billing_enabled=True,
        store_backend="memory",
        database_url="memory://",
        price_infer_micro_usd=100,
        min_charge_micro_usd=10,
    )
    store = MemoryStore()
    store.create_account(
        AccountRecord(
            account_id="acct-a",
            name="a",
            balance_micro_usd=1000,
            created_at=time.time(),
        )
    )
    billing = BillingService(store, cfg)
    rsv = billing.reserve("acct-a", "req-1", 500)
    assert store.get_account("acct-a").balance_micro_usd == 500
    result = billing.settle(
        rsv,
        actual_cost=120,
        kind="infer",
        units=1,
        provider_id="n1",
        latency_ms=12.0,
    )
    assert result.charged_micro_usd == 120
    assert result.refunded_micro_usd == 380
    assert store.get_account("acct-a").balance_micro_usd == 880


def test_insufficient_funds():
    cfg = load_config(
        billing_enabled=True,
        store_backend="memory",
        database_url="memory://",
    )
    store = MemoryStore()
    store.create_account(
        AccountRecord(
            account_id="acct-b",
            name="b",
            balance_micro_usd=10,
            created_at=time.time(),
        )
    )
    billing = BillingService(store, cfg)
    with pytest.raises(Exception) as exc:
        billing.reserve("acct-b", "req-2", 100)
    assert exc.value.status_code == 402


def test_api_key_auth_required():
    cfg = load_config(
        auth_disabled=False,
        billing_enabled=True,
        store_backend="memory",
        database_url="memory://",
        bootstrap_api_key="testsecretkeyvalue123",
        data_dir="/tmp/dico-auth-test",
    )
    store = MemoryStore()
    bootstrap_store(store, cfg)
    state = CoordinatorState(cfg, store)
    app = create_coordinator_app(state)
    client = TestClient(app)

    denied = client.get("/status")
    assert denied.status_code == 401

    raw = "sk-dico-testsecretkeyvalue123"
    ok = client.get("/status", headers={"Authorization": f"Bearer {raw}"})
    assert ok.status_code == 200


def test_create_key_admin():
    raw, h, prefix = generate_api_key()
    cfg = load_config(
        auth_disabled=False,
        store_backend="memory",
        database_url="memory://",
        data_dir="/tmp/dico-key-test",
    )
    store = MemoryStore()
    store.create_account(
        AccountRecord(
            account_id="acct-default",
            name="default",
            balance_micro_usd=1_000_000,
            created_at=time.time(),
        )
    )
    store.create_api_key(
        ApiKeyRecord(
            key_id="key-admin",
            key_hash=h,
            name="admin",
            role="admin",
            created_at=time.time(),
            key_prefix=prefix,
        ),
        account_id="acct-default",
    )
    state = CoordinatorState(cfg, store)
    app = create_coordinator_app(state)
    client = TestClient(app)
    resp = client.post(
        "/admin/keys",
        headers={"Authorization": f"Bearer {raw}"},
        json={"name": "user1", "role": "consumer"},
    )
    assert resp.status_code == 200
    assert resp.json()["api_key"].startswith("sk-dico-")
    assert hash_api_key(resp.json()["api_key"])

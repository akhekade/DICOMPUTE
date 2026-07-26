import httpx
import pytest
from fastapi.testclient import TestClient

from dico.config import load_config
from dico.coordinator import CoordinatorState, create_coordinator_app
from dico.hardware import probe_capabilities
from dico.node import ProviderState, create_provider_app
from dico.protocol import NodeInfo, NodeRole, RegisterRequest
from dico.store import MemoryStore
from dico.auth import bootstrap_store


def _coord_app():
    cfg = load_config(
        auth_disabled=True,
        billing_enabled=True,
        store_backend="memory",
        database_url="memory://",
        data_dir="/tmp/dico-test",
    )
    store = MemoryStore()
    bootstrap_store(store, cfg)
    state = CoordinatorState(cfg, store)
    return create_coordinator_app(state), state


@pytest.fixture
def silence_provider_network(monkeypatch):
    async def noop(self):
        return None

    monkeypatch.setattr(ProviderState, "register_http", noop)
    monkeypatch.setattr(ProviderState, "leave_http", noop)

    async def idle_heartbeat(self):
        await self._stop.wait()

    monkeypatch.setattr(ProviderState, "heartbeat_loop_http", idle_heartbeat)


def test_coordinator_health_and_status():
    app, state = _coord_app()
    client = TestClient(app)
    assert client.get("/health").json()["ok"] is True
    status = client.get("/status").json()
    assert status["online_providers"] == 0
    assert status["coordinator_id"] == state.node_id
    assert status["protocol_version"] == 1


def test_register_heartbeat_and_infer():
    app, _state = _coord_app()
    client = TestClient(app)

    node = NodeInfo(
        node_id="node-test",
        role=NodeRole.PROVIDER,
        host="127.0.0.1",
        port=9999,
        capabilities=probe_capabilities(),
    )
    reg = client.post(
        "/nodes/register", json=RegisterRequest(node=node).model_dump()
    )
    assert reg.status_code == 200
    assert "weights" in reg.json()
    assert "checksum_sha256" in reg.json()

    hb = client.post(
        "/nodes/heartbeat",
        json={
            "node_id": "node-test",
            "load": 0.2,
            "model_version": 0,
            "status": "online",
            "capacity": {
                "max_slots": 2,
                "free_slots": 2,
                "queued": 0,
                "warm_models": ["collective-mlp"],
            },
        },
    )
    assert hb.status_code == 200
    assert client.get("/status").json()["online_providers"] == 1

    resp = client.post(
        "/infer",
        json={
            "features": [0.1] * 8,
            "ensemble": True,
            "timeout_s": 0.2,
            "routing": "ensemble",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "prediction" in body
    assert len(body["probabilities"]) == 3
    assert body["contributors"]


def test_train_round_local_only():
    app, _state = _coord_app()
    client = TestClient(app)
    before = client.get("/model").json()["model_version"]
    resp = client.post(
        "/train/round",
        json={"epochs": 2, "samples": 64, "learning_rate": 0.1, "batch_size": 16},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["num_contributors"] >= 1
    assert data["model_version"] == before + 1
    assert data["checksum_sha256"]


def test_provider_infer_train_sync(silence_provider_network):
    state = ProviderState(
        coordinator_url="http://127.0.0.1:7400",
        host="127.0.0.1",
        port=7401,
        node_id="prov-1",
        transport="http",
    )
    app = create_provider_app(state)
    with TestClient(app) as client:
        health = client.get("/health").json()
        assert health["node_id"] == "prov-1"
        assert "capacity" in health

        inf = client.post("/infer", json={"features": [0.0] * 8})
        assert inf.status_code == 200
        assert inf.json()["node_id"] == "prov-1"

        tr = client.post(
            "/train",
            json={
                "epochs": 1,
                "samples": 32,
                "learning_rate": 0.05,
                "batch_size": 16,
            },
        )
        assert tr.status_code == 200
        assert "weights" in tr.json()

        weights = state.model.export_weights()
        from dico.checksum import weights_checksum

        synced = client.post(
            "/model/sync",
            json={
                "model_version": 3,
                "weights": weights,
                "checksum_sha256": weights_checksum(weights),
            },
        )
        assert synced.status_code == 200
        assert synced.json()["model_version"] == 3


@pytest.mark.asyncio
async def test_fedavg_submit_and_collective_infer():
    app, _coord_state = _coord_app()

    prov_state = ProviderState(
        coordinator_url="http://coordinator",
        host="127.0.0.1",
        port=7401,
        node_id="e2e-node",
        transport="http",
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://coordinator"
    ) as coord:
        reg = await coord.post(
            "/nodes/register",
            json=RegisterRequest(node=prov_state.info).model_dump(),
        )
        assert reg.status_code == 200
        prov_state.model.load_weights(reg.json()["weights"])

        x_loss = prov_state.model.train_batch(
            *__import__("dico.model", fromlist=["make_synthetic_dataset"]).make_synthetic_dataset(
                n=48, seed=7
            ),
            lr=0.1,
            epochs=2,
            batch_size=16,
        )
        update = {
            "node_id": prov_state.node_id,
            "model_version": prov_state.model.version,
            "num_samples": 48,
            "weights": prov_state.model.export_weights(),
            "loss": float(x_loss),
        }
        agg = await coord.post("/train/submit", json=update)
        assert agg.status_code == 200
        assert agg.json()["model_version"] >= 1

        inf = await coord.post(
            "/infer",
            json={
                "features": [0.2, -0.1, 0.3, 0.0, -0.4, 0.5, 0.1, -0.2],
                "ensemble": False,
                "routing": "local",
            },
        )
        assert inf.status_code == 200
        assert "prediction" in inf.json()


def test_openai_chat_and_metrics():
    app, _state = _coord_app()
    client = TestClient(app)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "collective-mlp",
            "messages": [{"role": "user", "content": "hello mesh"}],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["choices"][0]["message"]["content"]
    assert "dico" in body
    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    assert "dico_counter" in metrics.text or "dico_outcome" in metrics.text

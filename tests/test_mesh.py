import httpx
import pytest
from fastapi.testclient import TestClient

from dico.coordinator import CoordinatorState, create_coordinator_app
from dico.hardware import probe_capabilities
from dico.node import ProviderState, create_provider_app
from dico.protocol import NodeInfo, NodeRole, RegisterRequest


@pytest.fixture
def silence_provider_network(monkeypatch):
    async def noop(self):
        return None

    monkeypatch.setattr(ProviderState, "register", noop)
    monkeypatch.setattr(ProviderState, "leave", noop)

    async def idle_heartbeat(self):
        await self._stop.wait()

    monkeypatch.setattr(ProviderState, "heartbeat_loop", idle_heartbeat)


def test_coordinator_health_and_status():
    state = CoordinatorState(host="127.0.0.1", port=7400)
    app = create_coordinator_app(state)
    client = TestClient(app)
    assert client.get("/health").json()["ok"] is True
    status = client.get("/status").json()
    assert status["online_providers"] == 0
    assert status["coordinator_id"] == state.node_id


def test_register_heartbeat_and_infer():
    state = CoordinatorState(host="127.0.0.1", port=7400)
    app = create_coordinator_app(state)
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

    hb = client.post(
        "/nodes/heartbeat",
        json={
            "node_id": "node-test",
            "load": 0.2,
            "model_version": 0,
            "status": "online",
        },
    )
    assert hb.status_code == 200
    assert client.get("/status").json()["online_providers"] == 1

    resp = client.post(
        "/infer",
        json={"features": [0.1] * 8, "ensemble": True, "timeout_s": 0.2},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "prediction" in body
    assert len(body["probabilities"]) == 3
    assert body["contributors"]


def test_train_round_local_only():
    state = CoordinatorState(host="127.0.0.1", port=7400)
    app = create_coordinator_app(state)
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


def test_provider_infer_train_sync(silence_provider_network):
    state = ProviderState(
        coordinator_url="http://127.0.0.1:7400",
        host="127.0.0.1",
        port=7401,
        node_id="prov-1",
    )
    app = create_provider_app(state)
    with TestClient(app) as client:
        health = client.get("/health").json()
        assert health["node_id"] == "prov-1"

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

        synced = client.post(
            "/model/sync",
            json={"model_version": 3, "weights": state.model.export_weights()},
        )
        assert synced.status_code == 200
        assert synced.json()["model_version"] == 3


@pytest.mark.asyncio
async def test_fedavg_submit_and_collective_infer():
    coord_state = CoordinatorState(host="127.0.0.1", port=7400)
    coord_app = create_coordinator_app(coord_state)

    prov_state = ProviderState(
        coordinator_url="http://coordinator",
        host="127.0.0.1",
        port=7401,
        node_id="e2e-node",
    )

    transport = httpx.ASGITransport(app=coord_app)
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
            },
        )
        assert inf.status_code == 200
        assert "prediction" in inf.json()

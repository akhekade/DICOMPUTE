import time

from dico.config import load_config
from dico.hardware import probe_capabilities
from dico.protocol import BackendCapacity, Heartbeat, NodeInfo, NodeRole
from dico.registry import ProviderRegistry


def _node(nid: str, free: int = 2, load: float = 0.1) -> NodeInfo:
    return NodeInfo(
        node_id=nid,
        role=NodeRole.PROVIDER,
        host="127.0.0.1",
        port=7401,
        capabilities=probe_capabilities(),
        load=load,
        capacity=BackendCapacity(
            max_slots=2,
            free_slots=free,
            warm_models=["collective-mlp"],
        ),
        last_heartbeat=time.time(),
    )


def test_cost_scheduler_prefers_lighter_node():
    cfg = load_config(store_backend="memory", database_url="memory://")
    reg = ProviderRegistry(cfg)
    a = _node("a", free=2, load=0.1)
    b = _node("b", free=1, load=0.9)
    a.transport = "websocket"
    b.transport = "http"
    reg.register(a)
    reg.register(b)
    chosen = reg.select(k=1, strategy="cheapest")
    assert chosen[0].node.node_id == "a"


def test_cooldown_after_errors():
    cfg = load_config(
        store_backend="memory",
        database_url="memory://",
        max_consecutive_errors=2,
        error_cooldown_s=60,
    )
    reg = ProviderRegistry(cfg)
    n = _node("n1")
    reg.register(n)
    reg.record_error("n1")
    reg.record_error("n1")
    assert reg.nodes["n1"].status == "cooldown"
    assert reg.select(k=1) == []


def test_stale_eviction():
    cfg = load_config(
        store_backend="memory",
        database_url="memory://",
        heartbeat_ttl_s=0.01,
        heartbeat_interval_s=0.001,
    )
    reg = ProviderRegistry(cfg)
    n = _node("stale")
    n.last_heartbeat = time.time() - 1
    reg.nodes["stale"] = n
    assert reg.prune_stale() == ["stale"]


def test_heartbeat_updates_capacity():
    cfg = load_config(store_backend="memory", database_url="memory://")
    reg = ProviderRegistry(cfg)
    reg.register(_node("h1"))
    reg.heartbeat(
        Heartbeat(
            node_id="h1",
            load=0.5,
            capacity=BackendCapacity(max_slots=4, free_slots=3, queued=1),
        )
    )
    assert reg.nodes["h1"].capacity.free_slots == 3
    assert reg.nodes["h1"].capacity.queued == 1

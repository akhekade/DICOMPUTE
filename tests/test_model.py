import numpy as np

from dico.model import CollectiveMLP, federated_average, make_synthetic_dataset


def test_predict_shape():
    model = CollectiveMLP(seed=0)
    pred, probs = model.predict([0.1] * 8)
    assert pred in {0.0, 1.0, 2.0}
    assert len(probs) == 3
    assert abs(sum(probs) - 1.0) < 1e-6


def test_train_reduces_loss():
    model = CollectiveMLP(seed=1)
    x, y = make_synthetic_dataset(n=200, seed=2)
    loss0, _ = model.loss_and_grads(x, y)
    model.train_batch(x, y, lr=0.1, epochs=8, batch_size=32)
    loss1, _ = model.loss_and_grads(x, y)
    assert loss1 < loss0


def test_federated_average_versions():
    a = CollectiveMLP(seed=3)
    b = CollectiveMLP(seed=4)
    a.version = 2
    b.version = 2
    out = federated_average(
        [(a.export_weights(), 10), (b.export_weights(), 30)]
    )
    assert out["version"] == [3]
    # closer to b because of higher weight
    avg_w1 = np.asarray(out["w1"])
    expected = (np.asarray(a.w1) * 0.25) + (np.asarray(b.w1) * 0.75)
    assert np.allclose(avg_w1, expected)


def test_weight_roundtrip():
    model = CollectiveMLP(seed=5)
    model.version = 9
    clone = CollectiveMLP(seed=None)
    clone.load_weights(model.export_weights())
    assert clone.version == 9
    assert np.allclose(clone.w1, model.w1)

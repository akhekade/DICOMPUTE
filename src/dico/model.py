"""Tiny numpy MLP used for federated training and collective inference.

Designed to run on Mac Minis / laptops without heavy ML frameworks.
Weights serialize cleanly over JSON for FedAvg aggregation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class CollectiveMLP:
    """2-layer MLP for binary/multiclass toy intelligence tasks."""

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 16,
        output_dim: int = 3,
        seed: int | None = 42,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.version = 0
        rng = np.random.default_rng(seed)
        scale_w1 = np.sqrt(2.0 / input_dim)
        scale_w2 = np.sqrt(2.0 / hidden_dim)
        self.w1 = rng.normal(0, scale_w1, size=(input_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.w2 = rng.normal(0, scale_w2, size=(hidden_dim, output_dim))
        self.b2 = np.zeros(output_dim)

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        x = np.atleast_2d(x).astype(np.float64)
        z1 = x @ self.w1 + self.b1
        a1 = np.tanh(z1)
        logits = a1 @ self.w2 + self.b2
        # stable softmax
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / exp.sum(axis=1, keepdims=True)
        cache = {"x": x, "z1": z1, "a1": a1, "logits": logits, "probs": probs}
        return probs, cache

    def predict(self, features: list[float] | np.ndarray) -> tuple[float, list[float]]:
        probs, _ = self.forward(np.asarray(features, dtype=np.float64))
        p = probs[0]
        return float(np.argmax(p)), [float(v) for v in p]

    def loss_and_grads(
        self, x: np.ndarray, y: np.ndarray
    ) -> tuple[float, dict[str, np.ndarray]]:
        probs, cache = self.forward(x)
        n = x.shape[0]
        y = y.astype(np.int64)
        # cross-entropy
        log_likelihood = -np.log(probs[np.arange(n), y] + 1e-12)
        loss = float(log_likelihood.mean())

        dlogits = probs
        dlogits[np.arange(n), y] -= 1.0
        dlogits /= n

        grads = {
            "w2": cache["a1"].T @ dlogits,
            "b2": dlogits.sum(axis=0),
        }
        da1 = dlogits @ self.w2.T
        dz1 = da1 * (1.0 - np.tanh(cache["z1"]) ** 2)
        grads["w1"] = cache["x"].T @ dz1
        grads["b1"] = dz1.sum(axis=0)
        return loss, grads

    def train_batch(
        self,
        x: np.ndarray,
        y: np.ndarray,
        lr: float = 0.05,
        epochs: int = 3,
        batch_size: int = 32,
    ) -> float:
        n = x.shape[0]
        last_loss = 0.0
        for _ in range(epochs):
            idx = np.random.permutation(n)
            for start in range(0, n, batch_size):
                batch = idx[start : start + batch_size]
                last_loss, grads = self.loss_and_grads(x[batch], y[batch])
                self.w1 -= lr * grads["w1"]
                self.b1 -= lr * grads["b1"]
                self.w2 -= lr * grads["w2"]
                self.b2 -= lr * grads["b2"]
        return last_loss

    def export_weights(self) -> dict[str, list[Any]]:
        return {
            "w1": self.w1.tolist(),
            "b1": self.b1.tolist(),
            "w2": self.w2.tolist(),
            "b2": self.b2.tolist(),
            "input_dim": [self.input_dim],
            "hidden_dim": [self.hidden_dim],
            "output_dim": [self.output_dim],
            "version": [self.version],
        }

    def load_weights(self, weights: dict[str, list[Any]]) -> None:
        self.w1 = np.asarray(weights["w1"], dtype=np.float64)
        self.b1 = np.asarray(weights["b1"], dtype=np.float64)
        self.w2 = np.asarray(weights["w2"], dtype=np.float64)
        self.b2 = np.asarray(weights["b2"], dtype=np.float64)
        self.input_dim = int(weights["input_dim"][0])
        self.hidden_dim = int(weights["hidden_dim"][0])
        self.output_dim = int(weights["output_dim"][0])
        self.version = int(weights.get("version", [0])[0])

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.export_weights()), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CollectiveMLP":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        model = cls(
            input_dim=int(data["input_dim"][0]),
            hidden_dim=int(data["hidden_dim"][0]),
            output_dim=int(data["output_dim"][0]),
            seed=None,
        )
        model.load_weights(data)
        return model


def federated_average(updates: list[tuple[dict[str, list[Any]], int]]) -> dict[str, list[Any]]:
    """Weighted FedAvg over exported weight dicts."""
    if not updates:
        raise ValueError("no updates to average")
    total = sum(max(1, n) for _, n in updates)
    keys = ("w1", "b1", "w2", "b2")
    acc: dict[str, np.ndarray] = {}
    meta = updates[0][0]
    for weights, n in updates:
        w = max(1, n) / total
        for key in keys:
            arr = np.asarray(weights[key], dtype=np.float64) * w
            acc[key] = arr if key not in acc else acc[key] + arr
    return {
        "w1": acc["w1"].tolist(),
        "b1": acc["b1"].tolist(),
        "w2": acc["w2"].tolist(),
        "b2": acc["b2"].tolist(),
        "input_dim": meta["input_dim"],
        "hidden_dim": meta["hidden_dim"],
        "output_dim": meta["output_dim"],
        "version": [int(meta.get("version", [0])[0]) + 1],
    }


def make_synthetic_dataset(
    n: int = 256,
    input_dim: int = 8,
    output_dim: int = 3,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a linearly-ish separable toy dataset for local training demos."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, input_dim))
    # latent class directions
    centers = rng.normal(size=(output_dim, input_dim))
    scores = x @ centers.T
    y = scores.argmax(axis=1)
    return x.astype(np.float64), y.astype(np.int64)

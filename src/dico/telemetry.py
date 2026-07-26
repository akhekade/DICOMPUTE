"""Structured logging and in-process metrics (client / provider / billing outcomes)."""

from __future__ import annotations

import logging
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from threading import Lock
from typing import Any


@dataclass
class MetricsSnapshot:
    counters: dict[str, int]
    latencies_ms: dict[str, list[float]]
    outcomes: dict[str, int]


class Metrics:
    def __init__(self) -> None:
        self._lock = Lock()
        self._counters: dict[str, int] = defaultdict(int)
        self._latencies: dict[str, list[float]] = defaultdict(list)
        self._outcomes: dict[str, int] = defaultdict(int)

    def incr(self, name: str, n: int = 1, **tags: Any) -> None:
        key = name if not tags else f"{name}|{_tag_str(tags)}"
        with self._lock:
            self._counters[key] += n

    def observe_ms(self, name: str, value_ms: float, **tags: Any) -> None:
        key = name if not tags else f"{name}|{_tag_str(tags)}"
        with self._lock:
            bucket = self._latencies[key]
            bucket.append(value_ms)
            if len(bucket) > 500:
                del bucket[:250]

    def outcome(self, kind: str, value: str) -> None:
        """kind: client | provider | billing"""
        with self._lock:
            self._outcomes[f"{kind}:{value}"] += 1

    def snapshot(self) -> MetricsSnapshot:
        with self._lock:
            return MetricsSnapshot(
                counters=dict(self._counters),
                latencies_ms={k: list(v) for k, v in self._latencies.items()},
                outcomes=dict(self._outcomes),
            )

    def prometheus_text(self) -> str:
        snap = self.snapshot()
        lines = ["# HELP dico_info DICOMPUTE metrics", "# TYPE dico_info gauge"]
        for k, v in sorted(snap.counters.items()):
            lines.append(f'dico_counter{{name="{_esc(k)}"}} {v}')
        for k, v in sorted(snap.outcomes.items()):
            lines.append(f'dico_outcome{{name="{_esc(k)}"}} {v}')
        for k, values in sorted(snap.latencies_ms.items()):
            if not values:
                continue
            avg = sum(values) / len(values)
            lines.append(
                f'dico_latency_ms_avg{{name="{_esc(k)}"}} {avg:.3f}'
            )
            lines.append(
                f'dico_latency_ms_count{{name="{_esc(k)}"}} {len(values)}'
            )
        return "\n".join(lines) + "\n"


def _tag_str(tags: dict[str, Any]) -> str:
    return ",".join(f"{k}={v}" for k, v in sorted(tags.items()))


def _esc(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


METRICS = Metrics()


class RequestTimer:
    def __init__(self, name: str, **tags: Any) -> None:
        self.name = name
        self.tags = tags
        self._t0 = 0.0

    def __enter__(self) -> "RequestTimer":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        METRICS.observe_ms(
            self.name, (time.perf_counter() - self._t0) * 1000, **self.tags
        )


def setup_logging(level: str = "INFO", json_logs: bool = False) -> None:
    root = logging.getLogger()
    root.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    if json_logs:
        formatter = logging.Formatter(
            '{"ts":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s",'
            '"msg":%(message)s}'
        )
    else:
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s [%(name)s] %(message)s"
        )
    handler.setFormatter(formatter)
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)

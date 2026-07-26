"""Checksum helpers for weight integrity."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def weights_checksum(weights: dict[str, list[Any]]) -> str:
    payload = json.dumps(weights, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

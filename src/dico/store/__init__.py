"""Persistence layer: MemoryStore for tests, SQLiteStore for production."""

from __future__ import annotations

from dico.store.base import Store
from dico.store.memory import MemoryStore
from dico.store.sqlite_store import SQLiteStore


def open_store(database_url: str) -> Store:
    if database_url.startswith("memory:"):
        return MemoryStore()
    if database_url.startswith("sqlite:"):
        # sqlite:///absolute/or/relative/path
        path = database_url.removeprefix("sqlite:///")
        return SQLiteStore(path)
    raise ValueError(f"unsupported store backend: {database_url}")


__all__ = ["Store", "MemoryStore", "SQLiteStore", "open_store"]

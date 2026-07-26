"""Local capability probing for Mac Minis and laptops."""

from __future__ import annotations

import platform
import socket
import subprocess
from functools import lru_cache

import psutil

from dico.protocol import NodeCapabilities


@lru_cache(maxsize=1)
def detect_metal() -> bool:
    system = platform.system()
    machine = platform.machine().lower()
    if system != "Darwin":
        return False
    # Apple Silicon almost always has Metal; also check system_profiler lightly.
    if machine in {"arm64", "aarch64"}:
        return True
    try:
        out = subprocess.check_output(
            ["system_profiler", "SPDisplaysDataType"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=3,
        )
        return "Metal" in out
    except Exception:
        return False


@lru_cache(maxsize=1)
def detect_cuda() -> bool:
    try:
        subprocess.check_output(
            ["nvidia-smi"], stderr=subprocess.DEVNULL, timeout=2
        )
        return True
    except Exception:
        return False


def probe_capabilities(max_concurrent_tasks: int = 2) -> NodeCapabilities:
    from dico import __version__

    mem = psutil.virtual_memory()
    return NodeCapabilities(
        cpu_cores=psutil.cpu_count(logical=True) or 1,
        memory_gb=round(mem.total / (1024**3), 2),
        has_metal=detect_metal(),
        has_cuda=detect_cuda(),
        platform=f"{platform.system()}-{platform.machine()}",
        hostname=socket.gethostname(),
        max_concurrent_tasks=max_concurrent_tasks,
        software_version=__version__,
    )


def local_ip() -> str:
    """Best-effort LAN IP (not 127.0.0.1)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"

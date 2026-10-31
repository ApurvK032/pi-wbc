"""Record where a result came from: commit, package versions, GPU.

Every result file in this project embeds this record, so any number can be
traced back to the code and machine that produced it.
"""

from __future__ import annotations

import gzip
import json
import platform
import subprocess
import sys
from collections.abc import Iterable
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

DEFAULT_PACKAGES = (
    "pi-wbc",
    "isaaclab",
    "newton",
    "warp-lang",
    "torch",
    "rsl-rl-lib",
    "mujoco",
    "onnxruntime",
    "numpy",
)


def _run(cmd: list[str], cwd: Path | None = None) -> str | None:
    """Run a command and return its stripped stdout, or None if it fails."""
    try:
        out = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=10, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def git_info(repo: Path | None = None) -> dict[str, Any]:
    """Commit hash, whether the working tree differs from it, and the changed paths.

    Changes under `results/` (new or rewritten files) don't count: they are the outputs of the
    runs being recorded, never inputs to the code, and a batch of runs would otherwise mark
    every run after its first as dirty. Any other change, tracked or new, does. Defaults to
    the repository this file lives in; outside a git checkout every field is None.
    """
    repo = repo if repo is not None else Path(__file__).resolve().parent
    commit = _run(["git", "rev-parse", "HEAD"], cwd=repo)
    if commit is None:
        return {"commit": None, "dirty": None, "changes": None}
    # -z: NUL-separated entries "XY path", with paths not quoted or stripped.
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain", "-z", "--untracked-files=all"],
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {"commit": commit, "dirty": None, "changes": None}
    changes = [
        entry[3:]
        for entry in out.split("\0")
        if len(entry) > 3 and not entry[3:].startswith("results/")
    ]
    return {"commit": commit, "dirty": bool(changes), "changes": changes[:20]}


def package_versions(names: Iterable[str] = DEFAULT_PACKAGES) -> dict[str, str | None]:
    """Installed version of each package, or None if it isn't installed."""
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def parse_nvidia_smi(output: str) -> list[dict[str, str]]:
    """Parse `nvidia-smi --query-gpu=name,driver_version,memory.total` CSV output."""
    gpus = []
    for line in output.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            continue
        name, driver, memory_mib = parts
        gpus.append({"name": name, "driver": driver, "memory_mib": memory_mib})
    return gpus


def gpu_info() -> list[dict[str, str]]:
    """Name, driver and memory of each visible NVIDIA GPU (empty if none)."""
    out = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    return parse_nvidia_smi(out) if out else []


def torch_cuda() -> str | None:
    """CUDA version torch was built with, only if torch is already imported.

    Importing torch just to ask would add seconds to every call.
    """
    torch = sys.modules.get("torch")
    if torch is None:
        return None
    return getattr(torch.version, "cuda", None)


def collect(repo: Path | None = None, packages: Iterable[str] = DEFAULT_PACKAGES) -> dict[str, Any]:
    """Full provenance record for a result file."""
    return {
        "time_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git": git_info(repo),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "host": platform.node(),
        "packages": package_versions(packages),
        "torch_cuda": torch_cuda(),
        "gpus": gpu_info(),
    }


def save_json(path: str | Path, data: dict[str, Any], repo: Path | None = None) -> Path:
    """Write `data` plus a provenance record to a JSON file and return its path.

    A path ending in `.gz` is written gzip-compressed (for large result files, such as those
    with thousands of episodes), without a timestamp, so equal data gives equal bytes.
    """
    if "provenance" in data:
        raise ValueError("data already has a 'provenance' key")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**data, "provenance": collect(repo)}
    text = json.dumps(payload, indent=2) + "\n"
    if path.suffix == ".gz":
        path.write_bytes(gzip.compress(text.encode(), compresslevel=9, mtime=0))
    else:
        path.write_text(text)
    return path


def load_json(path: str | Path) -> Any:
    """Read a JSON file written by `save_json`, gzip-compressed (`.gz`) or not."""
    path = Path(path)
    raw = path.read_bytes()
    if path.suffix == ".gz":
        raw = gzip.decompress(raw)
    return json.loads(raw)

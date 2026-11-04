"""scripts/build_dataset_manifest.py: D-042's rule on small made-up runs."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_dataset_manifest.py"
GIT = {"git": {"commit": "abc", "dirty": False}}


def _run_files(tmp: Path, name: str, rows: list[tuple[int, str, bool]]) -> list[str]:
    """rows: (seed, expert label, replay passes)."""
    rec = tmp / name
    rec.mkdir()
    for seed, _, _ in rows:
        (rec / f"{seed}.npz").write_bytes(b"")
    expert = {"controller_run": "run", "provenance": GIT, "expert_params": {"lean": 0.28},
              "results": [{"seed": s, "label": lab, "failed_in": None, "time_s": 30.0,
                           "notes": {"hand": "right"}} for s, lab, _ in rows]}  # fmt: skip
    replay = {"provenance": GIT, "results": [
        {"seed": s, "recorded": lab, "replayed": lab, "first_second_drift_mm": 1.0 if ok else 20.0,
         "passes": ok} for s, lab, ok in rows]}  # fmt: skip
    (tmp / f"{name}-expert.json").write_text(json.dumps(expert))
    (tmp / f"{name}-replay.json").write_text(json.dumps(replay))
    return [str(tmp / f"{name}-expert.json"), str(tmp / f"{name}-replay.json"), str(rec)]


def _build(tmp: Path, runs: list[list[str]], count: int | None):
    out = tmp / "manifest.json"
    cmd = [sys.executable, str(SCRIPT), "--name", "test", "--out", str(out)]
    for r in runs:
        cmd += ["--run", *r]
    if count is not None:
        cmd += ["--count", str(count)]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    return proc, (json.loads(out.read_text()) if out.exists() else None)


def test_rule(tmp_path: Path) -> None:
    a = _run_files(tmp_path, "a", [(10, "success", True), (11, "missed_grasp", True),
                                   (12, "success", False), (13, "success", True)])  # fmt: skip
    b = _run_files(tmp_path, "b", [(14, "success", True), (15, "success", True)])
    proc, m = _build(tmp_path, [a, b], count=3)
    assert proc.returncode == 0, proc.stderr
    assert m["chosen_seeds"] == [10, 13, 14]  # seed order; 11 failed, 12 dropped by the replay
    assert m["dropped_by_replay_check"] == [12]
    assert m["recorded"] == 6 and m["successes"] == 5 and m["replay_passes"] == 5
    assert m["labels"] == {"success": 5, "missed_grasp": 1}
    assert m["rule"].startswith("D-042: the first 3 successful episodes")
    proc, m = _build(tmp_path, [a, b], count=None)
    assert m["chosen_seeds"] == [10, 13, 14, 15]


def test_too_few_and_mismatches_are_refused(tmp_path: Path) -> None:
    a = _run_files(tmp_path, "a", [(10, "success", True), (11, "success", True)])
    proc, _ = _build(tmp_path, [a], count=3)
    assert proc.returncode != 0 and "only 2 episodes qualify" in proc.stderr
    (Path(a[2]) / "99.npz").write_bytes(b"")  # a recording the expert file doesn't list
    proc, _ = _build(tmp_path, [a], count=None)
    assert proc.returncode != 0 and "differ" in proc.stderr


@pytest.mark.skipif(not (ROOT / "logs/expert/train-v1.json").exists(), reason="needs the logs")
def test_rebuilds_the_committed_v1_manifest(tmp_path: Path) -> None:
    run = ["logs/expert/train-v1.json", "logs/replay/train-v1.json", "logs/dataset/train-v1"]
    proc, m = _build(tmp_path, [run], count=1000)
    assert proc.returncode == 0, proc.stderr
    committed = json.loads((ROOT / "results/stage3/dataset/train-v1-manifest.json").read_text())
    assert m["chosen_seeds"] == committed["chosen_seeds"]
    assert m["dropped_by_replay_check"] == committed["dropped_by_replay_check"]

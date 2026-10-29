import json
import subprocess
from pathlib import Path

import pytest

from pi_wbc import provenance


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "-q")
    (tmp_path / "a.txt").write_text("one\n")
    _git(tmp_path, "add", "a.txt")
    _git(tmp_path, "commit", "-q", "-m", "First")
    return tmp_path


def test_git_info_clean_then_dirty(repo: Path) -> None:
    info = provenance.git_info(repo)
    assert len(info["commit"]) == 40
    assert info["dirty"] is False

    (repo / "a.txt").write_text("two\n")
    info = provenance.git_info(repo)
    assert info["dirty"] is True
    assert info["changes"] == ["a.txt"]


def test_result_files_do_not_make_the_tree_dirty(repo: Path) -> None:
    (repo / "results" / "stage0").mkdir(parents=True)
    (repo / "results" / "stage0" / "run.json").write_text("{}")
    assert provenance.git_info(repo)["dirty"] is False
    _git(repo, "add", "results")
    _git(repo, "commit", "-q", "-m", "Add a result")
    (repo / "results" / "stage0" / "run.json").write_text('{"rerun": true}')
    assert provenance.git_info(repo)["dirty"] is False  # rewriting a result doesn't count

    (repo / "new_module.py").write_text("")
    info = provenance.git_info(repo)
    assert info["dirty"] is True
    assert info["changes"] == ["new_module.py"]


def test_git_info_outside_repo(tmp_path: Path) -> None:
    assert provenance.git_info(tmp_path) == {"commit": None, "dirty": None, "changes": None}


def test_package_versions_reports_missing_as_none() -> None:
    versions = provenance.package_versions(["pytest", "surely-not-a-real-package-xyz"])
    assert versions["pytest"] == pytest.__version__
    assert versions["surely-not-a-real-package-xyz"] is None


def test_parse_nvidia_smi() -> None:
    out = "NVIDIA GeForce RTX 4070, 595.99.02, 12282\nNVIDIA A40, 580.65.06, 46068\n"
    assert provenance.parse_nvidia_smi(out) == [
        {"name": "NVIDIA GeForce RTX 4070", "driver": "595.99.02", "memory_mib": "12282"},
        {"name": "NVIDIA A40", "driver": "580.65.06", "memory_mib": "46068"},
    ]


def test_parse_nvidia_smi_skips_malformed_lines() -> None:
    assert provenance.parse_nvidia_smi("garbage\n\n") == []


def test_save_json_adds_provenance(repo: Path, tmp_path: Path) -> None:
    path = provenance.save_json(tmp_path / "out" / "result.json", {"steps_per_s": 1.5}, repo=repo)
    payload = json.loads(path.read_text())
    assert payload["steps_per_s"] == 1.5
    record = payload["provenance"]
    assert record["git"]["commit"] is not None
    assert {"time_utc", "python", "platform", "host", "packages", "gpus"} <= record.keys()


def test_save_json_refuses_to_overwrite_provenance(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        provenance.save_json(tmp_path / "r.json", {"provenance": {}})

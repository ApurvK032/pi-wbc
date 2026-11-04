"""The input files of a MuJoCo-bench job list, for copying it to the cluster.

    .venv/bin/python scripts/bench_inputs.py JOBS.json [EXTRA ...]

Writes, next to the job list, `upload.txt` (for `rsync --files-from`: every policy with its
external weight file (`policy.onnx.data`), every policy_io and model file the jobs name, the
folder of each MuJoCo scene (it holds the meshes), the job list itself and any EXTRA paths)
and `inputs.sha256` (the SHA-256 of every input file, for `sha256sum -c` on the cluster before
submitting). Paths are relative to the repository root, where this must run.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import sys
from pathlib import Path


def main() -> None:
    jobs_path = Path(sys.argv[1])
    extra = sys.argv[2:]
    files: set[str] = set()
    for job in json.loads(jobs_path.read_text()):
        args = job["args"]
        for flag, value in itertools.pairwise(args):
            if flag in ("--policy", "--policy-io"):
                files.add(value)
            if flag == "--policy":  # an ONNX file's weights may sit beside it (policy.onnx.data)
                files.update(str(f) for f in Path(value).parent.glob(Path(value).name + ".*"))
            elif flag == "--model":
                files.add(value.split("=", 1)[1])
    missing = [f for f in sorted(files) if not Path(f).is_file()]
    if missing:
        raise SystemExit(f"missing inputs: {missing}")
    # a scene that is not an export (Menagerie's) loads meshes from its own folder
    folders = sorted({str(Path(f).parent) for f in files if not f.startswith("logs/")})
    lines = sorted(f for f in files if f.startswith("logs/")) + folders
    lines += [str(jobs_path), str(jobs_path.parent / "inputs.sha256"), *extra]
    (jobs_path.parent / "upload.txt").write_text("\n".join(lines) + "\n")
    sums = [
        f"{hashlib.sha256(Path(f).read_bytes()).hexdigest()}  {f}"
        for f in sorted(files)
        + sorted(str(p) for d in folders for p in Path(d).rglob("*") if p.is_file())
    ]
    (jobs_path.parent / "inputs.sha256").write_text("\n".join(sums) + "\n")
    print(f"{len(files)} input files, {len(folders)} scene folder(s); {len(sums)} checksums")


if __name__ == "__main__":
    main()

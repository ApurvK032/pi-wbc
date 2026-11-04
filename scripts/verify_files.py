"""Check stored stage 3 files against their sources, reading each from disk (not the page cache).

On 2026-11-04 one recording's cached copy had a flipped bit in RAM (its disk copy was fine; the
workstation's memory has no error correction). Every file read here is first dropped from the
page cache (`posix_fadvise(DONTNEED)`, no root needed), so the check sees what is on disk:

- `--recordings DIR ...`: every .npz passes its zip CRC checks; their SHA-256 hashes are written
  to `--hashes` (a `sha256sum`-style list) for later checks;
- `--render DIR ...`: every video's hash equals its render manifest's (`render.json`);
- `--export DIR ...`: every episode's state and action rows equal those recomputed from its
  recording (pi_wbc.stage3.export), and its video's hash equals the one in `meta/pi_wbc.json`.

    uv run python scripts/verify_files.py --recordings logs/dataset/train-v1 \\
        --hashes results/stage3/dataset/recordings-sha256.txt
"""

import argparse
import hashlib
import io
import json
import os
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


def read_from_disk(path: Path) -> bytes:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        with os.fdopen(os.dup(fd), "rb") as f:
            data = f.read()
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        return data
    finally:
        os.close(fd)


def check_recording(path: str) -> tuple[str, str, str | None]:
    data = read_from_disk(Path(path))
    try:
        bad = zipfile.ZipFile(io.BytesIO(data)).testzip()
        err = None if bad is None else f"bad member {bad}"
    except zipfile.BadZipFile as e:
        err = repr(e)
    return path, hashlib.sha256(data).hexdigest(), err


def check_video(job: tuple[str, str]) -> tuple[str, str | None]:
    path, want = job
    got = hashlib.sha256(read_from_disk(Path(path))).hexdigest()
    return path, None if got == want else f"hash {got[:12]} != manifest {want[:12]}"


def check_export_episode(job: dict) -> tuple[int, str | None]:
    import pyarrow.parquet as pq

    from pi_wbc.stage3 import export as E

    d = np.load(io.BytesIO(read_from_disk(Path(job["recording"]))))
    meta = json.loads(str(d["meta"]))
    ep = {k: d[k] for k in ("phase", "joint_pos", "cmd_body", "cmd_arm_hand", "cmd_velocity")}
    state, action = E.episode_arrays(ep, meta)
    table = pq.read_table(io.BytesIO(read_from_disk(Path(job["parquet"]))))
    got_s = np.stack(table.column("observation.state").to_numpy(zero_copy_only=False))
    got_a = np.stack(table.column("action").to_numpy(zero_copy_only=False))
    if not (np.array_equal(got_s, state) and np.array_equal(got_a, action)):
        return job["episode_index"], "rows differ from the recording"
    if hashlib.sha256(read_from_disk(Path(job["video"]))).hexdigest() != job["video_sha256"]:
        return job["episode_index"], "video hash differs"
    return job["episode_index"], None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--recordings", type=Path, nargs="*", default=[])
    parser.add_argument("--hashes", type=Path, default=None)
    parser.add_argument("--render", type=Path, nargs="*", default=[])
    parser.add_argument("--export", type=Path, nargs="*", default=[])
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    problems = 0
    with ProcessPoolExecutor(args.workers) as pool:
        if args.recordings:
            files = sorted(str(p) for d in args.recordings for p in d.glob("*.npz"))
            rows = list(pool.map(check_recording, files, chunksize=16))
            bad = [(p, e) for p, _, e in rows if e]
            problems += len(bad)
            print(f"[verify] recordings: {len(rows)} files, {len(bad)} bad", flush=True)
            for p, e in bad:
                print(f"  {p}: {e}")
            if args.hashes is not None:
                args.hashes.write_text("".join(f"{h}  {p}\n" for p, h, _ in rows))
        for d in args.render:
            manifest = json.loads((d / "render.json").read_text())
            jobs = [(str(d / e["video"]), e["sha256"]) for e in manifest["episodes"]]
            bad = [(p, e) for p, e in pool.map(check_video, jobs, chunksize=16) if e]
            problems += len(bad)
            print(f"[verify] {d}: {len(jobs)} videos, {len(bad)} differ", flush=True)
            for p, e in bad:
                print(f"  {p}: {e}")
        for d in args.export:
            info = json.loads((d / "meta/info.json").read_text())
            pw = json.loads((d / "meta/pi_wbc.json").read_text())
            rec = Path(pw["recordings"])
            jobs = [{
                "episode_index": e["episode_index"], "video_sha256": e["video_sha256"],
                "recording": str(rec / f"{e['seed']}.npz"),
                "parquet": str(d / info["data_path"].format(
                    episode_chunk=e["episode_index"] // info["chunks_size"],
                    episode_index=e["episode_index"])),
                "video": str(d / info["video_path"].format(
                    episode_chunk=e["episode_index"] // info["chunks_size"],
                    video_key="observation.images.ego_view", episode_index=e["episode_index"])),
            } for e in pw["episodes"]]  # fmt: skip
            bad = [(i, e) for i, e in pool.map(check_export_episode, jobs, chunksize=8) if e]
            problems += len(bad)
            print(f"[verify] {d}: {len(jobs)} episodes, {len(bad)} differ", flush=True)
            for i, e in bad:
                print(f"  episode {i}: {e}")
    print(f"[verify] {'all files match' if problems == 0 else f'{problems} problems'}")
    raise SystemExit(1 if problems else 0)


if __name__ == "__main__":
    main()

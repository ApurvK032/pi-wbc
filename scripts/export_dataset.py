"""Export recorded, rendered episodes as a GR00T-flavoured LeRobot v2.1 dataset (D-045).

Reads each episode's recording (`--recordings`, from scripts/expert_rollout.py --record) and its
head-camera video (`--render`, from scripts/render_episodes.py) and writes the layout described
in `pi_wbc.stage3.export`. Videos are hard-linked (same file system) unless `--copy-videos`.
Every exported episode must be a recorded success, and its video must have exactly one frame per
row and the hash the render manifest lists.

    uv run python scripts/export_dataset.py --recordings logs/dataset/train-v1 \\
        --render logs/render/train-v1-fov87 \\
        --seeds results/stage3/dataset/train-v1-manifest.json \\
        --out logs/export/pi_wbc_g1_pick_place_train_v1
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from pi_wbc import provenance
from pi_wbc.stage3 import export as E


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def video_frames(path: Path) -> int:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
         "stream=nb_read_packets,width,height,r_frame_rate", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout  # fmt: skip
    s = json.loads(out)["streams"][0]
    if (s["width"], s["height"]) != (640, 480) or s["r_frame_rate"] != f"{E.FPS}/1":
        raise ValueError(f"{path}: {s}")
    return int(s["nb_read_packets"])


def export_episode(job: dict) -> dict:
    """Write one episode's parquet, link its video, and return its metadata and stats."""
    import pyarrow.parquet as pq

    d = np.load(job["recording"])
    meta = json.loads(str(d["meta"]))
    if meta["outcome"]["label"] != "success" or meta["seed"] != job["seed"]:
        raise ValueError(f"seed {job['seed']}: not a recorded success ({meta['outcome']})")
    ep = {k: d[k] for k in ("phase", "joint_pos", "cmd_body", "cmd_arm_hand", "cmd_velocity")}
    state, action = E.episode_arrays(ep, meta)
    n = len(state)
    video = Path(job["video"])
    if n != job["render_frames"] or video_frames(video) != n:
        raise ValueError(f"seed {job['seed']}: {n} rows, video frames differ")
    if sha256(video) != job["video_sha256"]:
        raise ValueError(f"seed {job['seed']}: video hash differs from the render manifest")
    i = job["episode_index"]
    out = Path(job["out"])
    chunk = i // E.CHUNK
    data_path = out / f"data/chunk-{chunk:03d}/episode_{i:06d}.parquet"
    pq.write_table(E.episode_table(state, action, i, job["first_index"], job["task_index"]),
                   data_path)  # fmt: skip
    video_path = out / f"videos/chunk-{chunk:03d}/{E.VIDEO_KEY}/episode_{i:06d}.mp4"
    if job["copy"]:
        shutil.copyfile(video, video_path)
    else:
        os.link(video, video_path)
    t = np.arange(n)
    done = t == n - 1
    stats = {
        "observation.state": E.numeric_stats(state),
        "action": E.numeric_stats(action),
        "timestamp": E.numeric_stats(t / E.FPS),
        "frame_index": E.numeric_stats(t),
        "episode_index": E.numeric_stats(np.full(n, i)),
        "index": E.numeric_stats(job["first_index"] + t),
        "task_index": E.numeric_stats(np.full(n, job["task_index"])),
        E.TASK_COLUMN: E.numeric_stats(np.full(n, job["task_index"])),
        "next.reward": E.numeric_stats(done.astype(float)),
        "next.done": E.numeric_stats(done.astype(float)),
        E.VIDEO_KEY: E.video_stats(video, n),
    }
    return {
        "episode_index": i, "seed": job["seed"], "length": n, "instruction": meta["instruction"],
        "hand": meta["outcome"].get("hand"), "steps": len(ep["phase"]),
        "video_sha256": job["video_sha256"], "stats": stats,
    }  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--recordings", type=Path, required=True)
    parser.add_argument("--render", type=Path, required=True, help="render output folder")
    parser.add_argument(
        "--seeds", type=Path, default=None, help="dataset manifest whose chosen_seeds to export"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--copy-videos", action="store_true")
    args = parser.parse_args()

    render = json.loads((args.render / "render.json").read_text())
    rendered = {e["seed"]: e for e in render["episodes"]}
    if args.seeds is not None:
        seeds = json.loads(args.seeds.read_text())["chosen_seeds"]
    else:
        seeds = sorted(rendered)
    missing = [s for s in seeds if s not in rendered]
    if missing:
        raise SystemExit(f"{len(missing)} seeds were not rendered, e.g. {missing[:5]}")
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; export into a new folder")

    # tasks, in order of first appearance; episode lengths for the global row index
    instructions, lengths = [], []
    for s in seeds:
        d = np.load(args.recordings / f"{s}.npz")
        instructions.append(json.loads(str(d["meta"]))["instruction"])
        lengths.append(len(E.frames_of(len(d["phase"]))))
    tasks = list(dict.fromkeys(instructions))
    first = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    for c in range((len(seeds) + E.CHUNK - 1) // E.CHUNK):
        (args.out / f"data/chunk-{c:03d}").mkdir(parents=True)
        (args.out / f"videos/chunk-{c:03d}/{E.VIDEO_KEY}").mkdir(parents=True)
    (args.out / "meta").mkdir()
    jobs = [{
        "seed": s, "episode_index": i, "first_index": int(first[i]),
        "task_index": tasks.index(instructions[i]),
        "recording": str(args.recordings / f"{s}.npz"),
        "video": str(args.render / rendered[s]["video"]),
        "render_frames": rendered[s]["frames"], "video_sha256": rendered[s]["sha256"],
        "out": str(args.out), "copy": args.copy_videos,
    } for i, s in enumerate(seeds)]  # fmt: skip
    with ProcessPoolExecutor(args.workers) as pool:
        episodes = list(pool.map(export_episode, jobs, chunksize=4))
    total = sum(e["length"] for e in episodes)
    if [e["length"] for e in episodes] != lengths:
        raise RuntimeError("episode lengths changed between passes")

    meta = args.out / "meta"
    w, h = render["camera"]["size"]
    info = {
        "codebase_version": "v2.1", "robot_type": "unitree_g1",
        "total_episodes": len(episodes), "total_frames": total, "total_tasks": len(tasks),
        "total_videos": len(episodes), "total_chunks": (len(episodes) + E.CHUNK - 1) // E.CHUNK,
        "chunks_size": E.CHUNK, "fps": E.FPS, "splits": {args.split: f"0:{len(episodes)}"},
        "data_path": E.DATA_PATH,
        "video_path": E.VIDEO_PATH,
        "features": E.features(w, h),
    }  # fmt: skip
    (meta / "info.json").write_text(json.dumps(info, indent=4) + "\n")
    (meta / "modality.json").write_text(json.dumps(E.modality(), indent=4) + "\n")
    E.write_jsonl(meta / "tasks.jsonl", [{"task_index": i, "task": t} for i, t in enumerate(tasks)])
    E.write_jsonl(meta / "episodes.jsonl", [
        {"episode_index": e["episode_index"], "tasks": [e["instruction"]], "length": e["length"]}
        for e in episodes
    ])  # fmt: skip
    stats = [{"episode_index": e["episode_index"], "stats": e["stats"]} for e in episodes]
    E.write_jsonl(meta / "episodes_stats.jsonl", stats)
    provenance.save_json(meta / "pi_wbc.json", {
        "name": args.out.name, "split": args.split,
        "recordings": str(args.recordings), "render": str(args.render),
        "render_manifest_sha256": sha256(args.render / "render.json"),
        "render_commit": render["provenance"]["git"]["commit"],
        "seeds_from": str(args.seeds), "decisions": ["D-037", "D-042", "D-043", "D-044", "D-045"],
        "frame_step": E.FRAME_STEP, "state_names": E.STATE_NAMES, "action_names": E.ACTION_NAMES,
        "episodes": [{k: e[k] for k in ("episode_index", "seed", "length", "steps", "hand",
                                        "instruction", "video_sha256")} for e in episodes],
    })  # fmt: skip
    print(f"[export] {len(episodes)} episodes, {total} frames, {len(tasks)} tasks -> {args.out}")


if __name__ == "__main__":
    main()

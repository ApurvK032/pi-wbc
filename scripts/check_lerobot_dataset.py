"""Load a converted LeRobot v3.0 dataset with LeRobot's own `LeRobotDataset` and check it (D-045).

Runs in LeRobot's environment, not this project's:

    third_party/lerobot-env/bin/python scripts/check_lerobot_dataset.py \\
        --dataset logs/export/lerobot/pi_wbc_g1_pick_place_train_v1 \\
        --reference logs/export/pi_wbc_g1_pick_place_train_v1 --add-quantiles \\
        --out logs/export/lerobot_check.json

`--add-quantiles` computes LeRobot's quantile statistics (q01 ... q99, which its `pi05` policy
normalizes with) and writes them locally; nothing is uploaded. It uses LeRobot's own functions
(`get_feature_stats`, `sample_indices`, `aggregate_stats`) on the same rows and sampled frames as
its `augment_dataset_quantile_stats.py`, but decodes each episode's sampled frames in one pass;
`--lerobot-sampler` runs LeRobot's per-episode function itself (random access, far slower).
Then, for a sample of episodes, items at the first, middle and last frame are compared with
the GR00T-format export they were converted from (`--reference`, LeRobot v2.1 layout):
state and action rows must be equal, the action chunk (π0.5's 50 steps) must follow the
episode's rows and be marked as padding past its end, the instruction must match, and the
image must match the reference video's frame (and not a neighbouring one).
"""

import argparse
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

FPS = 25
CHUNK = 50  # π0.5's action horizon in LeRobot
REPO_ID = "local/pi_wbc"
_worker_dataset = None
_worker_reference = None


def _init_worker(root: str, reference: str) -> None:
    global _worker_dataset, _worker_reference
    from lerobot.datasets import LeRobotDataset

    _worker_dataset = LeRobotDataset(REPO_ID, root=root)
    _worker_reference = Path(reference)


def _episode_stats_lerobot(episode_index: int) -> dict:
    """LeRobot's own per-episode function (random access into the concatenated videos)."""
    from lerobot.scripts.augment_dataset_quantile_stats import process_single_episode

    return process_single_episode(_worker_dataset, episode_index, use_sampling=True)


def _episode_stats(episode_index: int) -> dict:
    """What LeRobot's `process_single_episode` computes, with the sampled frames decoded in one
    sequential pass over the episode's own video (the same encoded frames as the converted
    dataset's, checked below) instead of one random seek each into the concatenated files."""
    from lerobot.datasets import DEFAULT_QUANTILES, get_feature_stats
    from lerobot.datasets.compute_stats import sample_indices

    ds = _worker_dataset
    ep = ds.meta.episodes[episode_index]
    start, end = ep["dataset_from_index"], ep["dataset_to_index"]
    numeric = [k for k, f in ds.features.items() if f["dtype"] not in ("image", "video", "string")]
    cols = ds.hf_dataset.select_columns(numeric)[start:end]
    stats = {}
    for key in numeric:
        data = np.stack([np.asarray(v) for v in cols[key]])
        stats[key] = get_feature_stats(data, axis=0, keepdims=data.ndim == 1,
                                       quantile_list=DEFAULT_QUANTILES)  # fmt: skip
    info = json.loads((_worker_reference / "meta/info.json").read_text())
    for key in ds.meta.video_keys:
        video = _worker_reference / info["video_path"].format(
            episode_chunk=episode_index // info["chunks_size"], video_key=key,
            episode_index=episode_index)  # fmt: skip
        offsets = sample_indices(end - start)
        frames = ffmpeg_frames(video, offsets).astype(np.float32) / 255.0
        data = frames.transpose(0, 3, 1, 2)  # (n, c, h, w), as LeRobot's items
        st = get_feature_stats(data, axis=(0, 2, 3), keepdims=True,
                               quantile_list=DEFAULT_QUANTILES)  # fmt: skip
        stats[key] = {k: v if k == "count" else np.squeeze(v, axis=0) for k, v in st.items()}
    return stats


def ffmpeg_frames(path: Path, indices) -> np.ndarray:
    """Frames at `indices` (sorted), decoded in one pass."""
    indices = sorted(int(i) for i in indices)
    sel = "+".join(f"eq(n\\,{i})" for i in indices)
    raw = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-threads", "1", "-i", str(path), "-vf",
         f"select='{sel}'", "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True, check=True,
    ).stdout  # fmt: skip
    out = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 480, 640, 3)
    if len(out) != len(indices):
        raise RuntimeError(f"{path}: {len(out)} frames decoded, {len(indices)} asked for")
    return out


def ffmpeg_frame(path: Path, index: int) -> np.ndarray:
    """One frame."""
    raw = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", str(path), "-vf", f"select=eq(n\\,{index})",
         "-vsync", "0", "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True, check=True,
    ).stdout  # fmt: skip
    return np.frombuffer(raw, dtype=np.uint8).reshape(480, 640, 3)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True, help="the v2.1 export")
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--add-quantiles", action="store_true")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument(
        "--lerobot-sampler", action="store_true", help="LeRobot's own random-access sampler"
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    from lerobot.datasets import LeRobotDataset
    from lerobot.datasets.io_utils import write_stats

    if args.add_quantiles:
        from lerobot.datasets import aggregate_stats
        from lerobot.scripts.augment_dataset_quantile_stats import has_quantile_stats

        ds = LeRobotDataset(REPO_ID, root=args.dataset)
        if not has_quantile_stats(ds.meta.stats):
            fn = _episode_stats_lerobot if args.lerobot_sampler else _episode_stats
            init = {"initializer": _init_worker,
                    "initargs": (str(args.dataset), str(args.reference))}  # fmt: skip
            with ProcessPoolExecutor(args.workers, **init) as pool:
                episode_stats = list(pool.map(fn, range(ds.num_episodes), chunksize=8))
            write_stats(aggregate_stats(episode_stats), ds.meta.root)
            print("[lerobot] quantile statistics written", flush=True)

    ds = LeRobotDataset(REPO_ID, root=args.dataset,
                        delta_timestamps={"action": [k / FPS for k in range(CHUNK)]})  # fmt: skip
    ref_info = json.loads((args.reference / "meta/info.json").read_text())
    ref_eps = [json.loads(line) for line in (args.reference / "meta/episodes.jsonl").open()]
    assert ds.meta.total_episodes == len(ref_eps), ds.meta.total_episodes
    assert ds.meta.total_frames == sum(e["length"] for e in ref_eps), ds.meta.total_frames
    assert ds.meta.fps == FPS
    stats = ds.meta.stats
    for key in ("action", "observation.state"):
        assert all(q in stats[key] for q in ("q01", "q99")), (key, sorted(stats[key]))

    rng = np.random.default_rng(0)
    n_eps = len(ref_eps)
    if args.episodes >= n_eps:
        pick = list(range(n_eps))
    else:
        pick = sorted({0, n_eps - 1, *rng.choice(n_eps, args.episodes - 2, replace=False).tolist()})
    first = np.concatenate([[0], np.cumsum([e["length"] for e in ref_eps])[:-1]])
    report = {"dataset": str(args.dataset), "frames": int(ds.meta.total_frames), "checked": []}
    for e in pick:
        n = ref_eps[e]["length"]
        ref = pd.read_parquet(args.reference / ref_info["data_path"].format(
            episode_chunk=e // ref_info["chunks_size"], episode_index=e))  # fmt: skip
        video = args.reference / ref_info["video_path"].format(
            episode_chunk=e // ref_info["chunks_size"],
            video_key="observation.images.ego_view", episode_index=e)  # fmt: skip
        state = np.stack(ref["observation.state"].to_numpy())
        action = np.stack(ref["action"].to_numpy())
        diffs = []
        for k in (0, n // 2, n - 1):
            item = ds[int(first[e] + k)]
            assert int(item["episode_index"]) == e and int(item["frame_index"]) == k
            assert np.array_equal(item["observation.state"].numpy(), state[k])
            chunk = item["action"].numpy()
            pad = item["action_is_pad"].numpy()
            valid = min(CHUNK, n - k)
            assert np.array_equal(chunk[:valid], action[k : k + valid])
            assert not pad[:valid].any() and pad[valid:].all()
            assert item["task"] == ref_eps[e]["tasks"][0]
            img = (item["observation.images.ego_view"].numpy().transpose(1, 2, 0) * 255).round()
            same = float(np.abs(img - ffmpeg_frame(video, k).astype(float)).mean())
            assert same < 1.0, (e, k, same)
            other = None
            if k == n // 2:  # mid-task the robot moves, so 5 frames earlier looks different
                other = float(np.abs(img - ffmpeg_frame(video, k - 5)).mean())
                assert other > 3 * same + 0.5, (e, k, same, other)
            diffs.append((round(same, 3), None if other is None else round(other, 2)))
        report["checked"].append({"episode_index": e, "rows": n, "image_diff_vs_ref": diffs})
        print(f"[lerobot] episode {e}: {n} rows ok; image vs reference {diffs}", flush=True)
    report["passed"] = True
    report["quantiles"] = sorted(k for k in stats["action"] if k.startswith("q"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"[lerobot] {len(pick)} episodes passed")


if __name__ == "__main__":
    main()

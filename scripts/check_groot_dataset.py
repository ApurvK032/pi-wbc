"""Load an exported dataset with GR00T N1.6's own data loader and check what comes out (D-045).

Runs in GR00T's environment, not this project's (it imports `gr00t`):

    third_party/Isaac-GR00T-n1d6/.venv/bin/python scripts/check_groot_dataset.py \\
        --dataset logs/export/pi_wbc_g1_pick_place_train_v1 --out logs/export/groot_check.json

Uses GR00T's `unitree_g1` modality config (the export follows its names). For a sample of
episodes it loads the whole episode through `LeRobotEpisodeLoader` (parquet plus every video
frame; the loader itself asserts one frame per row), and checks:

- every state and action group has the size modality.json gives, and equals the parquet's slice;
- the language is the episode's instruction (tasks.jsonl and meta/pi_wbc.json agree);
- the frames GR00T decodes (torchcodec) equal ffmpeg's at the same indices (first, middle and
  last), and where the video changes most between two frames GR00T's frame equals ffmpeg's same
  frame and not the next one, so rows and frames line up (not off by one);
- `extract_step_data`, which builds GR00T's training samples, returns the 30-step action chunks
  of the unitree_g1 config at the first and last full step;
- the dataset statistics GR00T computed load.
"""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag


def ffmpeg_frames(path: Path, indices: list[int]) -> np.ndarray:
    sel = "+".join(f"eq(n\\,{i})" for i in indices)
    raw = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", str(path), "-vf", f"select='{sel}'", "-vsync", "0",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True, check=True,
    ).stdout  # fmt: skip
    return np.frombuffer(raw, dtype=np.uint8).reshape(len(indices), 480, 640, 3)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=12, help="how many episodes to check")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    meta = args.dataset / "meta"
    modality = json.loads((meta / "modality.json").read_text())
    info = json.loads((meta / "info.json").read_text())
    tasks = {t["task_index"]: t["task"] for t in map(json.loads, (meta / "tasks.jsonl").open())}
    episodes = [json.loads(line) for line in (meta / "episodes.jsonl").open()]
    ours = {
        e["episode_index"]: e for e in json.loads((meta / "pi_wbc.json").read_text())["episodes"]
    }
    configs = MODALITY_CONFIGS["unitree_g1"]
    loader = LeRobotEpisodeLoader(args.dataset, configs, video_backend="torchcodec")
    horizon = len(configs["action"].delta_indices)
    n_eps = len(loader)
    if args.episodes >= n_eps:
        pick = list(range(n_eps))
    else:
        rng = np.random.default_rng(0)
        pick = sorted({0, n_eps - 1, *rng.choice(n_eps, args.episodes - 2, replace=False).tolist()})
    report = {"dataset": str(args.dataset), "episodes_in_dataset": n_eps, "checked": []}
    for i in pick:
        df = loader[i]
        n = episodes[i]["length"]
        assert len(df) == n, f"episode {i}: {len(df)} rows, episodes.jsonl says {n}"
        raw = pd.read_parquet(args.dataset / info["data_path"].format(
            episode_chunk=i // info["chunks_size"], episode_index=i))  # fmt: skip
        for kind, column in (("state", "observation.state"), ("action", "action")):
            for key in configs[kind].modality_keys:
                s = modality[kind][key]
                got = np.stack(df[f"{kind}.{key}"].to_numpy())
                assert got.shape == (n, s["end"] - s["start"]), (kind, key, got.shape)
                want = np.stack(raw[column].to_numpy())[:, s["start"] : s["end"]]
                assert np.array_equal(got, want), (kind, key)
        text = df["language.annotation.human.task_description"].iloc[0]
        assert text == tasks[int(raw["task_index"].iloc[0])] == ours[i]["instruction"], text
        # frames: GR00T's decode against ffmpeg's, at three places
        idx = [0, n // 2, n - 1]
        video = args.dataset / info["video_path"].format(
            episode_chunk=i // info["chunks_size"], video_key="observation.images.ego_view",
            episode_index=i)  # fmt: skip
        theirs = np.stack([np.asarray(df["video.ego_view"].iloc[k]) for k in idx]).astype(int)
        ref = ffmpeg_frames(video, idx).astype(int)
        same = [float(np.abs(theirs[j] - ref[j]).mean()) for j in range(3)]
        # off-by-one test: where the video changes most from one frame to the next, GR00T's frame
        # must equal ffmpeg's same frame and differ from ffmpeg's next one
        small = np.stack([np.asarray(f)[::8, ::8] for f in df["video.ego_view"]]).astype(np.int16)
        k = int(np.argmax(np.abs(np.diff(small, axis=0)).mean(axis=(1, 2, 3))))
        pair = ffmpeg_frames(video, [k, k + 1]).astype(int)
        mine = np.asarray(df["video.ego_view"].iloc[k]).astype(int)
        at_k = float(np.abs(mine - pair[0]).mean())
        off = [float(np.abs(mine - pair[1]).mean())]
        assert max(same) < 1.0 and at_k < 1.0 and off[0] > 3 * at_k + 0.5, (same, at_k, off)
        # training samples: action chunks of the config's horizon at the first and last full step
        for step in (0, n - horizon):
            vla = extract_step_data(df, step, configs, EmbodimentTag.UNITREE_G1)
            for key, a in vla.actions.items():
                s = modality["action"][key]
                assert a.shape == (horizon, s["end"] - s["start"]), (key, a.shape)
            assert len(vla.images["ego_view"]) == 1 and vla.text == text
        report["checked"].append({
            "episode_index": i, "seed": ours[i]["seed"], "rows": n,
            "frame_diff_vs_ffmpeg": [round(x, 3) for x in same],
            "frame_diff_at_largest_change_vs_next": [round(at_k, 3), round(off[0], 2)],
        })  # fmt: skip
        seed = ours[i]["seed"]
        print(f"[groot] episode {i} (seed {seed}): {n} rows ok; frames differ from ffmpeg's "
              f"by {max(same):.2f} (next frame: {off[0]:.1f})", flush=True)  # fmt: skip
    stats = loader.get_dataset_statistics()
    report["statistics_keys"] = sorted(stats)
    report["action_horizon"] = horizon
    report["passed"] = True
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"[groot] {len(pick)} episodes passed; statistics for {sorted(stats)}")


if __name__ == "__main__":
    main()

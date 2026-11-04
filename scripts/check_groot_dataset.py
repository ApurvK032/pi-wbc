"""Load an exported dataset with GR00T N1.6's own data loader and check what comes out (D-045).

Runs in GR00T's environment, not this project's (it imports `gr00t`):

    third_party/Isaac-GR00T-n1d6/.venv/bin/python scripts/check_groot_dataset.py \\
        --dataset logs/export/pi_wbc_g1_pick_place_train_v1 --out logs/export/groot_check.json

Uses GR00T's `unitree_g1` modality config (the export follows its names). For a sample of
episodes it loads the whole episode through `LeRobotEpisodeLoader` (parquet plus every video
frame; the loader itself asserts one frame per row), and checks:

- every state and action group has the size modality.json gives, and equals the parquet's slice;
- the language is the episode's instruction (tasks.jsonl and meta/pi_wbc.json agree);
- the frames GR00T decodes (torchcodec) match ffmpeg's frames at the same indices (first, middle
  and last), and the middle one differs from a frame 0.2, 0.4 or 0.8 s later, so rows and frames
  line up (the robot may stand still for a moment);
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
        # mid-task the robot moves: a frame 0.2, 0.4 or 0.8 s later must differ
        fps = info["fps"]
        later = [min(n // 2 + round(t * fps), n - 1) for t in (0.2, 0.4, 0.8)]
        shifted = ffmpeg_frames(video, later).astype(int)
        off = [float(np.abs(theirs[1] - f).mean()) for f in shifted]
        assert max(same) < 1.0 and max(off) > 3 * max(same) + 0.5, (same, off)
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
            "frame_diff_vs_0.2_0.4_0.8_s_later": [round(x, 2) for x in off],
        })  # fmt: skip
        seed = ours[i]["seed"]
        print(f"[groot] episode {i} (seed {seed}): {n} rows ok; frames differ from ffmpeg's "
              f"by {max(same):.2f} (later: {max(off):.1f})", flush=True)  # fmt: skip
    stats = loader.get_dataset_statistics()
    report["statistics_keys"] = sorted(stats)
    report["action_horizon"] = horizon
    report["passed"] = True
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"[groot] {len(pick)} episodes passed; statistics for {sorted(stats)}")


if __name__ == "__main__":
    main()

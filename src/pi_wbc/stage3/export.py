"""Write recorded, rendered episodes as a GR00T-flavoured LeRobot v2.1 dataset (D-045).

Layout (GR00T's `getting_started/data_preparation.md`, N1.6 branch; LeRobot v2.1):

    meta/info.json, episodes.jsonl, tasks.jsonl, episodes_stats.jsonl, modality.json, pi_wbc.json
    data/chunk-000/episode_000000.parquet ...
    videos/chunk-000/observation.images.ego_view/episode_000000.mp4 ...

One row per video frame: frame k shows the state that control step 2k's command was applied to
(25 frames per second from 50 Hz control, D-043), and the row holds that step's joint positions
(`observation.state`) and commands (`action`). The names follow GR00T N1.6's `unitree_g1` data
config, and the state's 43 joints are in the same order as NVIDIA's G1 datasets:

- state: left_leg, right_leg, waist, left_arm, left_hand, right_arm, right_hand (joint positions);
- action: left_arm, right_arm, left_hand, right_hand (joint targets), waist (yaw, roll, pitch
  goals), base_height_command (pelvis height goal), navigate_command (vx, vy, yaw rate).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

FPS = 25
FRAME_STEP = 2  # control steps per frame
CHUNK = 1000  # episodes per chunk folder
VIDEO_KEY = "observation.images.ego_view"
TASK_COLUMN = "annotation.human.task_description"
DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"

# GR00T's G1 joint order (NVIDIA's unitree_g1 datasets), which our recordings use too
SIDES = ("left", "right")
LEG = ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
ARM = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch",
       "wrist_yaw")  # fmt: skip
HAND = ("index_0", "index_1", "middle_0", "middle_1", "thumb_0", "thumb_1", "thumb_2")
WAIST = ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")


def _arm(side: str) -> list[str]:
    return [f"{side}_{j}_joint" for j in ARM]


def _hand(side: str) -> list[str]:
    return [f"{side}_hand_{j}_joint" for j in HAND]


STATE_GROUPS = {
    "left_leg": [f"left_{j}_joint" for j in LEG],
    "right_leg": [f"right_{j}_joint" for j in LEG],
    "waist": list(WAIST),
    "left_arm": _arm("left"),
    "left_hand": _hand("left"),
    "right_arm": _arm("right"),
    "right_hand": _hand("right"),
}
STATE_NAMES = [name for names in STATE_GROUPS.values() for name in names]
ACTION_GROUPS = {
    "left_arm": _arm("left"),
    "right_arm": _arm("right"),
    "left_hand": _hand("left"),
    "right_hand": _hand("right"),
    "waist": [f"{j}_goal" for j in WAIST],
    "base_height_command": ["pelvis_height_goal"],
    "navigate_command": ["vx", "vy", "yaw_rate"],
}
ACTION_NAMES = [name for names in ACTION_GROUPS.values() for name in names]


def _slices(groups: dict[str, list[str]], original_key: str) -> dict[str, dict]:
    out, start = {}, 0
    for key, names in groups.items():
        out[key] = {"original_key": original_key, "start": start, "end": start + len(names)}
        start += len(names)
    return out


def modality() -> dict:
    """GR00T's meta/modality.json."""
    return {
        "state": _slices(STATE_GROUPS, "observation.state"),
        "action": _slices(ACTION_GROUPS, "action"),
        "video": {"ego_view": {"original_key": VIDEO_KEY}},
        "annotation": {"human.task_description": {"original_key": TASK_COLUMN}},
    }


def features(width: int, height: int) -> dict:
    """LeRobot v2.1 info.json features: every parquet column, plus the video."""

    def scalar(dtype):
        return {"dtype": dtype, "shape": [1], "names": None}

    return {
        VIDEO_KEY: {
            "dtype": "video",
            "shape": [height, width, 3],
            "names": ["height", "width", "channel"],
            "info": {
                "video.height": height,
                "video.width": width,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": FPS,
                "video.channels": 3,
                "has_audio": False,
            },
        },
        "observation.state": {
            "dtype": "float32",
            "shape": [len(STATE_NAMES)],
            "names": STATE_NAMES,
        },
        "action": {"dtype": "float32", "shape": [len(ACTION_NAMES)], "names": ACTION_NAMES},
        "timestamp": scalar("float32"),
        "frame_index": scalar("int64"),
        "episode_index": scalar("int64"),
        "index": scalar("int64"),
        "task_index": scalar("int64"),
        TASK_COLUMN: scalar("int64"),
        "next.reward": scalar("float32"),
        "next.done": scalar("bool"),
    }


def frames_of(steps: int) -> np.ndarray:
    """Control steps shown by the frames of an episode with `steps` control steps."""
    return np.arange(0, steps, FRAME_STEP)


def episode_arrays(ep: dict, meta: dict) -> tuple[np.ndarray, np.ndarray]:
    """(state, action) per frame of one recorded episode, float32, in the export's layout."""
    joints = list(meta["joint_names"])
    if joints != STATE_NAMES:
        raise ValueError("the recording's joint order is not GR00T's G1 order")
    arm_hand = list(meta["arm_hand_names"])
    pick = [arm_hand.index(n) for g in ("left_arm", "right_arm", "left_hand", "right_hand")
            for n in ACTION_GROUPS[g]]  # fmt: skip
    k = frames_of(len(ep["phase"]))
    body = ep["cmd_body"][k]  # pelvis height, waist yaw, roll, pitch
    action = np.concatenate(
        [ep["cmd_arm_hand"][k][:, pick], body[:, 1:4], body[:, :1], ep["cmd_velocity"][k]], axis=1
    )
    return ep["joint_pos"][k].astype(np.float32), action.astype(np.float32)


def episode_table(state, action, episode_index: int, first_index: int, task: int):
    """One episode's parquet table."""
    import pyarrow as pa

    n = len(state)
    t = np.arange(n)
    done = t == n - 1
    return pa.table({
        "observation.state": pa.array(list(state), type=pa.list_(pa.float32())),
        "action": pa.array(list(action), type=pa.list_(pa.float32())),
        "timestamp": pa.array((t / FPS).astype(np.float32)),
        "frame_index": pa.array(t.astype(np.int64)),
        "episode_index": pa.array(np.full(n, episode_index, dtype=np.int64)),
        "index": pa.array((first_index + t).astype(np.int64)),
        "task_index": pa.array(np.full(n, task, dtype=np.int64)),
        TASK_COLUMN: pa.array(np.full(n, task, dtype=np.int64)),
        "next.reward": pa.array(done.astype(np.float32)),  # every exported episode succeeds
        "next.done": pa.array(done),
    })  # fmt: skip


def numeric_stats(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    return {"min": x.min(0).tolist(), "max": x.max(0).tolist(), "mean": x.mean(0).tolist(),
            "std": x.std(0).tolist(), "count": [len(x)]}  # fmt: skip


def video_stats(path: Path, frames: int, samples: int = 100) -> dict:
    """LeRobot's per-episode image stats: per channel, values in [0, 1], shape (3, 1, 1), from
    up to `samples` evenly spaced frames decoded at 160 x 120."""
    every = max(1, frames // samples)
    raw = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", str(path), "-vf",
         f"select=not(mod(n\\,{every})),scale=160:120", "-vsync", "0", "-f", "rawvideo",
         "-pix_fmt", "rgb24", "-"],
        capture_output=True, check=True,
    ).stdout  # fmt: skip
    img = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 120 * 160, 3).astype(np.float64) / 255
    flat = img.reshape(-1, 3)

    def c(v):
        return [[[float(x)]] for x in v]

    return {"min": c(flat.min(0)), "max": c(flat.max(0)), "mean": c(flat.mean(0)),
            "std": c(flat.std(0)), "count": [len(img)]}  # fmt: skip


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))

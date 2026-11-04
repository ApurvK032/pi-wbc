"""Record the expert's episodes, states first (STAGE3_DESIGN.md section 8; D-031).

Every control step of an episode stores the commands the expert sent (what a VLA learns to
send) and the simulator state they were applied to: robot root pose and velocity, joint
positions and velocities, object poses, and the expert's phase. Images are rendered afterwards
from these states, so rendering choices never need the physics to run again. One compressed
`.npz` per episode, named by its layout seed, with the episode's metadata (layout,
instruction, outcome, joint and command names, controller, provenance) as JSON inside.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pi_wbc import provenance
from pi_wbc.stage3 import layout as L
from pi_wbc.stage3.expert import PHASES

FIELDS = (
    "cmd_velocity",  # (T, 3) base velocity command: vx, vy, yaw rate
    "cmd_body",  # (T, 4) pelvis height, waist yaw, roll, pitch
    "cmd_arm_hand",  # (T, 28) arm and hand joint targets, in `arm_hand_names` order
    "root_pose",  # (T, 7) pelvis position (environment frame) and quaternion (x, y, z, w)
    "root_vel",  # (T, 6) pelvis linear and angular velocity (world axes)
    "joint_pos",  # (T, 43) in `joint_names` order
    "joint_vel",  # (T, 43)
    "object_pose",  # (T, 5, 7) every object, in `object_names` order
    "phase",  # (T,) the expert's phase index, in `phases` order
)


def _t(x) -> torch.Tensor:
    return x.torch if hasattr(x, "torch") else x


class EpisodeRecorder:
    """Buffers one batch's episodes, one per environment, and writes each when it ends."""

    def __init__(self, env, high_level, expert, out_dir: str | Path, run_info: dict[str, Any]):
        self.base = env.unwrapped
        self.hl = high_level
        self.expert = expert
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.robot = self.base.scene["robot"]
        self.objects = [self.base.scene[f"obj_{name}"] for name in L.OBJECT_NAMES]
        self.run_info = run_info  # controller, expert parameters
        self.provenance = provenance.collect()  # once per run: git, packages, GPU
        self.n = self.base.num_envs
        self.buf: list[dict[str, list[np.ndarray]]] = []
        self.reset()

    def reset(self) -> None:
        self.buf = [{f: [] for f in FIELDS} for _ in range(self.n)]

    def record(self, active: np.ndarray) -> None:
        """The state now and the commands about to be applied to it (call after
        `HighLevel.apply()`, before the environment steps), for the environments in `active`."""
        origins = self.base.scene.env_origins
        root = _t(self.robot.data.root_link_pose_w).clone()
        root[:, :3] -= origins
        objs = torch.stack(
            [
                torch.cat([_t(o.data.root_pos_w) - origins, _t(o.data.root_quat_w)], dim=-1)
                for o in self.objects
            ],
            dim=1,
        )
        now = {
            "cmd_velocity": self.hl.velocity,
            "cmd_body": self.hl.body_goal,
            "cmd_arm_hand": self.hl.arm_hand,
            "root_pose": root,
            "root_vel": _t(self.robot.data.root_link_vel_w),
            "joint_pos": _t(self.robot.data.joint_pos),
            "joint_vel": _t(self.robot.data.joint_vel),
            "object_pose": objs,
        }
        now = {k: v.detach().float().cpu().numpy() for k, v in now.items()}
        now["phase"] = self.expert.phase.astype(np.int16)
        for i in np.flatnonzero(active):
            buf = self.buf[i]
            for k, v in now.items():
                buf[k].append(v[i])

    def save(self, i: int, outcome: dict[str, Any]) -> Path:
        """Write environment i's episode with its outcome (label, phase it failed in, time)."""
        lay = self.base.pick_place_layouts[i]
        data = {k: np.stack(v) for k, v in self.buf[i].items()}
        meta = {
            "seed": lay.seed,
            "split": L.split_of(lay.seed),
            "instruction": lay.instruction,
            "layout": lay.to_dict(),
            "outcome": outcome,
            "steps": len(data["phase"]),
            "control_dt": float(self.base.step_dt),
            "joint_names": list(self.robot.joint_names),
            "arm_hand_names": list(self.hl.arm_hand_names),
            "object_names": list(L.OBJECT_NAMES),
            "phases": list(PHASES),
            "fields": {f: list(data[f].shape) for f in FIELDS},
            **self.run_info,
            "provenance": self.provenance,
        }
        path = self.out / f"{lay.seed}.npz"
        np.savez_compressed(path, meta=np.array(json.dumps(meta)), **data)
        self.buf[i] = {f: [] for f in FIELDS}
        return path

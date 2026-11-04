"""NVIDIA's Decoupled WBC lower body as the pick-place task's low level (STAGE3_DESIGN.md
sections 6-7, D-046).

The released Balance and Walk policies (scripts/fetch_nvidia_wbc.sh; "Balance" while the
velocity command is below 0.05, "Walk" otherwise, as in NVIDIA's `g1_gear_wbc_policy.py`) drive
the 12 leg and 3 waist joints. Their observation is built as in scripts/nvidia_wbc_test.py (86
values per step, the last 6 steps oldest first) from the task's own commands, the same ones our
controller gets:

- velocity: the `base_velocity` command (vx, vy, yaw rate), as written by the high level;
- height: the `body_pose` command's pelvis height minus `height_offset` (our controller's
  nominal is 0.78 m, NVIDIA's 0.74 m: the command interface's height is taken relative to each
  controller's nominal);
- torso roll, pitch, yaw: the `body_pose` command's waist roll, pitch and yaw.

The arms and hands are untouched: the task's `upper_body` command sets their targets directly,
for either controller. NVIDIA's targets (action x 0.25 + its default angles) are converted into
our action term's units, so `env.step` applies exactly those targets: raw = (target - default -
crouch offset at the commanded height) / 0.25 (`HeightPoseJointPositionAction`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

LEGS = ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
ARMS = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch",
        "wrist_yaw")  # fmt: skip
# NVIDIA's body joint order (Unitree's 29: legs L then R, waist, arms L then R)
BODY = (
    [f"left_{j}_joint" for j in LEGS]
    + [f"right_{j}_joint" for j in LEGS]
    + ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
    + [f"left_{j}_joint" for j in ARMS]
    + [f"right_{j}_joint" for j in ARMS]
)
ACTION_JOINTS = BODY[:15]
DEFAULT_ANGLES = np.array([-0.1, 0.0, 0.0, 0.3, -0.2, 0.0] * 2 + [0.0] * 3, dtype=np.float32)
CMD_SCALE = np.array([2.0, 2.0, 0.5], dtype=np.float32)
ACTION_SCALE, HISTORY, FRAME = 0.25, 6, 86
WALK_ABOVE = 0.05  # velocity command norm above which the Walk policy runs
NOMINAL_HEIGHT = {"ours": 0.78, "nvidia": 0.74}


def _t(x):
    return x.torch if hasattr(x, "torch") else x


class NvidiaLowLevel:
    """Computes the task's actions from NVIDIA's policies, once per control step (`act`)."""

    def __init__(self, env, policy_dir: Path, height_offset: float | None = None):
        import onnxruntime as ort

        base = env.unwrapped
        self.robot = base.scene["robot"]
        self.term = base.action_manager.get_term("joint_pos")
        self.vel = base.command_manager.get_term("base_velocity")
        self.body = base.command_manager.get_term("body_pose")
        self.n, self.dev = base.num_envs, base.device
        self.sessions = {
            k: ort.InferenceSession(str(Path(policy_dir) / f"GR00T-WholeBodyControl-{k}.onnx"),
                                    providers=["CPUExecutionProvider"])
            for k in ("Balance", "Walk")
        }  # fmt: skip
        self.policy_commit = (Path(policy_dir).parent / "COMMIT").read_text().strip()
        if height_offset is None:
            height_offset = NOMINAL_HEIGHT["ours"] - NOMINAL_HEIGHT["nvidia"]
        self.height_offset = height_offset
        names = list(self.robot.joint_names)
        self.body_ids = torch.tensor([names.index(j) for j in BODY], device=self.dev)
        term_names = list(self.term._joint_names)
        if sorted(term_names) != sorted(ACTION_JOINTS):
            raise ValueError(f"the action term drives {term_names}, not NVIDIA's 15 joints")
        # column of the action term for each of NVIDIA's 15 action joints
        self.term_cols = torch.tensor([term_names.index(j) for j in ACTION_JOINTS],
                                      device=self.dev)  # fmt: skip
        default = _t(self.robot.data.default_joint_pos)[0, self.body_ids[:15]].cpu().numpy()
        if not np.allclose(default, DEFAULT_ANGLES, atol=1e-6):
            raise ValueError(f"our default leg pose {default} is not NVIDIA's {DEFAULT_ANGLES}")
        self.hist = np.zeros((self.n, HISTORY, FRAME), dtype=np.float32)
        self.action = np.zeros((self.n, 15), dtype=np.float32)

    def reset(self, env_ids=None) -> None:
        ids = slice(None) if env_ids is None else list(env_ids)
        self.hist[ids] = 0.0
        self.action[ids] = 0.0

    def observation(self) -> tuple[np.ndarray, np.ndarray]:
        """(this step's 86-value frame, |velocity command|), from the task's state."""
        vel = self.vel.command.cpu().numpy()
        body = self.body.command.cpu().numpy()  # pelvis height, waist yaw, roll, pitch
        height = body[:, :1] - self.height_offset
        rpy = body[:, [2, 3, 1]]  # NVIDIA's torso command: roll, pitch, yaw
        q = _t(self.robot.data.joint_pos)[:, self.body_ids].cpu().numpy()
        dq = _t(self.robot.data.joint_vel)[:, self.body_ids].cpu().numpy()
        omega = _t(self.robot.data.root_ang_vel_b).cpu().numpy()
        grav = _t(self.robot.data.projected_gravity_b).cpu().numpy()
        q = q.copy()
        q[:, :15] -= DEFAULT_ANGLES
        frame = np.concatenate([vel * CMD_SCALE, height, rpy, omega * 0.5, grav, q, dq * 0.05,
                                self.action], axis=1).astype(np.float32)  # fmt: skip
        return frame, np.linalg.norm(vel, axis=1)

    def act(self) -> torch.Tensor:
        """The task's raw actions (num_envs, 15) that make the action term apply NVIDIA's
        joint targets this step."""
        frame, speed = self.observation()
        self.hist = np.concatenate([self.hist[:, 1:], frame[:, None]], axis=1)
        obs = self.hist.reshape(self.n, -1)
        out = np.zeros((self.n, 15), dtype=np.float32)
        walk = speed >= WALK_ABOVE
        for name, mask in (("Walk", walk), ("Balance", ~walk)):
            if mask.any():
                out[mask] = self.sessions[name].run(None, {"input": obs[mask]})[0]
        self.action = out
        target = torch.as_tensor(out * ACTION_SCALE + DEFAULT_ANGLES, device=self.dev)
        return self.to_raw(target)

    def to_raw(self, target: torch.Tensor) -> torch.Tensor:
        """Invert `HeightPoseJointPositionAction.process_actions` for targets given in NVIDIA's
        joint order."""
        term = self.term
        t = torch.zeros(self.n, 15, device=self.dev)
        t[:, self.term_cols] = target
        offset = term._offset.expand(self.n, -1).clone()
        height = self.body.command[:, 0]
        offset[:, term._pose_cols] += term.pose_offset(height)
        return (t - offset) / term._scale

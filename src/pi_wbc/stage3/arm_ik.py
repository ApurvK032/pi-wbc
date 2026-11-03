"""Inverse kinematics of one G1 arm for the scripted expert.

The arm's joint targets go straight to their PD controllers (D-012), so the expert plans the
palm's path and needs joint angles for each point of it. They are solved here on the MuJoCo model
that Newton builds from the robot for training (the `newton_model.xml` in a controller's export),
so links, frames and joint limits are the simulator's own. Each solve sets the pelvis pose and
the other joints (waist, the other arm) to the robot's current state and moves only the seven
joints of one arm: damped least squares on the palm's position and orientation, with radians
counting `w_rot` as much as metres, so an orientation that is hard to reach does not cost
position. Joints are clamped to the controller's trained arm ranges (from its `policy_io.json`),
so the arm stays where the controller learned to balance it.

MuJoCo quaternions are (w, x, y, z); this module takes and returns Isaac Lab's (x, y, z, w).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

ARM_JOINTS = (
    "shoulder_pitch_joint",
    "shoulder_roll_joint",
    "shoulder_yaw_joint",
    "elbow_joint",
    "wrist_roll_joint",
    "wrist_pitch_joint",
    "wrist_yaw_joint",
)


def arm_joint_names(side: str) -> list[str]:
    return [f"{side}_{j}" for j in ARM_JOINTS]


@dataclass
class IkResult:
    q: np.ndarray  # (7,) arm joint angles
    pos_err: float  # m
    rot_err: float  # rad


def _rot_err(target_r: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Rotation vector taking r to target_r (world frame), small-angle accurate."""
    re = target_r @ r.T
    angle = np.arccos(np.clip((np.trace(re) - 1.0) / 2.0, -1.0, 1.0))
    axis = np.array([re[2, 1] - re[1, 2], re[0, 2] - re[2, 0], re[1, 0] - re[0, 1]])
    s = np.linalg.norm(axis)
    if s < 1e-9:
        return np.zeros(3)
    return axis / s * angle


class ArmIK:
    """IK for the palm of one arm on Newton's model of the robot."""

    def __init__(
        self,
        model_xml: str | Path,
        policy_io: str | Path,
        side: str = "right",
        w_rot: float = 0.3,
        damping: float = 1e-2,
    ):
        self.model = mujoco.MjModel.from_xml_path(str(model_xml))
        self.data = mujoco.MjData(self.model)
        self.side = side
        self.w_rot = w_rot
        self.damping = damping
        m = self.model
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(m.njnt)]
        self._qadr: dict[str, int] = {}
        self._dadr: dict[str, int] = {}
        for j, full in enumerate(names):
            if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                self._free_qadr = m.jnt_qposadr[j]
                continue
            # "..._Robot_joints_<name>" or "..._Robot_right_hand_joints_<name>"
            short = full.split("Robot_")[-1].split("joints_")[-1]
            self._qadr[short] = m.jnt_qposadr[j]
            self._dadr[short] = m.jnt_dofadr[j]
        self.arm = arm_joint_names(side)
        self._arm_q = np.array([self._qadr[n] for n in self.arm])
        self._arm_d = np.array([self._dadr[n] for n in self.arm])
        bodies = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in range(m.nbody)]
        self.palm = next(b for b, n in enumerate(bodies) if n.endswith(f"{side}_hand_palm_link"))
        io = json.loads(Path(policy_io).read_text())
        ub = io["upper_body"]
        rng = {
            n: (lo, hi) for n, lo, hi in zip(ub["joint_names"], ub["low"], ub["high"], strict=True)
        }
        self.lower = np.array([rng[n][0] for n in self.arm])
        self.upper = np.array([rng[n][1] for n in self.arm])
        self.kp = np.array([io["joints"][n]["stiffness"] for n in self.arm])
        self._palm_mass0 = float(self.model.body_mass[self.palm])

    def set_state(
        self, base_pos, base_quat_xyzw, joint_pos: dict[str, float] | None = None
    ) -> None:
        """Pelvis pose (world) and any joint angles by name (others keep their last values)."""
        d = self.data
        a = self._free_qadr
        d.qpos[a : a + 3] = base_pos
        x, y, z, w = base_quat_xyzw
        d.qpos[a + 3 : a + 7] = (w, x, y, z)
        for name, value in (joint_pos or {}).items():
            if name in self._qadr:
                d.qpos[self._qadr[name]] = value

    def palm_pose(self, q: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Palm position and rotation matrix (world) for arm angles q (default: as set)."""
        if q is not None:
            self.data.qpos[self._arm_q] = q
        mujoco.mj_kinematics(self.model, self.data)
        return self.data.xpos[self.palm].copy(), self.data.xmat[self.palm].reshape(3, 3).copy()

    def gravity_offset(self, q: np.ndarray, payload_kg: float = 0.0) -> np.ndarray:
        """What to add to the arm's PD targets so that, at rest at `q`, gravity is held exactly:
        each joint's gravity torque divided by its stiffness (with `payload_kg` in the palm)."""
        m, d = self.model, self.data
        m.body_mass[self.palm] = self._palm_mass0 + payload_kg
        d.qpos[self._arm_q] = q
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        bias = d.qfrc_bias[self._arm_d].copy()  # gravity only: the robot is at rest
        m.body_mass[self.palm] = self._palm_mass0
        return bias / self.kp

    def solve(
        self,
        target_pos: np.ndarray,
        target_rot: np.ndarray,
        q_init: np.ndarray,
        iters: int = 60,
        tol_pos: float = 1e-4,
        tol_rot: float = 1e-3,
    ) -> IkResult:
        """Arm angles that put the palm at target_pos with rotation matrix target_rot (world)."""
        m, d = self.model, self.data
        q = np.clip(np.asarray(q_init, dtype=float), self.lower, self.upper)
        jp = np.zeros((3, m.nv))
        jr = np.zeros((3, m.nv))
        eye = np.eye(6) * self.damping**2
        for _ in range(iters):
            d.qpos[self._arm_q] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            ep = target_pos - d.xpos[self.palm]
            er = _rot_err(target_rot, d.xmat[self.palm].reshape(3, 3))
            if np.linalg.norm(ep) < tol_pos and np.linalg.norm(er) < tol_rot:
                break
            mujoco.mj_jacBody(m, d, jp, jr, self.palm)
            j = np.vstack([jp[:, self._arm_d], self.w_rot * jr[:, self._arm_d]])
            e = np.concatenate([ep, self.w_rot * er])
            dq = j.T @ np.linalg.solve(j @ j.T + eye, e)
            q = np.clip(q + dq, self.lower, self.upper)
        pos, rot = self.palm_pose(q)
        return IkResult(
            q=q,
            pos_err=float(np.linalg.norm(target_pos - pos)),
            rot_err=float(np.linalg.norm(_rot_err(target_rot, rot))),
        )

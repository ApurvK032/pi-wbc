"""The scripted expert of the pick-and-place task (STAGE3_DESIGN.md, section 5; D-036).

A state machine per environment over the simulator's true state. Every control step it writes
the high level's commands (`pi_wbc.tasks.g1_pick_place.HighLevel`): base velocity while walking,
the body pose (a torso lean to reach the table), and the right arm and hand joint targets. The
left arm and hand stay at the default pose.

Phases: walk to the object's stance, settle, raise the hand in front of the chest while leaning,
reach above the object, lower beside it, close, lift; straighten, walk to the bin's stance,
settle, lean and move the object above the bin, open the hand, retract. A failed check ends the
episode with a label (no retries): fell, knocked, missed grasp, dropped, timed out, wrong bin.

Arm motion follows the grasp test (results/stage3): the palm's path is planned in the table's
frame and solved by inverse kinematics on Newton's model of the robot (`ArmIK`), backwards from
the grasp so the arm stays on one smooth branch. While it moves, each step solves the IK again
from the robot's measured pelvis pose and waist angles, so the hand stays on its path while
the body sways; a slow bounded correction on the measured palm error removes the arm's sag.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from pi_wbc.stage3 import layout as L
from pi_wbc.stage3 import navigate as nav
from pi_wbc.stage3.arm_ik import ArmIK, arm_joint_names

RIGHT_ARM = arm_joint_names("right")
RIGHT_HAND = [
    "right_hand_index_0_joint", "right_hand_index_1_joint", "right_hand_middle_0_joint",
    "right_hand_middle_1_joint", "right_hand_thumb_0_joint", "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
]  # fmt: skip
HAND_OPEN = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.72, 0.0])  # fingers straight, thumb back
HAND_CLOSED = np.array([1.2, 1.2, 1.2, 1.2, 0.0, -1.0, -0.9])

PHASES = (
    "walk_obj", "settle_obj", "raise", "reach", "lower", "close", "lift", "straighten",
    "walk_bin", "settle_bin", "reach_bin", "release", "retract", "done", "failed",
)  # fmt: skip
P = {name: i for i, name in enumerate(PHASES)}
# "no_plan" is the expert's own failure (its stance left the target out of reach)
LABELS = ("success", "fell", "knocked", "missed_grasp", "dropped", "wrong_bin", "timed_out",
          "no_plan")  # fmt: skip


@dataclass(frozen=True)
class ExpertParams:
    stance_rel: tuple[float, float] = (0.40, -0.18)  # object in the robot's heading frame
    lean: float = 0.25  # waist pitch while reaching (rad)
    grasp_in_palm: tuple[float, float, float] = (0.085, 0.062, 0.0)  # grasp centre, palm frame
    grasp_height: float = 0.075  # grasp centre above the table top
    pre_up: float = 0.15  # start of the descent above the grasp
    lift_up: float = 0.15
    raise_x: float = 0.25  # robot frame: where the hand rises in front of the chest (0.20 is
    # 2.5-3.5 cm out of reach at that height in the handshake orientation)
    release_above: float = 0.05  # object bottom above the bin's rim when released
    settle_s: float = 1.0
    raise_s: float = 1.5
    reach_s: float = 1.5
    lower_s: float = 1.2
    close_s: float = 1.2
    lift_s: float = 1.2
    straighten_s: float = 0.8
    reach_bin_s: float = 2.0
    release_s: float = 1.5
    retract_s: float = 1.5
    walk_timeout_s: float = 30.0
    episode_timeout_s: float = 60.0
    corr_gain: float = 0.1
    max_corr: float = 0.15
    knock_dist: float = 0.03  # target object moved this far before the grasp: knocked
    nav: nav.NavParams = field(default_factory=nav.NavParams)


def _smooth(s: float) -> float:
    s = min(max(s, 0.0), 1.0)
    return 0.5 - 0.5 * math.cos(math.pi * s)


def _yaw_of(quat_xyzw: torch.Tensor) -> torch.Tensor:
    x, y, z, w = quat_xyzw.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _rz(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class Expert:
    def __init__(self, env, high_level, export_dir: str | Path, params: ExpertParams | None = None):
        self.base = env.unwrapped
        self.hl = high_level
        self.p = params if params is not None else ExpertParams()
        base = self.base
        self.n, self.dev = base.num_envs, base.device
        self.dt = base.step_dt
        self.robot = base.scene["robot"]
        export_dir = Path(export_dir)
        self.iks = [
            ArmIK(export_dir / "newton_model.xml", export_dir / "policy_io.json", "right")
            for _ in range(self.n)
        ]
        names = list(self.robot.joint_names)
        self.joint_names = names
        self.arm_ids = [names.index(j) for j in RIGHT_ARM]
        self.hand_slots = high_level.index(RIGHT_HAND)
        self.arm_slots = high_level.index(RIGHT_ARM)
        self.palm = self.robot.find_bodies("right_hand_palm_link")[0][0]
        self.objects = [base.scene[f"obj_{name}"] for name in L.OBJECT_NAMES]
        self.q_hang = np.array([float(high_level.upper.default[high_level.index([j])[0]])
                                for j in RIGHT_ARM])  # fmt: skip
        n = self.n
        self.phase = np.full(n, P["walk_obj"])
        self.t = np.zeros(n, dtype=int)  # steps in the current phase
        self.age = np.zeros(n, dtype=int)  # steps in the episode
        self.label = [None] * n
        self.failed_in: list[str | None] = [None] * n  # the phase a failure happened in
        self.q_ik = [self.q_hang.copy() for _ in range(n)]
        self.q_corr = torch.zeros(n, 7, device=self.dev)
        self.hand = np.tile(HAND_OPEN, (n, 1))
        self.plan: list[dict] = [{} for _ in range(n)]
        self.goal = torch.zeros(n, 3, device=self.dev)
        self.obj_start = torch.zeros(n, 3, device=self.dev)
        self.obj_in_palm = [np.zeros(3) for _ in range(n)]
        self.target_k = np.zeros(n, dtype=int)

    # -- state ----------------------------------------------------------------------------------
    def _env_pos(self, world: torch.Tensor) -> torch.Tensor:
        return world - self.base.scene.env_origins

    def _base_pose(self) -> torch.Tensor:
        pos = self._env_pos(self.robot.data.root_pos_w.torch)
        yaw = _yaw_of(self.robot.data.root_quat_w.torch)
        return torch.stack([pos[:, 0], pos[:, 1], yaw], dim=-1)

    def _target_obj_pos(self) -> torch.Tensor:
        pos = torch.stack([o.data.root_pos_w.torch for o in self.objects], dim=1)
        rows = torch.arange(self.n, device=self.dev)
        return self._env_pos(pos[rows, torch.as_tensor(self.target_k, device=self.dev)])

    def _palm_pose(self):
        return (self._env_pos(self.robot.data.body_link_pos_w.torch[:, self.palm]),
                self.robot.data.body_link_quat_w.torch[:, self.palm])  # fmt: skip

    def reset(self, env_ids) -> None:
        layouts = self.base.pick_place_layouts
        for i in env_ids:
            lay = layouts[i]
            self.phase[i] = P["walk_obj"]
            self.t[i] = self.age[i] = 0
            self.label[i] = None
            self.failed_in[i] = None
            self.q_ik[i] = self.q_hang.copy()
            self.hand[i] = HAND_OPEN
            self.plan[i] = {}
            self.target_k[i] = L.OBJECT_NAMES.index(lay.target_object)
            obj = torch.tensor(lay.object_pos(lay.target_object)[:2], device=self.dev)[None]
            self.goal[i] = nav.stance_for(obj, self.p.stance_rel, torch.zeros(1, device=self.dev))[
                0
            ]
            self.obj_start[i] = torch.tensor(lay.object_pos(lay.target_object), device=self.dev)
        self.q_corr[env_ids] = 0.0
        self.hl.reset(env_ids)

    # -- arm planning (CPU, one environment) ----------------------------------------------------
    def _ik_state(self, i: int, waist_pitch: float | None = None) -> None:
        pos = self._env_pos(self.robot.data.root_pos_w.torch)[i].tolist()
        quat = self.robot.data.root_quat_w.torch[i].tolist()
        q = self.robot.data.joint_pos.torch[i].tolist()
        joints = dict(zip(self.joint_names, q, strict=True))
        if waist_pitch is not None:  # plan for the lean the body is about to take
            joints = {**joints, "waist_pitch_joint": waist_pitch}
        self.iks[i].set_state(pos, quat, joints)

    def _plan_grasp(self, i: int) -> bool:
        p, ik = self.p, self.iks[i]
        yaw = float(self._base_pose()[i, 2])
        rot = _rz(yaw)  # handshake: palm frame = the robot's heading frame
        obj = self._target_obj_pos()[i].cpu().numpy()
        center = np.array([obj[0], obj[1], L.TABLE_TOP + p.grasp_height])
        grasp = center - rot @ np.array(p.grasp_in_palm)
        up = np.array([0.0, 0.0, 1.0])
        pre = grasp + p.pre_up * up
        pelvis = self._env_pos(self.robot.data.root_pos_w.torch)[i, :2].cpu().numpy()
        grasp_rel = rot[:2, :2].T @ (grasp[:2] - pelvis)  # the grasp in the robot's frame
        start = pre.copy()  # in front of the chest, on the grasp's line, at the descent's height
        start[:2] = pelvis + rot[:2, :2] @ np.array([p.raise_x, grasp_rel[1]])
        self._ik_state(i, waist_pitch=p.lean)
        g = ik.solve(grasp, rot, self.q_hang, iters=400)
        if g.pos_err > 0.01 or g.rot_err > 0.05:
            return False
        q = g.q
        for a, b in ((grasp, pre), (pre, start)):
            for s in np.linspace(0.0, 1.0, max(2, int(np.linalg.norm(b - a) / 0.005)))[1:]:
                q = ik.solve(a + s * (b - a), rot, q, iters=100).q
        q_now = self.robot.data.joint_pos.torch[i, self.arm_ids].cpu().numpy()
        self.plan[i] = {"rot": rot, "grasp": grasp, "pre": pre, "start": start, "q_start": q,
                        "q_from": q_now}  # fmt: skip
        return True

    def _plan_release(self, i: int) -> bool:
        p, ik = self.p, self.iks[i]
        lay = self.base.pick_place_layouts[i]
        yaw = float(self._base_pose()[i, 2])
        rot = _rz(yaw)
        bx, by = lay.bin_xy(lay.target_bin)
        h = L.OBJECTS[lay.target_object]["height"]
        obj_target = np.array(
            [bx, by, L.TABLE_TOP + 0.01 + L.BIN_WALL[1] + p.release_above + h / 2]
        )
        palm_target = obj_target - rot @ self.obj_in_palm[i]
        palm_now = self._palm_pose()[0][i].cpu().numpy()
        self._ik_state(i, waist_pitch=p.lean)
        r = ik.solve(palm_target, rot, self.q_ik[i], iters=300)
        if r.pos_err > 0.02:
            return False
        self.plan[i] = {"rot": rot, "from": palm_now, "to": palm_target}
        return True

    def _ik_follow(self, i: int, target: np.ndarray, rot: np.ndarray) -> np.ndarray:
        self._ik_state(i)
        self.q_ik[i] = self.iks[i].solve(target, rot, self.q_ik[i], iters=40).q
        return self.q_ik[i]

    # -- the step -------------------------------------------------------------------------------
    def _go(self, i: int, phase: str) -> None:
        self.phase[i] = P[phase]
        self.t[i] = 0

    def _fail(self, i: int, label: str) -> None:
        self.label[i] = label
        self.failed_in[i] = PHASES[self.phase[i]]
        self._go(i, "failed")

    def step(self) -> None:
        """Write this control step's commands for every environment."""
        p, hl = self.p, self.hl
        dt = self.dt
        pose = self._base_pose()
        z = self.robot.data.root_pos_w.torch[:, 2] - self.base.scene.env_origins[:, 2]
        tilt = self.robot.data.projected_gravity_b.torch[:, 2]
        fell = ((z < 0.45) | (tilt > -0.5)).cpu().numpy()
        obj = self._target_obj_pos()
        palm_pos, palm_quat = self._palm_pose()
        obj_palm = (obj - palm_pos).norm(dim=-1).cpu().numpy()
        moved = (obj - self.obj_start).norm(dim=-1).cpu().numpy()
        nav_cmd, arrived = nav.command(pose, self.goal, p.nav)
        arrived = arrived.cpu().numpy()
        hl.velocity[:] = 0.0
        targets = {}  # env -> (palm target, rot) for the sag correction
        arm_cmd = {}  # env -> arm joint targets before the correction

        for i in range(self.n):
            ph = PHASES[self.phase[i]]
            if ph in ("done", "failed"):
                continue
            self.age[i] += 1
            self.t[i] += 1
            k = self.t[i]
            if fell[i]:
                self._fail(i, "fell")
                continue
            if self.age[i] * dt > p.episode_timeout_s:
                self._fail(i, "timed_out")
                continue
            if self.phase[i] <= P["lower"] and moved[i] > p.knock_dist:
                self._fail(i, "knocked")
                continue
            if P["lift"] < self.phase[i] < P["release"] and obj_palm[i] > 0.12:
                self._fail(i, "dropped")
                continue
            arm = None
            if ph in ("walk_obj", "walk_bin"):
                hl.velocity[i] = nav_cmd[i]
                if arrived[i]:
                    self._go(i, "settle_obj" if ph == "walk_obj" else "settle_bin")
                elif k * dt > p.walk_timeout_s:
                    self._fail(i, "timed_out")
            elif ph in ("settle_obj", "settle_bin"):
                if k * dt >= p.settle_s:
                    if ph == "settle_obj":
                        if self._plan_grasp(i):
                            self.q_corr[i] = 0.0
                            self._go(i, "raise")
                        else:
                            self._fail(i, "no_plan")  # out of reach: the stance was off
                    else:
                        hl.body_goal[i, 3] = p.lean
                        if self._plan_release(i):
                            self._go(i, "reach_bin")
                        else:
                            self._fail(i, "no_plan")
            elif ph == "raise":
                s = _smooth(k * dt / p.raise_s)
                pl = self.plan[i]
                arm = pl["q_from"] + s * (pl["q_start"] - pl["q_from"])
                self.q_ik[i] = arm.copy()
                hl.body_goal[i, 3] = p.lean
                if k * dt >= p.raise_s:
                    self._go(i, "reach")
            elif ph in ("reach", "lower", "lift"):
                pl = self.plan[i]
                a, b, dur = {
                    "reach": (pl["start"], pl["pre"], p.reach_s),
                    "lower": (pl["pre"], pl["grasp"], p.lower_s),
                    "lift": (pl["grasp"], pl["grasp"] + [0.0, 0.0, p.lift_up], p.lift_s),
                }[ph]
                target = a + _smooth(k * dt / dur) * (b - a)
                arm = self._ik_follow(i, target, pl["rot"])
                targets[i] = (target, pl["rot"])
                if ph == "lift":
                    self.hand[i] = HAND_CLOSED
                if k * dt >= dur:
                    if ph == "lift":
                        lifted = float(obj[i, 2] - self.obj_start[i, 2])
                        if lifted < 0.08 or obj_palm[i] > 0.10:
                            self._fail(i, "missed_grasp")
                            continue
                        rot = pl["rot"]
                        self.obj_in_palm[i] = rot.T @ (obj[i] - palm_pos[i]).cpu().numpy()
                        self.plan[i]["q_hold"] = self.q_ik[i].copy()
                        self._go(i, "straighten")
                    else:
                        self._go(i, "lower" if ph == "reach" else "close")
            elif ph == "close":
                pl = self.plan[i]
                arm = self._ik_follow(i, pl["grasp"], pl["rot"])
                targets[i] = (pl["grasp"], pl["rot"])
                s = min(1.0, k * dt / p.close_s)  # fingers and thumb together
                self.hand[i] = HAND_OPEN + s * (HAND_CLOSED - HAND_OPEN)
                if k * dt >= p.close_s:
                    self._go(i, "lift")
            elif ph == "straighten":
                arm = self.plan[i]["q_hold"]
                hl.body_goal[i, 3] = 0.0
                if k * dt >= p.straighten_s:
                    lay = self.base.pick_place_layouts[i]
                    bxy = torch.tensor(lay.bin_xy(lay.target_bin), device=self.dev)[None]
                    zero = torch.zeros(1, device=self.dev)
                    self.goal[i] = nav.stance_for(bxy, p.stance_rel, zero)[0]
                    self._go(i, "walk_bin")
            elif ph == "reach_bin":
                pl = self.plan[i]
                target = pl["from"] + _smooth(k * dt / p.reach_bin_s) * (pl["to"] - pl["from"])
                arm = self._ik_follow(i, target, pl["rot"])
                targets[i] = (target, pl["rot"])
                hl.body_goal[i, 3] = p.lean
                if k * dt >= p.reach_bin_s:
                    self._go(i, "release")
            elif ph == "release":
                pl = self.plan[i]
                arm = self._ik_follow(i, pl["to"], pl["rot"])
                s = min(1.0, k * dt / 0.5)
                self.hand[i] = HAND_CLOSED + s * (HAND_OPEN - HAND_CLOSED)
                if k * dt >= p.release_s:
                    self._judge_place(i, obj[i].cpu().numpy())
                    self._go(i, "retract")
            elif ph == "retract":  # rise clear of the bin, then back to the hanging pose
                pl = self.plan[i]
                rise = 0.6
                if k * dt < rise:
                    target = pl["to"] + _smooth(k * dt / rise) * np.array([0.0, 0.0, 0.10])
                    arm = self._ik_follow(i, target, pl["rot"])
                    targets[i] = (target, pl["rot"])
                    pl["q_from"] = self.q_ik[i].copy()
                else:
                    s = _smooth((k * dt - rise) / p.retract_s)
                    arm = pl["q_from"] + s * (self.q_hang - pl["q_from"])
                    hl.body_goal[i, 3] = 0.0
                    if k * dt >= rise + p.retract_s:
                        self._go(i, "done")
            if ph in ("walk_bin", "settle_bin") and "q_hold" in self.plan[i]:
                arm = self.plan[i]["q_hold"]
            if arm is not None:
                arm_cmd[i] = arm
            hl.arm_hand[i, self.hand_slots] = torch.tensor(
                self.hand[i], device=self.dev, dtype=torch.float32
            )

        self._sag_correction(targets, palm_pos, palm_quat)
        for i, arm in arm_cmd.items():
            hl.arm_hand[i, self.arm_slots] = (
                torch.tensor(arm, device=self.dev, dtype=torch.float32) + self.q_corr[i]
            )

    def _judge_place(self, i: int, obj: np.ndarray) -> None:
        lay = self.base.pick_place_layouts[i]
        r = L.OBJECTS[lay.target_object]["radius"]
        inner = (L.BIN_SIZE[0] / 2 - L.BIN_WALL[0], L.BIN_SIZE[1] / 2 - L.BIN_WALL[0])

        def inside(bin_name):
            bx, by = lay.bin_xy(bin_name)
            resting = (
                obj[2] < L.TABLE_TOP + 0.01 + L.OBJECTS[lay.target_object]["height"] / 2 + 0.03
            )
            return (
                abs(obj[0] - bx) < inner[0] - r / 2
                and abs(obj[1] - by) < inner[1] - r / 2
                and resting
            )

        if inside(lay.target_bin):
            self.label[i] = "success"
        else:
            other = next(b for b in lay.bins if b != lay.target_bin)
            self.label[i] = "wrong_bin" if inside(other) else "dropped"

    def _sag_correction(self, targets, palm_pos, palm_quat) -> None:
        from isaaclab.utils.math import compute_pose_error, quat_from_matrix

        if not targets:
            return
        ids = torch.tensor(sorted(targets), device=self.dev)
        tgt = torch.tensor(
            np.stack([targets[int(i)][0] for i in ids]), device=self.dev, dtype=torch.float32
        )
        rots = torch.tensor(
            np.stack([targets[int(i)][1] for i in ids]), device=self.dev, dtype=torch.float32
        )
        q_tgt = quat_from_matrix(rots)
        jac = self.robot.data.body_link_jacobian_w.torch[ids, self.palm][
            :, :, [6 + j for j in self.arm_ids]
        ]
        ep, eo = compute_pose_error(
            palm_pos[ids], palm_quat[ids], tgt, q_tgt, rot_error_type="axis_angle"
        )
        w = 0.3
        e = torch.cat([ep, w * eo], dim=-1)[..., None]
        j = torch.cat([jac[:, :3], w * jac[:, 3:]], dim=1)
        eye = torch.eye(6, device=self.dev) * 0.05**2
        step = (j.transpose(1, 2) @ torch.linalg.solve(j @ j.transpose(1, 2) + eye, e))[..., 0]
        self.q_corr[ids] = (self.q_corr[ids] + self.p.corr_gain * step).clamp(
            -self.p.max_corr, self.p.max_corr
        )

    def finished(self) -> np.ndarray:
        return np.isin(self.phase, [P["done"], P["failed"]])

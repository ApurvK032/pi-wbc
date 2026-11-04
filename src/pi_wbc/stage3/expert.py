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

import collections
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

# Leaning moves the pelvis (the controller steps back a little), so the robot walks to its
# stance upright, leans, and then steps onto the stance again with the lean held ("align").
PHASES = (
    "walk_obj", "lean_obj", "align_obj", "settle_obj", "raise", "reach", "fine_pre", "lower",
    "fine_out", "slide", "fine_grasp", "close", "lift", "straighten", "walk_bin",
    "lean_bin", "align_bin", "settle_bin", "reach_bin", "release", "retract", "done", "failed",
)  # fmt: skip
P = {name: i for i, name in enumerate(PHASES)}
# "no_plan" is the expert's own failure (its stance left the target out of reach)
LABELS = ("success", "fell", "knocked", "missed_grasp", "dropped", "wrong_bin", "timed_out",
          "no_plan")  # fmt: skip


@dataclass(frozen=True)
class ExpertParams:
    stance_rel: tuple[float, float] = (0.40, -0.18)  # object in the robot's heading frame
    lean: float = 0.28  # waist pitch while reaching (rad; trained up to 0.45)
    # pelvis height at the table, from leaning to the lift (m; None: the nominal 0.78). Reaching
    # down beside the object tips the pelvis back up to 0.17 rad (the controller balancing the
    # arm), raising the shoulder until the low point is out of reach; a crouch lowers it again
    crouch: float | None = None
    grasp_in_palm: tuple[float, float, float] = (0.085, 0.062, 0.0)  # grasp centre, palm frame
    grasp_height: float = 0.07  # grasp centre above the table top
    pre_up: float = 0.15  # start of the descent above the grasp
    slide_in: float = 0.03  # come down this much further out along the palm normal, then slide in
    slide_s: float = 0.6
    lift_up: float = 0.10  # 0.15 (0.97 m) is at the edge of reach: the IK sometimes ran off
    raise_x: float = 0.25  # robot frame: where the hand rises in front of the chest (0.20 is
    # 2.5-3.5 cm out of reach at that height in the handshake orientation)
    release_above: float = 0.05  # object bottom above the bin's rim when released
    settle_s: float = 1.0
    lean_s: float = 0.6
    align_timeout_s: float = 10.0
    align_tries: int = 3
    raise_s: float = 1.5
    reach_s: float = 1.5
    lower_s: float = 1.6
    close_s: float = 1.2
    lift_s: float = 1.2
    straighten_s: float = 0.8
    reach_bin_s: float = 2.0
    release_s: float = 1.5
    retract_s: float = 1.5
    walk_timeout_s: float = 30.0
    episode_timeout_s: float = 60.0
    body_smooth: float = 1.0  # per step (1: the IK follows the measured body; 0.1, a ~0.2 s
    # average, made the hand lag the torso: 18 of 32 knocked)
    corr_gain: float = 0.05  # gravity is fed forward (ArmIK.gravity_offset); this trims the rest
    max_corr: float = 0.1
    fine_pre_tol: float = 0.01  # the hand waits above the object until this close (m)
    fine_out_tol: float = 0.008  # and beside it, before sliding in
    fine_grasp_tol: float = 0.005  # and at the grasp point before closing
    fine_max_s: float = 1.5
    knock_dist: float = 0.03  # target object moved this far before the grasp: knocked
    grab_tol: float = 0.015  # >0: close once the object is this close to the grasp point in the
    # hand (and not short of it), without waiting for the palm (the grasp target follows the
    # object, so a palm that pushes it would keep pushing)
    ik_w_rot: float = 0.3  # IK weight of the palm's orientation against its position
    # the controller tips the pelvis back as the arm reaches, and episodes whose pelvis stays
    # within 0.04 rad nearly always succeed (logs/expert/dev-v25-tilt). Two ways to tip it less:
    tilt_pause: float = 0.0  # >0: hold the hand's path while the pelvis tips back more than this
    tilt_pause_max_s: float = 2.0  # at most this long per phase
    counter_arm: float = 0.0  # swing the left arm back by this (shoulder pitch, rad) while leaning
    counter_rate: float = 1.0  # rad/s
    tilt_gain: float = 0.0  # >0: while the arm works at the table, lean the waist further by
    # this share of the pelvis's backward tilt (the controller tips it back as the arm reaches)
    release_away: float = 0.04  # after opening over the bin, move the palm this far off the
    # object before judging, so one resting on the open hand falls off
    nav: nav.NavParams = field(default_factory=nav.NavParams)
    # the bin's opening is much larger than the grasp's tolerance
    nav_bin: nav.NavParams = field(
        default_factory=lambda: nav.NavParams(tol_pos=0.05, tol_yaw=0.08)
    )


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
            ArmIK(
                export_dir / "newton_model.xml",
                export_dir / "policy_io.json",
                "right",
                w_rot=self.p.ik_w_rot,
            )
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
        self.nominal_height = float(high_level.body.nominal[0])
        self.left_pitch_slot = high_level.index(["left_shoulder_pitch_joint"])[0]
        self.left_pitch_default = float(high_level.upper.default[self.left_pitch_slot])
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
        self.tries = np.zeros(n, dtype=int)
        self.paused = np.zeros(n, dtype=int)  # steps the hand's path has been held in this phase
        self.left_back = np.zeros(n)  # the left arm's current counterbalance (rad)
        self.notes: list[dict] = [{} for _ in range(n)]  # measurements for failure analysis
        # the last 2 s of each episode, saved in its notes when it fails: phase, steps in it,
        # object in the heading frame from the palm (cm), object moved (cm), waist pitch,
        # pelvis pitch (rad), palm target minus palm in the heading frame (cm), and the IK's own
        # miss (cm)
        self.trace = [collections.deque(maxlen=100) for _ in range(n)]
        self.ik_err = np.zeros(n)
        self.waist_pitch_id = names.index("waist_pitch_joint")
        # a smoothed view of the body for the arm's IK: chasing every sway of the torso with
        # the arm excites the sway further (the arm's reaction moves the waist)
        self.body_pos = self.robot.data.root_pos_w.torch.clone()
        self.body_quat = self.robot.data.root_quat_w.torch.clone()
        self.body_q = self.robot.data.joint_pos.torch.clone()

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
        # tensors replaced during an inference-mode rollout can't be changed in place outside it
        for name in ("q_corr", "goal", "obj_start", "body_pos", "body_quat", "body_q"):
            setattr(self, name, getattr(self, name).clone())
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
            self.tries[i] = 0
            self.paused[i] = 0
            self.left_back[i] = 0.0
            self.notes[i] = {}
            self.trace[i].clear()
            self.target_k[i] = L.OBJECT_NAMES.index(lay.target_object)
            obj = torch.tensor(lay.object_pos(lay.target_object)[:2], device=self.dev)[None]
            self.goal[i] = nav.stance_for(obj, self.p.stance_rel, torch.zeros(1, device=self.dev))[
                0
            ]
            self.obj_start[i] = torch.tensor(lay.object_pos(lay.target_object), device=self.dev)
        self.q_corr[env_ids] = 0.0
        self._snap_body(env_ids)
        self.hl.reset(env_ids)

    def _snap_body(self, env_ids) -> None:
        self.body_pos[env_ids] = self.robot.data.root_pos_w.torch[env_ids]
        self.body_quat[env_ids] = self.robot.data.root_quat_w.torch[env_ids]
        self.body_q[env_ids] = self.robot.data.joint_pos.torch[env_ids]

    def _smooth_body(self) -> None:
        a = self.p.body_smooth
        self.body_pos += a * (self.robot.data.root_pos_w.torch - self.body_pos)
        q = self.robot.data.root_quat_w.torch
        q = torch.where((q * self.body_quat).sum(-1, keepdim=True) < 0, -q, q)
        bq = self.body_quat + a * (q - self.body_quat)
        self.body_quat = bq / bq.norm(dim=-1, keepdim=True)
        self.body_q += a * (self.robot.data.joint_pos.torch - self.body_q)

    # -- arm planning (CPU, one environment) ----------------------------------------------------
    def _ik_state(self, i: int, waist_pitch: float | None = None) -> None:
        pos = (self.body_pos[i] - self.base.scene.env_origins[i]).tolist()
        quat = self.body_quat[i].tolist()
        q = self.body_q[i].tolist()
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
        out = grasp - p.slide_in * rot[:, 1]  # 2.5 cm further from the object, then slide in
        up = np.array([0.0, 0.0, 1.0])
        pre = out + p.pre_up * up
        pelvis_pos = self._env_pos(self.robot.data.root_pos_w.torch)[i].cpu().numpy()
        self.notes[i]["pelvis_z_at_plan"] = round(float(pelvis_pos[2]), 3)
        pelvis = pelvis_pos[:2]
        grasp_rel = rot[:2, :2].T @ (grasp[:2] - pelvis)  # the grasp in the robot's frame
        start = pre.copy()  # in front of the chest, on the grasp's line, at the descent's height
        start[:2] = pelvis + rot[:2, :2] @ np.array([p.raise_x, grasp_rel[1]])
        self._ik_state(i)  # the IK's model at the measured body against the measured palm
        q_now = self.robot.data.joint_pos.torch[i, self.arm_ids].cpu().numpy()
        model_palm, _ = ik.palm_pose(q_now)
        miss = self._palm_pose()[0][i].cpu().numpy() - model_palm
        self.notes[i]["palm_model_err_cm"] = np.round(miss * 100, 1).tolist()
        self._ik_state(i, waist_pitch=p.lean)
        g = ik.solve(grasp, rot, self.q_hang, iters=400)
        if g.pos_err > 0.01 or g.rot_err > 0.05:
            return False
        q = g.q
        # the planned arm angles at each waypoint: each step's IK is pulled toward them (rest
        # posture), so the arm's spare freedom doesn't drift onto a joint limit on the way
        q_plan = {"grasp": g.q.copy()}
        for name, a, b in (("out", grasp, out), ("pre", out, pre), ("start", pre, start)):
            for s in np.linspace(0.0, 1.0, max(2, int(np.linalg.norm(b - a) / 0.005)))[1:]:
                q = ik.solve(a + s * (b - a), rot, q, iters=100).q
            q_plan[name] = q.copy()
        q_plan["lift"] = ik.solve(grasp + np.array([0.0, 0.0, p.lift_up]), rot, g.q, iters=200).q
        q_now = self.robot.data.joint_pos.torch[i, self.arm_ids].cpu().numpy()
        self.plan[i] = {"rot": rot, "grasp": grasp, "out": out, "pre": pre, "start": start,
                        "q_start": q, "q_plan": q_plan,
                        "q_from": q_now}  # fmt: skip
        return True

    def _plan_release(self, i: int) -> bool:
        p, ik = self.p, self.iks[i]
        lay = self.base.pick_place_layouts[i]
        yaw = float(self._base_pose()[i, 2])
        rot = _rz(yaw)
        bx, by = lay.bin_xy(lay.target_bin)
        h = L.OBJECTS[lay.target_object]["height"]
        # release anywhere inside the bin's opening: the point nearest where the hand reaches best
        # (the object's place relative to the robot at its stance), so a stance a few cm off
        # still reaches
        pelvis = self._env_pos(self.robot.data.root_pos_w.torch)[i, :2].cpu().numpy()
        ideal = pelvis + rot[:2, :2] @ np.array(p.stance_rel)
        r = L.OBJECTS[lay.target_object]["radius"]
        half = np.array([L.BIN_SIZE[0] / 2 - L.BIN_WALL[0] - r - 0.015,
                         L.BIN_SIZE[1] / 2 - L.BIN_WALL[0] - r - 0.015])  # fmt: skip
        bx, by = np.array([bx, by]) + np.clip(ideal - np.array([bx, by]), -half, half)
        obj_target = np.array(
            [bx, by, L.TABLE_TOP + 0.01 + L.BIN_WALL[1] + p.release_above + h / 2]
        )
        palm_target = obj_target - rot @ self.obj_in_palm[i]
        palm_now = self._palm_pose()[0][i].cpu().numpy()
        self._ik_state(i, waist_pitch=p.lean)
        seeds = [self.q_ik[i], self.plan[i].get("q_hold", self.q_ik[i]), self.q_hang]
        r = min((ik.solve(palm_target, rot, q0, iters=300) for q0 in seeds),
                key=lambda res: res.pos_err + 0.1 * res.rot_err)  # fmt: skip
        self.notes[i]["release_ik_error_cm"] = round(r.pos_err * 100, 1)
        self.notes[i]["release_target"] = np.round(palm_target, 3).tolist()
        self.notes[i]["pelvis_at_release_plan"] = np.round(pelvis, 3).tolist()
        if r.pos_err > 0.02:
            return False
        q_from = self.robot.data.joint_pos.torch[i, self.arm_ids].cpu().numpy()
        self.plan[i].update(
            {"rot": rot, "from": palm_now, "to": palm_target, "q_rel": r.q, "q_from": q_from}
        )
        return True

    def _ik_follow(
        self, i: int, target: np.ndarray, rot: np.ndarray, q_rest: np.ndarray | None = None
    ) -> np.ndarray:
        """One step of the arm along its path; a solution that misses the target by more than
        3 cm is not taken (the arm keeps its last angles instead of running to its limits)."""
        self._ik_state(i)
        r = self.iks[i].solve(target, rot, self.q_ik[i], iters=40, q_rest=q_rest)
        self.ik_err[i] = r.pos_err
        if r.pos_err > 0.01:  # which arm joints held it back (failure analysis)
            ik = self.iks[i]
            stuck = self.notes[i].setdefault("ik_limits", {})
            ph = PHASES[self.phase[i]]
            for j in np.where((r.q - ik.lower < 0.02) | (ik.upper - r.q < 0.02))[0]:
                side = "low" if r.q[j] - ik.lower[j] < 0.02 else "high"
                key = f"{ph}:{RIGHT_ARM[j][6:-6]}:{side}"
                stuck[key] = stuck.get(key, 0) + 1
        if r.pos_err < 0.03:
            self.q_ik[i] = r.q
        else:
            self.notes[i]["ik_misses"] = self.notes[i].get("ik_misses", 0) + 1
        return self.q_ik[i]

    # -- the step -------------------------------------------------------------------------------
    def _go(self, i: int, phase: str) -> None:
        self.phase[i] = P[phase]
        self.t[i] = 0
        self.paused[i] = 0

    def _fail(self, i: int, label: str) -> None:
        self.label[i] = label
        self.failed_in[i] = PHASES[self.phase[i]]
        self.notes[i]["trace"] = list(self.trace[i])
        self._go(i, "failed")

    def step(self) -> None:
        """Write this control step's commands for every environment."""
        p, hl = self.p, self.hl
        dt = self.dt
        self._smooth_body()
        pose = self._base_pose()
        z = self.robot.data.root_pos_w.torch[:, 2] - self.base.scene.env_origins[:, 2]
        tilt = self.robot.data.projected_gravity_b.torch[:, 2]
        g3 = self.robot.data.projected_gravity_b.torch
        back = torch.atan2(-g3[:, 0], -g3[:, 2]).cpu().numpy()  # pelvis tipped back (rad)
        fell = ((z < 0.45) | (tilt > -0.5)).cpu().numpy()
        obj = self._target_obj_pos()
        palm_pos, palm_quat = self._palm_pose()
        obj_palm = (obj - palm_pos).norm(dim=-1).cpu().numpy()
        moved = (obj - self.obj_start).norm(dim=-1).cpu().numpy()
        cmd_obj, arr_obj = nav.command(pose, self.goal, p.nav)
        cmd_bin, arr_bin = nav.command(pose, self.goal, p.nav_bin)
        at_bin = torch.as_tensor(self.phase >= P["walk_bin"], device=self.dev)
        nav_cmd = torch.where(at_bin[:, None], cmd_bin, cmd_obj)
        arrived = torch.where(at_bin, arr_bin, arr_obj).cpu().numpy()
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
            if self.phase[i] < P["close"] and moved[i] > p.knock_dist:
                self._fail(i, "knocked")
                continue
            if P["lift"] < self.phase[i] < P["release"] and obj_palm[i] > 0.12:
                self._fail(i, "dropped")
                continue
            arm = None
            if ph in ("walk_obj", "walk_bin"):
                hl.velocity[i] = nav_cmd[i]
                if arrived[i]:
                    self._go(i, "lean_obj" if ph == "walk_obj" else "lean_bin")
                elif k * dt > p.walk_timeout_s:
                    self._fail(i, "timed_out")
            elif ph in ("lean_obj", "lean_bin"):
                hl.body_goal[i, 3] = p.lean
                if ph == "lean_obj" and p.crouch is not None:
                    hl.body_goal[i, 0] = p.crouch
                if k * dt >= p.lean_s:
                    self._go(i, "align_obj" if ph == "lean_obj" else "align_bin")
            elif ph in ("align_obj", "align_bin"):
                hl.body_goal[i, 3] = p.lean
                hl.velocity[i] = nav_cmd[i]
                if arrived[i] and k * dt >= 0.2:
                    self._go(i, "settle_obj" if ph == "align_obj" else "settle_bin")
                elif k * dt > p.align_timeout_s:
                    self._fail(i, "timed_out")
            elif ph in ("settle_obj", "settle_bin"):
                hl.body_goal[i, 3] = p.lean
                if k * dt >= p.settle_s:
                    if not arrived[i] and self.tries[i] < p.align_tries:  # drifted: step back on
                        self.tries[i] += 1
                        self._go(i, "align_obj" if ph == "settle_obj" else "align_bin")
                    elif ph == "settle_obj":
                        if self._plan_grasp(i):
                            self.q_corr[i] = 0.0
                            self._go(i, "raise")
                        else:
                            self._fail(i, "no_plan")  # out of reach: the stance was off
                    elif self._plan_release(i):
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
            elif ph in ("reach", "lower", "slide", "lift"):
                pl = self.plan[i]
                if ph == "slide":  # aim at the object where it is now (the slide may nudge it)
                    self._update_grasp(i, obj[i])
                a, b, dur = {
                    "reach": (pl["start"], pl["pre"], p.reach_s),
                    "lower": (pl["pre"], pl["out"], p.lower_s),
                    "slide": (pl["out"], pl["grasp"], p.slide_s),
                    "lift": (pl["grasp"], pl["grasp"] + [0.0, 0.0, p.lift_up], p.lift_s),
                }[ph]
                ra, rb = {"reach": ("start", "pre"), "lower": ("pre", "out"),
                          "slide": ("out", "grasp"), "lift": ("grasp", "lift")}[ph]  # fmt: skip
                held = self.paused[i] * dt < p.tilt_pause_max_s
                if p.tilt_pause > 0 and ph != "lift" and back[i] > p.tilt_pause and held:
                    self.t[i] -= 1  # hold the path here until the pelvis comes back
                    k = self.t[i]
                    self.paused[i] += 1
                    note = self.notes[i]
                    note["paused_s"] = round(note.get("paused_s", 0.0) + dt, 2)
                s_ = _smooth(k * dt / dur)
                target = a + s_ * (b - a)
                q_rest = pl["q_plan"][ra] + s_ * (pl["q_plan"][rb] - pl["q_plan"][ra])
                arm = self._ik_follow(i, target, pl["rot"], q_rest)
                targets[i] = (target, pl["rot"])
                if ph == "lift":
                    self.hand[i] = HAND_CLOSED
                if k * dt >= dur:
                    if ph == "lift":
                        lifted = float(obj[i, 2] - self.obj_start[i, 2])
                        if lifted < 0.05 or obj_palm[i] > 0.10:  # lift is 10 cm, less sag
                            self._fail(i, "missed_grasp")
                            continue
                        rot = pl["rot"]
                        self.obj_in_palm[i] = rot.T @ (obj[i] - palm_pos[i]).cpu().numpy()
                        # walk with the arm's joints held at the lift pose: straightening
                        # raises the object, leaning at the bin brings it back to this height
                        self.plan[i]["q_hold"] = self.q_ik[i].copy()
                        self._go(i, "straighten")
                    else:
                        nxt = {"reach": "fine_pre", "lower": "fine_out", "slide": "fine_grasp"}
                        self._go(i, nxt[ph])
            elif ph in ("fine_pre", "fine_out", "fine_grasp"):  # wait for the hand to arrive
                pl = self.plan[i]
                if ph == "fine_grasp":
                    self._update_grasp(i, obj[i])
                where = {"fine_pre": "pre", "fine_out": "out", "fine_grasp": "grasp"}[ph]
                target = pl[where]
                arm = self._ik_follow(i, target, pl["rot"], pl["q_plan"][where])
                targets[i] = (target, pl["rot"])
                err = float(np.linalg.norm(palm_pos[i].cpu().numpy() - target))
                tol = {"fine_pre": p.fine_pre_tol, "fine_out": p.fine_out_tol,
                       "fine_grasp": p.fine_grasp_tol}[ph]  # fmt: skip
                in_hand = False
                if ph == "fine_grasp" and p.grab_tol > 0:
                    rel = pl["rot"].T @ (obj[i] - palm_pos[i]).cpu().numpy()
                    lay = self.base.pick_place_layouts[i]
                    want = np.array(p.grasp_in_palm)
                    want[2] += L.OBJECTS[lay.target_object]["height"] / 2 - p.grasp_height
                    off = rel - want
                    in_hand = (abs(off[0]) < p.grab_tol and abs(off[2]) < p.grab_tol
                               and off[1] < p.grab_tol / 2)  # fmt: skip
                if err < tol or in_hand or k * dt >= p.fine_max_s:
                    if ph == "fine_grasp":  # where the object sits in the hand as it closes
                        rel = pl["rot"].T @ (obj[i] - palm_pos[i]).cpu().numpy()
                        e3 = palm_pos[i].cpu().numpy() - pl["grasp"]
                        note = self.notes[i]
                        note["obj_in_palm_at_close_cm"] = np.round(rel * 100, 1).tolist()
                        note["palm_error_at_close_cm"] = np.round(e3 * 100, 1).tolist()
                        note["fine_grasp_s"] = round(k * dt, 2)
                        cmd = hl.arm_hand[i, self.arm_slots].cpu().numpy()
                        got = self.robot.data.joint_pos.torch[i, self.arm_ids].cpu().numpy()
                        note["arm_target_minus_angle_rad"] = np.round(cmd - got, 3).tolist()
                    self._go(
                        i, {"fine_pre": "lower", "fine_out": "slide", "fine_grasp": "close"}[ph]
                    )
            elif ph == "close":
                pl = self.plan[i]
                arm = self._ik_follow(i, pl["grasp"], pl["rot"], pl["q_plan"]["grasp"])
                targets[i] = (pl["grasp"], pl["rot"])
                s = min(1.0, k * dt / p.close_s)  # fingers and thumb together
                self.hand[i] = HAND_OPEN + s * (HAND_CLOSED - HAND_OPEN)
                if k * dt >= p.close_s:
                    self._go(i, "lift")
            elif ph == "straighten":
                arm = self.plan[i]["q_hold"]
                hl.body_goal[i, 3] = 0.0
                hl.body_goal[i, 0] = self.nominal_height
                if k * dt >= p.straighten_s:
                    lay = self.base.pick_place_layouts[i]
                    bxy = torch.tensor(lay.bin_xy(lay.target_bin), device=self.dev)[None]
                    zero = torch.zeros(1, device=self.dev)
                    self.goal[i] = nav.stance_for(bxy, p.stance_rel, zero)[0]
                    self._go(i, "walk_bin")
            elif ph == "reach_bin":  # in joint space, to the release plan's own solution
                pl = self.plan[i]
                s_ = _smooth(k * dt / p.reach_bin_s)
                arm = pl["q_from"] + s_ * (pl["q_rel"] - pl["q_from"])
                self.q_ik[i] = arm.copy()
                hl.body_goal[i, 3] = p.lean
                if k * dt >= p.reach_bin_s:
                    self._go(i, "release")
            elif ph == "release":
                pl = self.plan[i]
                away = _smooth(k * dt / 0.5 - 1.0) * p.release_away  # after the hand is open
                target = pl["to"] - away * pl["rot"][:, 1]
                arm = self._ik_follow(i, target, pl["rot"], pl["q_rel"])
                pl["release_end"] = target
                s = min(1.0, k * dt / 0.5)
                self.hand[i] = HAND_CLOSED + s * (HAND_OPEN - HAND_CLOSED)
                if k * dt >= p.release_s:
                    self._judge_place(i, obj[i].cpu().numpy())
                    self._go(i, "retract")
            elif ph == "retract":  # rise clear of the bin, then back to the hanging pose
                pl = self.plan[i]
                rise = 0.6
                if k * dt < rise:
                    start = pl.get("release_end", pl["to"])
                    target = start + _smooth(k * dt / rise) * np.array([0.0, 0.0, 0.10])
                    arm = self._ik_follow(i, target, pl["rot"], pl["q_rel"])
                    targets[i] = (target, pl["rot"])
                    pl["q_from"] = self.q_ik[i].copy()
                else:
                    s = _smooth((k * dt - rise) / p.retract_s)
                    arm = pl["q_from"] + s * (self.q_hang - pl["q_from"])
                    hl.body_goal[i, 3] = 0.0
                    if k * dt >= rise + p.retract_s:
                        self._go(i, "done")
            if ph in ("walk_bin", "lean_bin", "align_bin", "settle_bin"):
                arm = self.plan[i]["q_hold"]  # the carry pose, joint angles held
            if arm is not None:
                arm_cmd[i] = arm
            hl.arm_hand[i, self.hand_slots] = torch.tensor(
                self.hand[i], device=self.dev, dtype=torch.float32
            )

        if p.tilt_gain > 0:  # lean further as the pelvis tips back, at the table
            g = self.robot.data.projected_gravity_b.torch
            back = torch.atan2(-g[:, 0], -g[:, 2]).cpu().numpy()  # pelvis tipped back (rad)
            for i in range(self.n):
                if P["raise"] <= self.phase[i] <= P["lift"]:
                    extra = p.tilt_gain * min(max(float(back[i]), 0.0), 0.15)
                    hl.body_goal[i, 3] = min(p.lean + extra, 0.45)
        if p.counter_arm:  # the left arm swung back while leaning, against the reaching right arm
            for i in range(self.n):
                ph = self.phase[i]
                lean = P["lean_obj"] <= ph <= P["lift"] or P["lean_bin"] <= ph <= P["release"]
                want = p.counter_arm if lean else 0.0
                rate = p.counter_rate * dt
                self.left_back[i] += min(max(want - self.left_back[i], -rate), rate)
                hl.arm_hand[i, self.left_pitch_slot] = min(
                    self.left_pitch_default + float(self.left_back[i]), 1.0
                )  # the trained range ends at 1.0
        self._record(pose, obj, palm_pos, moved, targets)
        # palm height above the table at each bin phase's end (failure analysis)
        for i in range(self.n):
            ph = PHASES[self.phase[i]]
            if ph in ("straighten", "walk_bin", "lean_bin", "align_bin", "settle_bin",
                      "reach_bin", "release"):  # fmt: skip
                trace = self.notes[i].setdefault("palm_z_cm", {})
                trace[ph] = round(float(palm_pos[i, 2] - L.TABLE_TOP) * 100, 1)
        self._sag_correction(targets, palm_pos, palm_quat)
        for i, arm in arm_cmd.items():
            # hold the arm against gravity at its target pose. Not with the object's weight added:
            # that lifted the hand 10-20 cm above its targets (cause not found), while without it
            # the held object sags only about 2 cm
            self._ik_state(i)
            ff = self.iks[i].gravity_offset(arm)
            hl.arm_hand[i, self.arm_slots] = (
                torch.tensor(arm + ff, device=self.dev, dtype=torch.float32) + self.q_corr[i]
            )

    def _record(self, pose, obj, palm_pos, moved, targets) -> None:
        c, s = torch.cos(pose[:, 2]), torch.sin(pose[:, 2])
        d = obj - palm_pos
        rel = torch.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1], d[:, 2]], -1)
        rel = (rel * 100).cpu().numpy()
        palm = palm_pos.cpu().numpy()
        yaw = pose[:, 2].cpu().numpy()
        waist = self.robot.data.joint_pos.torch[:, self.waist_pitch_id].cpu().numpy()
        g = self.robot.data.projected_gravity_b.torch[:, :3].cpu().numpy()
        pitch = np.arctan2(-g[:, 0], -g[:, 2])
        for i in range(self.n):
            if PHASES[self.phase[i]] in ("done", "failed"):
                continue
            if P["raise"] <= self.phase[i] <= P["lift"]:  # the most the pelvis tipped back
                note = self.notes[i]  # while the arm worked at the table (every episode)
                tilt = max(note.get("back_tilt_max", -1.0), float(pitch[i]))
                note["back_tilt_max"] = round(tilt, 3)  # a Python float: the notes are JSON
            row = [int(self.phase[i]), int(self.t[i]), *np.round(rel[i], 1).tolist()]
            row += [round(float(moved[i]) * 100, 1), round(float(waist[i]), 3)]
            row.append(round(float(pitch[i]), 3))
            if i in targets:  # palm error in the heading frame, and its extremes per phase
                e = _rz(float(yaw[i])).T @ (targets[i][0] - palm[i]) * 100
                row += [*np.round(e, 1).tolist(), round(float(self.ik_err[i]) * 100, 1)]
                ph = PHASES[self.phase[i]]
                ext = (
                    self.notes[i]
                    .setdefault("palm_err_cm", {})
                    .setdefault(ph, [[e[0], e[0]], [e[1], e[1]], [e[2], e[2]]])
                )
                for a in range(3):
                    ext[a] = [round(min(ext[a][0], e[a]), 1), round(max(ext[a][1], e[a]), 1)]
            self.trace[i].append(row)

    def _update_grasp(self, i: int, obj: torch.Tensor) -> None:
        """Move the grasp point with the object (privileged), keeping its planned height."""
        pl = self.plan[i]
        o = obj.cpu().numpy()
        center = np.array([o[0], o[1], L.TABLE_TOP + self.p.grasp_height])
        pl["grasp"] = center - pl["rot"] @ np.array(self.p.grasp_in_palm)

    def _start_hold(self, i: int, palm: torch.Tensor, pose: torch.Tensor) -> None:
        """Remember the palm's position in the robot's heading frame and its height."""
        yaw = float(pose[2])
        rel = _rz(yaw)[:2, :2].T @ (palm[:2].cpu().numpy() - pose[:2].cpu().numpy())
        self.plan[i]["hold"] = (rel, float(palm[2]))
        self.q_corr[i] = 0.0

    def _hold_target(self, i: int, pose: torch.Tensor):
        rel, z = self.plan[i]["hold"]
        yaw = float(pose[2])
        rot = _rz(yaw)
        xy = pose[:2].cpu().numpy() + rot[:2, :2] @ rel
        return np.array([xy[0], xy[1], z]), rot

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

        bx, by = lay.bin_xy(lay.target_bin)
        self.notes[i]["obj_at_judge_rel_bin_cm"] = np.round(
            (obj - np.array([bx, by, L.TABLE_TOP])) * 100, 1
        ).tolist()
        self.notes[i]["palm_at_judge_rel_bin_cm"] = np.round(
            (self._palm_pose()[0][i].cpu().numpy() - np.array([bx, by, L.TABLE_TOP])) * 100, 1
        ).tolist()
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
        base = self.robot.num_base_dofs  # 6 for the floating base, 0 for a pinned (fixed) one
        jac = self.robot.data.body_link_jacobian_w.torch[ids, self.palm][
            :, :, [base + j for j in self.arm_ids]
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

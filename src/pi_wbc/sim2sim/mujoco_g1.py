"""The G1 in plain MuJoCo, set up the way Isaac Lab trained the policy.

Everything that must match training comes from `policy_io.json` (written by
`pi_wbc.tasks.policy_io` from the live Isaac Lab environment): joint order, default pose, PD
gains, armature, effort limits, action scale, observation layout, control rate and solver
settings. This module adds the conventions that differ between the two simulators:

- Joint names. Models name joints differently (Newton's export prefixes scene paths), so joints
  are matched by name ending, and each must match exactly once.
- Quaternions. Isaac Lab stores (x, y, z, w); MuJoCo stores (w, x, y, z).
- Base velocity. Isaac Lab's `base_lin_vel` is the pelvis centre-of-mass velocity expressed in
  the pelvis frame. MuJoCo's free-joint linear velocity is the pelvis frame origin's velocity
  in world coordinates, so it is not used directly.
- Time step. Newton splits each physics step into substeps; MuJoCo steps at the substep size.
- PD control. Isaac Lab's implicit actuators become MuJoCo position actuators (kp, kv), with
  the effort limit applied to the joint's total actuator force.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

GRAVITY_DIR = np.array([0.0, 0.0, -1.0])
SUPPORTED_OBS_TERMS = {
    "base_lin_vel",
    "base_ang_vel",
    "projected_gravity",
    "velocity_commands",
    "joint_pos",
    "joint_vel",
    "actions",
    "gait_clock",
    "body_pose_command",
}


def xyzw_to_wxyz(q: Sequence[float]) -> np.ndarray:
    """Isaac Lab quaternion order to MuJoCo's."""
    x, y, z, w = q
    return np.array([w, x, y, z], dtype=float)


def wxyz_to_xyzw(q: Sequence[float]) -> np.ndarray:
    """MuJoCo quaternion order to Isaac Lab's."""
    w, x, y, z = q
    return np.array([x, y, z, w], dtype=float)


def match_name(candidates: Sequence[str], target: str) -> int:
    """Index of the one candidate that is `target` or ends with `_target` or `/target`."""
    hits = [
        i
        for i, c in enumerate(candidates)
        if c == target or c.endswith("_" + target) or c.endswith("/" + target)
    ]
    if len(hits) != 1:
        found = [candidates[i] for i in hits]
        raise KeyError(f"{target!r} matched {len(hits)} names: {found}")
    return hits[0]


@dataclass(frozen=True)
class PolicyIO:
    """The parts of policy_io.json the MuJoCo side uses."""

    joint_names: list[str]  # every joint, in Isaac Lab's order
    action_joint_names: list[str]  # joints the policy drives, in action order
    default_pos: dict[str, float]
    stiffness: dict[str, float]
    damping: dict[str, float]
    armature: dict[str, float]
    friction: dict[str, float]
    effort_limit: dict[str, float]
    action_scale: float
    physics_dt: float
    decimation: int
    num_substeps: int
    obs_terms: list[tuple[str, int]]
    solver: dict[str, Any]
    root_pos: tuple[float, float, float]
    root_quat_wxyz: tuple[float, float, float, float]
    command_ranges: dict[str, tuple[float, float]]
    # joints of each joint observation, in observation order (stage 0: the action's joints)
    obs_joint_names: dict[str, list[str]] = field(default_factory=dict)
    gait: dict[str, float] | None = None  # gait clock settings (stage 1 tasks)
    # PD targets of joints the policy does not drive. Isaac Lab leaves these at 0 unless the
    # task sets them, so exports that don't record them (stage 0, the first gait runs) get 0.
    hold_targets: dict[str, float] = field(default_factory=dict)
    # body-pose command (pelvis height, waist yaw, roll, pitch) and random arm and hand motion
    # (stage 1, step 2): their settings as exported, or None
    body_pose: dict[str, Any] | None = None
    upper_body: dict[str, Any] | None = None
    # leg actions centred on a crouch that follows the commanded height: pelvis heights and
    # each listed joint's offset from the default pose at them; None for plain actions
    height_pose: dict[str, Any] | None = None
    # uniform noise range per observation term as trained ({} for exports that don't record it)
    obs_noise: dict[str, tuple[float, float]] = field(default_factory=dict)
    # frames of observation history the policy sees, oldest first (0: the current step only)
    obs_history: int = 0

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> PolicyIO:
        joints = raw["joints"]
        names = raw["joint_names"]

        def per_joint(key: str) -> dict[str, float]:
            return {n: float(joints[n][key]) for n in names}

        scale = raw["action"]["scale"]
        if not isinstance(scale, int | float):
            raise ValueError("only a single action scale is supported")
        if raw["action"].get("clip") is not None:
            raise ValueError("action clipping is not supported")
        if not raw["action"]["offset_is_default_pos"]:
            raise ValueError("actions must be offsets from the default pose")
        terms = [(t["name"], int(t["size"])) for t in raw["observation"]["terms"]]
        unknown = {n for n, _ in terms} - SUPPORTED_OBS_TERMS
        if unknown:
            raise ValueError(f"unsupported observation terms: {sorted(unknown)}")
        return cls(
            joint_names=list(names),
            action_joint_names=list(raw["action"]["joint_names"]),
            default_pos=per_joint("default_pos"),
            stiffness=per_joint("stiffness"),
            damping=per_joint("damping"),
            armature=per_joint("armature"),
            friction=per_joint("friction"),
            effort_limit=per_joint("effort_limit"),
            action_scale=float(scale),
            physics_dt=float(raw["physics_dt"]),
            decimation=int(raw["decimation"]),
            num_substeps=int(raw["solver"]["num_substeps"]),
            obs_terms=terms,
            solver=dict(raw["solver"]),
            root_pos=tuple(raw["root_init"]["pos"]),
            root_quat_wxyz=tuple(xyzw_to_wxyz(raw["root_init"]["rot_xyzw"])),
            command_ranges={k: tuple(v) for k, v in raw["commands"].items()},
            obs_joint_names={
                k: list(v) for k, v in raw["observation"].get("joint_names", {}).items()
            },
            gait=raw.get("gait"),
            hold_targets={k: float(v) for k, v in (raw.get("hold_targets") or {}).items()},
            body_pose=raw.get("body_pose"),
            upper_body=raw.get("upper_body"),
            height_pose=raw["action"].get("height_pose"),
            obs_noise={
                k: (float(v[0]), float(v[1]))
                for k, v in (raw["observation"].get("noise") or {}).items()
            },
            obs_history=int(raw["observation"].get("history_length") or 0),
        )

    @classmethod
    def load(cls, path: str | Path) -> PolicyIO:
        return cls.from_dict(json.loads(Path(path).read_text()))

    @property
    def obs_dim(self) -> int:
        return sum(size for _, size in self.obs_terms) * max(self.obs_history, 1)


def build_model(
    xml_path: str | Path,
    io: PolicyIO,
    floor_friction: float = 1.0,
    robot_friction: float | None = None,
    hand_payload_kg: float = 0.0,
) -> mujoco.MjModel:
    """Load an MJCF and give it the training's actuators, joint dynamics and solver settings.

    The model's own actuators and keyframes are removed. Joint damping is set to 0 and joint
    friction, armature and effort limits come from `io`. Planes (the floor) get
    `floor_friction`: Newton, like MuJoCo, uses the larger of two touching surfaces' friction,
    and Isaac Lab's ground is 1.0. For a slippery floor, set `robot_friction` too (every other
    geom), or the robot's own friction decides. `hand_payload_kg` is added to each palm (in
    Menagerie's model, the wrist link the palm belongs to).
    """
    spec = mujoco.MjSpec.from_file(str(xml_path))
    for actuator in list(spec.actuators):
        spec.delete(actuator)
    for key in list(spec.keys):
        spec.delete(key)

    opt = spec.option
    opt.timestep = io.physics_dt / io.num_substeps
    opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    opt.iterations = int(io.solver["iterations"])
    opt.ls_iterations = int(io.solver["ls_iterations"])
    opt.impratio = float(io.solver["impratio"])
    opt.tolerance = float(io.solver["tolerance"])

    joint_names = [j.name for j in spec.joints]
    for name in io.joint_names:
        joint = spec.joints[match_name(joint_names, name)]
        joint.armature = io.armature[name]
        joint.damping = [0.0, 0.0, 0.0]  # a 3-element array in MuJoCo 3.12
        joint.frictionloss = io.friction[name]
        limit = io.effort_limit[name]
        joint.actfrclimited = mujoco.mjtLimited.mjLIMITED_TRUE
        joint.actfrcrange = [-limit, limit]
        actuator = spec.add_actuator(
            name=f"pd_{name}", target=joint.name, trntype=mujoco.mjtTrn.mjTRN_JOINT
        )
        actuator.set_to_position(kp=io.stiffness[name], kv=io.damping[name])

    for geom in spec.geoms:
        if geom.type == mujoco.mjtGeom.mjGEOM_PLANE:
            geom.friction = [floor_friction, geom.friction[1], geom.friction[2]]
        elif robot_friction is not None:
            geom.friction = [robot_friction, geom.friction[1], geom.friction[2]]
    model = spec.compile()
    if hand_payload_kg:
        bodies = [model.body(i).name for i in range(model.nbody)]
        for side in ("left", "right"):
            # Menagerie's model has no separate palm: it is part of the wrist's last link
            try:
                hand = match_name(bodies, f"{side}_hand_palm_link")
            except KeyError:
                hand = match_name(bodies, f"{side}_wrist_yaw_link")
            model.body_mass[hand] += hand_payload_kg
    return model


class G1MujocoSim:
    """Steps a G1 MuJoCo model at the policy rate and builds Isaac Lab-style observations."""

    def __init__(self, model: mujoco.MjModel, io: PolicyIO) -> None:
        self.model, self.io = model, io
        self.data = mujoco.MjData(model)
        m = model

        joint_names = [m.joint(i).name for i in range(m.njnt)]
        joint_ids = [match_name(joint_names, n) for n in io.joint_names]
        self.qpos_adr = np.array([m.jnt_qposadr[j] for j in joint_ids])
        self.qvel_adr = np.array([m.jnt_dofadr[j] for j in joint_ids])
        actuator_names = [m.actuator(i).name for i in range(m.nu)]
        self.actuator_ids = np.array([actuator_names.index(f"pd_{n}") for n in io.joint_names])
        self.policy_ids = np.array([io.joint_names.index(n) for n in io.action_joint_names])
        self.default_pos = np.array([io.default_pos[n] for n in io.joint_names])
        # PD targets with a zero action: the default pose for driven joints, the hold target
        # (as in Isaac Lab) for the rest
        driven = set(io.action_joint_names)
        self.base_target = np.array(
            [
                io.default_pos[n] if n in driven else io.hold_targets.get(n, 0.0)
                for n in io.joint_names
            ]
        )
        self.obs_ids = {
            term: np.array([io.joint_names.index(n) for n in io.obs_joint_names.get(term, [])])
            if io.obs_joint_names.get(term)
            else self.policy_ids
            for term in ("joint_pos", "joint_vel")
        }
        if any(t == "gait_clock" for t, _ in io.obs_terms) and not io.gait:
            raise ValueError("the policy observes a gait clock but policy_io has no gait settings")
        self._base_target0 = self.base_target.copy()
        # body-pose command: what the policy is asked for now (nominal unless a scenario says
        # otherwise), and the waist joints it sets
        self.body_command: np.ndarray | None = None
        self.waist_ids = np.zeros(0, dtype=int)
        if io.body_pose:
            self.body_command = np.array(io.body_pose["nominal"], dtype=float)
            self.waist_ids = np.array(
                [io.joint_names.index(n) for n in io.body_pose["waist_joint_names"]]
            )
        elif any(t == "body_pose_command" for t, _ in io.obs_terms):
            raise ValueError("the policy observes a body-pose command but policy_io has none")
        self.pose_ids = np.zeros(0, dtype=int)
        if io.height_pose:
            if self.body_command is None:
                raise ValueError("crouch-centred actions need a body-pose command")
            offsets = io.height_pose["joint_offsets"]
            self.pose_ids = np.array([io.joint_names.index(n) for n in offsets])
            self.pose_heights = np.asarray(io.height_pose["heights"], dtype=float)
            self.pose_offsets = np.array(list(offsets.values()), dtype=float)
        # joints moved by the arm and hand motion (their PD targets bypass the policy)
        self.upper_ids = np.zeros(0, dtype=int)
        if io.upper_body:
            self.upper_ids = np.array(
                [io.joint_names.index(n) for n in io.upper_body["joint_names"]]
            )
            if set(self.upper_ids) & set(self.policy_ids):
                raise ValueError("arm and hand motion would move joints the policy drives")

        free = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
        if len(free) != 1:
            raise ValueError(f"expected one free joint, found {len(free)}")
        self.root_qpos_adr = int(m.jnt_qposadr[free[0]])
        self.root_qvel_adr = int(m.jnt_dofadr[free[0]])
        self.base_body = int(m.jnt_bodyid[free[0]])
        body_names = [m.body(i).name for i in range(m.nbody)]
        self.torso_body = match_name(body_names, "torso_link")
        self.floor_geoms = {
            g for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE
        }
        if not self.floor_geoms:
            raise ValueError("the model has no floor plane")

        self.steps_per_action = io.decimation * io.num_substeps
        self.policy_dt = io.physics_dt * io.decimation
        self.steps = 0  # policy steps since reset, as Isaac Lab's episode_length_buf
        self.last_action = np.zeros(len(self.policy_ids))
        self._vel6 = np.zeros(6)
        # test conditions (evaluate.py): action delay in physics steps, observation noise
        self.delay_steps = 0
        self.noise: dict[str, tuple[float, float]] = {}
        self.noise_rng = np.random.default_rng(0)
        self._last_target: np.ndarray | None = None
        self._history: list[np.ndarray] = []  # observation frames, oldest first

    # -- state -------------------------------------------------------------------------------

    def reset(
        self,
        joint_offsets: np.ndarray | None = None,
        yaw: float = 0.0,
        root_lin_vel: Sequence[float] = (0.0, 0.0, 0.0),
    ) -> None:
        """Default pose at Isaac Lab's spawn height, optionally perturbed."""
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        a = self.root_qpos_adr
        d.qpos[a : a + 3] = self.io.root_pos
        spawn = np.array(self.io.root_quat_wxyz)
        turn = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
        quat = np.zeros(4)
        mujoco.mju_mulQuat(quat, turn, spawn)
        d.qpos[a + 3 : a + 7] = quat
        q = self.default_pos.copy()
        if joint_offsets is not None:
            q[self.policy_ids] += joint_offsets
        d.qpos[self.qpos_adr] = q
        d.qvel[self.root_qvel_adr : self.root_qvel_adr + 3] = root_lin_vel
        self.base_target = self._base_target0.copy()
        d.ctrl[self.actuator_ids] = self.base_target
        if self.io.body_pose:
            self.body_command = np.array(self.io.body_pose["nominal"], dtype=float)
        self.last_action = np.zeros(len(self.policy_ids))
        self.steps = 0
        self._last_target = None
        self._history = []
        mujoco.mj_forward(m, d)

    def set_upper_body_targets(self, targets: np.ndarray) -> None:
        """PD targets of the arm and hand joints (in `io.upper_body` order), from the next step."""
        self.base_target[self.upper_ids] = targets

    def base_rotation(self) -> np.ndarray:
        """3x3 rotation from the pelvis frame to the world frame."""
        return self.data.xmat[self.base_body].reshape(3, 3).copy()

    def base_com_velocity_world(self) -> tuple[np.ndarray, np.ndarray]:
        """(angular, linear) velocity of the pelvis; linear is at its centre of mass. World axes."""
        # mjOBJ_BODY is the body's inertial frame, so the linear part is the COM velocity.
        mujoco.mj_objectVelocity(
            self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.base_body, self._vel6, 0
        )
        return self._vel6[:3].copy(), self._vel6[3:].copy()

    def observation(self, command: Sequence[float]) -> np.ndarray:
        """The policy observation, in the term order and frames Isaac Lab used in training."""
        d = self.data
        rot = self.base_rotation()
        ang_w, lin_w = self.base_com_velocity_world()
        q = d.qpos[self.qpos_adr]
        qd = d.qvel[self.qvel_adr]
        terms = {
            "base_lin_vel": rot.T @ lin_w,
            "base_ang_vel": rot.T @ ang_w,
            "projected_gravity": rot.T @ GRAVITY_DIR,
            "velocity_commands": np.asarray(command, dtype=float),
            "joint_pos": (q - self.default_pos)[self.obs_ids["joint_pos"]],
            "joint_vel": qd[self.obs_ids["joint_vel"]],
            "actions": self.last_action,
        }
        if self.io.gait:
            terms["gait_clock"] = self.gait_clock(command)
        if self.body_command is not None:
            terms["body_pose_command"] = self.body_command
        parts = []
        for name, size in self.io.obs_terms:
            value = np.asarray(terms[name], dtype=float).ravel()
            if value.size != size:
                raise ValueError(f"{name}: got {value.size} values, policy expects {size}")
            if name in self.noise:
                lo, hi = self.noise[name]
                value = value + self.noise_rng.uniform(lo, hi, value.size)
            parts.append(value)
        frame = np.concatenate(parts).astype(np.float32)
        history = self.io.obs_history
        if not history:
            return frame
        # as Isaac Lab's history buffer: the first frame after a reset fills every slot, then
        # each call shifts in one frame (call once per policy step)
        if not self._history:
            self._history = [frame] * history
        else:
            self._history = [*self._history[1:], frame]
        return np.concatenate(self._history)

    def step(self, action: np.ndarray) -> None:
        """Apply one policy action and advance one policy period."""
        action = np.asarray(action, dtype=float).ravel()
        target = self.base_target.copy()
        target[self.policy_ids] += self.io.action_scale * action
        if self.pose_ids.size:  # as pi_wbc.tasks.mdp.HeightPoseJointPositionAction
            height = self.body_command[0]
            target[self.pose_ids] += [
                np.interp(height, self.pose_heights, row) for row in self.pose_offsets
            ]
        # a delay of k physics steps keeps the previous targets for k x num_substeps MuJoCo steps
        held = self.delay_steps * self.io.num_substeps if self._last_target is not None else 0
        for i in range(self.steps_per_action):
            self.data.ctrl[self.actuator_ids] = self._last_target if i < held else target
            mujoco.mj_step(self.model, self.data)
        self._last_target = target
        self.last_action = action.copy()
        self.steps += 1

    def gait_clock(self, command: Sequence[float] = (0.0, 0.0, 0.0)) -> np.ndarray:
        """sin and cos of the left leg's gait phase, as `pi_wbc.tasks.mdp.gait_clock`.

        Policies trained with `clock_off_when_standing` see (0, 0) while the command says
        "stand" (speed and yaw rate both below the threshold).
        """
        gait = self.io.gait
        period = float(gait["period"])
        if gait.get("clock_off_when_standing"):
            threshold = float(gait["stand_threshold"])
            if np.hypot(command[0], command[1]) < threshold and abs(command[2]) < threshold:
                return np.zeros(2)
        phase = (self.steps * self.policy_dt % period) / period
        return np.array([np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)])

    # -- measurements ------------------------------------------------------------------------

    def base_height(self) -> float:
        return float(self.data.qpos[self.root_qpos_adr + 2])

    def floor_contacts(self) -> Iterator[Any]:
        """Contacts with the floor that carry force.

        Newton's exported model gives every geom a 1 cm `gap`: MuJoCo then also lists pairs up
        to 2 cm apart as inactive contacts (`exclude == 1`) that exert no force. Those are
        skipped here, so "touching" means the same on every model.
        """
        d = self.data
        for c in d.contact[: d.ncon]:
            if c.exclude == 0 and (c.geom1 in self.floor_geoms or c.geom2 in self.floor_geoms):
                yield c

    def torso_touches_floor(self) -> bool:
        """Isaac Lab's fall test: the torso link in contact with the ground."""
        m = self.model
        return any(
            self.torso_body in (m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2])
            for c in self.floor_contacts()
        )

    def velocity_yaw_frame(self) -> tuple[np.ndarray, float]:
        """Planar COM velocity in the heading frame and yaw rate, as the tracking rewards use."""
        ang_w, lin_w = self.base_com_velocity_world()
        rot = self.base_rotation()
        yaw = np.arctan2(rot[1, 0], rot[0, 0])
        c, s = np.cos(yaw), np.sin(yaw)
        vx = c * lin_w[0] + s * lin_w[1]
        vy = -s * lin_w[0] + c * lin_w[1]
        return np.array([vx, vy]), float(ang_w[2])

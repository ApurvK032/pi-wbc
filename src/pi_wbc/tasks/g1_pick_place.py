"""Stage 3: the pick-and-place task (STAGE3_DESIGN.md, sections 3, 4 and 7).

The stage 1 student task, nominal (no randomization, noise, delay or pushes), with a table,
five round objects (three on the table per layout, two parked out of the way) and two bins,
and with every command held by a high level (the scripted expert, later the VLA) instead of
being resampled:

- `base_velocity`: forward and sideways speed and yaw rate, as written;
- `body_pose`: pelvis height and waist yaw, roll, pitch: the high level sets the goal, and the
  command moves to it at training's rate limits;
- `upper_body`: the 28 arm and hand joint targets, applied straight to those joints' PD
  controllers (D-012).

`HighLevel` writes them. Layouts (`pi_wbc.stage3.layout`) are applied on reset by the
`reset_layout` event from `env.pick_place_layouts` (one per environment, set before the reset).
Physics: Newton at 4 substeps (D-034) and the feet's heel spheres at Unitree's URDF position
(D-038). Falls do not end an episode: the runner labels outcomes itself.
"""

from __future__ import annotations

from collections.abc import Sequence

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass

from pi_wbc.stage3 import layout as L
from pi_wbc.tasks.g1_student import G1StudentDistillationRunnerCfg, G1TeacherStudentEnvCfg

HISTORY_STEPS = 25  # the stage 1 controller's history (D-026)
SUBSTEPS = 4  # D-034
ROBOT_START_Z = 0.8  # pelvis height at reset, as in training
PARK = (-6.0, 6.0)  # where unused objects stand (env frame): far behind and to the side

# a bin is five kinematic boxes: a floor and four walls, offsets from the bin's centre on the
# table top. Parts keep 1 mm gaps from each other and from the table: Newton generates contacts
# between touching kinematic bodies too, and they use up the contact budget.
_FT, _WT, _WH = 0.01, L.BIN_WALL[0], L.BIN_WALL[1]
_BX, _BY = L.BIN_SIZE
_GAP = 0.001
_WZ = _GAP + _FT + _GAP + _WH / 2  # wall centres sit on top of the floor
BIN_PARTS = {
    "floor": ((_BX, _BY, _FT), (0.0, 0.0, _GAP + _FT / 2)),
    "wall_xm": ((_WT, _BY, _WH), (-(_BX - _WT) / 2, 0.0, _WZ)),
    "wall_xp": ((_WT, _BY, _WH), ((_BX - _WT) / 2, 0.0, _WZ)),
    "wall_ym": ((_BX - 2 * _WT - 2 * _GAP, _WT, _WH), (0.0, -(_BY - _WT) / 2, _WZ)),
    "wall_yp": ((_BX - 2 * _WT - 2 * _GAP, _WT, _WH), (0.0, (_BY - _WT) / 2, _WZ)),
}


def _kinematic_box(name: str, size, pos, rgb) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=sim_utils.CuboidCfg(
            size=size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.8, dynamic_friction=0.8
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=rgb, roughness=0.6),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
    )


def _object(name: str, k: int) -> RigidObjectCfg:
    spec = L.OBJECTS[name]
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/obj_{name}",
        spawn=sim_utils.CylinderCfg(
            radius=spec["radius"],
            height=spec["height"],
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.3),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.8, dynamic_friction=0.8
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=spec["rgb"], roughness=0.4),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(PARK[0], PARK[1] + 0.3 * k, spec["height"] / 2)
        ),
    )


def _yaw_quat(yaw: torch.Tensor) -> torch.Tensor:
    """(x, y, z, w) quaternions of rotations about z."""
    z = torch.zeros_like(yaw)
    return torch.stack([z, z, torch.sin(yaw / 2), torch.cos(yaw / 2)], dim=-1)


class LayoutState:
    """Poses and masses of one layout per environment, as tensors in env frames."""

    def __init__(self, layouts: Sequence[L.Layout], device):
        n = len(layouts)
        self.layouts = list(layouts)
        self.robot_xy = torch.tensor([lay.robot_xy for lay in layouts], device=device)
        self.robot_yaw = torch.tensor([lay.robot_yaw for lay in layouts], device=device)
        # every object: its place on the table, or parked
        pos = torch.zeros(n, len(L.OBJECT_NAMES), 3, device=device)
        mass = torch.full((n, len(L.OBJECT_NAMES)), 0.3, device=device)
        for i, lay in enumerate(layouts):
            for k, name in enumerate(L.OBJECT_NAMES):
                h = L.OBJECTS[name]["height"]
                if name in lay.objects:
                    pos[i, k] = torch.tensor(lay.object_pos(name))
                    mass[i, k] = lay.object_mass[lay.objects.index(name)]
                else:
                    pos[i, k] = torch.tensor((PARK[0], PARK[1] + 0.3 * k, h / 2))
        self.object_pos, self.object_mass = pos, mass
        self.bin_xy = torch.tensor(
            [[lay.bin_xy(b) for b in L.BIN_NAMES] for lay in layouts], device=device
        )


def reset_layout(env, env_ids: torch.Tensor | None) -> None:
    """Place robot, objects and bins of the environments `env_ids` as their layouts say."""
    layouts = getattr(env, "pick_place_layouts", None)
    if layouts is None:  # e.g. the first reset when the environment is created
        layouts = [L.sample(0)] * env.num_envs
        env.pick_place_layouts = layouts
    state = LayoutState(layouts, env.device)
    env.pick_place_state = state
    if env_ids is None or isinstance(env_ids, slice):  # a full reset passes slice(None)
        ids = torch.arange(env.num_envs, device=env.device)
    else:
        ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    origins = env.scene.env_origins[ids]
    n = len(ids)
    zeros6 = torch.zeros(n, 6, device=env.device)

    robot = env.scene["robot"]
    pose = torch.zeros(n, 7, device=env.device)
    pose[:, :2] = state.robot_xy[ids] + origins[:, :2]
    pose[:, 2] = ROBOT_START_Z + origins[:, 2]
    pose[:, 3:] = _yaw_quat(state.robot_yaw[ids])
    robot.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
    robot.write_root_velocity_to_sim_index(root_velocity=zeros6, env_ids=ids)

    unit = torch.tensor([0.0, 0.0, 0.0, 1.0], device=env.device)
    for k, name in enumerate(L.OBJECT_NAMES):
        obj = env.scene[f"obj_{name}"]
        p = torch.cat([state.object_pos[ids, k] + origins, unit.repeat(n, 1)], dim=-1)
        obj.write_root_pose_to_sim_index(root_pose=p, env_ids=ids)
        obj.write_root_velocity_to_sim_index(root_velocity=zeros6, env_ids=ids)
        base_mass = getattr(env, "_pick_place_base_mass", {})
        if name not in base_mass:
            base_mass[name] = (obj.data.body_mass.torch[0].clone(),
                               obj.data.body_inertia.torch[0].clone())  # fmt: skip
            env._pick_place_base_mass = base_mass
        m0, i0 = base_mass[name]
        m = state.object_mass[ids, k][:, None]
        obj.set_masses_index(masses=m, env_ids=ids)
        obj.set_inertias_index(inertias=i0[None] * (m / m0)[:, :, None], env_ids=ids)

    for b, bin_name in enumerate(L.BIN_NAMES):
        center = torch.zeros(n, 3, device=env.device)
        center[:, :2] = state.bin_xy[ids, b]
        center[:, 2] = L.TABLE_TOP
        for part, (_, offset) in BIN_PARTS.items():
            asset = env.scene[f"{bin_name}_{part}"]
            p = torch.cat(
                [center + origins + torch.tensor(offset, device=env.device), unit.repeat(n, 1)],
                dim=-1,
            )
            asset.write_root_pose_to_sim_index(root_pose=p, env_ids=ids)


def urdf_heel(env, env_ids) -> None:
    """Startup event: the feet's heel contact spheres where Unitree's URDF has them (D-038)."""
    from pi_wbc.stage3.feet import URDF_HEEL_X, set_heel_x

    set_heel_x(URDF_HEEL_X)


@configclass
class G1PickPlaceEnvCfg(G1TeacherStudentEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.make_nominal()
        self.observations.policy.history_length = HISTORY_STEPS
        self.episode_length_s = 1.0e5  # the runner ends episodes
        self.terminations.base_contact = None  # falls are labelled, not reset
        self.scene.env_spacing = 12.0

        # commands held by the high level (never resampled)
        never = (1.0e6, 1.0e6)
        vel = self.commands.base_velocity
        vel.resampling_time_range = never
        vel.rel_standing_envs = 0.0
        vel.rel_turning_envs = 0.0
        self.commands.body_pose.resampling_time_range = never
        self.commands.upper_body.resampling_time_range = never
        self.commands.upper_body.rel_still_envs = 1.0
        self.curriculum.upper_body_level = None

        # resets place the layout; the robot's joints start at the default pose
        self.events.reset_base = EventTerm(func=reset_layout, mode="reset")
        self.events.base_external_force_torque = None
        self.events.urdf_heel = EventTerm(func=urdf_heel, mode="startup")

        # scene: table, objects, bins
        sc = self.scene
        sc.table = _kinematic_box(
            "Table", L.TABLE_SIZE, (L.TABLE_CENTER[0], L.TABLE_CENTER[1], L.TABLE_TOP / 2),
            (0.55, 0.38, 0.22),
        )  # fmt: skip
        for k, name in enumerate(L.OBJECT_NAMES):
            setattr(sc, f"obj_{name}", _object(name, k))
        for b, bin_name in enumerate(L.BIN_NAMES):
            rgb = L.BINS[bin_name]["rgb"]
            for part, (size, offset) in BIN_PARTS.items():
                pos = (L.BIN_X + offset[0], (2 * b - 1) * L.BIN_Y + offset[1],
                       L.TABLE_TOP + offset[2])  # fmt: skip
                setattr(
                    sc, f"{bin_name}_{part}", _kinematic_box(f"{bin_name}_{part}", size, pos, rgb)
                )

        # physics: finer substeps for grasping, room for the extra contacts
        newton = self.sim.physics.newton_mjwarp
        newton.num_substeps = SUBSTEPS
        newton.solver_cfg.njmax = 1200  # per world
        newton.solver_cfg.nconmax = 300


class HighLevel:
    """Writes the high level's commands into the environment, every control step.

    Call `apply()` before `env.step(...)`. Commands (all (num_envs, ...) tensors):
    `velocity` (vx, vy, yaw rate), `body` (pelvis height, waist yaw, roll, pitch) and
    `arm_hand` (28 joint targets in `self.arm_hand_names` order).
    """

    def __init__(self, env):
        base = env.unwrapped
        self.vel = base.command_manager.get_term("base_velocity")
        self.body = base.command_manager.get_term("body_pose")
        self.upper = base.command_manager.get_term("upper_body")
        self.arm_hand_names = list(self.upper.joint_names)
        n, dev = base.num_envs, base.device
        self.velocity = torch.zeros(n, 3, device=dev)
        self.body_goal = self.body.nominal.repeat(n, 1)
        self.arm_hand = self.upper.default.repeat(n, 1)

    def index(self, names: Sequence[str]) -> list[int]:
        return [self.arm_hand_names.index(name) for name in names]

    def reset(self, env_ids) -> None:
        self.velocity[env_ids] = 0.0
        self.body_goal[env_ids] = self.body.nominal
        self.arm_hand[env_ids] = self.upper.default

    def apply(self) -> None:
        self.vel.vel_command_b[:] = self.velocity
        self.vel.is_standing_env[:] = False
        self.vel.is_turning_env[:] = False
        self.body.goal[:] = self.body_goal
        up = self.upper
        up.start[:] = self.arm_hand  # start == goal: the interpolation returns the target
        up.goal[:] = self.arm_hand


@configclass
class G1PickPlaceRunnerCfg(G1StudentDistillationRunnerCfg):
    """Only for loading the stage 1 student's checkpoint; nothing is trained on this task."""

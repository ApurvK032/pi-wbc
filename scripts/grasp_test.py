"""Stage 3 grasp test: can the Dex3 hand pick up and hold the task's objects in Newton?

A G1 with its base fixed (at a pelvis height and waist lean the controller is trained for)
stands at a table. The right hand takes a "handshake" side grasp: palm facing the robot's
midline, fingers forward. It rises in front of the chest, moves above the object, comes straight
down beside it, curls its two fingers round the object's front and then sweeps the thumb in
behind it, lifts the object and holds it. Every robot of the batch gets one variant: an object
mass and a placement error (the object sits off the spot the hand aims for, as when the walk to
the table stops a little off). Success: at the end the object is at least 10 cm above where it
started and within 10 cm of the palm.

    uv run python scripts/grasp_test.py --object block --waist-pitch 0.25 --out logs/grasp/block

How the arm is driven (the scripted expert does the same): the palm's path is planned in
Cartesian space and solved for joint angles on Newton's own model of the robot
(`pi_wbc.stage3.arm_ik`), backwards from the grasp so that the whole path stays on one smooth IK
branch; the joint targets follow that trajectory. A slow per-robot correction on the measured
palm error (damped least squares on Newton's Jacobian, bounded) works off the arm's sag under
gravity, since the PD controllers alone leave the hand 1-2 cm low.
"""

import argparse
import json
import math
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation

DEFAULT_EXPORT = "logs/rsl_rl/pi_wbc_g1_student/slurm-5628865-full-range-long-r/exported"

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--object", default="block", help="key of OBJECTS")
parser.add_argument("--masses", default="0.1,0.3,0.5", help="comma list of object masses (kg)")
parser.add_argument(
    "--offsets",
    default="0,0;0.015,0;-0.015,0;0,0.015;0,-0.015",
    help="semicolon list of x,y placement errors (m) of the object from the aimed spot",
)
parser.add_argument("--friction", type=float, default=0.8, help="object friction coefficient")
parser.add_argument("--table-top", type=float, default=0.75)
parser.add_argument("--pelvis-z", type=float, default=0.78, help="fixed pelvis height (crouch)")
parser.add_argument("--waist-pitch", type=float, default=0.25, help="forward lean of the torso")
parser.add_argument("--aim", default="0.40,-0.18", help="object spot the hand aims for (x,y)")
parser.add_argument(
    "--grasp-offset",
    default="0.07,0.065,0.0",
    help="grasp centre in the palm frame (x fingers, y palm normal, z index side)",
)
parser.add_argument(
    "--grasp-height",
    type=float,
    default=None,
    help="grasp centre above the table top (m); default min(7.5 cm, top - 3 cm)",
)
parser.add_argument(
    "--close",
    default="1.2,1.2,-1.0,-0.9",
    help="closed hand: finger_0, finger_1, thumb_1, thumb_2 (right-hand signs)",
)
parser.add_argument(
    "--export",
    type=Path,
    default=Path(DEFAULT_EXPORT),
    help="controller export with newton_model.xml and policy_io.json",
)
parser.add_argument(
    "--schedule",
    choices=["together", "fingers_first"],
    default="together",
    help="close fingers and thumb together, or fingers first then the thumb",
)
parser.add_argument("--substeps", type=int, default=None, help="Newton substeps (training: 2)")
parser.add_argument(
    "--mujoco-contacts",
    action="store_true",
    help="MuJoCo's collision detection instead of Newton's pipeline (training's)",
)
parser.add_argument(
    "--sweep",
    action="store_true",
    help="one robot per grasp point x closed pose (mass 0.3, no placement error)",
)
parser.add_argument("--debug", action="store_true", help="print robot 0's state per phase")
parser.add_argument("--shapes", action="store_true", help="print Newton's hand and object shapes")
parser.add_argument("--cam", choices=["side", "top"], default="side")
parser.add_argument("--video", action="store_true", help="camera frames of robot 0 -> GIF")
parser.add_argument("--out", type=Path, required=True)
add_launcher_args(parser)
args_cli = parser.parse_args()
# the launcher's physics argument stays unset: it would replace the preset (bench_physics, D-051)

import isaaclab.sim as sim_utils  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg  # noqa: E402
from isaaclab.scene import InteractiveSceneCfg  # noqa: E402
from isaaclab.sensors import CameraCfg, ContactSensorCfg  # noqa: E402
from isaaclab.utils import configclass, instantiate, replace  # noqa: E402
from isaaclab.utils.math import compute_pose_error, quat_from_matrix  # noqa: E402

from pi_wbc import provenance  # noqa: E402
from pi_wbc.stage3 import bench_physics  # noqa: E402
from pi_wbc.stage3.arm_ik import ArmIK, arm_joint_names  # noqa: E402
from pi_wbc.tasks.g1_gait import G1_29DOF_GAIT_CFG as G1_CFG  # noqa: E402  (stage 1's robot)

# Objects sized for a three-finger pinch (about 4.5-5 cm across where the hand closes), tall
# enough for a side grasp: the lower finger reaches 4.5 cm below the palm's centre.
OBJECTS = {
    "block": {"shape": "cuboid", "size": (0.045, 0.045, 0.12), "color": (0.95, 0.8, 0.1)},
    "box": {"shape": "cuboid", "size": (0.05, 0.065, 0.12), "color": (0.85, 0.1, 0.1)},
    "can": {"shape": "cylinder", "radius": 0.026, "height": 0.12, "color": (0.1, 0.6, 0.2)},
    "bottle": {"shape": "cylinder", "radius": 0.022, "height": 0.14, "color": (0.2, 0.4, 0.9)},
    "cube": {"shape": "cuboid", "size": (0.05, 0.05, 0.05), "color": (0.2, 0.3, 0.9)},
}
OBJ = OBJECTS[args_cli.object]
OBJ_H = OBJ["size"][2] if OBJ["shape"] == "cuboid" else OBJ["height"]
TABLE_X, TABLE_DEPTH = 0.55, 0.6  # table centre in front of the robot (front edge at 0.25 m)
PELVIS_Z = args_cli.pelvis_z  # stage 1's nominal walking height is 0.78 m
AIM = tuple(float(v) for v in args_cli.aim.split(","))
MASSES = [float(m) for m in args_cli.masses.split(",")]
OFFSETS = [tuple(float(v) for v in o.split(",")) for o in args_cli.offsets.split(";")]

RIGHT_ARM = arm_joint_names("right")
RIGHT_HAND = [
    "right_hand_index_0_joint", "right_hand_index_1_joint", "right_hand_middle_0_joint",
    "right_hand_middle_1_joint", "right_hand_thumb_0_joint", "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
]  # fmt: skip
f0, f1, t1, t2 = (float(v) for v in args_cli.close.split(","))
HAND_OPEN = [0.0, 0.0, 0.0, 0.0, 0.0, 0.72, 0.0]  # fingers straight, thumb swung back
HAND_CLOSED = [f0, f1, f0, f1, 0.0, t1, t2]
# Closed poses to compare, right hand, in RIGHT_HAND order (index 0-1, middle 0-1, thumb 0-2).
# The "nv_" ones are NVIDIA's binary teleop poses (decoupled_wbc g1_gripper_ik_solver.py,
# mirrored for the right hand); "wrap" curls the thumb's tip in hard.
CLOSED_POSES = {
    "ours": HAND_CLOSED,
    "nv_middle": [1.0, 1.5, 1.0, 1.5, 0.0, -0.7, -0.7],
    "nv_index": [0.6, 1.5, 1.5, 1.5, 0.5, -0.7, -0.7],
    "wrap": [0.8, 1.0, 0.8, 1.0, 0.0, -0.3, -1.5],
}

# Palm frame: x along the fingers, y out of the palm face, z across (index side +z on the right
# hand); the open thumb stands 7-9 cm out of the palm face. Handshake grasp: the palm frame is
# the world frame (palm facing +y, fingers forward, index finger up).
PALM_R = np.eye(3)
GRASP_IN_PALM = tuple(float(v) for v in args_cli.grasp_offset.split(","))
if args_cli.sweep:
    VARIANTS = [
        {"mass": 0.3, "offset": (0.0, 0.0), "grasp": (gx, gy, 0.0), "close": c}
        for gx in (0.06, 0.08, 0.10)
        for gy in (0.05, 0.065, 0.08)
        for c in CLOSED_POSES
    ]
else:
    VARIANTS = [
        {"mass": m, "offset": o, "grasp": GRASP_IN_PALM, "close": "ours"}
        for m in MASSES
        for o in OFFSETS
    ]
NUM_ENVS = len(VARIANTS)
GRASP_HEIGHT = (
    args_cli.grasp_height if args_cli.grasp_height is not None else min(0.075, OBJ_H - 0.03)
)
# The hand starts 15 cm above the grasp (its lower finger then clears a 12 cm object) and
# comes in from RAISE_X, in front of the chest.
PRE_GRASP_UP, LIFT_UP, RAISE_X = 0.15, 0.15, 0.20
MAX_CORR = 0.15  # rad: bound of the sag correction per joint
CORR_GAIN = 0.1  # share of the measured-error IK step added to the correction per step

# Timeline (s). "to_raise" moves in joint space from the hanging arm to the start of the
# planned path (and leans the waist); "reach", "lower" and "lift" follow the planned path.
PHASES = [("settle", 1.0), ("to_raise", 1.5), ("reach", 1.5), ("lower", 1.2), ("close", 1.2),
          ("lift", 1.2), ("hold", 3.0)]  # fmt: skip


def obj_spawn():
    common = {
        "rigid_props": sim_utils.RigidBodyPropertiesCfg(),
        "mass_props": sim_utils.MassPropertiesCfg(mass=0.3),
        "collision_props": sim_utils.CollisionPropertiesCfg(),
        "physics_material": sim_utils.RigidBodyMaterialCfg(
            static_friction=args_cli.friction, dynamic_friction=args_cli.friction
        ),
        "visual_material": sim_utils.PreviewSurfaceCfg(diffuse_color=OBJ["color"], roughness=0.5),
    }
    if OBJ["shape"] == "cuboid":
        return sim_utils.CuboidCfg(size=OBJ["size"], **common)
    return sim_utils.CylinderCfg(radius=OBJ["radius"], height=OBJ["height"], **common)


@configclass
class SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    sun = AssetBaseCfg(
        prim_path="/World/sun",
        spawn=sim_utils.DistantLightCfg(intensity=2500.0, color=(1.0, 0.97, 0.92), angle=0.5),
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.25, 0.25, 0.0, 0.933)),
    )
    sky = AssetBaseCfg(prim_path="/World/sky", spawn=sim_utils.DomeLightCfg(intensity=600.0))
    robot = replace(
        G1_CFG,
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=replace(G1_CFG.spawn, fix_root_link=True),
        init_state=replace(G1_CFG.init_state, pos=(0.0, 0.0, PELVIS_Z), rot=(0.0, 0.0, 0.0, 1.0)),
    )
    table = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(TABLE_DEPTH, 1.0, args_cli.table_top),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=20.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.38, 0.22)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(TABLE_X, 0.0, args_cli.table_top / 2)),
    )
    obj = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=obj_spawn(),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(AIM[0], AIM[1], args_cli.table_top + OBJ_H / 2 + 0.002)
        ),
    )


HAND_CONTACTS = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/right_hand/right_hand_.*_link",
    update_period=0.0,
    filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
)


@configclass
class GraspTestCfg:
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=0.005, render_interval=4, physics=bench_physics.newton()
    )
    scene: SceneCfg = SceneCfg(num_envs=NUM_ENVS, env_spacing=2.5)


def camera():
    from isaaclab_newton.renderers import NewtonWarpRendererCfg

    if args_cli.cam == "side":  # 0.9 m to the robot's right of the object, looking at it
        offset = CameraCfg.OffsetCfg(
            pos=(AIM[0], AIM[1] - 0.9, args_cli.table_top + 0.35),
            rot=(-0.1305, 0.1305, 0.6950, 0.6950),
            convention="world",
        )
    else:  # 0.6 m above the object looking down (image up = world +x)
        offset = CameraCfg.OffsetCfg(
            pos=(AIM[0], AIM[1], args_cli.table_top + 0.6),
            rot=(0.0, 0.7071, 0.0, 0.7071),
            convention="world",
        )
    return CameraCfg(
        prim_path="{ENV_REGEX_NS}/cam",
        update_period=0.0,
        width=480,
        height=360,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, clipping_range=(0.05, 20.0)),
        offset=offset,
        renderer_cfg=NewtonWarpRendererCfg(),
    )


def tt(x):
    return x.torch if hasattr(x, "torch") else x


def smooth(s: float) -> float:
    s = min(max(s, 0.0), 1.0)
    return 0.5 - 0.5 * math.cos(math.pi * s)


def dls_step(jac, pos, quat, target_p, target_q, w_rot=0.3, lam=0.05):
    """Batched joint step towards a palm pose (damped least squares, orientation weighted)."""
    ep, eo = compute_pose_error(pos, quat, target_p, target_q, rot_error_type="axis_angle")
    e = torch.cat([ep, w_rot * eo], dim=-1)[..., None]
    j = torch.cat([jac[:, :3], w_rot * jac[:, 3:]], dim=1)
    eye = torch.eye(6, device=jac.device)
    return (j.transpose(1, 2) @ torch.linalg.solve(j @ j.transpose(1, 2) + lam**2 * eye, e))[..., 0]


def plan(default: dict[str, float], dt: float, grasp_in_palm) -> dict:
    """The arm's joint trajectory per phase (same for every robot: the expert aims at the
    nominal spot, and the placement error tests how much it tolerates)."""
    ik = ArmIK(args_cli.export / "newton_model.xml", args_cli.export / "policy_io.json", "right")
    ik.set_state([0.0, 0.0, PELVIS_Z], [0.0, 0.0, 0.0, 1.0],
                 {**default, "waist_pitch_joint": args_cli.waist_pitch})  # fmt: skip
    q_hang = np.array([default[n] for n in RIGHT_ARM])
    center = np.array([AIM[0], AIM[1], args_cli.table_top + GRASP_HEIGHT])
    grasp = center - PALM_R @ np.asarray(grasp_in_palm)
    up = np.array([0.0, 0.0, 1.0])
    pre = grasp + PRE_GRASP_UP * up
    start = np.array([RAISE_X, pre[1], pre[2]])
    lift = grasp + LIFT_UP * up

    # one IK branch: solve the grasp, then walk back along the path in 5 mm steps
    g = ik.solve(grasp, PALM_R, q_hang, iters=400)
    if g.pos_err > 0.003 or g.rot_err > 0.02:
        raise SystemExit(f"[grasp] grasp pose out of reach: {g.pos_err:.3f} m, {g.rot_err:.3f} rad")

    def chain(q_from, p_from, p_to):
        n = max(2, int(np.linalg.norm(p_to - p_from) / 0.005))
        q, worst = q_from, 0.0
        for s in np.linspace(0.0, 1.0, n)[1:]:
            r = ik.solve(p_from + s * (p_to - p_from), PALM_R, q, iters=100)
            q, worst = r.q, max(worst, r.pos_err)
        return q, worst

    q_pre, e1 = chain(g.q, grasp, pre)
    q_start, e2 = chain(q_pre, pre, start)
    print(f"[grasp] plan: worst IK error on the way back {max(e1, e2) * 100:.2f} cm")

    def follow(q_from, p_from, p_to, duration):
        """Per control step: palm target and joint angles along a straight line."""
        steps = round(duration / dt)
        qs, ps, q = [], [], q_from
        for k in range(steps):
            p = p_from + smooth((k + 1) / steps) * (p_to - p_from)
            q = ik.solve(p, PALM_R, q, iters=40).q
            qs.append(q)
            ps.append(p)
        return np.array(qs), np.array(ps)

    traj = {}
    steps = round(1.5 / dt)
    qs = np.array([q_hang + smooth((k + 1) / steps) * (q_start - q_hang) for k in range(steps)])
    traj["to_raise"] = (qs, None)  # joint space; no palm target
    traj["reach"] = follow(q_start, start, pre, 1.5)
    traj["lower"] = follow(traj["reach"][0][-1], pre, grasp, 1.2)
    traj["lift"] = follow(traj["lower"][0][-1], grasp, lift, 1.2)
    return {"traj": traj, "grasp": grasp, "lift": lift, "q_grasp": traj["lower"][0][-1],
            "q_lift": traj["lift"][0][-1]}  # fmt: skip


def main() -> None:
    cfg = GraspTestCfg()
    if args_cli.video:
        cfg.scene.cam = camera()
    if args_cli.debug:
        cfg.scene.hand_contacts = HAND_CONTACTS
    args_cli.out.mkdir(parents=True, exist_ok=True)
    with launch_simulation(cfg, args_cli) as physics_cfg:
        if args_cli.substeps is not None:
            physics_cfg.num_substeps = args_cli.substeps
        if args_cli.mujoco_contacts:
            physics_cfg.solver_cfg.use_mujoco_contacts = True
        cfg.sim.physics = physics_cfg
        sim = sim_utils.SimulationContext(cfg.sim)
        scene = instantiate(cfg.scene)
        sim.reset()
        dev = sim.device
        dt = sim.get_physics_dt() * cfg.sim.render_interval  # control step, 50 Hz
        physics = bench_physics.record(cfg.sim.physics)
        bench_physics.check(physics, mujoco_contacts=args_cli.mujoco_contacts)
        print(f"[grasp] physics: {physics}")
        robot, obj = scene["robot"], scene["obj"]
        origins = scene.env_origins
        if args_cli.shapes:  # print Newton's collision shapes of hand and object, then stop
            from isaaclab_newton.physics.newton_manager import NewtonManager
            from newton import ShapeFlags

            model = NewtonManager.get_model()
            keys = list(model.body_key) if hasattr(model, "body_key") else list(model.body_label)
            flags = model.shape_flags.numpy()
            bodies = model.shape_body.numpy()
            types = model.shape_type.numpy()
            groups = model.shape_collision_group.numpy()
            for i in range(model.shape_count):
                b = bodies[i]
                name = keys[b] if b >= 0 else "world"
                if "env_0/" not in name and b >= 0:
                    continue
                if not any(t in name for t in ("right_hand", "Object", "Table", "world")):
                    continue
                collide = bool(flags[i] & ShapeFlags.COLLIDE_SHAPES)
                print(
                    f"shape {i:4d} body {name[-40:]:40s} type {int(types[i])} "
                    f"group {int(groups[i])} collide {collide}"
                )
            print(
                "shape types: see newton.GeoType",
                [(t.name, int(t)) for t in __import__("newton").GeoType],
            )
            return
        # start in the configured pose (a bare scene starts at the model's zero pose, where the
        # G1's forearms point straight ahead at table height) and hold it
        q0 = tt(robot.data.default_joint_pos).clone()
        robot.write_joint_position_to_sim_index(position=q0)
        robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q0))
        robot.set_joint_position_target_index(target=q0)
        default = dict(zip(robot.joint_names, q0[0].tolist(), strict=True))

        # per-robot variants: object mass (inertia scaled with it) and placement error
        mass = torch.tensor([v["mass"] for v in VARIANTS], device=dev)
        ratio = (mass[:, None] / tt(obj.data.body_mass)).reshape(NUM_ENVS, 1)
        obj.set_masses_index(masses=mass[:, None])
        obj.set_inertias_index(inertias=tt(obj.data.body_inertia).clone() * ratio[:, :, None])
        pose = tt(obj.data.root_link_pose_w).clone()
        pose[:, :2] += torch.tensor([v["offset"] for v in VARIANTS], device=dev)
        obj.write_root_pose_to_sim_index(root_pose=pose)
        obj.write_root_velocity_to_sim_index(root_velocity=torch.zeros(NUM_ENVS, 6, device=dev))

        arm_ids = torch.tensor(robot.find_joints(RIGHT_ARM, preserve_order=True)[0], device=dev)
        hand_ids = torch.tensor(robot.find_joints(RIGHT_HAND, preserve_order=True)[0], device=dev)
        waist_ids = torch.tensor(robot.find_joints(["waist_pitch_joint"])[0], device=dev)
        palm = robot.find_bodies("right_hand_palm_link")[0][0]
        jac_body = palm - (1 if robot.is_fixed_base else 0)
        jac_cols = arm_ids + (0 if robot.is_fixed_base else 6)

        plans = {g: plan(default, dt, g) for g in {v["grasp"] for v in VARIANTS}}
        per_env = [plans[v["grasp"]] for v in VARIANTS]

        def stack(get):
            return torch.tensor(np.stack([get(pl) for pl in per_env]), device=dev,
                                dtype=torch.float32)  # fmt: skip

        # trajectories as (robots, steps, 7) joint angles and (robots, steps, 3) palm targets
        traj = {
            k: (stack(lambda pl, k=k: pl["traj"][k][0]),
                None if per_env[0]["traj"][k][1] is None
                else stack(lambda pl, k=k: pl["traj"][k][1]))
            for k in per_env[0]["traj"]
        }  # fmt: skip
        q_palm = quat_from_matrix(torch.tensor(PALM_R, device=dev, dtype=torch.float32)[None])
        q_palm = q_palm.repeat(NUM_ENVS, 1)
        hold_q = {"close": stack(lambda pl: pl["q_grasp"]), "hold": stack(lambda pl: pl["q_lift"])}
        hold_p = {"close": stack(lambda pl: pl["grasp"]), "hold": stack(lambda pl: pl["lift"])}

        open_t = torch.tensor(HAND_OPEN, device=dev).repeat(NUM_ENVS, 1)
        closed_t = torch.tensor([CLOSED_POSES[v["close"]] for v in VARIANTS], device=dev)
        hand_default = tt(robot.data.default_joint_pos)[:, hand_ids].clone()
        q_corr = torch.zeros(NUM_ENVS, len(RIGHT_ARM), device=dev)

        def palm_pose():
            return tt(robot.data.body_link_pos_w)[:, palm], tt(robot.data.body_link_quat_w)[:, palm]

        def control_step():
            scene.write_data_to_sim()
            for _ in range(cfg.sim.render_interval):
                sim.step(render=False)
            if args_cli.video:
                sim.render()
            scene.update(dt)

        frames, log, marks = [], [], {}
        obj_start = tt(obj.data.root_pos_w).clone()
        hold_rel = None
        for name, duration in PHASES:
            marks[name] = len(frames)
            peak = None  # debug: largest hand-object force per hand link in this phase
            steps = round(duration / dt)
            for k in range(steps):
                s = (k + 1) / steps
                target_p = None
                if name == "settle":
                    hand = hand_default
                else:
                    if name == "to_raise":
                        q_plan = traj[name][0][:, k]
                        lean = torch.full(
                            (NUM_ENVS, 1), smooth(s) * args_cli.waist_pitch, device=dev
                        )
                        robot.set_joint_position_target_index(target=lean, joint_ids=waist_ids)
                    elif name in traj:
                        q_plan, target_p = traj[name][0][:, k], traj[name][1][:, k]
                    else:
                        q_plan, target_p = hold_q[name], hold_p[name]
                    if target_p is not None:
                        # slow, bounded correction of the measured palm error (gravity sag)
                        pos, quat = palm_pose()
                        jac = tt(robot.data.body_link_jacobian_w)[:, jac_body][:, :, jac_cols]
                        world_p = origins + target_p
                        step = dls_step(jac, pos, quat, world_p, q_palm)
                        q_corr = torch.clamp(q_corr + CORR_GAIN * step, -MAX_CORR, MAX_CORR)
                    target = q_plan + q_corr
                    robot.set_joint_position_target_index(target=target, joint_ids=arm_ids)
                    if name in ("to_raise", "reach", "lower"):
                        hand = open_t
                    elif name == "close":
                        if args_cli.schedule == "fingers_first":
                            sf = torch.full((NUM_ENVS, 7), min(1.0, 2.0 * s), device=dev)
                            sf[:, 4:] = max(0.0, 2.0 * s - 1.0)
                        else:  # together: on a can both reach it about half-way (MuJoCo sweep)
                            sf = torch.full((NUM_ENVS, 7), s, device=dev)
                        hand = open_t + sf * (closed_t - open_t)
                    else:
                        hand = closed_t
                robot.set_joint_position_target_index(target=hand, joint_ids=hand_ids)
                control_step()
                if args_cli.debug:
                    fm = tt(scene["hand_contacts"].data.force_matrix_w)[0].norm(dim=-1).sum(dim=-1)
                    peak = fm if peak is None else torch.maximum(peak, fm)
                if k % 5 == 0:
                    log.append({"phase": name, "palm_z": palm_pose()[0][:, 2].tolist(),
                                "obj_z": tt(obj.data.root_pos_w)[:, 2].tolist()})  # fmt: skip
                if args_cli.video and k % 2 == 0:
                    rgb = tt(scene["cam"].data.output["rgb"])[0, ..., :3]
                    frames.append(rgb.clamp(0, 255).to(torch.uint8).cpu().numpy())
            pos, quat = palm_pose()
            if name == "lift":
                hold_rel = tt(obj.data.root_pos_w) - pos
            goal = None
            if name in traj and traj[name][1] is not None:
                goal = origins + traj[name][1][:, -1]
            elif name in hold_p:
                goal = origins + hold_p[name]
            if goal is not None:
                ep, eo = compute_pose_error(pos, quat, goal, q_palm, rot_error_type="axis_angle")
                print(f"[grasp] {name:8s} palm error {ep.norm(dim=-1).mean() * 100:.1f} cm "
                      f"(max {ep.norm(dim=-1).max() * 100:.1f}), "
                      f"{eo.norm(dim=-1).mean():.3f} rad")  # fmt: skip
            if args_cli.debug:
                print(
                    f"   object 0 after {name}:",
                    [round(v, 3) for v in tt(obj.data.root_link_pose_w)[0].tolist()],
                )
                sensor = scene["hand_contacts"]
                per = {
                    n.split("right_hand_")[-1].removesuffix("_link"): round(float(peak[i]), 1)
                    for i, n in enumerate(sensor.body_names)
                }
                print(f"   peak hand-object forces during {name} (N): {per}")

        palm_pos = palm_pose()[0]
        obj_pos = tt(obj.data.root_pos_w)
        lifted = obj_pos[:, 2] - obj_start[:, 2]
        dist = torch.linalg.norm(obj_pos - palm_pos, dim=-1)
        slip = torch.linalg.norm((obj_pos - palm_pos) - hold_rel, dim=-1)
        success = (lifted > 0.10) & (dist < 0.10)
        rows = [
            {**v, "lifted_m": round(float(lifted[i]), 4), "palm_dist_m": round(float(dist[i]), 4),
             "slip_in_hold_m": round(float(slip[i]), 4), "success": bool(success[i])}
            for i, v in enumerate(VARIANTS)
        ]  # fmt: skip
        for r in rows:
            print("[grasp]", r)
        n_ok = int(success.sum())
        print(f"[grasp] {args_cli.object}: {n_ok}/{NUM_ENVS} held")
        result = {
            "object": args_cli.object, "object_spec": OBJ, "friction": args_cli.friction,
            "table_top": args_cli.table_top, "pelvis_z": PELVIS_Z,
            "waist_pitch": args_cli.waist_pitch, "aim": AIM,
            "grasp_height": GRASP_HEIGHT, "closed_poses": CLOSED_POSES, "success": n_ok,
            "robots": NUM_ENVS, "rows": rows,
            "frame_marks": marks,
        }  # fmt: skip
        result.update({"substeps": physics["solver_substeps"], "physics": physics,
                       "schedule": args_cli.schedule,
                       "mujoco_contacts": args_cli.mujoco_contacts,
                       "controller_export": str(args_cli.export)})  # fmt: skip
        provenance.save_json(args_cli.out / "result.json", result)
        (args_cli.out / "trace.json").write_text(json.dumps(log) + "\n")
        if frames:
            from PIL import Image

            imgs = [Image.fromarray(f) for f in frames]
            imgs[0].save(args_cli.out / "robot0.gif", save_all=True, append_images=imgs[1:],
                         duration=80, loop=0)  # fmt: skip
        print(f"[grasp] wrote {args_cli.out} (frame marks {marks})")


if __name__ == "__main__":
    main()

"""Run NVIDIA's Decoupled WBC lower-body policy (the baseline, D-004) on our G1 in Newton.

Each of the MuJoCo bench's command scenarios (`pi_wbc.sim2sim.commands`) gets `--seeds` robots
on flat ground for 20 s. The policy is NVIDIA's released pair (`scripts/fetch_nvidia_wbc.sh`):
"Balance" while the velocity command is below 0.05, "Walk" otherwise, as in their
`g1_gear_wbc_policy.py`. Its observation is rebuilt from Isaac Lab's state exactly as their code
builds it from MuJoCo's: 86 values per step, the last 6 steps oldest first (516):

    velocity command * (2, 2, 0.5) | height command (m) | torso roll, pitch, yaw commands |
    pelvis turn rate (body frame) * 0.5 | gravity in the pelvis frame |
    29 body joint angles minus the default (legs only have one) | joint speeds * 0.05 |
    the last 15 actions

and its 15 actions (legs, then waist yaw, roll, pitch) set joint targets action * 0.25 + the
default angles. Our G1 already uses NVIDIA's PD gains and default leg pose (D-012). Arms hold
stage 1's default pose, or move as in the bench's arm scenarios. A robot falls when its pelvis
drops below 0.45 m or tilts past 60 degrees.

    uv run python scripts/nvidia_wbc_test.py --out results/stage3/nvidia_wbc/newton.json --viz none
"""

import argparse
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--seeds", type=int, default=10, help="robots per scenario")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--duration-s", type=float, default=20.0)
parser.add_argument("--height", type=float, default=0.74, help="NVIDIA's nominal height command")
parser.add_argument("--substeps", type=int, default=None, help="Newton substeps (training: 2)")
parser.add_argument(
    "--arms",
    choices=["default", "zero"],
    default="default",
    help="held arm pose: stage 1's default (hanging) or all zero (NVIDIA's MuJoCo "
    "script; forearms forward)",
)
parser.add_argument("--scenarios", default=None, help="comma list (default: all)")
parser.add_argument("--check-targets", action="store_true", help="print held arm targets")
parser.add_argument("--policy-dir", type=Path, default=Path("third_party/gr00t_wbc/policy"))
parser.add_argument(
    "--io",
    type=Path,
    default=Path(
        "logs/rsl_rl/pi_wbc_g1_student/slurm-5628865-full-range-long-r/exported/policy_io.json"
    ),
    help="our controller's policy_io.json, for the arm-motion ranges of the arm scenarios",
)
parser.add_argument("--out", type=Path, required=True)
add_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.physics = "newton_mjwarp"

import json  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
import torch  # noqa: E402
from isaaclab.assets import AssetBaseCfg  # noqa: E402
from isaaclab.scene import InteractiveSceneCfg  # noqa: E402
from isaaclab.utils import configclass, instantiate, replace  # noqa: E402
from isaaclab.utils.math import quat_apply_inverse, yaw_quat  # noqa: E402
from isaaclab_tasks.core.velocity.velocity_env_cfg import RoughPhysicsCfg  # noqa: E402

from pi_wbc import provenance  # noqa: E402
from pi_wbc.sim2sim.commands import ALL_SCENARIOS, UpperBodyMotion  # noqa: E402
from pi_wbc.stats import wilson_interval  # noqa: E402
from pi_wbc.tasks.g1_gait import G1_29DOF_GAIT_CFG as G1_CFG  # noqa: E402

# NVIDIA's body joint order (Unitree's 29: legs L then R, waist, arms L then R)
LEGS = ["hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll"]
ARMS = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch",
        "wrist_yaw"]  # fmt: skip
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
SETTLE_S = 2.0

NAMES = args_cli.scenarios.split(",") if args_cli.scenarios else list(ALL_SCENARIOS)
N = len(NAMES) * args_cli.seeds


@configclass
class SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0, dynamic_friction=1.0
            )
        ),
    )
    sky = AssetBaseCfg(prim_path="/World/sky", spawn=sim_utils.DomeLightCfg(intensity=800.0))
    robot = replace(
        G1_CFG,
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=replace(G1_CFG.init_state, pos=(0.0, 0.0, 0.793), rot=(0.0, 0.0, 0.0, 1.0)),
    )


@configclass
class TestCfg:
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=0.005, render_interval=4, physics=RoughPhysicsCfg()
    )
    scene: SceneCfg = SceneCfg(num_envs=N, env_spacing=2.0)


def tt(x):
    return x.torch if hasattr(x, "torch") else x


def main() -> None:
    cfg = TestCfg()
    sessions = {
        k: ort.InferenceSession(
            str(args_cli.policy_dir / f"GR00T-WholeBodyControl-{k}.onnx"),
            providers=["CPUExecutionProvider"],
        )
        for k in ("Balance", "Walk")
    }
    io = json.loads(args_cli.io.read_text())
    with launch_simulation(cfg, args_cli) as physics_cfg:
        if args_cli.substeps is not None:
            physics_cfg.num_substeps = args_cli.substeps
        cfg.sim.physics = physics_cfg
        sim = sim_utils.SimulationContext(cfg.sim)
        scene = instantiate(cfg.scene)
        sim.reset()
        dev = sim.device
        dt = sim.get_physics_dt() * cfg.sim.render_interval
        robot = scene["robot"]
        q0 = tt(robot.data.default_joint_pos).clone()
        if args_cli.arms == "zero":
            arm_ids0 = robot.find_joints([".*_shoulder_.*", ".*_elbow_joint", ".*_wrist_.*"])[0]
            q0[:, arm_ids0] = 0.0
        robot.write_joint_position_to_sim_index(position=q0)
        robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q0))
        robot.set_joint_position_target_index(target=q0)
        body_ids = torch.tensor(robot.find_joints(BODY, preserve_order=True)[0], device=dev)
        act_ids = body_ids[:15]
        ub_names = io["upper_body"]["joint_names"]
        ub_ids = torch.tensor(robot.find_joints(ub_names, preserve_order=True)[0], device=dev)
        ub_default = q0[0, ub_ids].cpu().numpy()

        sc = [ALL_SCENARIOS[NAMES[i // args_cli.seeds]] for i in range(N)]
        vel = np.array([s.velocity for s in sc], dtype=np.float32)
        height = np.array([s.body[0] if s.body else args_cli.height for s in sc], np.float32)
        # NVIDIA's torso command is roll, pitch, yaw; the bench's body pose is waist yaw, roll,
        # pitch
        rpy = np.array([[s.body[2], s.body[3], s.body[1]] if s.body else [0, 0, 0] for s in sc],
                       dtype=np.float32)  # fmt: skip
        rng = np.random.default_rng(args_cli.seed)
        arm_motion = [
            UpperBodyMotion(io["upper_body"], ub_default, np.random.default_rng(rng.integers(1e9)))
            if s.arm_motion
            else None
            for s in sc
        ]
        hist = np.zeros((N, HISTORY, FRAME), dtype=np.float32)
        action = np.zeros((N, 15), dtype=np.float32)
        walk = np.linalg.norm(vel, axis=1) >= 0.05
        steps = round(args_cli.duration_s / dt)
        settle = round(SETTLE_S / dt)
        alive = np.ones(N, dtype=bool)
        sums = {k: np.zeros(N) for k in ("lin", "ang", "vx", "vy", "wz", "z")}
        count = np.zeros(N)
        with torch.inference_mode():
            for k in range(steps):
                q = tt(robot.data.joint_pos)[:, body_ids].cpu().numpy()
                dq = tt(robot.data.joint_vel)[:, body_ids].cpu().numpy()
                omega = tt(robot.data.root_ang_vel_b).cpu().numpy()
                grav = tt(robot.data.projected_gravity_b).cpu().numpy()
                qd = q.copy()
                qd[:, :15] -= DEFAULT_ANGLES
                frame = np.concatenate(
                    [vel * CMD_SCALE, height[:, None], rpy, omega * 0.5, grav, qd, dq * 0.05,
                     action],
                    axis=1,
                ).astype(np.float32)  # fmt: skip
                hist = np.concatenate([hist[:, 1:], frame[:, None]], axis=1)
                obs = hist.reshape(N, -1)
                out = np.zeros((N, 15), dtype=np.float32)
                for name, mask in (("Walk", walk), ("Balance", ~walk)):
                    if mask.any():
                        out[mask] = sessions[name].run(None, {"input": obs[mask]})[0]
                action = out
                target = torch.tensor(action * ACTION_SCALE + DEFAULT_ANGLES, device=dev)
                robot.set_joint_position_target_index(target=target, joint_ids=act_ids)
                for i, m in enumerate(arm_motion):
                    if m is not None:
                        ub = torch.tensor(m.update(dt), device=dev, dtype=torch.float32)
                        robot.set_joint_position_target_index(
                            target=ub[None], joint_ids=ub_ids, env_ids=torch.tensor([i], device=dev)
                        )
                scene.write_data_to_sim()
                for _ in range(cfg.sim.render_interval):
                    sim.step(render=False)
                scene.update(dt)
                if args_cli.check_targets and k in (0, 5, 50):
                    jl = robot.find_joints(["left_elbow_joint", "left_shoulder_pitch_joint"])[0]
                    print(
                        f"[check] step {k}: env 0 left elbow/shoulder target",
                        tt(robot.data.joint_pos_target)[0, jl].tolist(),
                        "actual",
                        tt(robot.data.joint_pos)[0, jl].tolist(),
                    )
                z = tt(robot.data.root_pos_w)[:, 2].cpu().numpy()
                g = tt(robot.data.projected_gravity_b)[:, 2].cpu().numpy()
                alive &= (z > 0.45) & (g < -0.5)
                if k < settle:
                    continue
                qy = yaw_quat(tt(robot.data.root_quat_w))
                v = quat_apply_inverse(qy, tt(robot.data.root_lin_vel_w))[:, :2].cpu().numpy()
                wz = tt(robot.data.root_ang_vel_w)[:, 2].cpu().numpy()
                a = alive.astype(float)
                sums["lin"] += np.linalg.norm(vel[:, :2] - v, axis=1) * a
                sums["ang"] += np.abs(vel[:, 2] - wz) * a
                sums["vx"] += v[:, 0] * a
                sums["vy"] += v[:, 1] * a
                sums["wz"] += wz * a
                sums["z"] += z * a
                count += a
        rows = []
        for j, name in enumerate(NAMES):
            idx = slice(j * args_cli.seeds, (j + 1) * args_cli.seeds)
            falls = int((~alive[idx]).sum())
            up = alive[idx] & (count[idx] > 0)

            def mean(key, idx=idx, up=up):
                return float(np.mean(sums[key][idx][up] / count[idx][up])) if up.any() else None

            lo, hi = wilson_interval(falls, args_cli.seeds)
            rows.append({
                "scenario": name, "episodes": args_cli.seeds, "falls": falls,
                "fall_rate_ci95": [lo, hi], "lin_err_mean": mean("lin"),
                "ang_err_mean": mean("ang"),
                "velocity_mean": [mean("vx"), mean("vy"), mean("wz")],
                "pelvis_z_mean": mean("z"),
                "command": {"velocity": list(map(float, vel[j * args_cli.seeds])),
                            "height": float(height[j * args_cli.seeds]),
                            "rpy": list(map(float, rpy[j * args_cli.seeds])),
                            "arm_motion": sc[j * args_cli.seeds].arm_motion},
            })  # fmt: skip
        print("| Command | Falls | Speed err | Turn err | Achieved vx, vy, wz | Pelvis z |")
        print("|---|---:|---:|---:|---|---:|")
        for r in rows:

            def f(v, fmt="{:.2f}"):
                return "—" if v is None else fmt.format(v)

            vm = ", ".join(f(v, "{:+.2f}") for v in r["velocity_mean"])
            print(f"| {r['scenario']} | {r['falls']}/{r['episodes']} | {f(r['lin_err_mean'])} | "
                  f"{f(r['ang_err_mean'])} | {vm} | {f(r['pelvis_z_mean'])} |")  # fmt: skip
        provenance.save_json(
            args_cli.out,
            {
                "controller": "NVIDIA Decoupled WBC (GR00T-WholeBodyControl, Balance + Walk)",
                "policy_commit": (args_cli.policy_dir.parent / "COMMIT").read_text().strip(),
                "physics": "newton_mjwarp",
                "substeps": cfg.sim.physics.num_substeps,
                "seeds_per_scenario": args_cli.seeds,
                "seed": args_cli.seed,
                "duration_s": args_cli.duration_s,
                "settle_s": SETTLE_S,
                "nominal_height_command": args_cli.height,
                "arms": args_cli.arms,
                "summary": rows,
            },
        )
        print(f"[nvidia-wbc] wrote {args_cli.out}")


if __name__ == "__main__":
    main()

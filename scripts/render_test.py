"""Render the stage 3 scene with one of Isaac Lab 3.0's camera renderers and time it.

A G1 (base fixed, default pose) stands at a table with three objects and two bins. Two cameras
per environment: the head camera (where Unitree's G1 model mounts its D435: torso link, 43 cm up,
tilted 47.6 degrees down) and a fixed side camera for showcase pictures. The script saves each
camera's image from environment 0 and reports render rate and GPU memory.

    .venv-isaacsim/bin/python scripts/render_test.py --renderer newton_renderer --num_envs 16 \
        --viz none --out logs/render/newton

Renderers: ``newton_renderer`` (Newton Warp, with Newton physics), ``ovrtx`` (OVRTX, with Newton
physics) and ``isaacsim_rtx`` (Isaac Sim RTX, with Isaac Sim PhysX). The Newton renderer also
runs in the training environment; the other two need scripts/isaacsim_env.sh. For RTX, dim the
lights: ``--sun 1000 --sky 250``.
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument(
    "--renderer", choices=["newton_renderer", "ovrtx", "isaacsim_rtx"], required=True
)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--width", type=int, default=640)
parser.add_argument("--height", type=int, default=480)
parser.add_argument("--steps", type=int, default=100, help="Timed render steps after warm-up.")
parser.add_argument("--warmup", type=int, default=20)
parser.add_argument("--shadows", action="store_true", help="Newton renderer: shadow rays on.")
parser.add_argument("--sun", type=float, default=2500.0, help="Distant light intensity.")
parser.add_argument("--sky", type=float, default=600.0, help="Dome light intensity.")
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--debug", default="", help="Comma list: no_objects, free_base, no_robot.")
add_launcher_args(parser)
args_cli = parser.parse_args()
DEBUG = set(filter(None, args_cli.debug.split(",")))
args_cli.physics = "isaacsim_physx" if args_cli.renderer == "isaacsim_rtx" else "newton_mjwarp"
if args_cli.renderer == "isaacsim_rtx":
    args_cli.enable_cameras = True

import isaaclab.sim as sim_utils  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg  # noqa: E402
from isaaclab.scene import InteractiveSceneCfg  # noqa: E402
from isaaclab.sensors import CameraCfg  # noqa: E402
from isaaclab.utils import configclass, instantiate, replace  # noqa: E402
from isaaclab_tasks.core.velocity.velocity_env_cfg import RoughPhysicsCfg  # noqa: E402

from pi_wbc.tasks.g1_gait import G1_29DOF_GAIT_CFG as G1_CFG  # noqa: E402  (stage 1's robot)


def renderer_cfg():
    if args_cli.renderer == "newton_renderer":
        from isaaclab_newton.renderers import NewtonWarpRendererCfg

        return NewtonWarpRendererCfg(enable_shadows=args_cli.shadows)
    if args_cli.renderer == "ovrtx":
        from isaaclab_ov.renderers import OVRTXRendererCfg

        return OVRTXRendererCfg()
    from isaaclab_physx.renderers import IsaacRtxRendererCfg

    return IsaacRtxRendererCfg()


def box(name, size, pos, color, kinematic=True):
    """A coloured box rigid body; kinematic ones don't move."""
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=sim_utils.CuboidCfg(
            size=size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=kinematic),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.5),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
    )


TABLE_X, TABLE_TOP = 0.55, 0.75  # table centre in front of the robot, and its height (m)


def camera(prim, offset, renderer):
    return CameraCfg(
        prim_path=prim,
        update_period=0.0,
        width=args_cli.width,
        height=args_cli.height,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0, horizontal_aperture=20.955, clipping_range=(0.05, 20.0)
        ),
        offset=offset,
        renderer_cfg=renderer,
    )


@configclass
class SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    sun = AssetBaseCfg(
        prim_path="/World/sun",
        spawn=sim_utils.DistantLightCfg(intensity=args_cli.sun, color=(1.0, 0.97, 0.92), angle=0.5),
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.25, 0.25, 0.0, 0.933)),
    )
    sky = AssetBaseCfg(prim_path="/World/sky", spawn=sim_utils.DomeLightCfg(intensity=args_cli.sky))
    robot = replace(
        G1_CFG,
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=replace(G1_CFG.spawn, fix_root_link="free_base" not in DEBUG),
        # Isaac Lab's G1 starts turned 90 degrees; face +x, towards the table
        init_state=replace(G1_CFG.init_state, rot=(0.0, 0.0, 0.0, 1.0)),
    )
    table = box("Table", (0.6, 1.0, TABLE_TOP), (TABLE_X, 0.0, TABLE_TOP / 2), (0.55, 0.38, 0.22))
    red_cube = box(
        "RedCube",
        (0.05, 0.05, 0.05),
        (TABLE_X - 0.12, 0.10, TABLE_TOP + 0.025),
        (0.85, 0.1, 0.1),
        kinematic=False,
    )
    green_box = box(
        "GreenBox",
        (0.04, 0.08, 0.10),
        (TABLE_X - 0.10, -0.05, TABLE_TOP + 0.05),
        (0.1, 0.7, 0.2),
        kinematic=False,
    )
    yellow_block = box(
        "YellowBlock",
        (0.10, 0.04, 0.04),
        (TABLE_X - 0.02, 0.05, TABLE_TOP + 0.02),
        (0.95, 0.8, 0.1),
        kinematic=False,
    )
    blue_bin = box(
        "BlueBin", (0.18, 0.22, 0.02), (TABLE_X + 0.10, -0.30, TABLE_TOP + 0.01), (0.1, 0.3, 0.85)
    )
    white_bin = box(
        "WhiteBin", (0.18, 0.22, 0.02), (TABLE_X + 0.10, 0.30, TABLE_TOP + 0.01), (0.9, 0.9, 0.9)
    )
    head_cam = camera(
        "{ENV_REGEX_NS}/Robot/torso_link/head_cam",
        # 4 cm ahead of Unitree's D435 mount (0.0576, 0.0175, 0.4299): at the mount itself the
        # camera sits inside the head mesh
        CameraCfg.OffsetCfg(
            pos=(0.0976, 0.0175, 0.4299), rot=(0.0, 0.4035, 0.0, 0.9150), convention="world"
        ),
        renderer_cfg(),
    )
    side_cam = camera(
        "{ENV_REGEX_NS}/side_cam",
        CameraCfg.OffsetCfg(
            pos=(0.6, -2.2, 1.5), rot=(-0.1393, 0.1393, 0.6933, 0.6933), convention="world"
        ),
        renderer_cfg(),
    )

    def __post_init__(self):
        if "no_objects" in DEBUG:
            self.red_cube = self.green_box = self.yellow_block = None
        if "no_robot" in DEBUG:
            self.robot = None
            self.head_cam = None


@configclass
class RenderTestCfg:
    # the velocity tasks' physics presets (training's Newton settings, or Isaac Sim PhysX)
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=0.005, render_interval=4, physics=RoughPhysicsCfg()
    )
    scene: SceneCfg = SceneCfg(num_envs=args_cli.num_envs, env_spacing=3.0)


def gpu_memory_mib() -> int | None:
    """This process's GPU memory as nvidia-smi reports it (all libraries, not only torch)."""
    import os

    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    for line in out.splitlines():
        pid, mem = (s.strip() for s in line.split(","))
        if int(pid) == os.getpid():
            return int(mem)
    return None


def save(image: torch.Tensor, path: Path) -> None:
    from PIL import Image

    Image.fromarray(image[..., :3].clamp(0, 255).to(torch.uint8).cpu().numpy()).save(path)


def main() -> None:
    cfg = RenderTestCfg()
    args_cli.out.mkdir(parents=True, exist_ok=True)
    with launch_simulation(cfg, args_cli) as physics_cfg:
        cfg.sim.physics = physics_cfg  # the preset chosen by args_cli.physics
        t0 = time.perf_counter()
        sim = sim_utils.SimulationContext(cfg.sim)
        scene = instantiate(cfg.scene)
        sim.reset()
        t_setup = time.perf_counter() - t0
        dt = sim.get_physics_dt()

        def step():
            scene.write_data_to_sim()
            for _ in range(cfg.sim.render_interval):
                sim.step(render=False)
            sim.render()
            scene.update(dt * cfg.sim.render_interval)

        for _ in range(args_cli.warmup):
            step()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args_cli.steps):
            step()
            if "head_cam" in scene.keys():  # noqa: SIM118 (InteractiveScene has no __contains__)
                _ = scene["head_cam"].data.output["rgb"]
            _ = scene["side_cam"].data.output["rgb"]
        torch.cuda.synchronize()
        t_steps = time.perf_counter() - t0

        def tt(x):
            return x.torch if hasattr(x, "torch") else x

        for name in ("table", "red_cube", "green_box", "yellow_block", "blue_bin", "white_bin"):
            if name not in scene.keys():  # noqa: SIM118 (InteractiveScene has no __contains__)
                continue
            print(f"[render_test] {name} at {tt(scene[name].data.root_pos_w)[0].tolist()}")
        robot = scene["robot"] if "robot" in scene.keys() else None  # noqa: SIM118
        pos = tt(robot.data.body_pos_w)[0] if robot is not None else None
        for i, body in enumerate(robot.body_names if robot is not None else []):
            if body in ("pelvis", "torso_link", "left_ankle_roll_link", "left_wrist_yaw_link"):
                print(f"[render_test] robot {body} at {pos[i].tolist()}")
        for name in ("head_cam", "side_cam"):
            if name not in scene.keys():  # noqa: SIM118 (InteractiveScene has no __contains__)
                continue
            rgb = scene[name].data.output["rgb"]
            rgb = rgb.torch if hasattr(rgb, "torch") else rgb
            save(rgb[0], args_cli.out / f"{name}.png")
            img = rgb[0, ..., :3].float()
            stats = {"mean": float(img.mean()), "std": float(img.std())}
            print(f"[render_test] {name}: shape {tuple(rgb.shape)}, pixel mean/std {stats}")

        frames = args_cli.steps * args_cli.num_envs * 2
        result = {
            "renderer": args_cli.renderer,
            "physics": args_cli.physics,
            "num_envs": args_cli.num_envs,
            "resolution": [args_cli.width, args_cli.height],
            "shadows": args_cli.shadows,
            "lights": [args_cli.sun, args_cli.sky],
            "setup_s": round(t_setup, 1),
            "steps_per_s": round(args_cli.steps / t_steps, 2),
            "camera_frames_per_s": round(frames / t_steps, 1),
            "gpu_memory_mib": gpu_memory_mib(),
            "gpu": torch.cuda.get_device_name(0),
            "torch_max_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20),
        }
        print("[render_test]", json.dumps(result))
        (args_cli.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        np.save(args_cli.out / "done.npy", np.zeros(1))


if __name__ == "__main__":
    main()

"""Record a GIF of a trained student being kicked while it walks, with an RTX renderer.

One robot, nominal (no randomization), commanded to walk forward; at the kick times its pelvis
gets a sideways velocity change (alternating left and right, so a side camera shows it). A camera
follows the robot. The start is reset seed `--seed`; the first one rendered is the one kept.

    OMNI_KIT_ACCEPT_EULA=YES .venv-isaacsim/bin/python scripts/kick_video.py \
        --checkpoint <run>/model_999.pt --renderer ovrtx --viz none \
        --out logs/render/kick_s1.gif physics=newton_mjwarp
"""

import argparse
import json
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="PiWbc-G1-Student")
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument(
    "--renderer", choices=["ovrtx", "isaacsim_rtx", "newton_renderer"], default="ovrtx"
)
parser.add_argument("--speed", type=float, default=0.5, help="forward command (m/s)")
parser.add_argument("--kick", type=float, default=2.0, help="kick speed (m/s)")
parser.add_argument("--kick_times", default="3,6,9", help="seconds")
parser.add_argument("--duration", type=float, default=12.0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--width", type=int, default=480)
parser.add_argument("--height", type=int, default=360)
parser.add_argument("--every", type=int, default=2, help="keep every n-th policy step (50 Hz)")
parser.add_argument("--sun", type=float, default=None)
parser.add_argument("--out", type=Path, required=True)
add_launcher_args(parser)
args, overrides = parser.parse_known_args()

import gymnasium as gym  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
import torch  # noqa: E402
from isaaclab.managers import ObservationTermCfg, SceneEntityCfg  # noqa: E402
from isaaclab.sensors import CameraCfg  # noqa: E402
from isaaclab_rl.rsl_rl import (  # noqa: E402
    RslRlVecEnvWrapper,
    check_rsl_rl_version,
    create_rsl_rl_runner,
    handle_deprecated_rsl_rl_cfg,
)
from isaaclab_tasks.utils import resolve_task_config  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

import pi_wbc.tasks  # noqa: E402, F401
from pi_wbc import isaac_eval  # noqa: E402


def renderer_cfg():
    if args.renderer == "ovrtx":
        from isaaclab_ov.renderers import OVRTXRendererCfg

        return OVRTXRendererCfg()
    if args.renderer == "isaacsim_rtx":
        from isaaclab_physx.renderers import IsaacRtxRendererCfg

        return IsaacRtxRendererCfg()
    from isaaclab_newton.renderers import NewtonWarpRendererCfg

    return NewtonWarpRendererCfg(enable_shadows=True)


env_cfg, agent_cfg = resolve_task_config(args.task, "rsl_rl_cfg_entry_point", overrides=overrides)
agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, check_rsl_rl_version())
env_cfg.scene.num_envs = 1
env_cfg.seed = args.seed
env_cfg.episode_length_s = args.duration + 5.0
env_cfg.make_nominal()
env_cfg.commands.base_velocity.resampling_time_range = (1.0e6, 1.0e6)
env_cfg.commands.body_pose.resampling_time_range = (1.0e6, 1.0e6)
env_cfg.curriculum.upper_body_level = None
physx = any(o.startswith("physics=") and "newton" not in o for o in overrides)
if physx:  # as pi_wbc.isaac_eval does for engines other than training's
    env_cfg.observations.teacher.friction = ObservationTermCfg(
        func=isaac_eval._constant_obs, params={"value": 1.0}
    )
    env_cfg.events.physics_material = None
    ground = env_cfg.scene.terrain.physics_material
    ground.static_friction = ground.dynamic_friction = 1.0
    ground.friction_combine_mode = "max"
    io = json.loads((args.checkpoint.resolve().parent / "exported" / "policy_io.json").read_text())
    env_cfg.actions.joint_pos.joint_names = list(io["action"]["joint_names"])
    env_cfg.actions.joint_pos.preserve_order = True
    for name, joints in io["observation"]["joint_names"].items():
        getattr(env_cfg.observations.policy, name).params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=list(joints), preserve_order=True
        )
if args.sun is not None and hasattr(env_cfg.scene, "light"):
    env_cfg.scene.light.spawn.intensity = args.sun
env_cfg.scene.video_cam = CameraCfg(
    prim_path="{ENV_REGEX_NS}/video_cam",
    update_period=0.0,
    width=args.width,
    height=args.height,
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=20.0, horizontal_aperture=20.955, clipping_range=(0.05, 50.0)
    ),
    renderer_cfg=renderer_cfg(),
)

with launch_simulation(env_cfg, args):
    env = gym.make(args.task, cfg=env_cfg)
    venv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = create_rsl_rl_runner(venv, agent_cfg)
    runner.load(str(args.checkpoint.resolve()))
    policy = runner.get_inference_policy(device=venv.unwrapped.device)
    base = venv.unwrapped
    robot, cam = base.scene["robot"], base.scene["video_cam"]
    vel_term = base.command_manager.get_term("base_velocity")
    body_term = base.command_manager.get_term("body_pose")
    arm_term = base.command_manager.get_term("upper_body")
    command = torch.tensor([[args.speed, 0.0, 0.0]], device=base.device)

    def hold():
        vel_term.vel_command_b[:] = command
        vel_term.is_standing_env[:] = False
        if hasattr(vel_term, "is_turning_env"):
            vel_term.is_turning_env[:] = False
        body_term.goal[:] = body_term.nominal
        arm_term.still[:] = True

    base.reset()
    hold()
    for buf in (arm_term.start, arm_term.goal, arm_term.current):
        buf[:] = arm_term.default
    obs = venv.get_observations()
    dt = base.step_dt
    kick_steps = {round(float(t) / dt): i for i, t in enumerate(args.kick_times.split(","))}
    frames, fell_at, label_until = [], None, -1
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except OSError:
        font = ImageFont.load_default()
    with torch.inference_mode():
        for k in range(round(args.duration / dt)):
            hold()
            if k in kick_steps:
                # sideways in the robot's heading frame, alternating left and right
                yaw = 2.0 * torch.atan2(
                    robot.data.root_quat_w.torch[0, 2], robot.data.root_quat_w.torch[0, 3]
                )
                side = 1.0 if kick_steps[k] % 2 == 0 else -1.0
                vel = robot.data.root_com_vel_w.torch.clone()
                vel[0, 0] += args.kick * side * -torch.sin(yaw)
                vel[0, 1] += args.kick * side * torch.cos(yaw)
                robot.write_root_velocity_to_sim_index(root_velocity=vel)
                label_until = k + round(0.6 / dt)
            obs, _, dones, _ = venv.step(policy(obs))
            if dones.any() and fell_at is None:
                fell_at = k * dt
            pos = robot.data.root_pos_w.torch[0]
            q = robot.data.root_quat_w.torch[0]
            yaw = 2.0 * torch.atan2(q[2], q[3])
            # a side view that turns with the robot: ahead-right of it, in its heading frame
            ox, oy = 1.7, -1.8
            eye = pos.clone()
            eye[0] += ox * torch.cos(yaw) - oy * torch.sin(yaw)
            eye[1] += ox * torch.sin(yaw) + oy * torch.cos(yaw)
            eye[2] = 1.25
            target = pos.clone()
            target[2] = 0.65
            cam.set_world_poses_from_view(eye.unsqueeze(0), target.unsqueeze(0))
            if k % args.every == 0:
                rgb = cam.data.output["rgb"]
                rgb = rgb.torch if hasattr(rgb, "torch") else rgb
                img = Image.fromarray(rgb[0, ..., :3].clamp(0, 255).to(torch.uint8).cpu().numpy())
                d = ImageDraw.Draw(img)
                d.text(
                    (10, 8),
                    f"t = {k * dt:4.1f} s   walk {args.speed:.1f} m/s",
                    fill="white",
                    font=font,
                    stroke_width=2,
                    stroke_fill="black",
                )
                if k <= label_until:
                    d.text(
                        (10, 34),
                        f"kick {args.kick:.1f} m/s",
                        fill=(255, 210, 0),
                        font=font,
                        stroke_width=2,
                        stroke_fill="black",
                    )
                frames.append(img)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        args.out,
        save_all=True,
        append_images=frames[1:],
        duration=round(1000 * dt * args.every),
        loop=0,
        optimize=True,
    )
    info = {
        "out": str(args.out),
        "frames": len(frames),
        "fell_at_s": fell_at,
        "seed": args.seed,
        "renderer": args.renderer,
        "overrides": overrides,
        "checkpoint": str(args.checkpoint),
    }
    args.out.with_suffix(".json").write_text(json.dumps(info, indent=1))
    print("[kick_video]", json.dumps(info))
    env.close()

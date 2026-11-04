"""Smoke test of the stage 3 task environment with the stage 1 controller in the loop.

Each robot gets a development layout, walks forward for `--walk-s` seconds at `--speed` (through
the high level's velocity command), then stands. Reports falls, how far the objects on the table
moved (they should not), and the simulation rate; `--gif` records robot 0 from the side.

    uv run python scripts/pick_place_smoke.py --num-envs 8 --viz none
"""

import argparse
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation

DEFAULT_CKPT = "logs/rsl_rl/pi_wbc_g1_student/slurm-5628865-full-range-long-r/model_2999.pt"
parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--num-envs", type=int, default=8)
parser.add_argument("--checkpoint", type=Path, default=Path(DEFAULT_CKPT))
parser.add_argument("--first-seed", type=int, default=50_000, help="first development layout")
parser.add_argument("--duration-s", type=float, default=8.0)
parser.add_argument("--walk-s", type=float, default=2.0)
parser.add_argument("--speed", type=float, default=0.4)
parser.add_argument("--gif", type=Path, default=None)
add_launcher_args(parser)
args, overrides = parser.parse_known_args()

import time  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_rl.rsl_rl import (  # noqa: E402
    RslRlVecEnvWrapper,
    check_rsl_rl_version,
    create_rsl_rl_runner,
    handle_deprecated_rsl_rl_cfg,
)
from isaaclab_tasks.utils import resolve_task_config  # noqa: E402

import pi_wbc.tasks  # noqa: E402, F401
from pi_wbc.stage3 import layout as L  # noqa: E402
from pi_wbc.tasks.g1_pick_place import HighLevel  # noqa: E402

TASK = "PiWbc-G1-PickPlace"


def tt(x):
    return x.torch if hasattr(x, "torch") else x


def main() -> None:
    env_cfg, agent_cfg = resolve_task_config(
        TASK, "rsl_rl_cfg_entry_point", overrides=[*overrides, "physics=newton_mjwarp"]
    )
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, check_rsl_rl_version())
    env_cfg.scene.num_envs = args.num_envs
    if args.gif is not None:
        import isaaclab.sim as sim_utils
        from isaaclab.sensors import CameraCfg
        from isaaclab_newton.renderers import NewtonWarpRendererCfg

        env_cfg.scene.debug_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/debug_cam",
            update_period=0.0,
            width=480,
            height=320,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.05, 30.0)),
            # to the robots' right, looking across the approach to the table
            offset=CameraCfg.OffsetCfg(
                pos=(-0.9, -3.4, 1.6), rot=(-0.1305, 0.1305, 0.6950, 0.6950), convention="world"
            ),
            renderer_cfg=NewtonWarpRendererCfg(),
        )
    with launch_simulation(env_cfg, args):
        env = gym.make(TASK, cfg=env_cfg)
        venv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = create_rsl_rl_runner(venv, agent_cfg)
        runner.load(str(args.checkpoint.resolve()))
        policy = runner.get_inference_policy(device=venv.unwrapped.device)
        base = venv.unwrapped
        n, dev, dt = base.num_envs, base.device, base.step_dt
        base.pick_place_layouts = [L.sample(args.first_seed + i) for i in range(n)]
        base.reset()
        hl = HighLevel(env)
        obs = venv.get_observations()
        robot = base.scene["robot"]
        objs = [base.scene[f"obj_{name}"] for name in L.OBJECT_NAMES]
        start_obj = torch.stack([tt(o.data.root_pos_w) for o in objs], dim=1).clone()
        start_xy = tt(robot.data.root_pos_w)[:, :2].clone()
        alive = torch.ones(n, dtype=torch.bool, device=dev)
        frames = []
        steps = round(args.duration_s / dt)
        t0 = time.perf_counter()
        with torch.inference_mode():
            for k in range(steps):
                walking = k * dt < args.walk_s
                hl.velocity[:, 0] = args.speed if walking else 0.0
                hl.apply()
                obs, _, _, _ = venv.step(policy(obs))
                z = tt(robot.data.root_pos_w)[:, 2]
                g = tt(robot.data.projected_gravity_b)[:, 2]
                alive &= (z > 0.45) & (g < -0.5)
                if args.gif is not None and k % 3 == 0:
                    rgb = tt(base.scene["debug_cam"].data.output["rgb"])[0, ..., :3]
                    frames.append(rgb.clamp(0, 255).to(torch.uint8).cpu().numpy())
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        obj_now = torch.stack([tt(o.data.root_pos_w) for o in objs], dim=1)
        moved = (obj_now - start_obj).norm(dim=-1)  # (envs, 5)
        walked = (tt(robot.data.root_pos_w)[:, :2] - start_xy).norm(dim=-1)
        rate = steps * n / wall
        print(f"[smoke] {n} robots, {steps} steps in {wall:.1f} s: {rate:.0f} robot-steps/s "
              f"({rate * dt:.1f}x real time summed over robots)")  # fmt: skip
        print(f"[smoke] falls {int((~alive).sum())}/{n}; walked {walked.mean():.2f} m "
              f"(commanded {args.speed * args.walk_s:.2f}); objects moved at most "
              f"{moved.max() * 100:.2f} cm")  # fmt: skip
        for i in range(n):
            far = [(L.OBJECT_NAMES[k], round(float(moved[i, k]) * 100, 1),
                    [round(v, 2) for v in (obj_now[i, k] - base.scene.env_origins[i]).tolist()])
                   for k in range(len(L.OBJECT_NAMES)) if moved[i, k] > 0.01]  # fmt: skip
            if far:
                print(f"   env {i} objects moved (cm, now at): {far}")
        for i in range(min(n, 4)):
            lay = base.pick_place_layouts[i]
            print(f"   env {i}: seed {lay.seed}: '{lay.instruction}', objects {lay.objects}")
        if frames:
            from PIL import Image

            imgs = [Image.fromarray(f) for f in frames]
            args.gif.parent.mkdir(parents=True, exist_ok=True)
            imgs[0].save(args.gif, save_all=True, append_images=imgs[1:], duration=60, loop=0)
            print(f"[smoke] wrote {args.gif}")
        env.close()


if __name__ == "__main__":
    main()

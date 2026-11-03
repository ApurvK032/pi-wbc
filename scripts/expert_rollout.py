"""Run the scripted expert through the stage 1 controller on pick-and-place layouts.

Batches of `--num-envs` layouts (consecutive seeds from `--first-seed`) run until every episode
has ended; the next batch follows until `--episodes` are done. Prints a table of outcomes
(success, or the failure label and the phase it happened in) with a 95% Wilson interval for
the success rate, and writes them to `--out`. `--gif` records robot 0 of the first batch.

    uv run python scripts/expert_rollout.py --episodes 16 --num-envs 8 \
        --out logs/expert/dev.json --viz none
"""

import argparse
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation

DEFAULT_RUN = "logs/rsl_rl/pi_wbc_g1_student/slurm-5628865-full-range-long-r"
parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--num-envs", type=int, default=8)
parser.add_argument("--episodes", type=int, default=8)
parser.add_argument("--first-seed", type=int, default=50_000, help="development layouts")
parser.add_argument("--run", type=Path, default=Path(DEFAULT_RUN), help="controller run folder")
parser.add_argument("--gif", type=Path, default=None)
parser.add_argument("--gif-env", type=int, default=0, help="which robot of the first batch")
parser.add_argument(
    "--fail-gifs",
    type=Path,
    default=None,
    help="save the last 9 s of every failed episode as a GIF in this folder",
)
parser.add_argument("--out", type=Path, required=True)
add_launcher_args(parser)
args, overrides = parser.parse_known_args()

import collections  # noqa: E402
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
from pi_wbc import provenance  # noqa: E402
from pi_wbc.stage3 import layout as L  # noqa: E402
from pi_wbc.stage3.expert import PHASES, Expert  # noqa: E402
from pi_wbc.stats import wilson_interval  # noqa: E402
from pi_wbc.tasks.g1_pick_place import HighLevel  # noqa: E402

TASK = "PiWbc-G1-PickPlace"


def tt(x):
    return x.torch if hasattr(x, "torch") else x


def look_at(eye, target):
    """(x, y, z, w) rotation of a camera at `eye` looking at `target` (Isaac Lab's "world"
    camera convention: forward +x, up +z)."""
    import numpy as np
    from scipy.spatial.transform import Rotation

    f = np.asarray(target, float) - np.asarray(eye, float)
    f /= np.linalg.norm(f)
    left = np.cross([0.0, 0.0, 1.0], f)
    left /= np.linalg.norm(left)
    up = np.cross(f, left)
    return tuple(Rotation.from_matrix(np.stack([f, left, up], axis=1)).as_quat())


EYE, TARGET = (-1.6, -2.4, 1.6), (-0.6, 0.0, 0.8)  # robot's right side, approach and table front


def side_camera(small: bool = False):
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import CameraCfg
    from isaaclab_newton.renderers import NewtonWarpRendererCfg

    return CameraCfg(
        prim_path="{ENV_REGEX_NS}/debug_cam",
        update_period=0.0,
        width=320 if small else 480,
        height=214 if small else 320,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.05, 30.0)),
        offset=CameraCfg.OffsetCfg(pos=EYE, rot=look_at(EYE, TARGET), convention="world"),
        renderer_cfg=NewtonWarpRendererCfg(),
    )


def save_gif(frames, path: Path) -> None:
    from PIL import Image

    if not frames:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=60, loop=0)


def main() -> None:
    env_cfg, agent_cfg = resolve_task_config(
        TASK, "rsl_rl_cfg_entry_point", overrides=[*overrides, "physics=newton_mjwarp"]
    )
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, check_rsl_rl_version())
    n = min(args.num_envs, args.episodes)
    env_cfg.scene.num_envs = n
    if args.gif is not None or args.fail_gifs is not None:
        env_cfg.scene.debug_cam = side_camera(small=args.fail_gifs is not None)
    with launch_simulation(env_cfg, args):
        env = gym.make(TASK, cfg=env_cfg)
        venv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = create_rsl_rl_runner(venv, agent_cfg)
        runner.load(str((args.run / "model_2999.pt").resolve()))
        policy = runner.get_inference_policy(device=venv.unwrapped.device)
        base = venv.unwrapped
        hl = HighLevel(env)
        expert = Expert(env, hl, args.run / "exported")
        dt = base.step_dt
        results, frames = [], []
        seed = args.first_seed
        t0 = time.perf_counter()
        sim_steps = 0
        while len(results) < args.episodes:
            m = min(n, args.episodes - len(results))
            seeds = [seed + i for i in range(n)]
            seed += n
            base.pick_place_layouts = [L.sample(s) for s in seeds]
            base.reset()
            expert.reset(list(range(n)))
            obs = venv.get_observations()
            clips = [collections.deque(maxlen=150) for _ in range(n)]
            saved = set()
            max_steps = round(expert.p.episode_timeout_s / dt) + 10
            # no_grad, not inference_mode: tensors made in inference mode (also inside the
            # environment's action term) can't be changed by the next batch's reset
            with torch.no_grad():
                for k in range(max_steps):
                    expert.step()
                    hl.apply()
                    obs, _, _, _ = venv.step(policy(obs))
                    sim_steps += 1
                    if args.fail_gifs is not None and k % 3 == 0:
                        rgb = tt(base.scene["debug_cam"].data.output["rgb"])[..., :3]
                        rgb = rgb.clamp(0, 255).to(torch.uint8).cpu().numpy()
                        for i in range(m):
                            if not expert.finished()[i]:
                                clips[i].append(rgb[i])
                            elif expert.label[i] != "success" and i not in saved:
                                saved.add(i)
                                save_gif(clips[i], args.fail_gifs / (
                                    f"{base.pick_place_layouts[i].seed}-{expert.label[i]}-"
                                    f"{expert.failed_in[i]}.gif"))  # fmt: skip
                    if args.gif is not None and not results and k % 3 == 0:
                        rgb = tt(base.scene["debug_cam"].data.output["rgb"])[args.gif_env, ..., :3]
                        frames.append(rgb.clamp(0, 255).to(torch.uint8).cpu().numpy())
                    if expert.finished()[:m].all():
                        break
            for i in range(m):
                lay = base.pick_place_layouts[i]
                label = expert.label[i] or "timed_out"
                results.append({
                    "seed": lay.seed, "instruction": lay.instruction, "label": label,
                    "phase": PHASES[expert.phase[i]], "failed_in": expert.failed_in[i],
                    "time_s": round(expert.age[i] * dt, 2), "notes": expert.notes[i],
                })  # fmt: skip
                print(f"[expert] seed {lay.seed}: {label:12s} (in {expert.failed_in[i]}) "
                      f"after {expert.age[i] * dt:5.1f} s  '{lay.instruction}'")  # fmt: skip
                if expert.notes[i]:
                    print(f"         {expert.notes[i]}")
        wall = time.perf_counter() - t0
        counts = collections.Counter(r["label"] for r in results)
        ok = counts.get("success", 0)
        lo, hi = wilson_interval(ok, len(results))
        print(f"[expert] success {ok}/{len(results)} ({ok / len(results):.0%}, 95% CI "
              f"{lo:.0%}-{hi:.0%}); labels {dict(counts)}; {wall:.0f} s wall, "
              f"{sim_steps * n * dt / wall:.1f}x real time summed over robots")  # fmt: skip
        args.out.parent.mkdir(parents=True, exist_ok=True)
        provenance.save_json(args.out, {
            "controller_run": str(args.run), "first_seed": args.first_seed,
            "episodes": len(results), "success": ok, "success_ci95": [lo, hi],
            "labels": dict(counts), "results": results,
            "expert_params": {k: (vars(v) if hasattr(v, "__dataclass_fields__") else v)
                              for k, v in vars(expert.p).items()},
        })  # fmt: skip
        if frames:
            from PIL import Image

            imgs = [Image.fromarray(f) for f in frames]
            args.gif.parent.mkdir(parents=True, exist_ok=True)
            imgs[0].save(args.gif, save_all=True, append_images=imgs[1:], duration=60, loop=0)
            print(f"[expert] wrote {args.gif}")
        env.close()


if __name__ == "__main__":
    main()

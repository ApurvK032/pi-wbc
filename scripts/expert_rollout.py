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
parser.add_argument(
    "--seed-list",
    type=str,
    default=None,
    help="comma-separated layout seeds to run instead of consecutive ones (sets --episodes)",
)
parser.add_argument("--run", type=Path, default=Path(DEFAULT_RUN), help="controller run folder")
parser.add_argument("--gif", type=Path, default=None)
parser.add_argument("--gif-env", type=int, default=0, help="which robot of the first batch")
parser.add_argument(
    "--fail-gifs",
    type=Path,
    default=None,
    help="save the last 9 s of every failed episode as a GIF in this folder",
)
parser.add_argument(
    "--cam",
    choices=["side", "front", "hero", "herd"],
    default="side",
    help="side: the robot's right, approach and table; front: across the table, at the hand; "
    "hero: across the table, robot, hand and both bins (640 x 400, for the front page)",
)
parser.add_argument(
    "--herd",
    action="store_true",
    help="with --gif: one high camera in the first robot's world that sees every robot "
    "(pi_wbc.stage3.herd), robots --herd-spacing apart; for front-page pictures",
)
parser.add_argument("--herd-spacing", type=float, default=5.0, help="metres between robots")
parser.add_argument(
    "--herd-eye",
    default="6,-22,8",
    help="camera position relative to the robots' centre (x toward the tables, z up)",
)
parser.add_argument("--herd-target", default="4,-2,0", help="where it looks, same frame")
parser.add_argument(
    "--param",
    action="append",
    default=[],
    metavar="NAME=VALUE",
    help="override an ExpertParams field (number, comma list or None), e.g. lean=0.3",
)
parser.add_argument(
    "--record",
    type=Path,
    default=None,
    metavar="DIR",
    help="record every episode (commands and simulator state per step) as DIR/<seed>.npz",
)
parser.add_argument(
    "--low-level",
    choices=["ours", "nvidia"],
    default="ours",
    help="the controller under the expert's commands: stage 1's (--run) or NVIDIA's Decoupled "
    "WBC (pi_wbc.stage3.nvidia_wbc, D-046)",
)
parser.add_argument("--nvidia-policy-dir", type=Path, default=Path("third_party/gr00t_wbc/policy"))
parser.add_argument("--out", type=Path, required=True)
add_launcher_args(parser)
args, overrides = parser.parse_known_args()

import collections  # noqa: E402
import dataclasses  # noqa: E402
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
from pi_wbc.stage3.expert import PHASES, Expert, ExpertParams  # noqa: E402
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


CAMERAS = {  # (eye, target) in the environment's frame; the table's front edge is at x = 0
    "side": ((-1.6, -2.4, 1.6), (-0.6, 0.0, 0.8)),  # robot's right side, approach and table front
    "front": ((0.75, -0.15, 1.15), (0.12, -0.10, 0.82)),  # across the table, at the hand
    "hero": ((0.75, -1.15, 1.35), (-0.45, 0.05, 0.9)),  # across the table, front right
    "herd": ((0.75, -1.15, 1.35), (-0.45, 0.05, 0.9)),  # moved over all robots by --herd
}


def side_camera(small: bool = False, cam: str = "side"):
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import CameraCfg
    from isaaclab_newton.renderers import NewtonWarpRendererCfg

    eye, target = CAMERAS[cam]
    return CameraCfg(
        prim_path="{ENV_REGEX_NS}/debug_cam",
        update_period=0.0,
        width=320 if small else {"hero": 640, "herd": 960}.get(cam, 480),
        height=214 if small else {"hero": 400, "herd": 540}.get(cam, 320),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0, clipping_range=(0.05, 150.0 if cam == "herd" else 30.0)
        ),
        offset=CameraCfg.OffsetCfg(pos=eye, rot=look_at(eye, target), convention="world"),
        renderer_cfg=NewtonWarpRendererCfg(),
    )


def aim_herd_camera(base, eye_offset, target_offset) -> None:
    """Point the first robot's camera over every robot, and let it see them all."""
    from pi_wbc.stage3.herd import see_all_worlds

    cam = base.scene["debug_cam"]
    see_all_worlds(cam._renderer.newton_sensor.model)
    origins = base.scene.env_origins
    centre = origins.mean(dim=0) + torch.tensor((-0.8, 0.0, 0.0), device=origins.device)
    eye = (centre + torch.tensor(eye_offset, device=origins.device)).tolist()
    target = (centre + torch.tensor(target_offset, device=origins.device)).tolist()
    pos = torch.tensor([eye], device=origins.device)
    rot = torch.tensor([look_at(eye, target)], device=origins.device)
    cam.set_world_poses(pos, rot, env_ids=[0], convention="world")


def save_gif(frames, path: Path) -> None:
    from PIL import Image

    if not frames:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=60, loop=0)


def main() -> None:
    if args.herd:  # unused objects stand 6 m from their table: in the next robot's patch here
        import pi_wbc.tasks.g1_pick_place as task_module

        task_module.PARK = (-400.0, 400.0)
    env_cfg, agent_cfg = resolve_task_config(
        TASK, "rsl_rl_cfg_entry_point", overrides=[*overrides, "physics=newton_mjwarp"]
    )
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, check_rsl_rl_version())
    seed_list = [int(s) for s in args.seed_list.split(",")] if args.seed_list else None
    if seed_list is not None:
        args.episodes = len(seed_list)
    n = min(args.num_envs, args.episodes)
    env_cfg.scene.num_envs = n
    if args.gif is not None or args.fail_gifs is not None:
        cam = "herd" if args.herd else args.cam
        env_cfg.scene.debug_cam = side_camera(small=args.fail_gifs is not None, cam=cam)
    if args.herd:
        env_cfg.scene.env_spacing = args.herd_spacing
    with launch_simulation(env_cfg, args):
        env = gym.make(TASK, cfg=env_cfg)
        venv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = create_rsl_rl_runner(venv, agent_cfg)
        runner.load(str((args.run / "model_2999.pt").resolve()))
        policy = runner.get_inference_policy(device=venv.unwrapped.device)
        base = venv.unwrapped
        hl = HighLevel(env)
        params = {}
        for item in args.param:
            name, _, value = item.partition("=")
            if value == "None":
                params[name] = None
            elif "," in value:  # a tuple, e.g. stance_rel=0.36,-0.18
                params[name] = tuple(float(v) for v in value.split(","))
            else:
                params[name] = float(value)
        expert_params = dataclasses.replace(ExpertParams(), **params)
        expert = Expert(env, hl, args.run / "exported", expert_params)
        nv = None
        if args.low_level == "nvidia":
            from pi_wbc.stage3.nvidia_wbc import NvidiaLowLevel

            nv = NvidiaLowLevel(env, args.nvidia_policy_dir)
        rec = None
        if args.record is not None:
            from pi_wbc.stage3.recorder import EpisodeRecorder

            run_info = {"controller_run": str(args.run),
                        "expert_params": dataclasses.asdict(expert_params)}  # fmt: skip
            if nv is not None:
                run_info["low_level"] = {"name": "nvidia", "policy_commit": nv.policy_commit,
                                         "height_offset": nv.height_offset}  # fmt: skip
            rec = EpisodeRecorder(env, hl, expert, args.record, run_info)
        dt = base.step_dt
        results, frames = [], []
        seed = args.first_seed
        t0 = time.perf_counter()
        sim_steps = 0
        while len(results) < args.episodes:
            m = min(n, args.episodes - len(results))
            if seed_list is not None:  # the next n of the list (the last batch is padded)
                done = len(results)
                seeds = (seed_list[done : done + n] + seed_list[:n])[:n]
            else:
                seeds = [seed + i for i in range(n)]
            seed += n
            base.pick_place_layouts = [L.sample(s) for s in seeds]
            base.reset()
            if args.herd and not results:
                aim_herd_camera(base, [float(v) for v in args.herd_eye.split(",")],
                                [float(v) for v in args.herd_target.split(",")])  # fmt: skip
            expert.reset(list(range(n)))
            if nv is not None:
                nv.reset()
            if rec is not None:
                rec.reset()
            in_batch = torch.arange(n).numpy() < m
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
                    if rec is not None:  # this step's state and the commands applied to it
                        rec.record(in_batch & ~expert.finished())
                    obs, _, _, _ = venv.step(policy(obs) if nv is None else nv.act())
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
                if rec is not None:
                    rec.save(i, {"label": label, "failed_in": expert.failed_in[i],
                                 "time_s": round(expert.age[i] * dt, 2),
                                 "hand": expert.side[i]})  # fmt: skip
        wall = time.perf_counter() - t0
        counts = collections.Counter(r["label"] for r in results)
        ok = counts.get("success", 0)
        lo, hi = wilson_interval(ok, len(results))
        print(f"[expert] success {ok}/{len(results)} ({ok / len(results):.0%}, 95% CI "
              f"{lo:.0%}-{hi:.0%}); labels {dict(counts)}; {wall:.0f} s wall, "
              f"{sim_steps * n * dt / wall:.1f}x real time summed over robots")  # fmt: skip
        args.out.parent.mkdir(parents=True, exist_ok=True)
        provenance.save_json(args.out, {
            "controller_run": str(args.run), "low_level": args.low_level,
            "nvidia_policy_commit": None if nv is None else nv.policy_commit,
            "first_seed": args.first_seed,
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

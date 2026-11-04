"""Replay recorded expert episodes without the expert (STAGE3_DESIGN.md section 8, check 2).

Each episode's layout is reset as recorded, and its recorded commands go step by step through
the same path the expert's did (`HighLevel` -> the stage 1 controller -> the physics). Newton
on the GPU does not repeat a run bit for bit (D-041: identical expert runs end differently in
about 8% of episodes), so the check measures how far each replay drifts from its recording
(pelvis and target object, every step) and whether it ends the same way (the expert's placement
judge, and no other object pushed more than 2 cm), instead of demanding equality.

An episode passes when its replay stays within `--max-early-drift-mm` of the recording over the
first second: a recording whose commands and states don't belong together drifts at once (17-124
mm in a test with mismatched recordings, against at most 2.7 mm for matching ones). Whether the
replay ends the same way is reported over the whole set: the expert corrects small differences
as it goes, a replay of its commands can't, so a few millimetres can become a missed grasp.

    uv run python scripts/replay_check.py --episodes logs/dataset/rec --out logs/replay/rec.json \\
        --viz none
"""

import argparse
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation

DEFAULT_RUN = "logs/rsl_rl/pi_wbc_g1_student/slurm-5628865-full-range-long-r"
parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--episodes", type=Path, required=True, help="folder of recorded .npz files")
parser.add_argument("--num-envs", type=int, default=32)
parser.add_argument("--limit", type=int, default=None, help="replay only the first N episodes")
parser.add_argument("--run", type=Path, default=Path(DEFAULT_RUN), help="controller run folder")
parser.add_argument("--other-knock-dist", type=float, default=0.02)
parser.add_argument("--max-early-drift-mm", type=float, default=10.0)
parser.add_argument("--low-level", choices=["ours", "nvidia"], default="ours",
                    help="the controller the episodes were recorded with (D-046)")  # fmt: skip
parser.add_argument("--nvidia-policy-dir", type=Path, default=Path("third_party/gr00t_wbc/policy"))
parser.add_argument("--out", type=Path, required=True)
add_launcher_args(parser)
args, overrides = parser.parse_known_args()

import collections  # noqa: E402
import json  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
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
from pi_wbc.stage3.expert import placement_label  # noqa: E402
from pi_wbc.stats import wilson_interval  # noqa: E402
from pi_wbc.tasks.g1_pick_place import HighLevel  # noqa: E402

TASK = "PiWbc-G1-PickPlace"


def tt(x):
    return x.torch if hasattr(x, "torch") else x


def _tuples(v):
    return tuple(_tuples(x) for x in v) if isinstance(v, list) else v


def load(path: Path) -> dict:
    d = np.load(path)
    meta = json.loads(str(d["meta"]))
    ep = {k: d[k] for k in d.files if k != "meta"}
    ep["meta"] = meta
    ep["layout"] = L.Layout(**{k: _tuples(v) for k, v in meta["layout"].items()})
    return ep


def main() -> None:
    files = sorted(args.episodes.glob("*.npz"), key=lambda p: int(p.stem))[: args.limit]
    if not files:
        raise SystemExit(f"no episodes in {args.episodes}")
    env_cfg, agent_cfg = resolve_task_config(
        TASK, "rsl_rl_cfg_entry_point", overrides=[*overrides, "physics=newton_mjwarp"]
    )
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, check_rsl_rl_version())
    n = min(args.num_envs, len(files))
    env_cfg.scene.num_envs = n
    with launch_simulation(env_cfg, args):
        env = gym.make(TASK, cfg=env_cfg)
        venv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = create_rsl_rl_runner(venv, agent_cfg)
        runner.load(str((args.run / "model_2999.pt").resolve()))
        policy = runner.get_inference_policy(device=venv.unwrapped.device)
        base = venv.unwrapped
        hl = HighLevel(env)
        nv = None
        if args.low_level == "nvidia":
            from pi_wbc.stage3.nvidia_wbc import NvidiaLowLevel

            nv = NvidiaLowLevel(env, args.nvidia_policy_dir)
        robot = base.scene["robot"]
        objects = [base.scene[f"obj_{name}"] for name in L.OBJECT_NAMES]
        results = []
        for start in range(0, len(files), n):
            batch = [load(f) for f in files[start : start + n]]
            m = len(batch)
            eps = batch + [batch[0]] * (n - m)  # unused environments repeat the first episode
            base.pick_place_layouts = [ep["layout"] for ep in eps]
            base.reset()
            hl.reset(list(range(n)))
            if nv is not None:
                nv.reset()
            obs = venv.get_observations()
            steps = np.array([len(ep["phase"]) for ep in eps])
            drift_root = np.zeros(n)
            early = np.zeros(n)  # largest pelvis or target-object drift over the first second
            curve = [[] for _ in range(n)]  # pelvis drift (cm) every 50 steps
            start_gap = [None] * n  # joints (rad) and pelvis (cm) against the recording, at t=0
            drift_obj = np.zeros(n)
            final = [None] * n
            pushed = np.zeros(n)
            start_obj = [ep["object_pose"][0, :, :3] for ep in eps]
            targets = [L.OBJECT_NAMES.index(ep["layout"].target_object) for ep in eps]
            others = [
                [k for k, name in enumerate(L.OBJECT_NAMES)
                 if name in ep["layout"].objects and name != ep["layout"].target_object]
                for ep in eps
            ]  # fmt: skip
            with torch.no_grad():
                for t in range(int(steps.max()) + 1):
                    origins = base.scene.env_origins
                    root = (tt(robot.data.root_pos_w) - origins).cpu().numpy()
                    every = torch.stack([tt(o.data.root_pos_w) for o in objects], dim=1)
                    every = (every - origins[:, None]).cpu().numpy()
                    for i, ep in enumerate(eps):
                        k = min(t, steps[i] - 1)
                        if t < steps[i]:  # the state each recorded command was applied to
                            rec_root = ep["root_pose"][t, :3]
                            rec_obj = ep["object_pose"][t, targets[i], :3]
                            gap = np.linalg.norm(root[i, :2] - rec_root[:2])
                            drift_root[i] = max(drift_root[i], gap)
                            if t <= 50:  # a broken or misaligned episode drifts at once
                                obj_gap = np.linalg.norm(every[i, targets[i]] - rec_obj)
                                early[i] = max(early[i], gap, obj_gap)
                            if t % 50 == 0:
                                curve[i].append(round(float(gap) * 100, 1))
                            if t in (0, 1):
                                q = tt(robot.data.joint_pos)[i].cpu().numpy()
                                dq = float(np.abs(q - ep["joint_pos"][t]).max())
                                dz = float(np.linalg.norm(root[i] - ep["root_pose"][t, :3]))
                                start_gap[i] = (start_gap[i] or []) + [
                                    round(dq, 4),
                                    round(dz * 100, 2),
                                ]
                            drift_obj[i] = max(
                                drift_obj[i], np.linalg.norm(every[i, targets[i]] - rec_obj)
                            )
                        if others[i]:
                            moved = np.linalg.norm(
                                every[i, others[i], :2] - start_obj[i][others[i], :2], axis=1
                            )
                            if t <= steps[i]:
                                pushed[i] = max(pushed[i], moved.max())
                        if t == steps[i]:  # where the recording ends
                            final[i] = every[i, targets[i]].copy()
                        hl.velocity[i] = torch.as_tensor(ep["cmd_velocity"][k], device=base.device)
                        hl.body_goal[i] = torch.as_tensor(ep["cmd_body"][k], device=base.device)
                        hl.arm_hand[i] = torch.as_tensor(ep["cmd_arm_hand"][k], device=base.device)
                    hl.apply()
                    obs, _, _, _ = venv.step(policy(obs) if nv is None else nv.act())
            for i, ep in enumerate(batch):
                meta, lay = ep["meta"], ep["layout"]
                label = placement_label(lay, final[i])
                if label == "success" and pushed[i] > args.other_knock_dist:
                    label = "knocked_other"
                rec_label = meta["outcome"]["label"]
                rec_final = ep["object_pose"][-1, targets[i], :3]
                gap = float(np.linalg.norm(final[i] - rec_final))
                results.append({
                    "seed": lay.seed, "recorded": rec_label, "replayed": label,
                    "same_outcome": (rec_label == "success") == (label == "success"),
                    "steps": int(steps[i]), "hand": meta["outcome"].get("hand"),
                    "max_root_drift_cm": round(float(drift_root[i]) * 100, 1),
                    "max_object_drift_cm": round(float(drift_obj[i]) * 100, 1),
                    "final_object_gap_cm": round(gap * 100, 1),
                    "other_pushed_cm": round(float(pushed[i]) * 100, 1),
                    "first_second_drift_mm": round(float(early[i]) * 1000, 2),
                    "passes": bool(early[i] * 1000 < args.max_early_drift_mm),
                    "root_drift_cm_every_50": curve[i],
                    "start_gap": start_gap[i],
                })  # fmt: skip
                r = results[-1]
                print(f"[replay] seed {r['seed']}: recorded {rec_label:13s} replayed {label:13s} "
                      f"root drift {r['max_root_drift_cm']:5.1f} cm, object drift "
                      f"{r['max_object_drift_cm']:5.1f} cm")  # fmt: skip
        passed = sum(r["passes"] for r in results)
        print(f"[replay] {passed}/{len(results)} episodes pass (first-second drift under "
              f"{args.max_early_drift_mm:g} mm)")  # fmt: skip
        ok = [r for r in results if r["recorded"] == "success"]
        kept = sum(r["replayed"] == "success" for r in ok)
        lo, hi = wilson_interval(kept, len(ok)) if ok else (0.0, 0.0)
        drift = np.array([r["max_object_drift_cm"] for r in ok]) if ok else np.zeros(1)
        print(f"[replay] {kept}/{len(ok)} recorded successes also succeed on replay "
              f"({100 * lo:.0f}-{100 * hi:.0f}%); object drift median {np.median(drift):.1f} cm, "
              f"90th percentile {np.percentile(drift, 90):.1f} cm; labels "
              f"{dict(collections.Counter(r['replayed'] for r in results))}")  # fmt: skip
        args.out.parent.mkdir(parents=True, exist_ok=True)
        provenance.save_json(args.out, {
            "episodes_dir": str(args.episodes), "controller_run": str(args.run),
            "low_level": args.low_level,
            "replayed": len(results), "passed": passed,
            "max_early_drift_mm": args.max_early_drift_mm, "recorded_successes": len(ok),
            "successes_kept": kept, "kept_ci95": [lo, hi], "results": results,
        })  # fmt: skip
        env.close()


if __name__ == "__main__":
    main()

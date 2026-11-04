"""Measure how well the head camera sees the grasp, for several lens widths (D-044).

For every recorded episode and every frame (each second control step), the recorded robot state
is written into the pick-place scene (no physics) to get the torso and palm poses; the head
camera's pose follows as in rendering (`pi_wbc.stage3.appearance.camera_poses`, with the
episode's mount error). The target object's centre and the working palm's origin are then
projected into a pinhole camera of each horizontal field of view in `--fov` (4:3 image), and
the script reports how often each lies inside the image, by phase of the expert.

    uv run python scripts/camera_view_check.py --episodes logs/dataset/train-v1 \\
        --seeds results/stage3/dataset/train-v1-manifest.json --out logs/render/view_check.json \\
        --viz none
"""

import argparse
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--episodes", type=Path, required=True)
parser.add_argument("--seeds", type=Path, default=None)
parser.add_argument("--limit", type=int, default=None)
parser.add_argument("--num-envs", type=int, default=64)
parser.add_argument("--fov", default="60.4,69.4,80,90", help="horizontal fields of view (deg)")
parser.add_argument("--margin-deg", type=float, default=2.0, help="inside by at least this much")
parser.add_argument("--out", type=Path, required=True)
add_launcher_args(parser)
args, overrides = parser.parse_known_args()

import json  # noqa: E402
import math  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils import instantiate  # noqa: E402
from isaaclab.utils.math import quat_apply_inverse  # noqa: E402
from isaaclab_tasks.utils import resolve_task_config  # noqa: E402

import pi_wbc.tasks  # noqa: E402, F401
from pi_wbc import provenance  # noqa: E402
from pi_wbc.stage3 import appearance as A  # noqa: E402
from pi_wbc.stage3 import layout as L  # noqa: E402
from pi_wbc.stage3.expert import PHASES  # noqa: E402

FRAME_STEP = 2
GROUPS = {  # phases of the expert, grouped
    "walk to the object": ("walk_obj", "lean_obj", "align_obj", "settle_obj"),
    "reach and grasp": ("raise", "reach", "fine_pre", "lower", "fine_out", "slide",
                        "fine_grasp", "close", "lift"),
    "carry": ("straighten", "walk_bin", "lean_bin", "align_bin", "settle_bin"),
    "place": ("reach_bin", "release", "retract"),
}  # fmt: skip


def tt(x):
    return x.torch if hasattr(x, "torch") else x


def padded(arrays: list[np.ndarray], steps: int, device) -> torch.Tensor:
    """Stack per-episode arrays, the shorter ones held at their last entry."""
    out = [np.concatenate([a, np.repeat(a[-1:], steps - len(a), axis=0)]) for a in arrays]
    return torch.as_tensor(np.stack(out), device=device)


def main() -> None:
    files = {int(p.stem): p for p in args.episodes.glob("*.npz")}
    seeds = json.loads(args.seeds.read_text())["chosen_seeds"] if args.seeds else sorted(files)
    seeds = seeds[: args.limit]
    fovs = [float(f) for f in args.fov.split(",")]
    half = torch.tensor([[math.radians(f) / 2 for f in fovs]])
    half_v = torch.atan(0.75 * torch.tan(half))
    margin = math.radians(args.margin_deg)
    names_g = list(GROUPS)
    group_of = np.full(len(PHASES) + 1, -1)  # phase index -> group index; -1: none or ended
    for gi, g in enumerate(names_g):
        for ph in GROUPS[g]:
            group_of[PHASES.index(ph)] = gi
    # frames and frames inside, per group, thing ("object", "palm") and field of view
    frames = np.zeros(len(names_g), dtype=np.int64)
    inside_n = np.zeros((len(names_g), 2, len(fovs)), dtype=np.int64)
    per_episode = []

    env_cfg, _ = resolve_task_config(
        "PiWbc-G1-PickPlace", "rsl_rl_cfg_entry_point",
        overrides=[*overrides, "physics=newton_mjwarp"],
    )  # fmt: skip
    n = min(args.num_envs, len(seeds))
    env_cfg.scene.num_envs = n
    with launch_simulation(env_cfg, args):
        sim = sim_utils.SimulationContext(env_cfg.sim)
        scene = instantiate(env_cfg.scene)
        sim.reset()
        dev = sim.device
        half, half_v = half.to(dev), half_v.to(dev)
        robot = scene["robot"]
        origins = scene.env_origins
        torso = robot.body_names.index("torso_link")
        palm = {s: robot.body_names.index(f"{s}_hand_palm_link") for s in ("left", "right")}
        for start in range(0, len(seeds), n):
            batch = seeds[start : start + n]
            m = len(batch)
            eps, metas = [], []
            for sd in batch:
                d = np.load(files[sd])
                metas.append(json.loads(str(d["meta"])))
                eps.append({k: d[k] for k in ("root_pose", "joint_pos", "object_pose", "phase")})
            eps += [eps[0]] * (n - m)
            metas += [metas[0]] * (n - m)
            names = metas[0]["joint_names"]
            jidx = torch.tensor([robot.joint_names.index(j) for j in names], device=dev)
            looks = [A.sample_look(s) for s in batch] + [A.sample_look(batch[0])] * (n - m)
            shift = torch.tensor([x.cam_shift for x in looks], device=dev)
            tilt = torch.tensor([x.cam_tilt for x in looks], device=dev)
            target = [L.OBJECT_NAMES.index(md["layout"]["target_object"]) for md in metas]
            hand_body = torch.tensor([palm[md["outcome"]["hand"]] for md in metas], device=dev)
            steps = np.array([len(ep["phase"]) for ep in eps])
            T = int(steps.max())
            root_all = padded([ep["root_pose"] for ep in eps], T, dev)
            joint_all = padded([ep["joint_pos"] for ep in eps], T, dev)
            obj_all = padded([ep["object_pose"][:, target[j], :3] for j, ep in enumerate(eps)], T,
                             dev)  # fmt: skip
            # group of every env and frame; -1 after an episode ends
            phase = np.full((n, T), len(PHASES), dtype=np.int64)
            for j, ep in enumerate(eps[:m]):
                phase[j, : steps[j]] = ep["phase"]
            groups = group_of[phase[:, ::FRAME_STEP]]  # (n, frames)
            groups[m:] = -1
            ins = []  # per frame: (n, 2, fovs) bool
            q = torch.zeros(n, robot.num_joints, device=dev)
            arange = torch.arange(n, device=dev)
            for t in range(0, T, FRAME_STEP):
                root = root_all[:, t].clone()
                root[:, :3] += origins
                robot.write_root_pose_to_sim_index(root_pose=root)
                q[:, jidx] = joint_all[:, t]
                robot.write_joint_position_to_sim_index(position=q)
                sim.forward()
                robot.update(0.0)
                body_p = tt(robot.data.body_pos_w)
                body_q = tt(robot.data.body_quat_w)
                cam_p, cam_q = A.camera_poses(body_p[:, torso], body_q[:, torso], shift, tilt)
                pts = torch.stack([obj_all[:, t] + origins, body_p[arange, hand_body]], dim=1)
                local = quat_apply_inverse(cam_q[:, None].expand(-1, 2, -1).reshape(-1, 4),
                                           (pts - cam_p[:, None]).reshape(-1, 3))  # fmt: skip
                local = local.reshape(n, 2, 3)  # x forward, y left, z up
                h = torch.atan2(-local[..., 1], local[..., 0])[..., None]
                v = torch.atan2(local[..., 2], local[..., 0])[..., None]
                ins.append((local[..., :1] > 0) & (h.abs() < half - margin)
                           & (v.abs() < half_v - margin))  # fmt: skip
            ins = torch.stack(ins, dim=1).cpu().numpy()  # (n, frames, 2, fovs)
            for gi in range(len(names_g)):
                sel = groups == gi
                frames[gi] += int(sel.sum())
                inside_n[gi] += ins[sel].sum(axis=0)
            grasp = names_g.index("reach and grasp")
            for j, sd in enumerate(batch):
                sel = groups[j] == grasp
                share = ins[j, sel, 0].mean(axis=0) if sel.any() else np.zeros(len(fovs))
                per_episode.append({
                    "seed": sd, "hand": metas[j]["outcome"]["hand"],
                    "grasp_object_inside": [round(float(x), 3) for x in share],
                })  # fmt: skip
            print(f"[view] {start + m}/{len(seeds)} episodes", flush=True)
        things = ("object", "palm")
        counts = {g: {t: [int(frames[gi]), inside_n[gi, ti]] for ti, t in enumerate(things)}
                  for gi, g in enumerate(names_g)}  # fmt: skip
        table = {g: {t: {"frames": c[0], "inside": {f"{f:g}": round(float(x / max(c[0], 1)), 4)
                                                    for f, x in zip(fovs, c[1], strict=True)}}
                     for t, c in v.items()} for g, v in counts.items()}  # fmt: skip
        for g, v in table.items():
            for thing, r in v.items():
                print(f"[view] {g:20s} {thing:6s} " + "  ".join(
                    f"{f} deg: {100 * x:5.1f}%" for f, x in r["inside"].items()))  # fmt: skip
        for hand in ("right", "left"):
            eps = [e for e in per_episode if e["hand"] == hand]
            for i, f in enumerate(fovs):
                most = sum(e["grasp_object_inside"][i] >= 0.9 for e in eps)
                print(f"[view] {hand} hand, {f:g} deg: target inside for 90% of the reach and "
                      f"grasp in {most}/{len(eps)} episodes")  # fmt: skip
        args.out.parent.mkdir(parents=True, exist_ok=True)
        provenance.save_json(args.out, {
            "episodes_dir": str(args.episodes), "fov_deg": fovs, "margin_deg": args.margin_deg,
            "groups": GROUPS, "by_phase": table, "episodes": per_episode,
        })  # fmt: skip


if __name__ == "__main__":
    main()

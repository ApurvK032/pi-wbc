"""Render recorded episodes from the head camera (STAGE3_DESIGN.md section 8, D-043).

Each episode's recorded states (robot root pose, joint positions, object poses) are written into
a render-only copy of the pick-place scene, without physics steps, every second control step:
25 frames per second, frame k showing the state that control step 2k's command was applied to.
Each episode gets its look from its layout seed (`pi_wbc.stage3.appearance`): floor, table and
wall colours and patterns, room size, sun height and direction, light brightness and colour,
camera mount error. Episodes with the same sun height are rendered together, since the sun is
shared by all environments. Output: one H.264 video per episode (system ffmpeg, libx264,
yuv420p), and a manifest with each episode's look, frame count and file hash.

    uv run python scripts/render_episodes.py --episodes logs/dataset/train-v1 \\
        --seeds results/stage3/dataset/train-v1-manifest.json --out logs/render/train-v1 --viz none
"""

import argparse
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--episodes", type=Path, required=True, help="folder of recorded .npz files")
parser.add_argument(
    "--seeds", type=Path, default=None, help="dataset manifest whose chosen_seeds to render"
)
parser.add_argument("--limit", type=int, default=None, help="render only the first N episodes")
parser.add_argument("--num-envs", type=int, default=32)
parser.add_argument("--render-seed", type=int, default=0, help="selects another set of looks")
parser.add_argument("--crf", type=int, default=20, help="libx264 quality (lower is better)")
parser.add_argument("--gop", type=int, default=25, help="frames between key frames")
parser.add_argument("--no-shadows", action="store_true")
parser.add_argument("--sheets", type=int, default=0,
                    help="save a picture of 6 frames for each of the first N episodes")  # fmt: skip
parser.add_argument("--out", type=Path, required=True)
add_launcher_args(parser)
args, overrides = parser.parse_known_args()

import collections  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils import instantiate  # noqa: E402
from isaaclab_newton.renderers import NewtonWarpRendererCfg  # noqa: E402
from isaaclab_tasks.utils import resolve_task_config  # noqa: E402
from PIL import Image  # noqa: E402

import pi_wbc.tasks  # noqa: E402, F401
from pi_wbc import provenance  # noqa: E402
from pi_wbc.stage3 import appearance as A  # noqa: E402
from pi_wbc.stage3 import layout as L  # noqa: E402
from pi_wbc.tasks.g1_pick_place import BIN_PARTS  # noqa: E402

TASK = "PiWbc-G1-PickPlace"
FRAME_STEP = 2  # control steps per frame: 50 Hz control, 25 frames per second
FPS = 25


def tt(x):
    return x.torch if hasattr(x, "torch") else x


def _tuples(v):
    return tuple(_tuples(x) for x in v) if isinstance(v, list) else v


def load(path: Path) -> dict:
    d = np.load(path)
    meta = json.loads(str(d["meta"]))
    ep = {k: d[k] for k in ("root_pose", "joint_pos", "object_pose")}
    ep["steps"] = len(d["phase"])
    ep["meta"] = meta
    ep["layout"] = L.Layout(**{k: _tuples(v) for k, v in meta["layout"].items()})
    return ep


def ffmpeg(path: Path) -> subprocess.Popen:
    w, h = A.HEAD_CAM_SIZE
    cmd = ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "veryfast",
           "-crf", str(args.crf), "-g", str(args.gop), "-pix_fmt", "yuv420p", "-threads", "1",
           "-movflags", "+faststart", str(path)]  # fmt: skip
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def frame_count(path: Path) -> int:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=nb_frames",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout  # fmt: skip
    return int(out.strip())


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def place(asset, pose_env, yaw, origins) -> None:
    """Write kinematic poses (n, 7), given in env frames before the scene turn."""
    pose = A.turn_poses(pose_env, yaw)
    pose[:, :3] += origins
    asset.write_root_pose_to_sim_index(root_pose=pose)


def padded(eps: list[dict], key: str, steps: int, device) -> torch.Tensor:
    """One recorded field of every episode, the shorter ones held at their last state."""
    out = []
    for ep in eps:
        a = ep[key]
        out.append(np.concatenate([a, np.repeat(a[-1:], steps - len(a), axis=0)]))
    return torch.as_tensor(np.stack(out), device=device)


def main() -> None:
    files = {int(p.stem): p for p in args.episodes.glob("*.npz")}
    if args.seeds is not None:
        seeds = json.loads(args.seeds.read_text())["chosen_seeds"]
        missing = [s for s in seeds if s not in files]
        if missing:
            raise SystemExit(f"{len(missing)} chosen seeds have no recording, e.g. {missing[:5]}")
    else:
        seeds = sorted(files)
    seeds = seeds[: args.limit]
    looks = {s: A.sample_look(s, args.render_seed) for s in seeds}
    steps = {s: len(np.load(files[s])["phase"]) for s in seeds}
    # batches: same sun height together, longest first, so a batch's episodes end close together
    batches = []
    for e in range(len(A.SUN_ELEVATIONS_DEG)):
        group = sorted((s for s in seeds if looks[s].sun_elevation == e), key=lambda s: -steps[s])
        batches += [(e, group[i : i + args.num_envs]) for i in range(0, len(group), args.num_envs)]
    n = min(args.num_envs, max(len(b) for _, b in batches))
    videos = args.out / "videos"
    videos.mkdir(parents=True, exist_ok=True)
    sheet_seeds = set(seeds[: args.sheets])

    env_cfg, _ = resolve_task_config(
        TASK, "rsl_rl_cfg_entry_point", overrides=[*overrides, "physics=newton_mjwarp"]
    )
    env_cfg.scene.num_envs = n
    A.add_look_assets(env_cfg.scene)
    env_cfg.scene.head_cam = A.head_camera_cfg(
        NewtonWarpRendererCfg(enable_shadows=not args.no_shadows)
    )
    results = []
    with launch_simulation(env_cfg, args):
        sim = sim_utils.SimulationContext(env_cfg.sim)
        scene = instantiate(env_cfg.scene)
        sim.reset()
        dev = sim.device
        robot, cam = scene["robot"], scene["head_cam"]
        objects = [scene[f"obj_{name}"] for name in L.OBJECT_NAMES]
        torso = robot.body_names.index("torso_link")
        origins = scene.env_origins
        looks_out = A.LookRenderer(cam, n)
        looks_out.install_textures()
        unit = torch.tensor([0.0, 0.0, 0.0, 1.0], device=dev)
        t_start = time.perf_counter()
        frames_done = 0
        for b, (elev, batch_seeds) in enumerate(batches):
            t_batch = time.perf_counter()
            eps = [load(files[s]) for s in batch_seeds]
            m = len(eps)
            eps += [eps[0]] * (n - m)  # unused environments repeat the first episode, unsaved
            lk = [looks[s] for s in batch_seeds] + [looks[batch_seeds[0]]] * (n - m)
            looks_out.set_looks(lk)
            looks_out.set_sun(A.SUN_ELEVATIONS_DEG[elev])
            yaw = torch.tensor([x.scene_yaw for x in lk], device=dev)

            def at(xyz):  # (n, 3) positions with unit quaternions -> (n, 7)
                return torch.cat([torch.as_tensor(xyz, device=dev, dtype=torch.float32),
                                  unit.repeat(n, 1)], dim=-1)  # fmt: skip

            table = (L.TABLE_CENTER[0], L.TABLE_CENTER[1], L.TABLE_TOP / 2)
            place(scene["table"], at([table] * n), yaw, origins)
            floor = (*A.FLOOR_CENTER, A.FLOOR_TOP - A.FLOOR_SIZE[2] / 2)
            place(scene["look_floor"], at([floor] * n), yaw, origins)
            for wall in ("wall_back", "wall_left", "wall_right"):
                place(scene[f"look_{wall}"], at([A.wall_poses(x)[wall] for x in lk]), yaw, origins)
            for bin_name in L.BIN_NAMES:
                centre = np.array([(*ep["layout"].bin_xy(bin_name), L.TABLE_TOP) for ep in eps])
                for part, (_, offset) in BIN_PARTS.items():
                    place(scene[f"{bin_name}_{part}"], at(centre + np.array(offset)), yaw, origins)

            names = eps[0]["meta"]["joint_names"]
            jidx = torch.tensor([robot.joint_names.index(j) for j in names], device=dev)
            T = max(ep["steps"] for ep in eps)

            root_all, joint_all, obj_all = (
                padded(eps, k, T, dev) for k in ("root_pose", "joint_pos", "object_pose")
            )
            ep_steps = np.array([ep["steps"] for ep in eps])
            shift = torch.tensor([x.cam_shift for x in lk], device=dev)
            tilt = torch.tensor([x.cam_tilt for x in lk], device=dev)
            gain = torch.tensor([x.gain for x in lk], device=dev)
            tint = torch.tensor([x.tint for x in lk], device=dev)
            paths = [videos / f"{s}.mp4" for s in batch_seeds]
            procs = [ffmpeg(p) for p in paths]
            sheet = {i: [] for i, s in enumerate(batch_seeds) if s in sheet_seeds}
            sheet_at = {i: set(np.linspace(0, (ep_steps[i] - 1) // FRAME_STEP, 6).astype(int))
                        for i in sheet}  # fmt: skip
            q = torch.zeros(n, robot.num_joints, device=dev)
            zeros6 = torch.zeros(n, 6, device=dev)
            for t in range(0, T, FRAME_STEP):
                root = A.turn_poses(root_all[:, t].clone(), yaw)
                root[:, :3] += origins
                robot.write_root_pose_to_sim_index(root_pose=root)
                robot.write_root_velocity_to_sim_index(root_velocity=zeros6)
                q[:, jidx] = joint_all[:, t]
                robot.write_joint_position_to_sim_index(position=q)
                robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q))
                obj = A.turn_poses(obj_all[:, t].clone(), yaw)
                obj[..., :3] += origins[:, None]
                for k, o in enumerate(objects):
                    o.write_root_pose_to_sim_index(root_pose=obj[:, k])
                sim.forward()
                robot.update(0.0)
                torso_pos = tt(robot.data.body_pos_w)[:, torso]
                pos, quat = A.camera_poses(
                    torso_pos, tt(robot.data.body_quat_w)[:, torso], shift, tilt
                )
                cam.set_world_poses(pos, quat, convention="world")
                cam.update(1.0 / FPS, force_recompute=True)
                rgb = A.apply_light(tt(cam.data.output["rgb"])[..., :3], gain, tint)
                rgb = rgb.cpu().numpy()
                k = t // FRAME_STEP
                for i in range(m):
                    if t < ep_steps[i]:
                        procs[i].stdin.write(rgb[i].tobytes())
                        frames_done += 1
                        if i in sheet and k in sheet_at[i]:
                            sheet[i].append(rgb[i])
            for p in procs:
                p.stdin.close()
            for i, (p, path) in enumerate(zip(procs, paths, strict=True)):
                if p.wait() != 0:
                    raise RuntimeError(f"ffmpeg failed for {path}")
                expected = (ep_steps[i] + FRAME_STEP - 1) // FRAME_STEP
                got = frame_count(path)
                if got != expected:
                    raise RuntimeError(f"{path}: {got} frames, expected {expected}")
                s = batch_seeds[i]
                results.append({
                    "seed": s, "video": f"videos/{s}.mp4", "frames": got,
                    "steps": int(ep_steps[i]), "bytes": path.stat().st_size,
                    "sha256": sha256(path), "look": looks[s].to_json(),
                })  # fmt: skip
            for i, frames in sheet.items():
                w, h = A.HEAD_CAM_SIZE
                img = Image.new("RGB", (3 * w // 2, 2 * h // 2))
                for j, f in enumerate(frames):
                    img.paste(Image.fromarray(f).resize((w // 2, h // 2)),
                              ((j % 3) * w // 2, (j // 3) * h // 2))  # fmt: skip
                img.save(args.out / f"sheet-{batch_seeds[i]}.png")
            dt = time.perf_counter() - t_batch
            frames = sum((x + 1) // FRAME_STEP for x in ep_steps[:m])
            print(f"[render] batch {b + 1}/{len(batches)} (sun {A.SUN_ELEVATIONS_DEG[elev]:g} deg,"
                  f" {m} episodes, {frames} frames) in {dt:.0f} s: {frames / dt:.0f} frames/s; "
                  f"{len(results)}/{len(seeds)} episodes done", flush=True)  # fmt: skip
        elapsed = time.perf_counter() - t_start
        results.sort(key=lambda r: seeds.index(r["seed"]))
        total = sum(r["bytes"] for r in results)
        print(f"[render] {len(results)} videos, {frames_done} frames, {total / 2**30:.2f} GiB, "
              f"{elapsed / 60:.1f} min ({frames_done / elapsed:.0f} frames/s)")  # fmt: skip
        provenance.save_json(args.out / "render.json", {
            "episodes_dir": str(args.episodes), "seeds_from": str(args.seeds),
            "render_seed": args.render_seed, "fps": FPS, "frame_step": FRAME_STEP,
            "camera": {"size": A.HEAD_CAM_SIZE, "pos": A.HEAD_CAM_POS, "rot": A.HEAD_CAM_ROT,
                       "focal_mm": A.HEAD_CAM_FOCAL, "aperture_mm": A.HEAD_CAM_APERTURE},
            "renderer": {"name": "newton_warp", "shadows": not args.no_shadows},
            "video": {"codec": "libx264", "crf": args.crf, "gop": args.gop,
                      "pix_fmt": "yuv420p", "preset": "veryfast"},
            "sun_elevations_deg": A.SUN_ELEVATIONS_DEG,
            "sun_counts": dict(collections.Counter(
                A.SUN_ELEVATIONS_DEG[looks[s].sun_elevation] for s in seeds)),
            "videos": len(results), "frames": frames_done, "bytes": total,
            "elapsed_s": round(elapsed, 1), "episodes": results,
        })  # fmt: skip


if __name__ == "__main__":
    main()

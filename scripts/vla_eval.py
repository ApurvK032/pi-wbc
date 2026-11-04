"""Closed-loop evaluation of a VLA in the pick-place task (stage 4 draft, design section 6).

The stage 1 controller keeps the robot balanced (as for the expert); a policy sends the task's
commands. Every `--execute` frames (25 per second) the head camera is rendered live with each
episode's look (pi_wbc.stage3.appearance: colours, patterns, light, camera mount; the sun's
direction is global per batch, since live physics can't turn the scene as rendering did), and
the policy gets the image, the 43 joint positions and the instruction and returns a chunk of
commands in the export's 35-value layout (pi_wbc.stage3.export); each command is held for two
50 Hz control steps.

An episode succeeds once the target object rests in the named bin (the expert's judge,
`placement_label`), no other object has moved more than 2 cm and the robot is up, for
`--hold-s` without a break; otherwise it fails when `--max-s` runs out (label from the final
state) or the robot falls.

Policies:
- `replay`: the expert's recorded commands for the same layout, as the dataset holds them
  (frame k: control step 2k), from `--recordings`; it tests this loop, not a model.
- `groot`: a GR00T policy server (GR00T's `gr00t/eval/run_gr00t_server.py`, run in GR00T's own
  environment), over ZeroMQ, `--host`/`--port`.

    uv run python scripts/vla_eval.py --policy replay --recordings logs/dataset/val-v1 \\
        --seeds 40000-40031 --out logs/vla/replay-val.json --viz none
"""

import argparse
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--policy", choices=["replay", "groot"], required=True)
parser.add_argument("--recordings", type=Path, default=None, help="replay: recorded episodes")
parser.add_argument("--replay-rate", type=int, choices=[25, 50], default=25)
parser.add_argument(
    "--between",
    choices=["hold", "mean"],
    default="hold",
    help="25 Hz policies: the second control step of a frame repeats its command (hold) or "
    "takes the mean of it and the next one (mean)",
)
parser.add_argument("--host", default="localhost")
parser.add_argument("--port", type=int, default=5555)
parser.add_argument("--seeds", required=True, help="layout seeds: a-b (inclusive) or a,b,c")
parser.add_argument("--num-envs", type=int, default=16)
parser.add_argument("--execute", type=int, default=8, help="frames executed per policy call")
parser.add_argument("--max-s", type=float, default=60.0)
parser.add_argument("--hold-s", type=float, default=1.0)
parser.add_argument("--knock-dist", type=float, default=0.02)
parser.add_argument("--render-seed", type=int, default=0)
parser.add_argument("--gif", type=Path, default=None, help="save each episode's head camera here")
RUN = "logs/rsl_rl/pi_wbc_g1_student/slurm-5628865-full-range-long-r"
parser.add_argument("--run", type=Path, default=Path(RUN), help="controller run folder")
parser.add_argument("--out", type=Path, required=True)
add_launcher_args(parser)
args, overrides = parser.parse_known_args()

import collections  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab_newton.renderers import NewtonWarpRendererCfg  # noqa: E402
from isaaclab_rl.rsl_rl import (  # noqa: E402
    RslRlVecEnvWrapper,
    check_rsl_rl_version,
    create_rsl_rl_runner,
    handle_deprecated_rsl_rl_cfg,
)
from isaaclab_tasks.utils import resolve_task_config  # noqa: E402

import pi_wbc.tasks  # noqa: E402, F401
from pi_wbc import provenance  # noqa: E402
from pi_wbc.stage3 import appearance as A  # noqa: E402
from pi_wbc.stage3 import export as E  # noqa: E402
from pi_wbc.stage3 import layout as L  # noqa: E402
from pi_wbc.stage3.expert import placement_label  # noqa: E402
from pi_wbc.stats import wilson_interval  # noqa: E402
from pi_wbc.tasks.g1_pick_place import HighLevel  # noqa: E402

TASK = "PiWbc-G1-PickPlace"
FRAME_STEP = 2
FALL_Z = 0.45  # pelvis height (m) below which the robot has fallen


def tt(x):
    return x.torch if hasattr(x, "torch") else x


def parse_seeds(text: str) -> list[int]:
    if "-" in text and "," not in text:
        a, b = (int(x) for x in text.split("-"))
        return list(range(a, b + 1))
    return [int(x) for x in text.split(",")]


class ReplayPolicy:
    """The expert's recorded commands, in the dataset's layout. At 25 Hz (as the dataset holds
    them) frame k gives control step 2k's command, held for two control steps; at 50 Hz every
    control step gets its own recorded command (as in scripts/replay_check.py)."""

    def __init__(self, folder: Path, rate: int):
        self.folder = folder
        self.per_step = rate == 50

    def reset(self, seeds: list[int]) -> None:
        self.actions = []
        for s in seeds:
            d = np.load(self.folder / f"{s}.npz")
            meta = json.loads(str(d["meta"]))
            ep = {k: d[k] for k in ("phase", "joint_pos", "cmd_body", "cmd_arm_hand",
                                     "cmd_velocity")}  # fmt: skip
            even = E.episode_arrays(ep, meta)[1]  # control steps 0, 2, 4, ...
            if self.per_step:  # the same rows for steps 1, 3, 5, ... (the arrays shifted by one)
                odd = E.episode_arrays({k: v[1:] for k, v in ep.items()}, meta)[1]
                full = np.empty((len(even) + len(odd), even.shape[1]), dtype=np.float32)
                full[0::2], full[1::2] = even, odd
                self.actions.append(full)
            else:
                self.actions.append(even)

    def act(self, images, state, texts, frame: int, horizon: int) -> np.ndarray:
        scale = 2 if self.per_step else 1
        out = []
        for a in self.actions:
            idx = np.arange(frame * scale, (frame + horizon) * scale)
            out.append(a[np.minimum(idx, len(a) - 1)])  # past the end: hold the last
        return np.stack(out)


class GrootPolicy:
    """A GR00T policy server (GR00T's run_gr00t_server.py): its ZeroMQ + msgpack protocol."""

    per_step = False

    def __init__(self, host: str, port: int):
        import zmq

        self.zmq = zmq
        self.socket = zmq.Context().socket(zmq.REQ)
        self.socket.connect(f"tcp://{host}:{port}")

    @staticmethod
    def _encode(obj):
        if isinstance(obj, np.ndarray):
            buf = io.BytesIO()
            np.save(buf, obj, allow_pickle=False)
            return {"__ndarray_class__": True, "as_npy": buf.getvalue()}
        return obj

    @staticmethod
    def _decode(obj):
        if isinstance(obj, dict) and "__ndarray_class__" in obj:
            return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
        return obj

    def _call(self, endpoint: str, data=None):
        import msgpack

        request = {"endpoint": endpoint}
        if data is not None:
            request["data"] = data
        self.socket.send(msgpack.packb(request, default=self._encode))
        reply = self.socket.recv()
        if reply == b"ERROR":
            raise RuntimeError("GR00T server error")
        out = msgpack.unpackb(reply, object_hook=self._decode)
        if isinstance(out, dict) and "error" in out:
            raise RuntimeError(f"GR00T server error: {out['error']}")
        return out

    def reset(self, seeds: list[int]) -> None:
        self._call("reset", {"options": None})

    def act(self, images, state, texts, frame: int, horizon: int) -> np.ndarray:
        m = E.modality()
        obs = {
            "video": {"ego_view": images[:, None]},
            "state": {k: state[:, None, s["start"]:s["end"]].astype(np.float32)
                      for k, s in m["state"].items()},
            "language": {"annotation.human.task_description": [[t] for t in texts]},
        }  # fmt: skip
        action, _ = self._call("get_action", {"observation": obs, "options": None})
        parts = [np.asarray(action[k], dtype=np.float32) for k in E.ACTION_GROUPS]
        chunk = np.concatenate(parts, axis=-1)  # (n, model horizon, 35)
        if chunk.shape[1] < horizon:
            raise ValueError(f"the model returns {chunk.shape[1]} steps, {horizon} are executed")
        return chunk[:, :horizon]


def main() -> None:
    seeds_all = parse_seeds(args.seeds)
    env_cfg, agent_cfg = resolve_task_config(
        TASK, "rsl_rl_cfg_entry_point", overrides=[*overrides, "physics=newton_mjwarp"]
    )
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, check_rsl_rl_version())
    n = min(args.num_envs, len(seeds_all))
    env_cfg.scene.num_envs = n
    A.add_look_assets(env_cfg.scene)
    # rendered only when asked (the scene updates sensors lazily); a long update period would
    # leave every environment "up to date" and force_recompute would then render nothing
    env_cfg.scene.head_cam = A.head_camera_cfg(NewtonWarpRendererCfg(enable_shadows=True))
    policy = (ReplayPolicy(args.recordings, args.replay_rate) if args.policy == "replay"
              else GrootPolicy(args.host, args.port))  # fmt: skip
    results, gif_frames = [], {}
    with launch_simulation(env_cfg, args):
        env = gym.make(TASK, cfg=env_cfg)
        venv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = create_rsl_rl_runner(venv, agent_cfg)
        runner.load(str((args.run / "model_2999.pt").resolve()))
        controller = runner.get_inference_policy(device=venv.unwrapped.device)
        base = venv.unwrapped
        dev = base.device
        hl = HighLevel(env)
        robot, cam = base.scene["robot"], base.scene["head_cam"]
        objects = [base.scene[f"obj_{name}"] for name in L.OBJECT_NAMES]
        torso = robot.body_names.index("torso_link")
        joint_ids = [robot.joint_names.index(j) for j in E.STATE_NAMES]
        arm_hand_cols = [hl.arm_hand_names.index(j) for g in ("left_arm", "right_arm",
                         "left_hand", "right_hand") for j in E.ACTION_GROUPS[g]]  # fmt: skip
        looks_out = A.LookRenderer(cam, n)
        looks_out.install_textures()
        origins = base.scene.env_origins
        unit = torch.tensor([0.0, 0.0, 0.0, 1.0], device=dev)
        max_frames = round(args.max_s * 25)
        hold = round(args.hold_s * 25)
        t0 = time.perf_counter()
        for start in range(0, len(seeds_all), n):
            seeds = seeds_all[start : start + n]
            m = len(seeds)
            seeds_p = seeds + [seeds[0]] * (n - m)
            lays = [L.sample(s) for s in seeds_p]
            base.pick_place_layouts = lays
            base.reset()
            hl.reset(list(range(n)))
            policy.reset(seeds_p)
            looks = [A.sample_look(s, args.render_seed) for s in seeds_p]
            looks_out.set_looks(looks)
            # the sun: the first episode's height, its scene turn as the direction (from -x)
            e = math.radians(A.SUN_ELEVATIONS_DEG[looks[0].sun_elevation])
            az = looks[0].scene_yaw
            d = np.array([[math.cos(e) * math.cos(az), math.cos(e) * math.sin(az),
                           -math.sin(e)]], dtype=np.float32)  # fmt: skip
            looks_out.ctx.lights_orientation.assign(d)

            def place(asset, xyz):
                pose = torch.cat([torch.as_tensor(xyz, device=dev, dtype=torch.float32),
                                  unit.repeat(n, 1)], dim=-1)  # fmt: skip
                pose[:, :3] += origins
                asset.write_root_pose_to_sim_index(root_pose=pose)

            place(base.scene["look_floor"],
                  [(*A.FLOOR_CENTER, A.FLOOR_TOP - A.FLOOR_SIZE[2] / 2)] * n)  # fmt: skip
            for wall in ("wall_back", "wall_left", "wall_right"):
                place(base.scene[f"look_{wall}"], [A.wall_poses(x)[wall] for x in looks])
            shift = torch.tensor([x.cam_shift for x in looks], device=dev)
            tilt = torch.tensor([x.cam_tilt for x in looks], device=dev)
            gain = torch.tensor([x.gain for x in looks], device=dev)
            tint = torch.tensor([x.tint for x in looks], device=dev)
            target = [L.OBJECT_NAMES.index(lay.target_object) for lay in lays]
            others = [
                [k for k, name in enumerate(L.OBJECT_NAMES)
                 if name in lay.objects and name != lay.target_object]
                for lay in lays
            ]  # fmt: skip
            texts = [lay.instruction for lay in lays]
            obs = venv.get_observations()
            every = torch.stack([tt(o.data.root_pos_w) for o in objects], dim=1)
            start_xy = (every - origins[:, None]).cpu().numpy()[..., :2].copy()
            ok_run = np.zeros(n, dtype=int)
            done = np.zeros(n, dtype=bool)
            label = ["timed_out"] * n
            end_frame = np.full(n, max_frames)
            fell = np.zeros(n, dtype=bool)
            chunk = None
            calls, call_s = 0, 0.0
            with torch.no_grad():
                for k in range(max_frames):
                    if k % args.execute == 0:  # render, ask the policy
                        root = tt(robot.data.body_pos_w)
                        quat_t = tt(robot.data.body_quat_w)[:, torso]
                        pos, quat = A.camera_poses(root[:, torso], quat_t, shift, tilt)
                        cam.set_world_poses(pos, quat, convention="world")
                        cam.update(0.04, force_recompute=True)
                        rgb = A.apply_light(tt(cam.data.output["rgb"])[..., :3], gain, tint)
                        images = rgb.cpu().numpy()
                        state = tt(robot.data.joint_pos)[:, joint_ids].cpu().numpy()
                        tc = time.perf_counter()
                        chunk = policy.act(images, state, texts, k, args.execute + 1)
                        call_s += time.perf_counter() - tc
                        calls += 1
                        if args.gif is not None:
                            for i in range(m):
                                if not done[i]:
                                    gif_frames.setdefault(seeds[i], []).append(images[i, ::2, ::2])
                    for sub in range(FRAME_STEP):
                        i = k % args.execute
                        if policy.per_step:
                            a = chunk[:, i * FRAME_STEP + sub]
                        elif sub == 1 and args.between == "mean":  # halfway to the next command
                            a = 0.5 * (chunk[:, i] + chunk[:, i + 1])
                        else:
                            a = chunk[:, i]
                        a = torch.as_tensor(a, device=dev)
                        hl.arm_hand[:, arm_hand_cols] = a[:, :28]
                        hl.body_goal[:] = torch.cat([a[:, 31:32], a[:, 28:31]], dim=1)
                        hl.velocity[:] = a[:, 32:35]
                        hl.apply()
                        obs, _, _, _ = venv.step(controller(obs))
                    # judge
                    every = torch.stack([tt(o.data.root_pos_w) for o in objects], dim=1)
                    every = (every - origins[:, None]).cpu().numpy()
                    z = tt(robot.data.root_pos_w)[:, 2].cpu().numpy()
                    for i in range(m):
                        if done[i]:
                            continue
                        moved = (np.linalg.norm(every[i, others[i], :2] - start_xy[i, others[i]],
                                                axis=1).max() if others[i] else 0.0)  # fmt: skip
                        if z[i] < FALL_Z:
                            fell[i], done[i], label[i], end_frame[i] = True, True, "fell", k
                            continue
                        placed = placement_label(lays[i], every[i, target[i]])
                        if moved > args.knock_dist:
                            done[i], label[i], end_frame[i] = True, "knocked_other", k
                            continue
                        ok_run[i] = ok_run[i] + 1 if placed == "success" else 0
                        if ok_run[i] >= hold:
                            done[i], label[i], end_frame[i] = True, "success", k
                        elif k == max_frames - 1:
                            label[i] = placed if placed != "success" else "timed_out"
                    if done[:m].all():
                        break
            for i in range(m):
                results.append({
                    "seed": seeds[i], "instruction": texts[i], "label": label[i],
                    "time_s": round(float(end_frame[i] + 1) / 25, 2), "fell": bool(fell[i]),
                })  # fmt: skip
                print(f"[vla] seed {seeds[i]}: {label[i]:13s} at {results[-1]['time_s']:5.1f} s "
                      f"'{texts[i]}'", flush=True)  # fmt: skip
            print(f"[vla] batch: {calls} policy calls, {1000 * call_s / max(calls, 1):.0f} ms each",
                  flush=True)  # fmt: skip
        wall = time.perf_counter() - t0
        ok = sum(r["label"] == "success" for r in results)
        lo, hi = wilson_interval(ok, len(results))
        labels = dict(collections.Counter(r["label"] for r in results))
        print(f"[vla] success {ok}/{len(results)} ({lo:.0%}-{hi:.0%}); {labels}; {wall:.0f} s")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        provenance.save_json(args.out, {
            "policy": args.policy, "recordings": str(args.recordings),
            "replay_rate": args.replay_rate if args.policy == "replay" else None,
            "between": args.between,
            "server": None if args.policy == "replay" else f"{args.host}:{args.port}",
            "controller_run": str(args.run), "execute_frames": args.execute,
            "max_s": args.max_s, "hold_s": args.hold_s, "knock_dist": args.knock_dist,
            "render_seed": args.render_seed, "episodes": len(results), "success": ok,
            "success_ci95": [lo, hi], "labels": labels, "results": results,
        })  # fmt: skip
        if args.gif is not None:
            from PIL import Image

            args.gif.mkdir(parents=True, exist_ok=True)
            for s, frames in gif_frames.items():
                imgs = [Image.fromarray(f) for f in frames]
                imgs[0].save(args.gif / f"{s}.gif", save_all=True, append_images=imgs[1:],
                             duration=40 * args.execute, loop=0)  # fmt: skip
        env.close()


if __name__ == "__main__":
    main()

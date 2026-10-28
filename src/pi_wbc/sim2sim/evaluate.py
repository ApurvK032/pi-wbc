"""Evaluate an exported policy in plain MuJoCo on fixed commands and seeds.

    uv run pi-wbc-sim2sim --policy <run>/exported/policy.onnx \
        --policy-io <run>/exported/policy_io.json \
        --model newton=<run>/exported/newton_model.xml \
        --model menagerie=third_party/mujoco_menagerie/unitree_g1/scene_with_hands.xml \
        --seeds 10 --out results/stage0/sim2sim_mujoco.json

Each episode starts from the default pose with a small seeded perturbation (joint offsets,
heading, base velocity), holds one velocity command, and runs until the time limit or a fall.
A fall is the torso touching the floor (Isaac Lab's termination) or the pelvis dropping below
0.35 m. Tracking error is measured after a 2 s settling period, the way the training rewards
measure it: planar velocity in the heading frame and yaw rate.
"""

from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pi_wbc import provenance
from pi_wbc.sim2sim.mujoco_g1 import G1MujocoSim, PolicyIO, build_model
from pi_wbc.stats import wilson_interval

# (vx m/s, vy m/s, yaw rate rad/s), all inside the training command ranges.
SCENARIOS: dict[str, tuple[float, float, float]] = {
    "stand": (0.0, 0.0, 0.0),
    "walk_0.5": (0.5, 0.0, 0.0),
    "walk_1.0": (1.0, 0.0, 0.0),
    "side_0.3": (0.0, 0.3, 0.0),
    "turn_0.8": (0.0, 0.0, 0.8),
    "walk_turn": (0.6, 0.0, 0.5),
}
FALL_HEIGHT = 0.35  # m, pelvis height
SETTLE_S = 2.0


class OnnxPolicy:
    """An exported rsl_rl actor: observation (1, n) float32 in, action (1, m) out."""

    def __init__(self, path: str | Path) -> None:
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.input_dim = self.session.get_inputs()[0].shape[-1]

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        out = self.session.run(None, {self.input_name: obs[None, :].astype(np.float32)})
        return np.asarray(out[0][0], dtype=float)


@dataclass
class Episode:
    model: str
    scenario: str
    seed: int
    fell: bool
    fall_time_s: float | None
    duration_s: float
    lin_err: float | None  # mean |command_xy - velocity_xy| after settling, m/s
    ang_err: float | None  # mean |command_yaw_rate - yaw_rate| after settling, rad/s


def run_episode(
    sim: G1MujocoSim,
    policy: Callable[[np.ndarray], np.ndarray],
    command: tuple[float, float, float],
    seed: int,
    duration_s: float,
) -> tuple[bool, float | None, float | None, float | None]:
    """One episode: (fell, fall time, mean linear error, mean yaw-rate error)."""
    rng = np.random.default_rng(seed)
    n = len(sim.policy_ids)
    sim.reset(
        joint_offsets=rng.uniform(-0.05, 0.05, n),
        yaw=rng.uniform(-np.pi, np.pi),
        root_lin_vel=(*rng.uniform(-0.1, 0.1, 2), 0.0),
    )
    dt = sim.io.physics_dt * sim.io.decimation
    lin_errs, ang_errs = [], []
    for k in range(round(duration_s / dt)):
        sim.step(policy(sim.observation(command)))
        t = (k + 1) * dt
        if sim.torso_touches_floor() or sim.base_height() < FALL_HEIGHT:
            return True, t, _mean(lin_errs), _mean(ang_errs)
        if t >= SETTLE_S:
            v_xy, yaw_rate = sim.velocity_yaw_frame()
            lin_errs.append(float(np.linalg.norm(np.asarray(command[:2]) - v_xy)))
            ang_errs.append(abs(command[2] - yaw_rate))
    return False, None, _mean(lin_errs), _mean(ang_errs)


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def summarize(episodes: list[Episode]) -> list[dict[str, Any]]:
    """Per (model, scenario): fall rate with a 95% Wilson interval, mean tracking errors."""
    order = list(SCENARIOS)
    models = list(dict.fromkeys(e.model for e in episodes))  # first-seen order
    scenarios = sorted(
        {e.scenario for e in episodes},
        key=lambda s: (s not in order, order.index(s) if s in order else 0, s),
    )
    rows = []
    for model, scenario in ((m, s) for m in models for s in scenarios):
        group = [e for e in episodes if e.model == model and e.scenario == scenario]
        falls = sum(e.fell for e in group)
        lo, hi = wilson_interval(falls, len(group))
        lin = [e.lin_err for e in group if e.lin_err is not None and not e.fell]
        ang = [e.ang_err for e in group if e.ang_err is not None and not e.fell]
        rows.append(
            {
                "model": model,
                "scenario": scenario,
                "command": SCENARIOS.get(scenario),
                "episodes": len(group),
                "falls": falls,
                "fall_rate": falls / len(group),
                "fall_rate_ci95": [lo, hi],
                "lin_err_mean": _mean(lin),
                "ang_err_mean": _mean(ang),
            }
        )
    return rows


def markdown_table(rows: list[dict[str, Any]]) -> str:
    out = [
        "| Model | Command | Falls | Fall rate (95% CI) | Speed error | Turn-rate error |",
        "|---|---|---:|---|---:|---:|",
    ]
    for r in rows:
        lo, hi = r["fall_rate_ci95"]
        lin = "—" if r["lin_err_mean"] is None else f"{r['lin_err_mean']:.2f} m/s"
        ang = "—" if r["ang_err_mean"] is None else f"{r['ang_err_mean']:.2f} rad/s"
        out.append(
            f"| {r['model']} | {r['scenario']} | {r['falls']}/{r['episodes']} "
            f"| {r['fall_rate']:.0%} ({lo:.0%}-{hi:.0%}) | {lin} | {ang} |"
        )
    return "\n".join(out)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--policy", type=Path, required=True, help="exported policy.onnx")
    parser.add_argument("--policy-io", type=Path, required=True, help="policy_io.json")
    parser.add_argument(
        "--model", action="append", required=True, help="NAME=path/to/scene.xml (repeatable)"
    )
    parser.add_argument("--seeds", type=int, default=10, help="episodes per command and model")
    parser.add_argument("--duration-s", type=float, default=20.0)
    parser.add_argument("--scenario", action="append", choices=list(SCENARIOS), default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--gif",
        metavar="MODEL:SCENARIO=PATH",
        action="append",
        default=[],
        help="also record a GIF, e.g. menagerie:walk_0.5=results/stage0/walk.gif (needs MUJOCO_GL)",
    )
    args = parser.parse_args(argv)

    io = PolicyIO.load(args.policy_io)
    policy = OnnxPolicy(args.policy)
    if policy.input_dim != io.obs_dim:
        raise SystemExit(f"policy takes {policy.input_dim} inputs, policy_io says {io.obs_dim}")
    scenarios = args.scenario or list(SCENARIOS)

    gifs: dict[tuple[str, str], Path] = {}
    for item in args.gif:
        key, _, gif_path = item.partition("=")
        model_name, _, scenario = key.partition(":")
        gifs[(model_name, scenario)] = Path(gif_path)

    episodes: list[Episode] = []
    model_info: dict[str, Any] = {}
    for spec in args.model:
        name, _, path = spec.partition("=")
        model = build_model(path, io)
        sim = G1MujocoSim(model, io)
        for (gif_model, scenario), gif_path in gifs.items():
            if gif_model == name:
                from pi_wbc.sim2sim.render import record_gif

                record_gif(sim, policy, SCENARIOS[scenario], gif_path)
                print(f"[sim2sim] wrote {gif_path}", flush=True)
        start = time.monotonic()
        for scenario in scenarios:
            for seed in range(args.seeds):
                fell, t_fall, lin, ang = run_episode(
                    sim, policy, SCENARIOS[scenario], seed, args.duration_s
                )
                episodes.append(
                    Episode(name, scenario, seed, fell, t_fall, args.duration_s, lin, ang)
                )
        model_info[name] = {
            "path": path,
            "total_mass_kg": float(model.body_mass.sum()),
            "timestep_s": float(model.opt.timestep),
            "wall_s": round(time.monotonic() - start, 1),
        }
        print(f"[sim2sim] {name}: {len(scenarios) * args.seeds} episodes done", flush=True)

    rows = summarize(episodes)
    print(markdown_table(rows))
    provenance.save_json(
        args.out,
        {
            "policy": str(args.policy),
            "policy_io": str(args.policy_io),
            "seeds": args.seeds,
            "duration_s": args.duration_s,
            "fall_height_m": FALL_HEIGHT,
            "settle_s": SETTLE_S,
            "models": model_info,
            "summary": rows,
            "episodes": [asdict(e) for e in episodes],
        },
    )
    print(f"[sim2sim] wrote {args.out}")


if __name__ == "__main__":
    main()

"""Summarize Isaac Lab training runs from their TensorBoard logs.

    uv run pi-wbc-train-summary logs/rsl_rl/pi_wbc_g1_flat_stage0/slurm-* --out summary.json

For each run: the mean over the last `--last` iterations of the training reward, episode
length, the reward terms that shape the gait, and Isaac Lab's own tracking-error metrics.
These are training-side numbers (Newton, with the task's randomization); the MuJoCo evaluation
is the separate, fixed-protocol check.
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path
from typing import Any

COLUMNS = (  # header, TensorBoard tag, format
    ("Reward", "Train/mean_reward", "{:.1f}"),
    ("Episode length", "Train/mean_episode_length", "{:.0f}"),
    ("Falls (share of episode ends)", "Episode_Termination/base_contact", "{:.1%}"),
    ("Speed error", "Metrics/base_velocity/error_vel_xy", "{:.2f} m/s"),
    ("Turn-rate error", "Metrics/base_velocity/error_vel_yaw", "{:.2f} rad/s"),
    ("track_lin_vel", "Episode_Reward/track_lin_vel_xy_exp", "{:.2f}"),
    ("feet_air_time", "Episode_Reward/feet_air_time", "{:.3f}"),
    ("feet_slide", "Episode_Reward/feet_slide", "{:.3f}"),
    ("action_rate", "Episode_Reward/action_rate_l2", "{:.3f}"),
    ("dof_torques", "Episode_Reward/dof_torques_l2", "{:.4f}"),
)


def read_scalars(run_dir: Path) -> dict[str, list[float]]:
    """All scalar series of a run, by tag, in step order."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    files = sorted(run_dir.glob("events.out.tfevents*"))
    if not files:
        raise FileNotFoundError(f"no TensorBoard log in {run_dir}")
    series: dict[str, dict[int, float]] = {}
    for f in files:
        acc = EventAccumulator(str(f), size_guidance={"scalars": 0})
        acc.Reload()
        for tag in acc.Tags()["scalars"]:
            series.setdefault(tag, {}).update({e.step: e.value for e in acc.Scalars(tag)})
    return {tag: [v for _, v in sorted(s.items())] for tag, s in series.items()}


def summarize_run(run_dir: Path, last: int) -> dict[str, Any]:
    scalars = read_scalars(run_dir)
    return {
        "run": run_dir.name,
        "iterations": len(scalars.get("Train/mean_reward", [])),
        "last": last,
        "means": {
            tag: statistics.fmean(scalars[tag][-last:]) for _, tag, _ in COLUMNS if tag in scalars
        },
    }


def markdown_table(rows: list[dict[str, Any]]) -> str:
    out = [
        "| Run | Iterations | " + " | ".join(h for h, *_ in COLUMNS) + " |",
        "|---|---:|" + "---:|" * len(COLUMNS),
    ]
    for r in rows:
        cells = [
            "—" if tag not in r["means"] else fmt.format(r["means"][tag]) for _, tag, fmt in COLUMNS
        ]
        out.append(f"| {r['run']} | {r['iterations']} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("runs", type=Path, nargs="+", help="rsl_rl run folders")
    parser.add_argument("--last", type=int, default=50, help="iterations to average over")
    parser.add_argument("--out", type=Path, default=None, help="also write the rows as JSON")
    args = parser.parse_args(argv)

    rows = [summarize_run(r, args.last) for r in args.runs]
    print(markdown_table(rows))
    if args.out:
        from pi_wbc import provenance

        provenance.save_json(args.out, {"runs": [str(r) for r in args.runs], "rows": rows})
        print(f"\n[train-summary] wrote {args.out}")


if __name__ == "__main__":
    main()

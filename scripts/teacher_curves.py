"""Collect the stage 1 controller's teacher training curves (D-028) from the TensorBoard logs
into one result file, for the training figure in scripts/figures.py.

    .venv/bin/python scripts/teacher_curves.py   (from the repository root)

Each seed's teacher trained in two parts: the B2 teacher (D-022, iterations 0-2999) and its
resumption at the full command ranges (D-024, D-028; iterations 2999-9998). Seed 1's first
full-range job stopped at iteration 5432 on NaN observations and was resumed from iteration
5000, so its curve takes iterations up to 4999 from the first job and the rest from the second.
Writes results/stage1/full_range_long/teacher_training.json.gz.
"""

from __future__ import annotations

from pathlib import Path

from pi_wbc.provenance import save_json

T = Path("logs/rsl_rl/pi_wbc_g1_teacher")
# seed: [(run folder, first iteration kept, last iteration kept)]
RUNS = {
    1: [
        ("pc-speed-fine05-teacher-s1-3000it", 0, 2998),
        ("slurm-5571784-full-range-long", 2999, 4999),
        ("slurm-5628865-full-range-long-r", 5000, 9998),
    ],
    2: [
        ("pc-speed-fine05-teacher-s2-3000it", 0, 2998),
        ("slurm-5571880-full-range-long", 2999, 9998),
    ],
    3: [
        ("slurm-5445344-speed-fine05", 0, 2998),
        ("slurm-5538135-full-range-long", 2999, 9998),
    ],
}
TAGS = {
    "turn_rate_error": "Metrics/base_velocity/error_vel_yaw",
    "speed_error": "Metrics/base_velocity/error_vel_xy",
    "curriculum_level": "Curriculum/upper_body_level",
    "episode_length": "Train/mean_episode_length",
}


def main() -> None:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    seeds = {}
    for seed, parts in RUNS.items():
        series: dict[str, dict[int, float]] = {k: {} for k in TAGS}
        for folder, first, last in parts:
            (events,) = sorted((T / folder).glob("events.out.tfevents*"))
            acc = EventAccumulator(str(events), size_guidance={"scalars": 0})
            acc.Reload()
            for key, tag in TAGS.items():
                for e in acc.Scalars(tag):
                    if first <= e.step <= last:
                        series[key][e.step] = round(float(e.value), 5)
        steps = sorted(series["turn_rate_error"])
        seeds[str(seed)] = {
            "runs": [p[0] for p in parts],
            "iteration": steps,
            **{k: [series[k].get(s) for s in steps] for k in TAGS},
        }
        print(f"seed {seed}: iterations {steps[0]}-{steps[-1]} ({len(steps)})")
    save_json(
        "results/stage1/full_range_long/teacher_training.json.gz",
        {"tags": TAGS, "full_ranges_from": 2999, "seeds": seeds},
    )


if __name__ == "__main__":
    main()

"""The three-seed ablation table of results/stage1/README.md (step 6), from the MuJoCo test seeds.

    .venv/bin/python scripts/ablation_table.py          (from the repository root)

Reads each configuration's test-seed results (1000-1199, both robot models, 4400 episodes per
seed and condition): results/stage1/ablation_test/abl-<config>-s{1,2,3}-<condition>.json.gz,
and for the full recipe the step 4 warm-up students in results/stage1/student_test_ramp/.
Prints one markdown table: falls per training seed 1 / 2 / 3 with the pooled 95% Wilson
interval, then the nominal robot's speed error at 1 m/s, turn-rate error at 0.8 rad/s in place
and crouch error at 0.60 m, per seed, each the mean of the two models.
"""

from __future__ import annotations

from pathlib import Path

from pi_wbc.provenance import load_json
from pi_wbc.stats import wilson_interval

R = Path("results/stage1")
SEEDS = (1, 2, 3)
CONFIGS = (  # row label, folder, label prefix
    ("nothing", R / "student_test_ramp", "ts-full-ramp"),
    ("all six groups", R / "ablation_test", "abl-none"),
    ("pushes", R / "ablation_test", "abl-pushes"),
    ("latency", R / "ablation_test", "abl-latency"),
    ("friction", R / "ablation_test", "abl-friction"),
    ("mass", R / "ablation_test", "abl-mass"),
    ("actuators", R / "ablation_test", "abl-actuators"),
    ("sensor noise", R / "ablation_test", "abl-sensor-noise"),
    ("curriculum", R / "ablation_test", "abl-curriculum"),
    ("teacher (PPO alone)", R / "ablation_test", "abl-teacher"),
)
CONDITIONS = (
    ("Nominal", "nominal"),
    ("Friction 0.4", "friction_0.4"),
    ("Sensor noise", "noise"),
    ("1 kg per hand", "payload_1kg"),
    ("20 ms delay", "delay_20ms"),
    ("1 m/s kicks", "push_1.0"),
    ("2 m/s kicks", "push_2.0"),
)
ERRORS = (  # column, scenario, summary key, scale, digits
    ("Speed error, 1 m/s", "walk_1.0", "lin_err_mean", 1.0, 2),
    ("Turn", "turn_0.8", "ang_err_mean", 1.0, 2),
    ("Crouch 0.60 m, cm", "crouch_0.6", "height_err_mean", 100.0, 1),
)


def falls_cell(folder: Path, prefix: str, cond: str) -> str:
    counts, total = [], 0
    for s in SEEDS:
        eps = load_json(folder / f"{prefix}-s{s}-{cond}.json.gz")["episodes"]
        counts.append(sum(e["fell"] for e in eps))
        total += len(eps)
    lo, hi = wilson_interval(sum(counts), total)
    return f"{' / '.join(map(str, counts))} ({100 * lo:.2f}%-{100 * hi:.2f}%)"


def error_cell(folder: Path, prefix: str, scenario: str, key: str, scale: float, digits: int):
    cells = []
    for s in SEEDS:
        rows = load_json(folder / f"{prefix}-s{s}-nominal.json.gz")["summary"]
        vals = [r[key] for r in rows if r["scenario"] == scenario]
        assert len(vals) == 2, (prefix, s, scenario)  # both robot models
        cells.append(f"{scale * sum(vals) / 2:.{digits}f}")
    return " / ".join(cells)


def main() -> None:
    head = ["Trained without", *(c for c, _ in CONDITIONS), *(e[0] for e in ERRORS)]
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for label, folder, prefix in CONFIGS:
        cells = [falls_cell(folder, prefix, cond) for _, cond in CONDITIONS]
        cells += [error_cell(folder, prefix, *e[1:]) for e in ERRORS]
        print(f"| {label} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()

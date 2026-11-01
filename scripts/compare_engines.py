"""Tables for results/stage1/README.md step 7: the B2 students on Newton, PhysX and MuJoCo.

    python3 scripts/compare_engines.py

Reads the Isaac Lab evaluations in results/stage1/physx/ (50 robots per command; nominal and
with kicks) and the MuJoCo bench's development-seed results in results/stage1/speed_fix/.
"""

import json
import math

SEEDS = (1, 2, 3)
IL = "results/stage1/physx/b2-s{s}-{e}-{c}.json"
MJ = "results/stage1/speed_fix/speed-fine05-s{s}-nominal.json"
ROWS = [
    ("walk_0.5", "lin_err_mean", "Speed error, 0.5 m/s", 1.0),
    ("walk_1.0", "lin_err_mean", "Speed error, 1 m/s", 1.0),
    ("side_0.3", "lin_err_mean", "Sideways error, 0.3 m/s", 1.0),
    ("turn_0.8", "ang_err_mean", "Turn-rate error, 0.8 rad/s (rad/s)", 1.0),
    ("crouch_0.6", "height_err_mean", "Crouch error (cm)", 100.0),
]


def wilson(k: int, n: int, z: float = 1.96) -> str:
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return f"{100 * max(0.0, c - h):.1f}%-{100 * (c + h):.1f}%"


def by_scenario(path: str, model: str | None = None) -> dict:
    with open(path) as f:
        rows = json.load(f)["summary"]
    return {r["scenario"]: r for r in rows if model is None or r["model"] == model}


def main() -> None:
    cols = [
        ("Isaac Lab, Newton", lambda s: by_scenario(IL.format(s=s, e="newton", c="50"))),
        ("Isaac Sim, PhysX", lambda s: by_scenario(IL.format(s=s, e="isaacsim", c="50"))),
        ("MuJoCo, Menagerie", lambda s: by_scenario(MJ.format(s=s), "menagerie")),
        ("MuJoCo, Newton's model", lambda s: by_scenario(MJ.format(s=s), "newton")),
    ]
    data = {name: {s: load(s) for s in SEEDS} for name, load in cols}
    print("| Measure (seeds 1 / 2 / 3, mean) | " + " | ".join(n for n, _ in cols) + " |")
    print("|---|" + "---|" * len(cols))
    for sc, key, label, scale in ROWS:
        cells = []
        for name, _ in cols:
            v = [scale * data[name][s][sc][key] for s in SEEDS]
            fmt = "{:.1f}" if scale != 1.0 else "{:.2f}"
            cells.append(" / ".join(fmt.format(x) for x in v) + f" ({fmt.format(sum(v) / 3)})")
        print(f"| {label} | " + " | ".join(cells) + " |")
    for name, _ in cols:
        f = [sum(r["falls"] for r in data[name][s].values()) for s in SEEDS]
        n = [sum(r["episodes"] for r in data[name][s].values()) for s in SEEDS]
        print(f"{name}: falls {f} of {n[0]} per seed; pooled {sum(f)}/{sum(n)}")
    print()
    print("| Kicks every 4 s | Isaac Lab, Newton | Isaac Sim, PhysX |")
    print("|---|---|---|")
    for v in ("1.0", "2.0"):
        cells = []
        for e in ("newton", "isaacsim"):
            runs = [by_scenario(IL.format(s=s, e=e, c=f"push{v}")) for s in SEEDS]
            f = [sum(r["falls"] for r in d.values()) for d in runs]
            n = [sum(r["episodes"] for r in d.values()) for d in runs]
            cells.append(" / ".join(map(str, f)) + f" ({sum(f)}: {wilson(sum(f), sum(n))})")
        print(f"| {v} m/s | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()

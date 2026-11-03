"""Stage 1 finish-line tables for one recipe, from its MuJoCo test-seed results.

    .venv/bin/python scripts/finish_line.py DIR PREFIX [--compare DIR:PREFIX:NAME ...]

Reads DIR/PREFIX-s{1,2,3}-<condition>.json.gz (scripts/test_seeds.sh) and the kick sweep in
DIR/push/PREFIX-s{1,2,3}-push_<speed>.json.gz, and prints three markdown tables:

1. the design's finish line (docs/STAGE1_DESIGN.md, section 1): each measure per training seed,
   each cell Menagerie / Newton model, with whether the target is met;
2. falls per test condition, per seed and pooled with 95% Wilson intervals, next to the recipes
   given with --compare;
3. the kick sweep: episodes that fell and the share of kicks survived, per kick speed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pi_wbc.provenance import load_json
from pi_wbc.stats import wilson_interval

SEEDS = (1, 2, 3)
MODELS = ("menagerie", "newton")
CONDITIONS = (
    ("nominal", "nominal"),
    ("floor friction 0.4", "friction_0.4"),
    ("1 kg in each hand", "payload_1kg"),
    ("20 ms extra delay", "delay_20ms"),
    ("sensor noise", "noise"),
    ("1 m/s kicks every 4 s", "push_1.0"),
    ("2 m/s kicks every 4 s", "push_2.0"),
)
KICKS = ("0.5", "1.0", "1.5", "2.0", "2.5", "3.0")


def interval(k: int, n: int) -> str:
    lo, hi = wilson_interval(k, n)
    return f"{100 * lo:.2f}%-{100 * hi:.2f}%"


def summary(folder: Path, prefix: str, seed: int, cond: str) -> dict:
    d = load_json(folder / f"{prefix}-s{seed}-{cond}.json.gz")
    return {(r["model"], r["scenario"]): r for r in d["summary"]}


def kicks(folder: Path, prefix: str, seed: int, speed: str) -> tuple[int, int, float]:
    """(episodes that fell, episodes, share of kicks survived) of one kick-sweep run."""
    d = load_json(folder / "push" / f"{prefix}-s{seed}-push_{speed}.json.gz")
    times = d["condition_settings"]["push_times"]
    eps = d["episodes"]
    falls = sum(e["fell"] for e in eps)
    given = sum(sum(1 for t in times if not e["fell"] or t <= e["fall_time_s"]) for e in eps)
    return falls, len(eps), 1 - falls / given


def finish_line(folder: Path, prefix: str) -> None:
    nom = {s: summary(folder, prefix, s, "nominal") for s in SEEDS}

    def cells(fn, fmt):
        return [f"{fmt(fn(nom[s], 'menagerie'))} / {fmt(fn(nom[s], 'newton'))}" for s in SEEDS]

    def met(fn, ok):
        bad = [s for s in SEEDS if not all(ok(fn(nom[s], m)) for m in MODELS)]
        if not bad:
            return "yes"
        good = [str(s) for s in SEEDS if s not in bad]
        return f"seed{'s' if len(good) > 1 else ''} {' and '.join(good)}" if good else "no"

    def gait(scen, key):
        return lambda x, m: x[(m, scen)]["gait"][key]

    def err(scen, key):
        return lambda x, m: x[(m, scen)][key]

    rows = []
    falls = [sum(r["falls"] for r in nom[s].values()) for s in SEEDS]
    n = [sum(r["episodes"] for r in nom[s].values()) for s in SEEDS]
    hi = wilson_interval(sum(falls), sum(n))[1]
    rows.append(
        (
            "Falls, all commands, both models",
            "upper 95% bound below 5%",
            [f"{f}/{e}" for f, e in zip(falls, n, strict=True)],
            f"{'yes' if hi < 0.05 else 'no'}: {100 * sum(falls) / sum(n):.2f}% pooled "
            f"({interval(sum(falls), sum(n))})",
        )
    )
    measures = (
        (
            "Speed error, walking 1 m/s",
            "at most 0.15 m/s",
            err("walk_1.0", "lin_err_mean"),
            "{:.2f}",
            lambda v: v <= 0.15,
        ),
        (
            "Speed error, walking 0.5 m/s",
            "at most 0.15 m/s",
            err("walk_0.5", "lin_err_mean"),
            "{:.2f}",
            lambda v: v <= 0.15,
        ),
        (
            "Speed error, sideways 0.3 m/s",
            "at most 0.15 m/s",
            err("side_0.3", "lin_err_mean"),
            "{:.2f}",
            lambda v: v <= 0.15,
        ),
        (
            "Turn-rate error, 0.8 rad/s in place",
            "at most 0.2 rad/s",
            err("turn_0.8", "ang_err_mean"),
            "{:.2f}",
            lambda v: v <= 0.2,
        ),
        (
            "Step rate at 1 m/s",
            "1.5-3 steps/s",
            gait("walk_1.0", "step_rate_hz"),
            "{:.1f}",
            lambda v: 1.5 <= v <= 3.0,
        ),
        (
            "Step length at 1 m/s",
            "at least 0.3 m",
            gait("walk_1.0", "step_length_m"),
            "CM0",
            lambda v: v >= 0.3,
        ),
        (
            "Foot lift at 1 m/s",
            "at least 5 cm",
            gait("walk_1.0", "foot_lift_m"),
            "CM1",
            lambda v: v >= 0.05,
        ),
        (
            "Slip at the sole, 1 m/s (D-016)",
            "at most 0.1 m/s",
            gait("walk_1.0", "sole_slip_mps"),
            "{:.2f}",
            lambda v: v <= 0.1,
        ),
    )

    def formatter(fmt):
        if fmt.startswith("CM"):  # metres shown in centimetres, with this many decimals
            return lambda v: f"{100 * v:.{fmt[2]}f} cm"
        return fmt.format

    for label, target, fn, fmt, ok in measures:
        rows.append((label, target, cells(fn, formatter(fmt)), met(fn, ok)))
    rows.append(
        (
            "Arm joint error, walking with the arms moving",
            "report",
            cells(err("arms_walk", "arm_err_mean"), "{:.3f} rad".format),
            "-",
        )
    )
    survived = []
    for s in SEEDS:
        a, b = kicks(folder, prefix, s, "2.0"), kicks(folder, prefix, s, "2.5")
        survived.append(f"{100 * a[2]:.1f}% / {100 * b[2]:.1f}%")
    rows.append(
        (
            "Kicks survived at 2.0 / 2.5 m/s",
            "report the largest kick survived 95% of the time",
            survived,
            "see the kick sweep",
        )
    )
    print("| Measure | Target | Seed 1 | Seed 2 | Seed 3 | Met |")
    print("|---|---|---|---|---|---|")
    for label, target, vals, ok in rows:
        print(f"| {label} | {target} | " + " | ".join(vals) + f" | {ok} |")


def falls_table(recipes: list[tuple[Path, str, str]]) -> None:
    print("| Condition | " + " | ".join(name for _, _, name in recipes) + " |")
    print("|---|" + "---|" * len(recipes))
    for label, cond in CONDITIONS:
        cells = []
        for folder, prefix, _ in recipes:
            per = [
                sum(r["falls"] for r in summary(folder, prefix, s, cond).values()) for s in SEEDS
            ]
            n = sum(
                sum(r["episodes"] for r in summary(folder, prefix, s, cond).values()) for s in SEEDS
            )
            cells.append(" / ".join(map(str, per)) + f" ({sum(per)}: {interval(sum(per), n)})")
        print(f"| {label} | " + " | ".join(cells) + " |")


def kick_table(folder: Path, prefix: str) -> None:
    print(
        "| Kick | Episodes that fell (seeds 1 / 2 / 3, of 2000 each) | Pooled, 95% "
        "| Kicks survived |"
    )
    print("|---|---|---|---|")
    for speed in KICKS:
        r = [kicks(folder, prefix, s, speed) for s in SEEDS]
        k, n = sum(x[0] for x in r), sum(x[1] for x in r)
        print(
            f"| {speed} m/s | "
            + " / ".join(str(x[0]) for x in r)
            + f" | {interval(k, n)} | "
            + " / ".join(f"{100 * x[2]:.1f}%" for x in r)
            + " |"
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dir", type=Path)
    parser.add_argument("prefix")
    parser.add_argument("--name", default="this recipe")
    parser.add_argument("--compare", action="append", default=[], help="DIR:PREFIX:NAME")
    args = parser.parse_args(argv)
    finish_line(args.dir, args.prefix)
    print()
    recipes = [(Path(d), p, n) for d, p, n in (c.split(":", 2) for c in args.compare)]
    falls_table([*recipes, (args.dir, args.prefix, args.name)])
    print()
    kick_table(args.dir, args.prefix)


if __name__ == "__main__":
    main()

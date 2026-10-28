"""Compare policy variants from several `pi-wbc-sim2sim` result files.

    uv run pi-wbc-sim2sim-compare results/stage0/reward_pilot/*.json \
        --model menagerie --scenario walk_1.0

Each result file holds one trained policy and carries a label `<variant>-s<seed>`; several
runs may share a label. Files with the same variant are grouped: falls are pooled over all
their commands, models and episodes (with a 95% Wilson interval), and tracking error and gait
are averaged per run on one model and command, then over runs. With a few training runs per
variant, differences smaller than the spread between runs are not evidence of anything.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any

from pi_wbc.sim2sim.evaluate import GAIT_COLUMNS, format_gait
from pi_wbc.sim2sim.gait import METRICS
from pi_wbc.stats import wilson_interval

LABEL = re.compile(r"^(?P<variant>.+)-s(?P<seed>\d+)$")


def variant_of(label: str) -> tuple[str, int | None]:
    m = LABEL.match(label)
    return (m["variant"], int(m["seed"])) if m else (label, None)


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def run_name(result: dict[str, Any]) -> str:
    """The training run folder a result's policy came from (.../<run>/exported/policy.onnx)."""
    return Path(result["policy"]).parent.parent.name


def compare(
    results: list[dict[str, Any]], model: str, scenario: str, per_run: bool = False
) -> list[dict[str, Any]]:
    """One row per variant (or per training run), in first-seen order."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in results:
        label = r.get("label") or run_name(r)
        key = f"{label} ({run_name(r)})" if per_run else variant_of(label)[0]
        groups.setdefault(key, []).append(r)

    rows = []
    for variant, runs in groups.items():
        falls = sum(row["falls"] for r in runs for row in r["summary"])
        episodes = sum(row["episodes"] for r in runs for row in r["summary"])
        picked = [
            row
            for r in runs
            for row in r["summary"]
            if row["model"] == model and row["scenario"] == scenario
        ]
        values = {
            k: [row["gait"][k] for row in picked if row["gait"].get(k) is not None] for k in METRICS
        }
        lo, hi = wilson_interval(falls, episodes)
        seeds = sorted({s for s in (variant_of(r.get("label") or "")[1] for r in runs) if s})
        lin = [p["lin_err_mean"] for p in picked]
        rows.append(
            {
                "variant": variant,
                "runs": len(runs),
                "seeds": seeds,
                "falls": falls,
                "episodes": episodes,
                "fall_rate_ci95": [lo, hi],
                "lin_err_mean": _mean([v for v in lin if v is not None]),
                "lin_err_runs": lin,
                "gait": {k: _mean(v) for k, v in values.items()},
                "gait_min": {k: min(v) if v else None for k, v in values.items()},
                "gait_max": {k: max(v) if v else None for k, v in values.items()},
                "gait_runs": [p["gait"] for p in picked],
            }
        )
    return rows


def _number(cell: str) -> str:
    """The number at the start of a formatted cell, without its unit."""
    m = re.match(r"-?[\d.]+", cell)
    return m.group(0) if m else cell


def markdown_table(rows: list[dict[str, Any]], spread: bool = True) -> str:
    """With `spread`, a group of several runs shows its mean and (lowest-highest) run."""
    headers = [h for h, *_ in GAIT_COLUMNS]
    out = [
        "| Variant | Runs | Seeds | Falls (all commands) | Speed error | "
        + " | ".join(headers)
        + " |",
        "|---|---:|---|---|---:|" + "---:|" * len(headers),
    ]
    for r in rows:
        lo, hi = r["fall_rate_ci95"]
        lin = "—" if r["lin_err_mean"] is None else f"{r['lin_err_mean']:.2f} m/s"
        cells = format_gait(r["gait"])
        if spread and r["runs"] > 1:
            lows, highs = format_gait(r["gait_min"]), format_gait(r["gait_max"])
            cells = [
                c if c == "—" else f"{c} ({_number(a)}-{_number(b)})"
                for c, a, b in zip(cells, lows, highs, strict=True)
            ]
        seeds = ", ".join(str(s) for s in r["seeds"])
        out.append(
            f"| {r['variant']} | {r['runs']} | {seeds} "
            f"| {r['falls']}/{r['episodes']} ({lo:.0%}-{hi:.0%}) "
            f"| {lin} | " + " | ".join(cells) + " |"
        )
    return "\n".join(out)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("results", type=Path, nargs="+", help="pi-wbc-sim2sim result files")
    parser.add_argument("--model", default="menagerie", help="model for tracking and gait")
    parser.add_argument("--scenario", default="walk_1.0", help="command for tracking and gait")
    parser.add_argument("--per-run", action="store_true", help="one row per training run")
    parser.add_argument("--out", type=Path, default=None, help="also write the rows as JSON")
    args = parser.parse_args(argv)

    results = [json.loads(p.read_text()) for p in args.results]
    rows = compare(results, args.model, args.scenario, per_run=args.per_run)
    print(f"Tracking and gait: model {args.model}, command {args.scenario}\n")
    print(markdown_table(rows))
    if args.out:
        from pi_wbc import provenance

        provenance.save_json(
            args.out,
            {
                "model": args.model,
                "scenario": args.scenario,
                "inputs": [str(p) for p in args.results],
                "rows": rows,
            },
        )
        print(f"\n[compare] wrote {args.out}")


if __name__ == "__main__":
    main()

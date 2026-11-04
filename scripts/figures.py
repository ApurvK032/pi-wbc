"""Figures for the README and results/stage1/README.md, from the result files in results/.

    .venv/bin/python scripts/figures.py          (from the repository root)

Writes results/figures/<name>-light.png and <name>-dark.png (transparent background, ink for
GitHub's light and dark themes):

- turning: turn rate reached in place against the rate asked, MuJoCo turn sweep, before and
  after the longer full-range teacher training (D-026 against D-028).
- kicks: share of kicks survived against kick speed, MuJoCo test seeds, the B2 students
  (D-022) against the stage 1 controller (D-028).
- ablation: falls on the MuJoCo test seeds when one part of training is switched off (step 6).
- tracking: one run through a sequence of commands in MuJoCo, asked against reached, and the
  footfalls (scripts/tracking_trace.py).
- training: the teachers' curriculum level and turn-rate error over training (D-022 to D-028;
  scripts/teacher_curves.py).
- engines: tracking errors of the stage 1 controller on Newton, PhysX and the two MuJoCo models.
- falls: falls per test condition on the MuJoCo test seeds, three generations of students.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm

from pi_wbc.provenance import load_json
from pi_wbc.sim2sim.evaluate import kick_counts

R = Path("results/stage1")
OUT = Path("results/figures")
SEEDS = (1, 2, 3)

# Reference palette (slots 1 and 2 of the validated categorical order) and chart ink.
THEMES = {
    "light": {
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "after": "#2a78d6",
        "before": "#eb6834",
        "third": "#1baf7a",
        "ramp": ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
    },
    "dark": {
        "ink": "#f0efec",
        "ink2": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "after": "#3987e5",
        "before": "#d95926",
        "third": "#199e70",
        "ramp": ["#104281", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"],
    },
}


def style(ax, t):
    ax.set_facecolor("none")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["axis"])
    ax.tick_params(colors=t["ink2"], labelsize=9, length=0, pad=6)
    ax.grid(True, color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def finish(fig, ax, t, title, subtitle):
    fig.patch.set_alpha(0)
    ax.set_title(title, loc="left", color=t["ink"], fontsize=12, fontweight="bold", pad=24)
    ax.text(0, 1.03, subtitle, transform=ax.transAxes, color=t["ink2"], fontsize=9, va="bottom")


# --- turning -----------------------------------------------------------------------------


def sweep_in_place(path: Path, policy_filter=None) -> dict[float, float]:
    """Mean turn rate reached over both MuJoCo models, per in-place command."""
    rows = [r for r in load_json(path)["rows"] if r["command"][0] == 0.0]
    if policy_filter:
        rows = [r for r in rows if policy_filter(r["policy"])]
    out: dict[float, list[float]] = {}
    for r in rows:
        out.setdefault(r["command"][2], []).append(r["achieved_yaw_rate"])
    assert all(len(v) == 2 for v in out.values()), path
    return {k: sum(v) / 2 for k, v in sorted(out.items())}


def turning(t, mode):
    before = [
        sweep_in_place(R / f"full_range/turn_sweep/turn_sweep_s{s}.json", lambda p: "25" in p)
        for s in SEEDS
    ]
    after = [
        sweep_in_place(R / f"full_range_long/turn_sweep/turn_sweep_long-s{s}.json") for s in SEEDS
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=150)
    style(ax, t)
    xs = sorted(before[0])
    ax.plot([0, 1.05], [0, 1.05], color=t["muted"], linewidth=1.2, linestyle=(0, (4, 3)))
    ax.text(0.93, 0.98, "asked", color=t["muted"], fontsize=9, ha="right", va="bottom", rotation=0)
    for runs, key, label in (
        (before, "before", "1000 iterations (D-026)"),
        (after, "after", "7000 iterations (D-028)"),
    ):
        mean = [sum(r[x] for r in runs) / 3 for x in xs]
        lo = [min(r[x] for r in runs) for x in xs]
        hi = [max(r[x] for r in runs) for x in xs]
        ax.fill_between(xs, lo, hi, color=t[key], alpha=0.18, linewidth=0)
        ax.plot(
            xs,
            mean,
            color=t[key],
            linewidth=2,
            marker="o",
            markersize=6,
            label=label,
            markeredgecolor="none",
        )
        ax.text(xs[-1] + 0.02, mean[-1], label, color=t["ink"], fontsize=9, va="center")
    ax.set_xlim(0.15, 1.42)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(xs)
    ax.set_xlabel("turn rate asked, in place (rad/s)", color=t["ink2"], fontsize=9)
    ax.set_ylabel("turn rate reached (rad/s)", color=t["ink2"], fontsize=9)
    leg = ax.legend(
        loc="upper left", frameon=False, fontsize=9, title="teacher trained at full ranges for"
    )
    plt.setp(leg.get_texts(), color=t["ink"])
    leg.get_title().set_color(t["ink2"])
    leg.get_title().set_fontsize(9)
    finish(
        fig,
        ax,
        t,
        "Longer teacher training brings turning back",
        "MuJoCo turn sweep, mean of both robot models; "
        "line: mean of 3 training seeds, band: their range",
    )
    fig.tight_layout()
    fig.savefig(OUT / f"turning-{mode}.png", transparent=True)
    plt.close(fig)


# --- kicks -------------------------------------------------------------------------------

KICKS = ("0.5", "1.0", "1.5", "2.0", "2.5", "3.0")


def kicks_survived(path: Path) -> float | None:
    try:
        d = load_json(path)
    except (OSError, ValueError):  # missing, or a run still going
        return None
    falls, given = kick_counts(d["episodes"], d["condition_settings"]["push_times"])
    return 1 - falls / given


def kicks(t, mode):
    recipes = (
        ("before", "B2 students (D-022)", R / "speed_fix_test/push", "speed-fine05"),
        ("after", "stage 1 controller (D-028)", R / "full_range_long_test/push", "full-range-long"),
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=150)
    style(ax, t)
    xs = [float(k) for k in KICKS]
    ax.axhline(95, color=t["muted"], linewidth=1.2, linestyle=(0, (4, 3)))
    ax.text(0.55, 95.6, "95% of kicks survived", color=t["muted"], fontsize=9, va="bottom")
    for key, label, folder, prefix in recipes:
        per = [
            [kicks_survived(folder / f"{prefix}-s{s}-push_{k}.json.gz") for k in KICKS]
            for s in SEEDS
        ]
        if any(v is None for row in per for v in row):
            print(f"kicks: {label} incomplete, left out")
            continue
        mean = [100 * sum(per[i][j] for i in range(3)) / 3 for j in range(len(KICKS))]
        lo = [100 * min(per[i][j] for i in range(3)) for j in range(len(KICKS))]
        hi = [100 * max(per[i][j] for i in range(3)) for j in range(len(KICKS))]
        ax.fill_between(xs, lo, hi, color=t[key], alpha=0.18, linewidth=0)
        ax.plot(
            xs,
            mean,
            color=t[key],
            linewidth=2,
            marker="o",
            markersize=6,
            label=label,
            markeredgecolor="none",
        )
        ax.text(xs[-1] + 0.06, mean[-1], label, color=t["ink"], fontsize=9, va="center")
    ax.set_xlim(0.35, 4.2)
    ax.set_ylim(60, 101)
    ax.set_xticks(xs)
    ax.set_xlabel(
        "kick: speed added to the pelvis in a random horizontal direction (m/s)",
        color=t["ink2"],
        fontsize=9,
    )
    ax.set_ylabel("kicks survived (%)", color=t["ink2"], fontsize=9)
    leg = ax.legend(loc="lower left", frameon=False, fontsize=9)
    plt.setp(leg.get_texts(), color=t["ink"])
    finish(
        fig,
        ax,
        t,
        "Push recovery in plain MuJoCo",
        "MuJoCo test seeds, 5 kicks per episode; line: mean of 3 training seeds, band: range",
    )
    fig.tight_layout()
    fig.savefig(OUT / f"kicks-{mode}.png", transparent=True)
    plt.close(fig)


# --- ablation ----------------------------------------------------------------------------

ABLATIONS = (
    ("nothing (full recipe)", R / "student_test_ramp", "ts-full-ramp"),
    ("kicks", R / "ablation_test", "abl-pushes"),
    ("friction", R / "ablation_test", "abl-friction"),
    ("masses", R / "ablation_test", "abl-mass"),
    ("motor gains", R / "ablation_test", "abl-actuators"),
    ("action delay", R / "ablation_test", "abl-latency"),
    ("sensor noise", R / "ablation_test", "abl-sensor-noise"),
    ("all six groups", R / "ablation_test", "abl-none"),
    ("curriculum", R / "ablation_test", "abl-curriculum"),
    ("teacher (PPO alone)", R / "ablation_test", "abl-teacher"),
)
CONDITIONS = (
    ("sensor noise", "noise"),
    ("1 kg per hand", "payload_1kg"),
    ("20 ms delay", "delay_20ms"),
    ("1 m/s kicks", "push_1.0"),
    ("2 m/s kicks", "push_2.0"),
)


def fall_rate(folder: Path, prefix: str, cond: str) -> float:
    falls = n = 0
    for s in SEEDS:
        d = load_json(folder / f"{prefix}-s{s}-{cond}.json.gz")
        falls += sum(x["falls"] for x in d["summary"])
        n += sum(x["episodes"] for x in d["summary"])
    return 100 * falls / n


def ablation(t, mode):
    grid = [[fall_rate(f, p, c) for _, c in CONDITIONS] for _, f, p in ABLATIONS]
    cmap = LinearSegmentedColormap.from_list("seq", t["ramp"])
    norm = LogNorm(vmin=0.01, vmax=100)
    fig, ax = plt.subplots(figsize=(7.6, 5.0), dpi=150)
    ax.set_facecolor("none")
    for i, row in enumerate(grid):
        for j, v in enumerate(row):
            c = cmap(norm(max(v, 0.01)))
            ax.add_patch(plt.Rectangle((j + 0.03, i + 0.04), 0.94, 0.92, color=c, linewidth=0))
            lum = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
            txt = f"{v:.1f}%" if v >= 1 else (f"{v:.2f}%" if v > 0 else "0%")
            ax.text(
                j + 0.5,
                i + 0.5,
                txt,
                ha="center",
                va="center",
                fontsize=9,
                color="#0b0b0b" if lum > 0.5 else "#ffffff",
            )
    ax.set_xlim(0, len(CONDITIONS))
    ax.set_ylim(len(ABLATIONS), 0)
    ax.set_xticks([j + 0.5 for j in range(len(CONDITIONS))], [c for c, _ in CONDITIONS])
    ax.set_yticks([i + 0.5 for i in range(len(ABLATIONS))], [a for a, _, _ in ABLATIONS])
    ax.xaxis.tick_top()
    ax.tick_params(colors=t["ink"], labelsize=9, length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.axhline(1, color=t["ink2"], linewidth=1)
    ax.axhline(8, color=t["ink2"], linewidth=1)
    ax.text(
        -0.012,
        1.0,
        "trained without",
        transform=ax.transAxes,
        color=t["ink2"],
        fontsize=9,
        ha="right",
        va="bottom",
        style="italic",
    )
    fig.patch.set_alpha(0)
    fig.text(
        0.02,
        0.975,
        "What each part of training buys",
        color=t["ink"],
        fontsize=12,
        fontweight="bold",
        va="top",
    )
    fig.text(
        0.02,
        0.928,
        "Episodes that fell on the MuJoCo test seeds, 3 training seeds pooled (13,200 per cell)",
        color=t["ink2"],
        fontsize=9,
        va="top",
    )
    fig.subplots_adjust(left=0.2, right=0.98, top=0.80, bottom=0.02)
    fig.savefig(OUT / f"ablation-{mode}.png", transparent=True)
    plt.close(fig)


# --- tracking ----------------------------------------------------------------------------

SEGMENTS = (
    (0, 2, "stand"),
    (2, 6, "walk 0.5"),
    (6, 10, "walk 1.0"),
    (10, 14, "walk + turn"),
    (14, 18, "turn in place"),
    (18, 21, "sideways"),
    (21, 25, "crouch"),
    (25, 28, "stand"),
)


def moving_average(x: list[float], n: int) -> list[float]:
    out, acc = [], 0.0
    for i, v in enumerate(x):
        acc += v
        if i >= n:
            acc -= x[i - n]
        out.append(acc / min(i + 1, n))
    # centre the window
    shift = n // 2
    return out[shift:] + [out[-1]] * shift


def tracking(t, mode, seed=2):
    d = load_json(R / f"full_range_long/tracking_trace_s{seed}.json.gz")
    rows = d["rows"]
    ts = [r["t"] for r in rows]
    panels = (
        ("forward speed\n(m/s)", [r["command"][0] for r in rows], [r["vx"] for r in rows]),
        ("sideways speed\n(m/s)", [r["command"][1] for r in rows], [r["vy"] for r in rows]),
        ("turn rate\n(rad/s)", [r["command"][2] for r in rows], [r["yaw_rate"] for r in rows]),
        ("pelvis height\n(m)", [r["height_command"] for r in rows], [r["height"] for r in rows]),
    )
    fig, axes = plt.subplots(
        5,
        1,
        figsize=(7.6, 7.4),
        dpi=150,
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1, 1, 1, 0.55], "hspace": 0.18},
    )
    for ax, (label, asked, reached) in zip(axes, panels, strict=False):
        style(ax, t)
        ax.plot(ts, reached, color=t["after"], linewidth=0.8, alpha=0.35)
        ax.plot(ts, moving_average(reached, 25), color=t["after"], linewidth=2, label="reached")
        ax.plot(ts, asked, color=t["ink"], linewidth=1.3, linestyle=(0, (4, 2)), label="asked")
        ax.set_ylabel(label, color=t["ink2"], fontsize=8.5)
        ax.yaxis.set_label_coords(-0.07, 0.5)
    axes[1].set_ylim(-0.2, 0.5)
    axes[3].set_ylim(0.5, 0.85)
    # footfalls
    ax = axes[4]
    style(ax, t)
    ax.grid(False)
    dt = d["policy_dt"]
    for i, y in enumerate((1, 0)):  # left foot on top
        spans, start = [], None
        for r in rows:
            down = r["feet_down"][i]
            if down and start is None:
                start = r["t"] - dt
            if not down and start is not None:
                spans.append((start, r["t"] - dt - start))
                start = None
        if start is not None:
            spans.append((start, rows[-1]["t"] - start))
        ax.broken_barh(
            spans, (y + 0.15, 0.7), color=t["after"] if i == 0 else t["before"], linewidth=0
        )
    ax.set_yticks([0.5, 1.5], ["right foot", "left foot"])
    ax.set_ylim(0, 2)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_xlabel("time (s)", color=t["ink2"], fontsize=9)
    ax.set_xlim(0, 28)
    ax.set_xticks(range(0, 29, 2))
    for a in axes:
        for x0, _, _ in SEGMENTS[1:]:
            a.axvline(x0, color=t["axis"], linewidth=0.8)
    for x0, x1, name in SEGMENTS:
        axes[0].text(
            (x0 + x1) / 2,
            1.04,
            name,
            transform=axes[0].get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=8,
            color=t["ink2"],
        )
    leg = axes[0].legend(loc="center right", frameon=False, fontsize=8.5)
    plt.setp(leg.get_texts(), color=t["ink"])
    fig.patch.set_alpha(0)
    fig.text(
        0.02,
        0.985,
        "Following a sequence of commands in plain MuJoCo",
        color=t["ink"],
        fontsize=12,
        fontweight="bold",
        va="top",
    )
    fig.text(
        0.02,
        0.955,
        f"Stage 1 controller (training seed {seed}) on MuJoCo Menagerie's G1, "
        "nominal robot; thick line: 0.5 s moving average",
        color=t["ink2"],
        fontsize=9,
        va="top",
    )
    fig.subplots_adjust(left=0.15, right=0.98, top=0.88, bottom=0.07)
    fig.savefig(OUT / f"tracking-{mode}.png", transparent=True)
    plt.close(fig)


# --- training ----------------------------------------------------------------------------


def training(t, mode):
    d = load_json(R / "full_range_long/teacher_training.json.gz")
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(7.6, 5.4),
        dpi=150,
        sharex=True,
        gridspec_kw={"height_ratios": [0.7, 1.3], "hspace": 0.12},
    )
    colors = (t["after"], t["before"], t["third"])
    for (seed, run), color in zip(d["seeds"].items(), colors, strict=True):
        it = run["iteration"]
        for ax, key, n in ((ax1, "curriculum_level", 1), (ax2, "turn_rate_error", 100)):
            ys = [v if v is not None else float("nan") for v in run[key]]
            ax.plot(it, moving_average(ys, n), color=color, linewidth=1.6, label=f"seed {seed}")
    for ax in (ax1, ax2):
        style(ax, t)
        ax.axvspan(0, 2999, color=t["grid"], alpha=0.45, linewidth=0)
        for x in (2999, 3999):
            ax.axvline(x, color=t["muted"], linewidth=1, linestyle=(0, (4, 3)))
    ax1.set_ylabel("curriculum level", color=t["ink2"], fontsize=8.5)
    ax2.set_ylabel("turn-rate error in training (rad/s)", color=t["ink2"], fontsize=8.5)
    for ax in (ax1, ax2):
        ax.yaxis.set_label_coords(-0.07, 0.5)
    ax2.set_ylim(0.6, 1.1)
    ax1.set_ylim(0, 1.05)
    ax2.set_xlabel("PPO iteration (4096 robots each)", color=t["ink2"], fontsize=9)
    ax2.set_xlim(0, 10000)
    top = ax1.get_xaxis_transform()
    ax1.text(
        1500,
        1.04,
        "first training (D-022)",
        transform=top,
        ha="center",
        va="bottom",
        fontsize=8,
        color=t["ink2"],
    )
    ax1.text(
        6500,
        1.04,
        "at the full command ranges: 7000 iterations (D-028)",
        transform=top,
        ha="center",
        va="bottom",
        fontsize=8,
        color=t["ink2"],
    )
    ax2.text(
        4080,
        1.08,
        "1000 iterations:\nstopped here before (D-024)",
        ha="left",
        va="top",
        fontsize=8,
        color=t["muted"],
    )
    leg = ax2.legend(loc="upper right", frameon=False, fontsize=8.5)
    plt.setp(leg.get_texts(), color=t["ink"])
    fig.patch.set_alpha(0)
    fig.text(
        0.02,
        0.985,
        "The teachers keep improving their turning long after the ranges are full",
        color=t["ink"],
        fontsize=12,
        fontweight="bold",
        va="top",
    )
    fig.text(
        0.02,
        0.945,
        "Teacher training, 3 seeds; turn-rate error on random training commands, "
        "100-iteration average",
        color=t["ink2"],
        fontsize=9,
        va="top",
    )
    fig.subplots_adjust(left=0.17, right=0.98, top=0.85, bottom=0.1)
    fig.savefig(OUT / f"training-{mode}.png", transparent=True)
    plt.close(fig)


# --- engines -----------------------------------------------------------------------------

ENGINE_MEASURES = (
    ("speed error (m/s)\nwalking 0.5 m/s", "walk_0.5", "lin_err_mean"),
    ("speed error (m/s)\nwalking 1 m/s", "walk_1.0", "lin_err_mean"),
    ("speed error (m/s)\nsideways 0.3 m/s", "side_0.3", "lin_err_mean"),
    ("turn-rate error (rad/s)\nturning 0.8 rad/s in place", "turn_0.8", "ang_err_mean"),
)


def engines(t, mode):
    def isaac(engine, s):
        rows = json.loads((R / f"physx/long-s{s}-{engine}-50.json").read_text())["summary"]
        return {r["scenario"]: r for r in rows}

    def mujoco(model, s):
        path = R / f"full_range_long/full-range-long-s{s}-nominal.json"
        rows = json.loads(path.read_text())["summary"]
        return {r["scenario"]: r for r in rows if r["model"] == model}

    sims = (
        ("Isaac Lab, Newton (training)", lambda s: isaac("newton", s)),
        ("Isaac Sim, PhysX", lambda s: isaac("isaacsim", s)),
        ("MuJoCo, Menagerie's G1", lambda s: mujoco("menagerie", s)),
        ("MuJoCo, Newton's G1 model", lambda s: mujoco("newton", s)),
    )
    data = {name: {s: f(s) for s in SEEDS} for name, f in sims}
    fig, axes = plt.subplots(1, 4, figsize=(9.6, 3.3), dpi=150, sharey=True)
    for ax, (label, scen, key) in zip(axes, ENGINE_MEASURES, strict=True):
        style(ax, t)
        ax.grid(axis="y", visible=False)
        for i, (name, _) in enumerate(sims):
            vals = [data[name][s][scen][key] for s in SEEDS]
            y = len(sims) - 1 - i
            ax.scatter(vals, [y] * 3, s=30, color=t["after"], alpha=0.75, linewidths=0, zorder=3)
            m = sum(vals) / 3
            ax.plot([m, m], [y - 0.28, y + 0.28], color=t["ink"], linewidth=2, zorder=4)
        ax.set_title(label, color=t["ink2"], fontsize=8.5, loc="left")
        top_value = max(data[n][s][scen][key] for n, _ in sims for s in SEEDS)
        ax.set_xlim(0, top_value * 1.2)
    axes[0].set_yticks(range(len(sims)), [n for n, _ in reversed(sims)])
    axes[0].set_ylim(-0.6, len(sims) - 0.4)
    fig.patch.set_alpha(0)
    fig.text(
        0.01,
        0.98,
        "One policy, three simulators, four setups",
        color=t["ink"],
        fontsize=12,
        fontweight="bold",
        va="top",
    )
    fig.text(
        0.01,
        0.9,
        "MuJoCo with two robot models; nominal robot, no retraining; dots: training seeds, "
        "bar: their mean",
        color=t["ink2"],
        fontsize=9,
        va="top",
    )
    fig.subplots_adjust(left=0.22, right=0.99, top=0.7, bottom=0.12, wspace=0.12)
    fig.savefig(OUT / f"engines-{mode}.png", transparent=True)
    plt.close(fig)


# --- falls by condition ------------------------------------------------------------------

FALL_RECIPES = (
    ("warm-up students (D-021)", R / "student_test_ramp", "ts-full-ramp", "third"),
    ("B2 students (D-022)", R / "speed_fix_test", "speed-fine05", "before"),
    ("stage 1 controller (D-028)", R / "full_range_long_test", "full-range-long", "after"),
)
FALL_CONDITIONS = (
    ("nominal", "nominal"),
    ("floor friction 0.4", "friction_0.4"),
    ("1 kg in each hand", "payload_1kg"),
    ("20 ms extra delay", "delay_20ms"),
    ("sensor noise", "noise"),
    ("1 m/s kicks", "push_1.0"),
    ("2 m/s kicks", "push_2.0"),
)


def falls(t, mode):
    from pi_wbc.stats import wilson_interval

    def pooled(folder, prefix, cond):
        k = n = 0
        for s in SEEDS:
            try:
                d = load_json(folder / f"{prefix}-s{s}-{cond}.json.gz")
            except (OSError, ValueError):
                return None
            k += sum(x["falls"] for x in d["summary"])
            n += sum(x["episodes"] for x in d["summary"])
        return k, n

    fig, ax = plt.subplots(figsize=(7.6, 4.8), dpi=150)
    style(ax, t)
    ax.grid(axis="y", visible=False)
    h = 0.26
    shown = 0
    for i, (label, folder, prefix, key) in enumerate(FALL_RECIPES):
        rows = [pooled(folder, prefix, c) for _, c in FALL_CONDITIONS]
        if any(r is None for r in rows):
            print(f"falls: {label} incomplete, left out")
            continue
        shown += 1
        for j, (k, n) in enumerate(rows):
            y = j + (i - 1) * h
            p = 100 * k / n
            lo, hi = (100 * v for v in wilson_interval(k, n))
            x = max(p, 0.004)
            ax.plot([max(lo, 0.004), hi], [y, y], color=t[key], linewidth=1.2, zorder=3)
            ax.scatter(
                [x],
                [y],
                s=26,
                color=t[key],
                zorder=4,
                linewidths=0,
                label=label if j == 0 else None,
            )
    ax.set_xscale("log")
    ax.set_xlim(0.004, 100)
    ax.set_xticks([0.01, 0.1, 1, 10, 100], ["0.01%", "0.1%", "1%", "10%", "100%"])
    ax.set_yticks(range(len(FALL_CONDITIONS)), [c for c, _ in FALL_CONDITIONS])
    ax.invert_yaxis()
    ax.set_xlabel(
        "episodes that fell (log scale; line: 95% Wilson interval; 0 falls drawn at the left edge)",
        color=t["ink2"],
        fontsize=8.5,
    )
    leg = ax.legend(loc="upper right", frameon=False, fontsize=8.5)
    plt.setp(leg.get_texts(), color=t["ink"])
    fig.patch.set_alpha(0)
    fig.text(
        0.02,
        0.985,
        "Falls under each test condition",
        color=t["ink"],
        fontsize=12,
        fontweight="bold",
        va="top",
    )
    fig.text(
        0.02,
        0.94,
        "MuJoCo test seeds 1000-1199; 3 training seeds pooled, 13,200 episodes per dot",
        color=t["ink2"],
        fontsize=9,
        va="top",
    )
    fig.subplots_adjust(left=0.22, right=0.98, top=0.86, bottom=0.12)
    if shown:
        fig.savefig(OUT / f"falls-{mode}.png", transparent=True)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for mode, t in THEMES.items():
        turning(t, mode)
        kicks(t, mode)
        ablation(t, mode)
        tracking(t, mode)
        training(t, mode)
        engines(t, mode)
        falls(t, mode)
    print("wrote", ", ".join(sorted(p.name for p in OUT.glob("*.png"))))


if __name__ == "__main__":
    main()

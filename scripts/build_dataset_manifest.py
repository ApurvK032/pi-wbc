"""Choose a dataset's episodes by D-042's rule and write its manifest.

Inputs, one triple per recording run: the expert's result file (scripts/expert_rollout.py
--out), the replay check's result file (scripts/replay_check.py --out) and the recordings
folder. Every recorded episode must have been replayed. The chosen episodes are the successful
ones whose replay passes the first-second check, in seed order, the first `--count` of them (all
if not given). Failures stay in the recordings, with their labels, for stage 5.

    uv run python scripts/build_dataset_manifest.py --name "pi-wbc stage 3 training set v1" \\
        --run logs/expert/train-v1.json logs/replay/train-v1.json logs/dataset/train-v1 \\
        --count 1000 --out results/stage3/dataset/train-v1-manifest.json
"""

import argparse
import collections
import json
from pathlib import Path

from pi_wbc import provenance
from pi_wbc.stats import wilson_interval


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", required=True)
    parser.add_argument("--run", nargs=3, action="append", required=True,
                        metavar=("EXPERT_JSON", "REPLAY_JSON", "RECORDINGS"))  # fmt: skip
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    episodes, runs = [], []
    for expert_path, replay_path, rec_dir in args.run:
        expert = json.loads(Path(expert_path).read_text())
        replay = json.loads(Path(replay_path).read_text())
        rec_dir = Path(rec_dir)
        replayed = {r["seed"]: r for r in replay["results"]}
        files = {int(p.stem) for p in rec_dir.glob("*.npz")}
        seeds = [r["seed"] for r in expert["results"]]
        if set(seeds) != files or set(replayed) != files:
            raise SystemExit(f"{rec_dir}: the expert's, the replay's and the recorded seeds differ")
        if expert.get("low_level", "ours") != replay.get("low_level", "ours"):
            raise SystemExit(f"{rec_dir}: recorded and replayed with different controllers")
        for r in expert["results"]:
            rp = replayed[r["seed"]]
            if rp["recorded"] != r["label"]:
                raise SystemExit(f"seed {r['seed']}: expert says {r['label']}, recording "
                                 f"{rp['recorded']}")  # fmt: skip
            episodes.append({
                "seed": r["seed"], "file": f"{rec_dir}/{r['seed']}.npz", "label": r["label"],
                "failed_in": r["failed_in"], "hand": r["notes"].get("hand"),
                "time_s": r["time_s"], "replay_first_second_drift_mm": rp["first_second_drift_mm"],
                "replay_passes": rp["passes"], "replay_label": rp["replayed"],
            })  # fmt: skip
        runs.append({
            "expert": expert_path, "replay": replay_path, "recordings": str(rec_dir),
            "low_level": expert.get("low_level", "ours"),
            "controller_run": expert["controller_run"],
            "record_commit": expert["provenance"]["git"]["commit"],
            "record_dirty": expert["provenance"]["git"]["dirty"],
            "replay_commit": replay["provenance"]["git"]["commit"],
            "replay_dirty": replay["provenance"]["git"]["dirty"],
            "expert_params": expert["expert_params"],
        })  # fmt: skip
    episodes.sort(key=lambda e: e["seed"])
    if len({e["seed"] for e in episodes}) != len(episodes):
        raise SystemExit("a seed was recorded twice")
    params = {json.dumps(r["expert_params"], sort_keys=True) for r in runs}
    if len(params) != 1:
        raise SystemExit("the runs used different expert parameters")
    good = [e for e in episodes if e["label"] == "success" and e["replay_passes"]]
    chosen = good[: args.count] if args.count is not None else good
    if args.count is not None and len(chosen) < args.count:
        raise SystemExit(f"only {len(chosen)} episodes qualify, {args.count} asked for")
    kept = sum(e["replay_label"] == "success" for e in chosen)
    lo, hi = wilson_interval(kept, len(chosen))
    first = episodes[0]["seed"]
    rule = ("D-042: the successful episodes, in seed order from seed "
            f"{first}, whose replay passes the first-second check (drift under 10 mm)")  # fmt: skip
    if args.count is not None:
        rule = rule.replace("the successful", f"the first {args.count:,} successful")
    provenance.save_json(args.out, {
        "name": args.name, "rule": rule, "runs": runs,
        "recorded": len(episodes),
        "successes": sum(e["label"] == "success" for e in episodes),
        "labels": dict(collections.Counter(e["label"] for e in episodes)),
        "replay_passes": sum(e["replay_passes"] for e in episodes),
        "dropped_by_replay_check": [e["seed"] for e in episodes
                                    if e["label"] == "success" and not e["replay_passes"]],
        "chosen_seeds": [e["seed"] for e in chosen],
        "chosen_seed_range": [chosen[0]["seed"], chosen[-1]["seed"]],
        "chosen_hands": dict(collections.Counter(e["hand"] for e in chosen)),
        "chosen_replay_outcome_agreement": {"success_on_replay": kept, "of": len(chosen),
                                            "ci95": [lo, hi]},
        "episodes": episodes,
    })  # fmt: skip
    span = f"{chosen[0]['seed']}-{chosen[-1]['seed']}"
    print(f"[manifest] {len(chosen)} chosen of {len(episodes)} recorded (seeds {span}); "
          f"{kept} also succeed on replay")  # fmt: skip


if __name__ == "__main__":
    main()

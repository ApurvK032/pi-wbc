"""Judge recorded expert episodes again with the shared judge (D-052) and compare with the
expert's own labels.

The recordings hold the pelvis pose, the joint angles and every object's pose at each control
step. The palms are not recorded: they are computed from the pelvis pose and joint angles on the
controller's own MuJoCo model (`pi_wbc.stage3.arm_ik`). A recording ends where the expert's
episode ended (the end of its retract, or its failure), so the judge's last few hundred
milliseconds after a placement are not seen; `finish()` labels the episode there.

    uv run python scripts/judge_recordings.py --recordings logs/dataset/train-v1 \\
        --out results/stage4/judge/train-v1.json
"""

import argparse
import collections
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from pi_wbc import provenance
from pi_wbc.stage3 import layout as L
from pi_wbc.stage3.judge import Judge, JudgeParams, up_z_of

RUN = Path("logs/rsl_rl/pi_wbc_g1_student/slurm-5628865-full-range-long-r/exported")
_ik = {}


def _init(export: str) -> None:
    from pi_wbc.stage3.arm_ik import ArmIK

    for side in ("left", "right"):
        _ik[side] = ArmIK(Path(export) / "newton_model.xml", Path(export) / "policy_io.json", side)


def palms_of(root: np.ndarray, joint_pos: np.ndarray, names: list[str]) -> np.ndarray:
    """(T, 2, 3) left and right palm positions, in the recording's frame."""
    out = np.zeros((len(root), 2, 3))
    for t in range(len(root)):
        q = dict(zip(names, joint_pos[t].tolist(), strict=True))
        for k, side in enumerate(("left", "right")):
            _ik[side].set_state(root[t, :3], root[t, 3:7], q)
            out[t, k] = _ik[side].palm_pose()[0]
    return out


def judge_one(path: str) -> dict:
    d = np.load(path)
    meta = json.loads(str(d["meta"]))
    lay = L.sample(int(meta["seed"]))
    root, objects = d["root_pose"], d["object_pose"][..., :3]
    palms = palms_of(root, d["joint_pos"], meta["joint_names"])
    dt = float(meta["control_dt"])
    judge = Judge([lay], objects[:1], JudgeParams())
    up = up_z_of(root[:, 3:7])
    for t in range(len(root)):
        judge.step((t + 1) * dt, root[t : t + 1, 2], up[t : t + 1], objects[t : t + 1],
                   palms[t : t + 1])  # fmt: skip
        if judge.done[0]:
            break
    judge.finish(len(root) * dt, "end of recording")
    r = judge.result(0)
    out = meta["outcome"]
    return {
        "seed": lay.seed, "expert": out["label"], "expert_failed_in": out.get("failed_in"),
        "expert_time_s": out.get("time_s"), "judge": r["label"], "judge_time_s": r["time_s"],
        "ended_by": r["ended_by"], "placed_s": r["placed_s"], "steps": len(root),
    }  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--recordings", type=Path, nargs="+", required=True)
    parser.add_argument("--export", type=Path, default=RUN, help="controller export (model)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    files = sorted(str(p) for d in args.recordings for p in d.glob("*.npz"))
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(str(args.export),)) as pool:
        rows = list(pool.map(judge_one, files, chunksize=4))
    pairs = collections.Counter((r["expert"], r["judge"]) for r in rows)
    agree = sum(n for (a, b), n in pairs.items() if a == b)
    same_success = sum(n for (a, b), n in pairs.items() if (a == "success") == (b == "success"))
    print(f"[judge] {len(rows)} episodes: same label {agree}, same success/failure {same_success}")
    for (a, b), n in sorted(pairs.items(), key=lambda x: -x[1]):
        print(f"  expert {a:13s} judge {b:13s} {n:5d}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    provenance.save_json(args.out, {
        "recordings": [str(d) for d in args.recordings], "export": str(args.export),
        "params": JudgeParams().__dict__, "episodes": len(rows), "same_label": agree,
        "same_success": same_success,
        "pairs": [{"expert": a, "judge": b, "n": n} for (a, b), n in sorted(pairs.items())],
        "results": rows,
    })  # fmt: skip


if __name__ == "__main__":
    main()

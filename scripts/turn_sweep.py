"""Turn-rate sweep in MuJoCo: achieved steady yaw rate vs the command, per student and model.

    python scripts/turn_sweep.py OUT.json NAME=EXPORTED_DIR...   (from the repository root)

Development seeds 0-9, nominal condition, 20 s episodes; the steady rate is the bench's yaw
drift (mean yaw rate after settling) plus the command. OUT holds the rows, the settings and the
provenance (commit, code hash, package versions).
"""

import sys

import numpy as np

from pi_wbc import provenance
from pi_wbc.sim2sim.commands import Scenario
from pi_wbc.sim2sim.evaluate import OnnxPolicy, run_episode
from pi_wbc.sim2sim.mujoco_g1 import G1MujocoSim, PolicyIO, build_model

POLICIES = dict(a.split("=", 1) for a in sys.argv[2:])
MENAGERIE = "third_party/mujoco_menagerie/unitree_g1/scene_with_hands.xml"
COMMANDS = [(0.0, 0.0, r) for r in (0.2, 0.4, 0.6, 0.8, 1.0)] + [
    (0.5, 0.0, 0.4),
    (0.5, 0.0, 0.8),
]
SEEDS = range(10)

rows = []
for name, exp in POLICIES.items():
    io = PolicyIO.load(f"{exp}/policy_io.json")
    policy = OnnxPolicy(f"{exp}/policy.onnx")
    for model_name, path in (("menagerie", MENAGERIE), ("newton", f"{exp}/newton_model.xml")):
        sim = G1MujocoSim(build_model(path, io), io)
        for cmd in COMMANDS:
            rates, errs, falls, steps = [], [], 0, []
            for seed in SEEDS:
                result = run_episode(sim, policy, Scenario(cmd), seed, 20.0)
                fell, _, _, ang, gait, drift, _ = result
                if fell:
                    falls += 1
                    continue
                rates.append(cmd[2] + drift)
                errs.append(ang)
                steps.append(gait.get("step_rate_hz") or 0.0)
            row = {
                "policy": name,
                "model": model_name,
                "command": cmd,
                "achieved_yaw_rate": float(np.mean(rates)) if rates else None,
                "ratio": float(np.mean(rates)) / cmd[2] if rates else None,
                "ang_err": float(np.mean(errs)) if errs else None,
                "step_rate_hz": float(np.mean(steps)) if steps else None,
                "falls": falls,
            }
            rows.append(row)
            print(
                f"{name:26s} {model_name:9s} cmd {cmd} -> {row['achieved_yaw_rate']:.2f} rad/s"
                f" ({row['ratio']:.0%}), err {row['ang_err']:.2f},"
                f" steps {row['step_rate_hz']:.1f}/s, falls {falls}",
                flush=True,
            )
provenance.save_json(
    sys.argv[1],
    {
        "policies": POLICIES,
        "menagerie": MENAGERIE,
        "condition": "nominal",
        "seeds": list(SEEDS),
        "duration_s": 20.0,
        "rows": rows,
    },
)

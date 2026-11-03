"""Drive an exported student through a sequence of commands in plain MuJoCo and record what it
was asked and what it did, step by step (for the tracking figure in scripts/figures.py).

    .venv/bin/python scripts/tracking_trace.py EXPORT_DIR OUT.json [--model menagerie|newton]

The sequence (seconds): stand 0-2, walk 0.5 m/s 2-6, walk 1.0 m/s 6-10, walk 0.6 m/s while
turning 0.5 rad/s 10-14, turn in place 0.8 rad/s 14-18, sideways 0.3 m/s 18-21, crouch to a
0.60 m pelvis height 21-25, stand up 25-28. The arms hold their default pose. Nominal robot,
no noise or kicks, reset with seed 0 and heading 0. Every policy step (50 Hz) records the time,
the command, the measured forward and sideways speed (heading frame) and yaw rate, the pelvis
height and which feet carry load.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pi_wbc.provenance import save_json
from pi_wbc.sim2sim.evaluate import OnnxPolicy
from pi_wbc.sim2sim.gait import FOOT_BODIES
from pi_wbc.sim2sim.mujoco_g1 import G1MujocoSim, PolicyIO, build_model, match_name

MENAGERIE = "third_party/mujoco_menagerie/unitree_g1/scene_with_hands.xml"
# (start s, (vx, vy, yaw rate), pelvis height or None for the nominal one)
SEQUENCE = (
    (0.0, (0.0, 0.0, 0.0), None),
    (2.0, (0.5, 0.0, 0.0), None),
    (6.0, (1.0, 0.0, 0.0), None),
    (10.0, (0.6, 0.0, 0.5), None),
    (14.0, (0.0, 0.0, 0.8), None),
    (18.0, (0.0, 0.3, 0.0), None),
    (21.0, (0.0, 0.0, 0.0), 0.60),
    (25.0, (0.0, 0.0, 0.0), None),
)
DURATION_S = 28.0


def segment(t: float) -> tuple[tuple[float, float, float], float | None]:
    current = SEQUENCE[0]
    for row in SEQUENCE:
        if t >= row[0]:
            current = row
    return current[1], current[2]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("export_dir", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--model", choices=("menagerie", "newton"), default="menagerie")
    args = parser.parse_args(argv)

    io = PolicyIO.load(args.export_dir / "policy_io.json")
    policy = OnnxPolicy(args.export_dir / "policy.onnx")
    xml = MENAGERIE if args.model == "menagerie" else args.export_dir / "newton_model.xml"
    sim = G1MujocoSim(build_model(xml, io), io)
    rng = np.random.default_rng(0)
    sim.reset(joint_offsets=rng.uniform(-0.05, 0.05, len(sim.policy_ids)), yaw=0.0)
    nominal = None if sim.body_command is None else sim.body_command.copy()
    max_rate = None if io.body_pose is None else np.array(io.body_pose["max_rates"])
    body_names = [sim.model.body(i).name for i in range(sim.model.nbody)]
    feet = [match_name(body_names, b) for b in FOOT_BODIES]

    dt = sim.policy_dt
    rows = []
    for k in range(round(DURATION_S / dt)):
        t = k * dt
        command, height = segment(t)
        if nominal is not None:
            goal = nominal.copy()
            if height is not None:
                goal[0] = height
            step = max_rate * dt
            sim.body_command = sim.body_command + np.clip(goal - sim.body_command, -step, step)
        sim.step(policy(sim.observation(command)))
        v_xy, yaw_rate = sim.velocity_yaw_frame()
        touching = {
            int(sim.model.geom_bodyid[g]) for c in sim.floor_contacts() for g in (c.geom1, c.geom2)
        }
        rows.append(
            {
                "t": round(t + dt, 4),
                "command": list(command),
                "height_command": None if nominal is None else float(sim.body_command[0]),
                "vx": float(v_xy[0]),
                "vy": float(v_xy[1]),
                "yaw_rate": yaw_rate,
                "height": sim.base_height(),
                "feet_down": [f in touching for f in feet],
            }
        )
        if sim.torso_touches_floor() or sim.base_height() < 0.35:
            print(f"fell at {t + dt:.2f} s")
            break

    out = {
        "export_dir": str(args.export_dir),
        "model": args.model,
        "policy_dt": dt,
        "sequence": [list(map(str, s)) for s in SEQUENCE],
        "rows": rows,
    }
    save_json(args.out, out)
    print(f"wrote {args.out}: {len(rows)} steps")


if __name__ == "__main__":
    main()

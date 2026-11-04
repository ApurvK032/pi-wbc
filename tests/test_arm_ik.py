"""ArmIK on Newton's model of the G1 (needs a controller export; skipped without one)."""

import json
from pathlib import Path

import numpy as np
import pytest

EXPORT = Path("logs/rsl_rl/pi_wbc_g1_student/slurm-5628865-full-range-long-r/exported")
pytestmark = pytest.mark.skipif(
    not (EXPORT / "newton_model.xml").exists(), reason="no controller export on this machine"
)


@pytest.fixture(scope="module")
def ik():
    from pi_wbc.stage3.arm_ik import ArmIK

    solver = ArmIK(EXPORT / "newton_model.xml", EXPORT / "policy_io.json", "right")
    io = json.loads((EXPORT / "policy_io.json").read_text())
    default = {n: io["joints"][n]["default_pos"] for n in io["joint_names"]}
    solver.set_state([0.0, 0.0, 0.78], [0.0, 0.0, 0.0, 1.0], {**default, "waist_pitch_joint": 0.25})
    solver.q_default = np.array([default[n] for n in solver.arm])
    return solver


def test_joint_names_cover_body_and_hands(ik):
    assert "right_hand_thumb_2_joint" in ik._qadr
    assert "waist_pitch_joint" in ik._qadr
    assert len(ik._qadr) == 43


def test_solves_a_reachable_grasp_pose(ik):
    target = np.array([0.315, -0.242, 0.825])  # the grasp test's can grasp, palm facing +y
    r = ik.solve(target, np.eye(3), ik.q_default, iters=400)
    assert r.pos_err < 1e-3 and r.rot_err < 1e-2
    assert np.all(r.q >= ik.lower - 1e-9) and np.all(r.q <= ik.upper + 1e-9)
    pos, _ = ik.palm_pose(r.q)
    assert np.linalg.norm(pos - target) < 1e-3


def test_stays_inside_trained_ranges_when_out_of_reach(ik):
    r = ik.solve(np.array([0.8, -0.2, 0.8]), np.eye(3), ik.q_default, iters=100)
    assert r.pos_err > 0.1  # too far: reports the miss instead of leaving the ranges
    assert np.all(r.q >= ik.lower - 1e-9) and np.all(r.q <= ik.upper + 1e-9)


def test_rest_posture_moves_only_the_spare_freedom(ik):
    target = np.array([0.315, -0.242, 0.825])
    free = ik.solve(target, np.eye(3), ik.q_default, iters=400)
    # a rest posture away from that solution: the palm still lands on the target, and the
    # arm ends closer to the rest posture than without it
    q_rest = free.q + np.array([0.0, 0.0, 0.4, 0.0, 0.0, 0.0, 0.0])
    held = ik.solve(target, np.eye(3), free.q, iters=400, q_rest=q_rest)
    assert held.pos_err < 2e-3 and held.rot_err < 2e-2
    assert np.linalg.norm(held.q - q_rest) < np.linalg.norm(free.q - q_rest) - 0.05

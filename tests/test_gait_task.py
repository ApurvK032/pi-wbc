"""Checks of the stage 1 gait clock (no simulator needed, but Isaac Lab must be installed)."""

import pytest
import torch

pytest.importorskip("isaaclab")

from pi_wbc.tasks import mdp

DT, PERIOD, OFFSET, STANCE = 0.02, 0.8, 0.5, 0.55


def test_phase_runs_once_per_period() -> None:
    steps = torch.tensor([0, 10, 20, 39, 40, 60])
    phase = mdp.gait_phase(steps, DT, PERIOD)
    assert phase.tolist() == pytest.approx([0.0, 0.25, 0.5, 0.975, 0.0, 0.5], abs=1e-6)


def test_feet_alternate_with_short_double_support() -> None:
    phase = mdp.gait_phase(torch.arange(40), DT, PERIOD)  # one full period
    stance = mdp.expected_stance(phase, OFFSET, STANCE, torch.zeros(40, dtype=torch.bool))
    left, right = stance[:, 0], stance[:, 1]
    assert left.float().mean().item() == pytest.approx(STANCE, abs=0.03)
    assert right.float().mean().item() == pytest.approx(STANCE, abs=0.03)
    assert (left & right).float().mean().item() == pytest.approx(2 * STANCE - 1, abs=0.03)
    assert not (~left & ~right).any()  # never both feet in the air
    # the right foot does what the left did half a period earlier
    assert torch.equal(right, torch.roll(left, 20))


def test_standing_keeps_both_feet_down() -> None:
    phase = torch.tensor([0.0, 0.3, 0.7])
    stance = mdp.expected_stance(phase, OFFSET, STANCE, torch.ones(3, dtype=torch.bool))
    assert stance.all()


def test_is_standing_needs_both_speed_and_turn_rate_small() -> None:
    cmd = torch.tensor([[0.0, 0.0, 0.0], [0.05, 0.05, 0.05], [0.3, 0.0, 0.0], [0.0, 0.0, 0.5]])
    assert mdp.is_standing(cmd, 0.1).tolist() == [True, True, False, False]


def test_scaled_range_shrinks_towards_the_nominal_value() -> None:
    nominal, lo, hi = (
        torch.tensor([0.78, 0.0]),
        torch.tensor([0.5, -0.6]),
        torch.tensor([0.78, 0.6]),
    )
    low, high = mdp.scaled_range(nominal, lo, hi, 0.0)
    assert torch.equal(low, nominal) and torch.equal(high, nominal)
    low, high = mdp.scaled_range(nominal, lo, hi, 1.0)
    assert torch.allclose(low, lo) and torch.allclose(high, hi)
    low, high = mdp.scaled_range(nominal, lo, hi, 0.5)
    assert low.tolist() == pytest.approx([0.64, -0.3]) and high.tolist() == pytest.approx(
        [0.78, 0.3]
    )


def test_interpolate_moves_in_a_straight_line_then_holds() -> None:
    start, goal = torch.zeros(3, 2), torch.ones(3, 2)
    out = mdp.interpolate(start, goal, torch.tensor([0.0, 0.5, 2.0]), torch.tensor([1.0, 1.0, 1.0]))
    assert out[:, 0].tolist() == pytest.approx([0.0, 0.5, 1.0])

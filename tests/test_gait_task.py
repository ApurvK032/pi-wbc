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


def _curriculum_env(falls: list[bool], track_per_s: list[float], step: int):
    """A stand-in env with what `upper_body_level` reads, for envs that just ended."""
    from types import SimpleNamespace

    terms = {n: SimpleNamespace(level=0.5, last_level_change=0) for n in ("body", "arms")}
    return SimpleNamespace(
        device="cpu",
        command_manager=SimpleNamespace(get_term=terms.__getitem__),
        reward_manager=SimpleNamespace(
            _episode_sums={"track": torch.tensor(track_per_s) * 20.0},
            get_term_cfg=lambda name: SimpleNamespace(weight=1.0),
        ),
        termination_manager=SimpleNamespace(terminated=torch.tensor(falls)),
        max_episode_length_s=20.0,
        common_step_counter=step,
    ), terms


CURRICULUM = {
    "command_names": ["body", "arms"],
    "reward_term": "track",
    "track_threshold": 0.6,
    "raise_below": 0.1,
    "lower_above": 0.25,
    "level_step": 0.05,
    "min_interval_steps": 480,
    "min_episodes": 4,
}


def test_curriculum_rises_when_few_fall_and_waits_for_its_window() -> None:
    env, terms = _curriculum_env([False] * 4, [0.8] * 4, step=100)
    ids = torch.arange(4)
    assert mdp.upper_body_level(env, ids, **CURRICULUM).item() == pytest.approx(0.5)  # too soon
    env.common_step_counter = 480
    assert mdp.upper_body_level(env, ids, **CURRICULUM).item() == pytest.approx(0.55)
    assert terms["arms"].level == pytest.approx(0.55)  # every named command moves together
    assert terms["body"].tally_episodes == 0  # the tally restarts


def test_curriculum_drops_when_many_fall_and_holds_in_between() -> None:
    env, _ = _curriculum_env([True, True, False, False], [0.8] * 4, step=480)
    assert mdp.upper_body_level(env, slice(None), **CURRICULUM).item() == pytest.approx(0.45)
    env, _ = _curriculum_env([True, False, False, False, False, False], [0.8] * 6, step=480)
    level = mdp.upper_body_level(env, slice(None), **CURRICULUM).item()
    assert level == pytest.approx(0.5)  # 17% fell: neither raise nor drop
    env, _ = _curriculum_env([False] * 4, [0.5] * 4, step=480)
    assert mdp.upper_body_level(env, slice(None), **CURRICULUM).item() == pytest.approx(0.5)


def test_interp_table_is_linear_between_rows_and_clamped_outside() -> None:
    xs = torch.tensor([0.5, 0.6, 0.78])
    ys = torch.tensor([[2.0, 1.6, 0.3], [0.0, 1.0, 2.0]])
    out = mdp.interp_table(torch.tensor([0.4, 0.55, 0.6, 0.69, 0.9]), xs, ys)
    assert out[:, 0].tolist() == pytest.approx([2.0, 1.8, 1.6, 0.95, 0.3])
    assert out[:, 1].tolist() == pytest.approx([0.0, 0.5, 1.0, 1.5, 2.0])

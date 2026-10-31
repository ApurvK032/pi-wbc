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


def _curriculum_env(
    falls: list[bool], track_per_s: list[float], step: int, turn_per_s: list[float] | None = None
):
    """A stand-in env with what `upper_body_level` reads, for envs that just ended."""
    from types import SimpleNamespace

    terms = {n: SimpleNamespace(level=0.5, last_level_change=0) for n in ("body", "arms")}
    turn = [0.5] * len(track_per_s) if turn_per_s is None else turn_per_s
    return SimpleNamespace(
        device="cpu",
        command_manager=SimpleNamespace(get_term=terms.__getitem__),
        reward_manager=SimpleNamespace(
            _episode_sums={
                "track": torch.tensor(track_per_s) * 20.0,
                "turn": torch.tensor(turn) * 20.0,
            },
            get_term_cfg=lambda name: SimpleNamespace(weight=1.0),
        ),
        termination_manager=SimpleNamespace(terminated=torch.tensor(falls)),
        max_episode_length_s=20.0,
        common_step_counter=step,
    ), terms


CURRICULUM = {
    "command_names": ["body", "arms"],
    "reward_thresholds": {"track": 0.6, "turn": 0.35},
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


def test_curriculum_holds_while_turning_is_poor() -> None:
    env, _ = _curriculum_env([False] * 4, [0.8] * 4, step=480, turn_per_s=[0.2] * 4)
    assert mdp.upper_body_level(env, slice(None), **CURRICULUM).item() == pytest.approx(0.5)


def test_delayed_targets_hold_the_previous_action_for_the_delay() -> None:
    previous, current = torch.zeros(3, 2), torch.ones(3, 2)
    delay = torch.tensor([0, 2, 4])
    rows = [mdp.delayed_targets(previous, current, k, delay)[:, 0].tolist() for k in range(4)]
    # env 0: no delay; env 1: 2 physics steps; env 2: the whole 4-step policy period
    assert rows == [[1, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 0]]


def test_curriculum_raises_the_push_speed_with_the_level() -> None:
    from types import SimpleNamespace

    env, _ = _curriculum_env([False] * 4, [0.8] * 4, step=480)
    push = SimpleNamespace(params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}})
    env.event_manager = SimpleNamespace(
        active_terms={"interval": ["push_robot"]}, get_term_cfg=lambda name: push
    )
    level = mdp.upper_body_level(
        env, slice(None), **CURRICULUM, push_event="push_robot", push_speeds=(0.5, 1.5)
    ).item()
    assert level == pytest.approx(0.55)
    assert push.params["velocity_range"]["x"] == pytest.approx((-1.05, 1.05))


def test_curriculum_holds_its_level_with_out_of_reach_thresholds() -> None:
    # D-019: a student keeps its teacher's level, all falling or none falling
    held = {**CURRICULUM, "raise_below": -1.0, "lower_above": 2.0}
    for falls in ([True] * 4, [False] * 4):
        env, _ = _curriculum_env(falls, [0.8] * 4, step=480)
        assert mdp.upper_body_level(env, slice(None), **held).item() == pytest.approx(0.5)


@pytest.mark.parametrize("config", ["none", "full", "curriculum", "friction", "pushes"])
def test_ablation_overrides_resolve_on_the_student_task(config: str, monkeypatch) -> None:
    import sys

    from isaaclab_tasks.utils import resolve_task_config

    monkeypatch.setattr(sys, "argv", ["test"])  # Isaac Lab reads these when overrides are empty

    import pi_wbc.tasks  # noqa: F401
    from pi_wbc.tasks.g1_robust import ABLATION_OVERRIDES, TRAINING_ABLATIONS

    assert not set(ABLATION_OVERRIDES) & set(TRAINING_ABLATIONS)
    groups = {"none": list(ABLATION_OVERRIDES), "full": []}.get(config, [config])
    table = {**ABLATION_OVERRIDES, **TRAINING_ABLATIONS}
    overrides = [o for g in groups for o in table[g]]
    env_cfg, _ = resolve_task_config(
        "PiWbc-G1-Student", "rsl_rl_cfg_entry_point", overrides=overrides
    )
    if config in ("none", "pushes"):
        assert env_cfg.events.push_robot is None
    if config == "curriculum":
        assert env_cfg.commands.body_pose.initial_level == 1.0
        assert env_cfg.curriculum.upper_body_level.params["lower_above"] == 2.0
    if config == "full":
        assert env_cfg.events.push_robot is not None


def test_ppo_student_trains_the_student_inputs_with_a_privileged_critic(monkeypatch) -> None:
    import sys

    from isaaclab_tasks.utils import resolve_task_config

    import pi_wbc.tasks  # noqa: F401

    monkeypatch.setattr(sys, "argv", ["test"])
    _, agent_cfg = resolve_task_config(
        "PiWbc-G1-Student-PPO", "rsl_rl_cfg_entry_point", overrides=[]
    )
    assert agent_cfg.obs_groups == {"actor": ["policy"], "critic": ["teacher"]}
    assert agent_cfg.actor.hidden_dims == [512, 256, 128]  # the distilled student's network


def test_curriculum_ramp_climbs_to_the_starting_level_on_schedule_then_holds() -> None:
    ramp = {**CURRICULUM, "ramp_steps": 480}
    env, terms = _curriculum_env([True] * 4, [0.0] * 4, step=0)  # starts at level 0.5
    assert mdp.upper_body_level(env, slice(None), **ramp).item() == pytest.approx(0.0)
    env.common_step_counter = 240
    assert mdp.upper_body_level(env, slice(None), **ramp).item() == pytest.approx(0.25)
    assert terms["arms"].level == pytest.approx(0.25)
    env.common_step_counter = 5000  # all falling, poor rewards: still held at the target
    assert mdp.upper_body_level(env, slice(None), **ramp).item() == pytest.approx(0.5)

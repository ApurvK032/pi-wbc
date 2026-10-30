"""Commands for the MuJoCo bench: fixed scenarios, fed to the policy the way training feeds them.

A scenario holds a velocity command, optionally a body pose (pelvis height and waist yaw, roll,
pitch) and whether the arms and hands move. The body-pose command starts at the nominal pose
and moves to the scenario's pose at the training command's rate limits; arm and hand targets
follow the training's random motion at its full range (`UpperBodyMotionCommand` at curriculum
level 1), seeded per episode.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pi_wbc.sim2sim.mujoco_g1 import G1MujocoSim


@dataclass(frozen=True)
class Scenario:
    velocity: tuple[float, float, float]  # vx m/s, vy m/s, yaw rate rad/s
    body: tuple[float, float, float, float] | None = None  # height m, waist yaw, roll, pitch rad
    arm_motion: bool = False

    @property
    def needs_body_commands(self) -> bool:
        return self.body is not None or self.arm_motion


class UpperBodyMotion:
    """Random arm and hand targets, as `pi_wbc.tasks.mdp.UpperBodyMotionCommand` makes them.

    A new pose, uniform within [low, high], every 1-3 s (the exported resampling range),
    reached in a straight line by the next resample.
    """

    def __init__(self, upper_body: dict, default: np.ndarray, rng: np.random.Generator) -> None:
        self.low = np.asarray(upper_body["low"], dtype=float)
        self.high = np.asarray(upper_body["high"], dtype=float)
        self.time_range = tuple(upper_body["resampling_time_range"])
        self.rng = rng
        self.start = default.astype(float).copy()
        self.current = self.start.copy()
        self._resample()

    def _resample(self) -> None:
        self.start = self.current.copy()
        self.goal = self.rng.uniform(self.low, self.high)
        self.duration = float(self.rng.uniform(*self.time_range))
        self.time_left = self.duration
        self.elapsed = 0.0

    def update(self, dt: float) -> np.ndarray:
        """Advance one policy step and return the targets for it."""
        self.time_left -= dt
        if self.time_left <= 0.0:
            self._resample()
        self.elapsed += dt
        frac = min(self.elapsed / self.duration, 1.0)
        self.current = self.start + frac * (self.goal - self.start)
        return self.current


class CommandDriver:
    """Feeds one scenario's body-pose and arm commands to a `G1MujocoSim`, step by step.

    Call `before_step()` once per policy step, before reading the observation.
    """

    def __init__(self, sim: G1MujocoSim, scenario: Scenario, seed: int) -> None:
        io = sim.io
        if scenario.needs_body_commands and not (io.body_pose and io.upper_body):
            raise ValueError("this scenario needs a policy trained with body and arm commands")
        self.sim, self.scenario = sim, scenario
        self.dt = sim.policy_dt
        self.body_goal = None if scenario.body is None else np.array(scenario.body, dtype=float)
        self.max_rate = None if io.body_pose is None else np.array(io.body_pose["max_rates"])
        self.arms = None
        if scenario.arm_motion:
            default = sim._base_target0[sim.upper_ids]
            self.arms = UpperBodyMotion(io.upper_body, default, np.random.default_rng(seed))

    def before_step(self) -> None:
        sim = self.sim
        if self.body_goal is not None:
            step = self.max_rate * self.dt
            sim.body_command = sim.body_command + np.clip(
                self.body_goal - sim.body_command, -step, step
            )
        if self.arms is not None:
            sim.set_upper_body_targets(self.arms.update(self.dt))


# All inside the training command ranges.
SCENARIOS: dict[str, Scenario] = {
    "stand": Scenario((0.0, 0.0, 0.0)),
    "walk_0.5": Scenario((0.5, 0.0, 0.0)),
    "walk_1.0": Scenario((1.0, 0.0, 0.0)),
    "side_0.3": Scenario((0.0, 0.3, 0.0)),
    "turn_0.8": Scenario((0.0, 0.0, 0.8)),
    "walk_turn": Scenario((0.6, 0.0, 0.5)),
}
# For policies trained with body-pose commands and moving arms (stage 1, step 2 onwards).
BODY_SCENARIOS: dict[str, Scenario] = {
    "crouch_0.6": Scenario((0.0, 0.0, 0.0), body=(0.60, 0.0, 0.0, 0.0)),
    "walk_low": Scenario((0.5, 0.0, 0.0), body=(0.65, 0.0, 0.0, 0.0)),
    "lean_0.4": Scenario((0.0, 0.0, 0.0), body=(0.78, 0.0, 0.0, 0.4)),
    "arms_stand": Scenario((0.0, 0.0, 0.0), arm_motion=True),
    "arms_walk": Scenario((0.5, 0.0, 0.0), arm_motion=True),
}
ALL_SCENARIOS = SCENARIOS | BODY_SCENARIOS


def scenarios_for(io) -> list[str]:
    """Scenario names a policy can be tested on: the body ones only if it was trained for them."""
    names = list(SCENARIOS)
    if io.body_pose and io.upper_body:
        names += list(BODY_SCENARIOS)
    return names

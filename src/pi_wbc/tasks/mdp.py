"""Observation, reward and command terms for the stage 1 controller.

The gait terms follow Unitree's G1 training code (`unitree_rl_gym`, legged_gym/envs/g1): a
clock with a fixed period, the right leg half a period behind the left, and each leg in stance
for the first part of its cycle. When the command is close to zero the clock asks for both
feet down (standing), which Unitree's code does not do.

The pure functions at the top have no simulator dependency, so they are unit-tested directly.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import MISSING, field
from typing import TYPE_CHECKING

import torch
from isaaclab.envs.mdp.commands import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# -- gait clock (pure) ---------------------------------------------------------------------------


def gait_phase(steps: torch.Tensor, step_dt: float, period: float) -> torch.Tensor:
    """Phase in [0, 1) of the left leg after `steps` policy steps since the episode started.

    Computed in float64, as the MuJoCo bench does, so both give the same phase at cycle
    boundaries (in float32, 40 x 0.02 s is just under 0.8 s).
    """
    return (torch.remainder(steps.double() * step_dt, period) / period).float()


def expected_stance(
    phase: torch.Tensor, offset: float, stance_fraction: float, standing: torch.Tensor
) -> torch.Tensor:
    """(num_envs, 2) bool: whether the left and right foot should be on the ground.

    The right leg runs `offset` of a period behind the left. A standing robot keeps both feet
    down.
    """
    leg_phase = torch.stack([phase, torch.remainder(phase + offset, 1.0)], dim=-1)
    return (leg_phase < stance_fraction) | standing.unsqueeze(-1)


def is_standing(command: torch.Tensor, threshold: float) -> torch.Tensor:
    """True where the velocity command (vx, vy, yaw rate) asks the robot to stand still."""
    return (torch.linalg.norm(command[:, :2], dim=-1) < threshold) & (
        command[:, 2].abs() < threshold
    )


# -- helpers that read the simulation ------------------------------------------------------------


def _feet_in_contact(env: ManagerBasedRLEnv, sensor_cfg) -> torch.Tensor:
    """(num_envs, num_feet) bool: contact force above 1 N, as in Unitree's code."""
    sensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w.torch[:, sensor_cfg.body_ids, :]
    return forces.norm(dim=-1) > 1.0


def _phase(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    return gait_phase(env.episode_length_buf, env.step_dt, period)


# -- observations --------------------------------------------------------------------------------


def gait_clock(
    env: ManagerBasedRLEnv,
    period: float,
    command_name: str | None = None,
    stand_threshold: float = 0.1,
) -> torch.Tensor:
    """sin and cos of the left leg's phase, as Unitree's G1 policy sees it.

    With `command_name`, the clock reads (0, 0) while the command says "stand": an explicit
    stand signal, since a running clock invites stepping in place.
    """
    angle = 2.0 * torch.pi * _phase(env, period)
    clock = torch.stack([torch.sin(angle), torch.cos(angle)], dim=-1)
    if command_name is not None:
        standing = is_standing(env.command_manager.get_command(command_name), stand_threshold)
        clock = clock * (~standing).unsqueeze(-1)
    return clock


# -- rewards -------------------------------------------------------------------------------------


def gait_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg,
    command_name: str,
    period: float,
    offset: float,
    stance_fraction: float,
    stand_threshold: float,
) -> torch.Tensor:
    """Number of feet (0-2) whose contact matches the gait clock."""
    standing = is_standing(env.command_manager.get_command(command_name), stand_threshold)
    expected = expected_stance(_phase(env, period), offset, stance_fraction, standing)
    return (_feet_in_contact(env, sensor_cfg) == expected).float().sum(dim=-1)


def feet_swing_height(
    env: ManagerBasedRLEnv, sensor_cfg, asset_cfg, target_height: float
) -> torch.Tensor:
    """Sum over feet in the air of (foot height - target)², heights of the foot link origin."""
    asset = env.scene[asset_cfg.name]
    height = asset.data.body_pos_w.torch[:, asset_cfg.body_ids, 2]
    in_air = ~_feet_in_contact(env, sensor_cfg)
    return (torch.square(height - target_height) * in_air).sum(dim=-1)


def feet_contact_velocity(env: ManagerBasedRLEnv, sensor_cfg, asset_cfg) -> torch.Tensor:
    """Sum over feet on the ground of their squared linear velocity (Unitree's contact_no_vel)."""
    asset = env.scene[asset_cfg.name]
    velocity = asset.data.body_lin_vel_w.torch[:, asset_cfg.body_ids, :]
    in_contact = _feet_in_contact(env, sensor_cfg)
    return (torch.square(velocity).sum(dim=-1) * in_contact).sum(dim=-1)


def heading_tracking_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """exp(-heading error² / std²), against the heading the commanded yaw rates add up to.

    Rewarding the instantaneous yaw rate alone lets a steady drift through: a bias of
    0.3 rad/s costs little next to the rocking of each step, and the robot walks in circles.
    The integrated target penalizes drift directly.
    """
    term = env.command_manager.get_term(command_name)
    heading = env.scene[term.cfg.asset_name].data.heading_w.torch
    error = wrap_to_pi(term.target_heading - heading)
    return torch.exp(-torch.square(error) / std**2)


def stand_still(env: ManagerBasedRLEnv, command_name: str, stand_threshold: float, asset_cfg):
    """L1 joint offset from the default pose while the command says "stand".

    Like Isaac Lab's stand_still_joint_deviation_l1, but "stand" also needs a small yaw rate,
    so turning in place is not penalized.
    """
    asset = env.scene[asset_cfg.name]
    ids = asset_cfg.joint_ids
    offset = asset.data.joint_pos.torch[:, ids] - asset.data.default_joint_pos.torch[:, ids]
    standing = is_standing(env.command_manager.get_command(command_name), stand_threshold)
    return offset.abs().sum(dim=-1) * standing


def joint_deviation_l2(env: ManagerBasedRLEnv, asset_cfg) -> torch.Tensor:
    """Sum of squared joint offsets from the default pose (Unitree's hip_pos uses squares)."""
    asset = env.scene[asset_cfg.name]
    ids = asset_cfg.joint_ids
    offset = asset.data.joint_pos.torch[:, ids] - asset.data.default_joint_pos.torch[:, ids]
    return torch.square(offset).sum(dim=-1)


# -- events --------------------------------------------------------------------------------------


def hold_joints_at_default(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None, asset_cfg) -> None:
    """Set the PD targets of joints no action drives to the default pose (reset event).

    Isaac Lab otherwise leaves their targets at 0: the first gait runs trained with the arms
    hanging straight down instead of in the default pose.
    """
    asset = env.scene[asset_cfg.name]
    ids = asset_cfg.joint_ids
    if isinstance(ids, slice):
        ids = list(range(asset.num_joints))
    rows = slice(None) if env_ids is None else env_ids
    default = asset.data.default_joint_pos.torch[rows][:, ids]
    asset.set_joint_position_target_index(target=default, joint_ids=ids, env_ids=env_ids)


# -- commands ------------------------------------------------------------------------------------


class GaitVelocityCommand(UniformVelocityCommand):
    """Uniform velocity commands with a share of turn-in-place commands and a deadband.

    Yaw rate is sampled directly (no heading controller). Turning envs get zero linear velocity
    and a yaw rate from the usual range. Linear commands slower than `deadband` are set to
    zero, so "walk very slowly" and "stand" are not confused.

    `target_heading` is where the robot should face if it had followed every yaw-rate command
    exactly since the command was sampled (used by `heading_tracking_exp`).
    """

    cfg: GaitVelocityCommandCfg

    def __init__(self, cfg: GaitVelocityCommandCfg, env):
        super().__init__(cfg, env)
        self.is_turning_env = torch.zeros_like(self.is_standing_env)
        self.target_heading = torch.zeros(self.num_envs, device=self.device)

    def _update_command(self):
        super()._update_command()  # zeroes the command of standing envs
        heading = self.robot.data.heading_w.torch
        # re-anchor just after an episode reset, once the reset pose has reached the data
        fresh = self._env.episode_length_buf <= 1
        self.target_heading = torch.where(
            fresh,
            heading,
            wrap_to_pi(self.target_heading + self.vel_command_b[:, 2] * self._env.step_dt),
        )

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        self.target_heading[env_ids] = self.robot.data.heading_w.torch[env_ids]
        cmd = self.vel_command_b[env_ids]
        turning = torch.rand(cmd.shape[0], device=self.device) < self.cfg.rel_turning_envs
        self.is_turning_env[env_ids] = turning
        slow = torch.linalg.norm(cmd[:, :2], dim=-1) < self.cfg.deadband
        cmd[:, :2] = torch.where((slow | turning).unsqueeze(-1), 0.0, cmd[:, :2])
        self.vel_command_b[env_ids] = cmd


@configclass
class GaitVelocityCommandCfg(UniformVelocityCommandCfg):
    class_type: type = GaitVelocityCommand
    rel_turning_envs: float = 0.1
    """Share of envs that turn in place (zero linear velocity)."""
    deadband: float = 0.1
    """Linear commands slower than this (m/s) are set to zero."""


# -- upper body and body pose (stage 1, step 2) --------------------------------------------------


def scaled_range(
    nominal: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor, level: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """The range [lo, hi] shrunk towards `nominal` by the curriculum `level` (0 to 1).

    At level 0 the range is the nominal value alone; at level 1 it is [lo, hi].
    """
    return nominal + level * (lo - nominal), nominal + level * (hi - nominal)


def _count(env_ids: Sequence[int] | slice, num_envs: int) -> int:
    """Number of envs in `env_ids`, which Isaac Lab may pass as a slice."""
    return len(range(num_envs)[env_ids]) if isinstance(env_ids, slice) else len(env_ids)


def interpolate(
    start: torch.Tensor, goal: torch.Tensor, elapsed: torch.Tensor, duration: torch.Tensor
) -> torch.Tensor:
    """Straight-line move from `start` to `goal`, `elapsed` seconds into `duration`.

    Rows are envs; `elapsed` and `duration` have one value per env. Holds `goal` once done.
    """
    frac = torch.clamp(elapsed / duration.clamp_min(1e-6), 0.0, 1.0)
    return start + frac.unsqueeze(-1) * (goal - start)


class BodyPoseCommand(CommandTerm):
    """Pelvis height and torso orientation, as NVIDIA's controller takes them (D-012).

    The command is (pelvis height in m, waist yaw, waist roll, waist pitch in rad). The torso's
    orientation relative to the pelvis is set by the G1's three waist joints, which turn in the
    order yaw, roll, pitch, so the angles are those joints' targets.

    Each resample picks a goal: the nominal pose for a share of envs, otherwise uniform within
    the ranges shrunk towards the nominal pose by `level` (set by the curriculum). The command
    then moves towards the goal at limited rates, so the robot is asked to crouch or lean
    smoothly. Every episode starts at the nominal pose.
    """

    cfg: BodyPoseCommandCfg

    def __init__(self, cfg: BodyPoseCommandCfg, env):
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.asset_name]
        self.waist_ids, _ = self.robot.find_joints(list(cfg.waist_joint_names), preserve_order=True)
        r = cfg.ranges
        bounds = [r.height, r.waist_yaw, r.waist_roll, r.waist_pitch]
        self.lo = torch.tensor([b[0] for b in bounds], device=self.device)
        self.hi = torch.tensor([b[1] for b in bounds], device=self.device)
        self.nominal = torch.tensor(cfg.nominal, device=self.device)
        self.max_rate = torch.tensor(cfg.max_rates, device=self.device)
        self.goal = self.nominal.repeat(self.num_envs, 1)
        self.current = self.goal.clone()
        self.level = cfg.initial_level
        self.last_level_change = 0
        self.metrics["error_height"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_waist"] = torch.zeros(self.num_envs, device=self.device)
        self._height_sum = torch.zeros(self.num_envs, device=self.device)
        self._waist_sum = torch.zeros(self.num_envs, device=self.device)
        self._count = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.current

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        ids = slice(None) if env_ids is None else env_ids
        denom = self._count[ids].clamp_min(1.0)
        self.metrics["error_height"][ids] = self._height_sum[ids] / denom
        self.metrics["error_waist"][ids] = self._waist_sum[ids] / denom
        self.current[ids] = self.nominal
        extras = super().reset(env_ids)
        for buf in (self._height_sum, self._waist_sum, self._count):
            buf[ids] = 0.0
        return extras

    def _resample_command(self, env_ids: Sequence[int]):
        n = _count(env_ids, self.num_envs)
        lo, hi = scaled_range(self.nominal, self.lo, self.hi, self.level)
        goal = lo + torch.rand(n, 4, device=self.device) * (hi - lo)
        nominal = torch.rand(n, device=self.device) < self.cfg.rel_nominal_envs
        self.goal[env_ids] = torch.where(nominal.unsqueeze(-1), self.nominal, goal)

    def _update_command(self):
        step = self.max_rate * self._env.step_dt
        self.current += torch.clamp(self.goal - self.current, -step, step)

    def _update_metrics(self):
        height = self.robot.data.root_pos_w.torch[:, 2]
        waist = self.robot.data.joint_pos.torch[:, self.waist_ids]
        self._height_sum += (height - self.current[:, 0]).abs()
        self._waist_sum += (waist - self.current[:, 1:]).abs().mean(dim=-1)
        self._count += 1.0


@configclass
class BodyPoseCommandCfg(CommandTermCfg):
    class_type: type = BodyPoseCommand
    asset_name: str = "robot"
    waist_joint_names: tuple[str, ...] = (
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
    )
    """The waist joints in command order: yaw, roll, pitch."""
    nominal: tuple[float, float, float, float] = (0.78, 0.0, 0.0, 0.0)
    """Height (m) and waist angles (rad) of the default standing pose."""
    max_rates: tuple[float, float, float, float] = (0.25, 1.0, 1.0, 1.0)
    """How fast the command moves towards a new goal (m/s, rad/s)."""
    rel_nominal_envs: float = 0.3
    """Share of resamples that ask for the nominal pose."""
    initial_level: float = 0.0
    """Curriculum level at the start of training (0: nominal pose only, 1: full ranges)."""

    @configclass
    class Ranges:
        height: tuple[float, float] = (0.50, 0.78)
        waist_yaw: tuple[float, float] = (-0.6, 0.6)
        waist_roll: tuple[float, float] = (-0.2, 0.2)
        waist_pitch: tuple[float, float] = (0.0, 0.45)

    ranges: Ranges = Ranges()


class UpperBodyMotionCommand(CommandTerm):
    """Random arm and hand motion, applied straight to those joints' PD targets.

    The VLA will send arm and hand joint targets that bypass the policy (D-012), so in training
    the arms move on their own and the legs learn to keep balance (HOMIE's recipe). At each
    resample a new random pose is drawn, uniform within each joint's range shrunk towards the
    default pose by `level` (set by the curriculum), and the targets move to it in a straight
    line over the time until the next resample. A share of episodes keeps the arms still at the
    default pose. Targets are written every policy step; the policy never sees them, only the
    joint positions.
    """

    cfg: UpperBodyMotionCommandCfg

    def __init__(self, cfg: UpperBodyMotionCommandCfg, env):
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.asset_name]
        self.joint_ids, self.joint_names = self.robot.find_joints(
            cfg.joint_names, preserve_order=True
        )
        self.default = self.robot.data.default_joint_pos.torch[0, self.joint_ids].clone()
        limits = self.robot.data.joint_pos_limits.torch[0, self.joint_ids]
        self.lo, self.hi = limits[:, 0].clone(), limits[:, 1].clone()
        for pattern, (lo, hi) in cfg.ranges.items():
            hits = [i for i, n in enumerate(self.joint_names) if re.fullmatch(pattern, n)]
            if not hits:
                raise ValueError(f"upper-body range {pattern!r} matches no joint")
            for i in hits:
                self.lo[i] = max(lo, float(limits[i, 0]))
                self.hi[i] = min(hi, float(limits[i, 1]))
        n = len(self.joint_ids)
        self.start = self.default.repeat(self.num_envs, 1)
        self.goal = self.start.clone()
        self.current = self.start.clone()
        self.elapsed = torch.zeros(self.num_envs, device=self.device)
        self.duration = torch.ones(self.num_envs, device=self.device)
        self.still = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.level = cfg.initial_level
        self.last_level_change = 0
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self._error_sum = torch.zeros(self.num_envs, device=self.device)
        self._count = torch.zeros(self.num_envs, device=self.device)
        assert self.current.shape[1] == n

    @property
    def command(self) -> torch.Tensor:
        return self.current

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        ids = slice(None) if env_ids is None else env_ids
        self.metrics["error_joint_pos"][ids] = self._error_sum[ids] / self._count[ids].clamp_min(
            1.0
        )
        for buf in (self.start, self.goal, self.current):
            buf[ids] = self.default
        rows = self.still[ids]
        self.still[ids] = torch.rand(rows.shape[0], device=self.device) < self.cfg.rel_still_envs
        extras = super().reset(env_ids)
        self._error_sum[ids] = 0.0
        self._count[ids] = 0.0
        return extras

    def _resample_command(self, env_ids: Sequence[int]):
        n = _count(env_ids, self.num_envs)
        lo, hi = scaled_range(self.default, self.lo, self.hi, self.level)
        goal = lo + torch.rand(n, len(self.joint_ids), device=self.device) * (hi - lo)
        still = self.still[env_ids].unsqueeze(-1)
        self.start[env_ids] = self.current[env_ids]
        self.goal[env_ids] = torch.where(still, self.default, goal)
        self.elapsed[env_ids] = 0.0
        self.duration[env_ids] = self.time_left[env_ids]

    def _update_command(self):
        self.elapsed += self._env.step_dt
        self.current = interpolate(self.start, self.goal, self.elapsed, self.duration)
        self.robot.set_joint_position_target_index(target=self.current, joint_ids=self.joint_ids)

    def _update_metrics(self):
        pos = self.robot.data.joint_pos.torch[:, self.joint_ids]
        self._error_sum += (pos - self.current).abs().mean(dim=-1)
        self._count += 1.0


@configclass
class UpperBodyMotionCommandCfg(CommandTermCfg):
    class_type: type = UpperBodyMotionCommand
    asset_name: str = "robot"
    joint_names: list[str] = MISSING
    """Joints moved by this command (arms and hands); no action term may drive them."""
    ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    """Target range per joint-name pattern at curriculum level 1, kept inside the joint limits.
    Joints not listed use their full limits."""
    rel_still_envs: float = 0.2
    """Share of episodes whose arms and hands stay at the default pose."""
    initial_level: float = 0.0


def upper_body_level(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | slice,
    command_names: Sequence[str],
    reward_term: str,
    track_threshold: float,
    raise_below: float,
    lower_above: float,
    level_step: float,
    min_interval_steps: int,
    min_episodes: int = 100,
) -> torch.Tensor:
    """Curriculum: widen the body-pose and arm-motion ranges while the robot walks well.

    Every episode that ends adds to a tally: whether it ended in a fall, and its velocity-
    tracking reward per second. At most every `min_interval_steps` policy steps, and once at
    least `min_episodes` have ended, the tally decides: if under `raise_below` of them fell and
    tracking averaged over `track_threshold` x the reward's weight, the level of every named
    command rises by `level_step` (up to 1); if over `lower_above` fell, it drops by the same
    step. Then the tally restarts. Judging a whole window, not each step's few resetting envs,
    keeps the level from moving on noise. Returns the level, which Isaac Lab logs.
    """
    terms = [env.command_manager.get_term(n) for n in command_names]
    lead = terms[0]
    if not hasattr(lead, "tally_episodes"):
        lead.tally_episodes = 0
        lead.tally_falls = torch.zeros((), device=env.device)
        lead.tally_track = torch.zeros((), device=env.device)
    sums = env.reward_manager._episode_sums[reward_term][env_ids]  # env_ids may be a slice
    lead.tally_episodes += sums.numel()
    lead.tally_falls += env.termination_manager.terminated[env_ids].sum()
    lead.tally_track += sums.sum() / env.max_episode_length_s
    level = lead.level
    since = env.common_step_counter - lead.last_level_change
    if since >= min_interval_steps and lead.tally_episodes >= min_episodes:
        n = lead.tally_episodes
        fall_share = lead.tally_falls.item() / n
        track = lead.tally_track.item() / n
        weight = env.reward_manager.get_term_cfg(reward_term).weight
        if fall_share < raise_below and track > track_threshold * weight:
            level = min(1.0, level + level_step)
        elif fall_share > lower_above:
            level = max(0.0, level - level_step)
        for term in terms:
            term.level = level
            term.last_level_change = env.common_step_counter
        lead.tally_episodes = 0
        lead.tally_falls.zero_()
        lead.tally_track.zero_()
    return torch.tensor(level)


def base_height_command_l2(env: ManagerBasedRLEnv, command_name: str, asset_cfg) -> torch.Tensor:
    """(pelvis height - commanded height)², the commanded height being the command's first value."""
    asset = env.scene[asset_cfg.name]
    target = env.command_manager.get_command(command_name)[:, 0]
    return torch.square(asset.data.root_pos_w.torch[:, 2] - target)


def waist_command_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, asset_cfg
) -> torch.Tensor:
    """exp(-|waist angles - commanded|² / std²); asset_cfg lists the waist yaw, roll, pitch."""
    asset = env.scene[asset_cfg.name]
    target = env.command_manager.get_command(command_name)[:, 1:4]
    error = asset.data.joint_pos.torch[:, asset_cfg.joint_ids] - target
    return torch.exp(-torch.square(error).sum(dim=-1) / std**2)


def stand_still_joint_vel(
    env: ManagerBasedRLEnv, command_name: str, stand_threshold: float, asset_cfg
) -> torch.Tensor:
    """L1 joint speed while the command says "stand".

    Step 1's `stand_still` penalized the offset from the default pose, which would also
    penalize crouching on command. Speed stops stepping in place but lets the robot settle
    into any commanded height.
    """
    asset = env.scene[asset_cfg.name]
    speed = asset.data.joint_vel.torch[:, asset_cfg.joint_ids].abs().sum(dim=-1)
    standing = is_standing(env.command_manager.get_command(command_name), stand_threshold)
    return speed * standing

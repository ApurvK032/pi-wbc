"""Observation, reward and command terms for the stage 1 controller.

The gait terms follow Unitree's G1 training code (`unitree_rl_gym`, legged_gym/envs/g1): a
clock with a fixed period, the right leg half a period behind the left, and each leg in stance
for the first part of its cycle. When the command is close to zero the clock asks for both
feet down (standing), which Unitree's code does not do.

The pure functions at the top have no simulator dependency, so they are unit-tested directly.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab.envs.mdp.commands import UniformVelocityCommand, UniformVelocityCommandCfg
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

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


def gait_clock(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    """sin and cos of the left leg's phase, as Unitree's G1 policy sees it."""
    angle = 2.0 * torch.pi * _phase(env, period)
    return torch.stack([torch.sin(angle), torch.cos(angle)], dim=-1)


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


def joint_deviation_l2(env: ManagerBasedRLEnv, asset_cfg) -> torch.Tensor:
    """Sum of squared joint offsets from the default pose (Unitree's hip_pos uses squares)."""
    asset = env.scene[asset_cfg.name]
    ids = asset_cfg.joint_ids
    offset = asset.data.joint_pos.torch[:, ids] - asset.data.default_joint_pos.torch[:, ids]
    return torch.square(offset).sum(dim=-1)


# -- commands ------------------------------------------------------------------------------------


class GaitVelocityCommand(UniformVelocityCommand):
    """Uniform velocity commands with a share of turn-in-place commands and a deadband.

    Yaw rate is sampled directly (no heading controller). Turning envs get zero linear velocity
    and a yaw rate from the usual range. Linear commands slower than `deadband` are set to
    zero, so "walk very slowly" and "stand" are not confused.
    """

    cfg: GaitVelocityCommandCfg

    def __init__(self, cfg: GaitVelocityCommandCfg, env):
        super().__init__(cfg, env)
        self.is_turning_env = torch.zeros_like(self.is_standing_env)

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
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

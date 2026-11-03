"""The scripted expert's walking: velocity commands that bring the pelvis to a stance pose.

Far from the goal the robot turns to face it and walks there; near it, it steps sideways and
forwards and turns to the goal's heading. The stage 1 controller treats commands below 0.1 m/s
and 0.1 rad/s as "stand" (its gait clock stops), so a moving command is never smaller than
`min_speed` / `min_turn`: within tolerance the command is zero, outside it at least that.
Batched over environments; poses are (x, y, yaw) in one frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class NavParams:
    far: float = 0.45  # m: beyond this, face the goal and walk
    max_speed: float = 0.6  # m/s forward when far
    max_side: float = 0.3  # m/s sideways when near
    max_turn: float = 0.8  # rad/s
    min_speed: float = 0.15  # m/s: smallest moving command (stand threshold: 0.1)
    min_turn: float = 0.15  # rad/s
    gain: float = 1.2  # 1/s on position error
    turn_gain: float = 1.5  # 1/s on heading error
    face_first: float = 0.6  # rad: when far, turn in place until the goal is this close to ahead
    tol_pos: float = 0.03  # m
    tol_yaw: float = 0.05  # rad


def wrap(a: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(a), torch.cos(a))


def stance_for(
    target_xy: torch.Tensor, rel_xy: tuple[float, float], yaw: torch.Tensor
) -> torch.Tensor:
    """Pelvis pose (x, y, yaw) that puts `target_xy` at `rel_xy` in the robot's heading frame."""
    c, s = torch.cos(yaw), torch.sin(yaw)
    x = target_xy[:, 0] - (c * rel_xy[0] - s * rel_xy[1])
    y = target_xy[:, 1] - (s * rel_xy[0] + c * rel_xy[1])
    return torch.stack([x, y, yaw], dim=-1)


def _at_least(v: torch.Tensor, err: torch.Tensor, tol: float, vmin: float, vmax: float):
    """Zero within tolerance, otherwise |v| clipped into [vmin, vmax] with the error's sign."""
    mag = v.abs().clamp(vmin, vmax)
    return torch.where(err.abs() <= tol, torch.zeros_like(v), torch.sign(err) * mag)


DEFAULT = NavParams()


def command(pose: torch.Tensor, goal: torch.Tensor, p: NavParams = DEFAULT):
    """Velocity commands (vx, vy, yaw rate in the robot's frame) and whether each robot has
    arrived (within tolerance of position and heading)."""
    d = goal[:, :2] - pose[:, :2]
    dist = d.norm(dim=-1)
    yaw = pose[:, 2]
    c, s = torch.cos(yaw), torch.sin(yaw)
    ex = c * d[:, 0] + s * d[:, 1]  # goal in the robot's frame
    ey = -s * d[:, 0] + c * d[:, 1]
    e_goal_yaw = wrap(goal[:, 2] - yaw)
    e_face = wrap(torch.atan2(d[:, 1], d[:, 0]) - yaw)

    cmd = torch.zeros_like(pose)
    far = dist > p.far
    # far: face the goal, then walk to it while steering
    turn_far = (p.turn_gain * e_face).clamp(-p.max_turn, p.max_turn)
    walk = (p.gain * dist).clamp(p.min_speed, p.max_speed) * (e_face.abs() < p.face_first)
    # near: step to the position and turn to the goal heading
    vx = _at_least(p.gain * ex, ex, p.tol_pos, p.min_speed, p.max_side)
    vy = _at_least(p.gain * ey, ey, p.tol_pos, p.min_speed, p.max_side)
    wz = _at_least(p.turn_gain * e_goal_yaw, e_goal_yaw, p.tol_yaw, p.min_turn, p.max_turn)
    cmd[:, 0] = torch.where(far, walk, vx)
    cmd[:, 1] = torch.where(far, torch.zeros_like(vy), vy)
    turn_far = torch.where(
        turn_far.abs() < p.min_turn, torch.sign(turn_far) * p.min_turn * (e_face.abs() > p.tol_yaw),
        turn_far,
    )  # fmt: skip
    cmd[:, 2] = torch.where(far, turn_far, wz)
    arrived = (ex.abs() <= p.tol_pos) & (ey.abs() <= p.tol_pos) & (e_goal_yaw.abs() <= p.tol_yaw)
    cmd[arrived] = 0.0
    return cmd, arrived

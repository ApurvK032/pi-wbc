"""The expert's walking commands bring a simple simulated robot to its stance."""

import math

import torch

from pi_wbc.stage3 import navigate as nav


def simulate(start, goal, turn_gain=0.7, speed_gain=0.9, lag=0.0, steps=2500, dt=0.02):
    """A kinematic robot that turns at `turn_gain` and walks at `speed_gain` of the command and
    treats commands below 0.1 as standing, like the stage 1 controller."""
    pose = start.clone()
    for k in range(steps):
        cmd, arrived = nav.command(pose, goal)
        if bool(arrived.all()):
            return pose, k * dt
        stand = (cmd[:, :2].norm(dim=-1) < 0.1) & (cmd[:, 2].abs() < 0.1)
        v = torch.where(stand[:, None], torch.zeros_like(cmd), cmd)
        c, s = torch.cos(pose[:, 2]), torch.sin(pose[:, 2])
        pose[:, 0] += dt * speed_gain * (c * v[:, 0] - s * v[:, 1])
        pose[:, 1] += dt * speed_gain * (s * v[:, 0] + c * v[:, 1])
        pose[:, 2] = nav.wrap(pose[:, 2] + dt * turn_gain * v[:, 2])
    return pose, None


def test_reaches_stances_from_task_starts():
    g = torch.Generator().manual_seed(0)
    n = 64
    start = torch.stack(
        [
            -1.5 - 1.5 * torch.rand(n, generator=g),
            -1.0 + 2.0 * torch.rand(n, generator=g),
            (torch.rand(n, generator=g) - 0.5) * math.pi,
        ],
        dim=-1,
    )
    target = torch.stack([0.12 + 0.08 * torch.rand(n, generator=g),
                          -0.3 + 0.6 * torch.rand(n, generator=g)], dim=-1)  # fmt: skip
    goal = nav.stance_for(target, (0.40, -0.18), torch.zeros(n))
    pose, t = simulate(start, goal)
    assert t is not None and t < 40.0
    assert (pose[:, :2] - goal[:, :2]).norm(dim=-1).max() < 0.05
    assert nav.wrap(pose[:, 2] - goal[:, 2]).abs().max() < 0.06


def test_stance_puts_target_at_relative_position():
    target = torch.tensor([[0.15, 0.1]])
    yaw = torch.tensor([0.3])
    st = nav.stance_for(target, (0.40, -0.18), yaw)
    c, s = math.cos(0.3), math.sin(0.3)
    dx, dy = target[0, 0] - st[0, 0], target[0, 1] - st[0, 1]
    assert abs(c * dx + s * dy - 0.40) < 1e-6 and abs(-s * dx + c * dy + 0.18) < 1e-6


def test_moving_commands_are_never_below_the_stand_threshold():
    pose = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    goal = torch.tensor([[0.05, 0.0, 0.0], [0.0, 0.04, 0.0], [0.0, 0.0, 0.08]])
    cmd, arrived = nav.command(pose, goal)
    assert not arrived.any()
    for row in cmd:
        moving = row.abs() > 0
        assert moving.any() and (row.abs()[moving] >= 0.15 - 1e-6).all()

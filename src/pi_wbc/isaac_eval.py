"""Evaluate a policy in Isaac Lab on the MuJoCo bench's fixed commands (design, section 9).

    uv run pi-wbc-isaac-eval --task PiWbc-G1-Teacher --checkpoint <run>/model_1399.pt \
        --seeds 10 --out results/stage1/<name>.json physics=newton_mjwarp

Every scenario of the bench (`pi_wbc.sim2sim.commands`) gets `--seeds` robots, all in one
Isaac Lab environment, each holding its scenario's commands for 20 s: velocity fixed (no
resampling), the body pose moving to the scenario's at the training's rate limits, arms moving
at full range only in the arm scenarios. Starts are Isaac Lab's reset randomization, seeded by
`--seed`. A robot's first fall ends its count. Tracking errors are taken after 2 s, as in the
bench: planar velocity in the heading frame, yaw rate, pelvis height, waist angles, arm joints.

By default the robots are nominal (`make_nominal`: no randomization, noise or delay) and are not
pushed; `--randomized` keeps the task's randomization and pushes. This is the only test the
teacher can take, since it needs privileged observations.
"""

from __future__ import annotations

import argparse
import statistics
from dataclasses import asdict
from pathlib import Path
from typing import Any

SETTLE_S = 2.0


def main(argv: list[str] | None = None) -> None:
    from isaaclab.app import add_launcher_args, launch_simulation

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--task", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=10, help="robots per scenario")
    parser.add_argument("--seed", type=int, default=0, help="seed of the reset randomization")
    parser.add_argument("--duration-s", type=float, default=20.0)
    parser.add_argument("--randomized", action="store_true", help="keep randomization and pushes")
    parser.add_argument("--label", default=None)
    parser.add_argument("--out", type=Path, required=True)
    add_launcher_args(parser)
    args, overrides = parser.parse_known_args(argv)

    import gymnasium as gym
    import torch
    from isaaclab.utils.math import quat_apply_inverse, yaw_quat
    from isaaclab_rl.rsl_rl import (
        RslRlVecEnvWrapper,
        check_rsl_rl_version,
        create_rsl_rl_runner,
        handle_deprecated_rsl_rl_cfg,
    )
    from isaaclab_tasks.utils import resolve_task_config

    import pi_wbc.tasks  # noqa: F401
    from pi_wbc import provenance
    from pi_wbc.sim2sim.commands import BODY_SCENARIOS, SCENARIOS
    from pi_wbc.stats import wilson_interval

    env_cfg, agent_cfg = resolve_task_config(
        args.task, "rsl_rl_cfg_entry_point", overrides=overrides
    )
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, check_rsl_rl_version())
    has_body = hasattr(env_cfg.commands, "body_pose")
    scenarios = {**SCENARIOS, **(BODY_SCENARIOS if has_body else {})}
    names = list(scenarios)
    n = len(names) * args.seeds
    env_cfg.scene.num_envs = n
    env_cfg.seed = args.seed
    env_cfg.episode_length_s = args.duration_s + 1.0  # no time-out inside the window
    if not args.randomized:
        if hasattr(env_cfg, "make_nominal"):
            env_cfg.make_nominal()
        # tasks without make_nominal still draw the base task's torso mass and the payload
        for name in ("push_robot", "add_base_mass", "hand_payload"):
            if hasattr(env_cfg.events, name):
                setattr(env_cfg.events, name, None)
    # fixed commands: no resampling, no curriculum
    env_cfg.commands.base_velocity.resampling_time_range = (1.0e6, 1.0e6)
    if has_body:
        env_cfg.commands.body_pose.resampling_time_range = (1.0e6, 1.0e6)
        env_cfg.curriculum.upper_body_level = None

    with launch_simulation(env_cfg, args):
        env = gym.make(args.task, cfg=env_cfg)
        venv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = create_rsl_rl_runner(venv, agent_cfg)
        runner.load(str(args.checkpoint.resolve()))
        policy = runner.get_inference_policy(device=venv.unwrapped.device)
        base = venv.unwrapped
        device = base.device
        robot = base.scene["robot"]

        scenario_of = torch.arange(n, device=device) // args.seeds
        velocity = torch.tensor([scenarios[s].velocity for s in names], device=device)[scenario_of]
        vel_term = base.command_manager.get_term("base_velocity")
        body_term = arm_term = None
        if has_body:
            body_term = base.command_manager.get_term("body_pose")
            arm_term = base.command_manager.get_term("upper_body")
            nominal = body_term.nominal
            goals = torch.stack(
                [
                    torch.tensor(scenarios[s].body, device=device)
                    if scenarios[s].body is not None
                    else nominal
                    for s in names
                ]
            )[scenario_of]
            moving_arms = torch.tensor([scenarios[s].arm_motion for s in names], device=device)[
                scenario_of
            ]
            body_term.level = arm_term.level = 1.0

        def hold_commands() -> None:
            vel_term.vel_command_b[:] = velocity
            vel_term.is_standing_env[:] = False
            if hasattr(vel_term, "is_turning_env"):
                vel_term.is_turning_env[:] = False
            if body_term is not None:
                body_term.goal[:] = goals
                arm_term.still[:] = ~moving_arms

        base.reset()
        hold_commands()
        if arm_term is not None:
            # the reset drew arm goals before the still flags were set: robots without arm
            # motion go back to the default pose now (the others keep their first pose)
            still = ~moving_arms
            for buf in (arm_term.start, arm_term.goal, arm_term.current):
                buf[still] = arm_term.default
            arm_cols = [i for i, j in enumerate(arm_term.joint_names) if "_hand_" not in j]
        obs = venv.get_observations()
        dt = base.step_dt
        steps = round(args.duration_s / dt)
        settle = round(SETTLE_S / dt)
        alive = torch.ones(n, dtype=torch.bool, device=device)
        fall_step = torch.full((n,), -1, device=device)
        keys = ("lin", "ang", "vx", "vy", "wz", "height", "waist", "arm")
        sums = {k: torch.zeros(n, device=device) for k in keys}
        count = torch.zeros(n, device=device)
        with torch.inference_mode():
            for k in range(steps):
                hold_commands()
                obs, _, dones, _ = venv.step(policy(obs))
                fell = dones.bool() & alive
                fall_step[fell] = k
                alive &= ~dones.bool()
                if k < settle:
                    continue
                q_yaw = yaw_quat(robot.data.root_quat_w.torch)
                v = quat_apply_inverse(q_yaw, robot.data.root_lin_vel_w.torch)[:, :2]
                wz = robot.data.root_ang_vel_w.torch[:, 2]
                sums["lin"] += torch.linalg.norm(velocity[:, :2] - v, dim=-1) * alive
                sums["ang"] += (velocity[:, 2] - wz).abs() * alive
                sums["vx"] += v[:, 0] * alive
                sums["vy"] += v[:, 1] * alive
                sums["wz"] += wz * alive
                if body_term is not None:
                    cmd = body_term.command
                    height = robot.data.root_pos_w.torch[:, 2]
                    waist = robot.data.joint_pos.torch[:, body_term.waist_ids]
                    arms = robot.data.joint_pos.torch[:, arm_term.joint_ids]
                    sums["height"] += (height - cmd[:, 0]).abs() * alive
                    sums["waist"] += (waist - cmd[:, 1:]).abs().mean(dim=-1) * alive
                    arm_err = (arms - arm_term.command)[:, arm_cols].abs().mean(dim=-1)
                    sums["arm"] += arm_err * alive
                count += alive.float()
        env.close()

    rows = summarize(names, args.seeds, fall_step, sums, count, has_body, wilson_interval)
    provenance.save_json(
        args.out,
        {
            "label": args.label,
            "task": args.task,
            "checkpoint": str(args.checkpoint),
            "seeds_per_scenario": args.seeds,
            "reset_seed": args.seed,
            "duration_s": args.duration_s,
            "settle_s": SETTLE_S,
            "randomized": args.randomized,
            "scenarios": {s: asdict(scenarios[s]) for s in names},
            "summary": rows,
        },
    )
    print_table(rows)
    print(f"[isaac-eval] wrote {args.out}")


def _mean_over(sums, count, idx, up, key: str) -> float | None:
    """Mean over the robots `up` of slice `idx` of their per-step mean `key` error."""
    if not up.any():
        return None
    return statistics.fmean((sums[key][idx][up] / count[idx][up]).tolist())


def summarize(names, seeds, fall_step, sums, count, has_body, wilson_interval) -> list[dict]:
    """Per scenario: falls with a 95% Wilson interval and mean errors of the robots that stood."""
    rows = []
    for i, name in enumerate(names):
        idx = slice(i * seeds, (i + 1) * seeds)
        fell = fall_step[idx] >= 0
        falls = int(fell.sum())
        lo, hi = wilson_interval(falls, seeds)
        up = ~fell & (count[idx] > 0)
        row: dict[str, Any] = {
            "model": "isaac",
            "scenario": name,
            "episodes": seeds,
            "falls": falls,
            "fall_rate": falls / seeds,
            "fall_rate_ci95": [lo, hi],
            "lin_err_mean": _mean_over(sums, count, idx, up, "lin"),
            "ang_err_mean": _mean_over(sums, count, idx, up, "ang"),
            # achieved velocity in the heading frame (m/s, m/s, rad/s)
            "velocity_mean": [_mean_over(sums, count, idx, up, k) for k in ("vx", "vy", "wz")],
        }
        if has_body:
            for key in ("height", "waist", "arm"):
                row[f"{key}_err_mean"] = _mean_over(sums, count, idx, up, key)
        rows.append(row)
    return rows


def print_table(rows: list[dict]) -> None:
    def cell(value, fmt: str, scale: float = 1.0) -> str:
        return "—" if value is None else fmt.format(scale * value)

    print(
        "| Command | Falls | Speed error | Turn-rate error | Achieved vx, vy, yaw rate "
        "| Height error | Waist error | Arm error |"
    )
    print("|---|---:|---:|---:|---|---:|---:|---:|")
    for r in rows:
        print(
            f"| {r['scenario']} | {r['falls']}/{r['episodes']} "
            f"| {cell(r['lin_err_mean'], '{:.2f} m/s')} "
            f"| {cell(r['ang_err_mean'], '{:.2f} rad/s')} "
            f"| {', '.join(cell(v, '{:+.2f}') for v in r['velocity_mean'])} "
            f"| {cell(r.get('height_err_mean'), '{:.1f} cm', 100.0)} "
            f"| {cell(r.get('waist_err_mean'), '{:.3f} rad')} "
            f"| {cell(r.get('arm_err_mean'), '{:.3f} rad')} |"
        )


if __name__ == "__main__":
    main()

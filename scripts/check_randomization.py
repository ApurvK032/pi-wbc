"""Check stage 1's domain randomization in a running Isaac Lab simulation (design, section 6).

    uv run python scripts/check_randomization.py --out results/stage1/randomization_check/all.json \
        physics=newton_mjwarp
    uv run python scripts/check_randomization.py --slide --out <file> physics=newton_mjwarp \
        'env.events.physics_material.params.static_friction_range=[0.1,0.1]'

Default mode builds PiWbc-G1-Robust (256 robots, zero actions) and checks each group where the
simulation holds it: masses, torso centre of mass, PD gains and armature read back per robot;
the action delay seen in the PD targets at each physics step; sensor noise seen as two reads
of the same state differing; pushes seen as jumps in pelvis speed; and that robots stand for
10 s as often as with every group off (the centre-of-mass change can destabilize Newton).

--slide pushes standing robots sideways at 1.5 m/s and measures how far their feet slide in
0.4 s; run it at a low and a high robot friction (and the ground's default) to see which
surface's friction Newton uses. Extra arguments are Isaac Lab (Hydra) overrides.
"""

from __future__ import annotations

import argparse
from pathlib import Path

TASK = "PiWbc-G1-Robust"


def main() -> None:
    from isaaclab.app import add_launcher_args, launch_simulation

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--slide", action="store_true", help="friction slide test only")
    parser.add_argument("--nominal", action="store_true", help="every group off (baseline)")
    parser.add_argument("--num-envs", type=int, default=256)
    add_launcher_args(parser)
    args, overrides = parser.parse_known_args()

    import gymnasium as gym
    import torch
    from isaaclab_tasks.utils import resolve_task_config

    import pi_wbc.tasks  # noqa: F401
    from pi_wbc import provenance

    env_cfg, _ = resolve_task_config(TASK, "rsl_rl_cfg_entry_point", overrides=overrides)
    env_cfg.scene.num_envs = args.num_envs
    # fixed nominal pose and commands: the checks look at the randomization, not the task
    env_cfg.commands.base_velocity.rel_standing_envs = 1.0
    env_cfg.commands.upper_body.rel_still_envs = 1.0
    env_cfg.commands.body_pose.rel_nominal_envs = 1.0
    if args.slide or args.nominal:
        env_cfg.make_nominal()
        if args.slide:  # make_nominal fixes friction at 1.0; the slide test sets its own
            pm = env_cfg.events.physics_material.params
            for key in ("static_friction_range", "dynamic_friction_range"):
                value = next((o.split("=", 1)[1] for o in overrides if key in o), None)
                if value is not None:
                    lo, hi = (float(v) for v in value.strip("[]").split(","))
                    pm[key] = (lo, hi)

    report: dict = {"task": TASK, "overrides": overrides, "num_envs": args.num_envs}
    with launch_simulation(env_cfg, args):
        env = gym.make(TASK, cfg=env_cfg).unwrapped
        robot = env.scene["robot"]
        zero = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
        env.reset()
        report["friction_setting"] = {
            "robot": list(env.cfg.events.physics_material.params["static_friction_range"]),
            "ground": env.cfg.scene.terrain.physics_material.static_friction,
        }
        if args.slide:
            report["slide"] = slide_test(env, robot, zero)
        else:
            report["stand_10s"] = stand_test(env, robot, zero)
            if not args.nominal:
                report.update(read_back(env, robot))
                report["latency"] = latency_test(env, robot, zero)
                report["sensor_noise"] = noise_test(env)
                report["pushes"] = push_test(env, robot, zero)
        env.close()
    provenance.save_json(args.out, report)
    print(f"[check] wrote {args.out}")
    for key, value in report.items():
        print(f"[check] {key}: {value}")


def _range(t) -> list[float]:
    return [round(float(t.min()), 4), round(float(t.max()), 4)]


def read_back(env, robot) -> dict:
    """Per-robot values as the simulation now holds them, against the nominal ones.

    Nominal gains and armature come from the body task's export (tests/data/), which has the
    same robot and no randomization.
    """
    import json

    import torch

    data = robot.data
    names = list(robot.body_names)
    torso, palms = names.index("torso_link"), [i for i, n in enumerate(names) if "palm" in n]
    mass = data.body_mass.torch
    first_read = env.event_manager.get_term_cfg("link_masses").func.default_mass
    others = [i for i in range(len(names)) if i != torso and i not in palms]
    link_ratio = mass[:, others] / first_read[:, others]
    nominal = json.loads(Path("tests/data/policy_io_g1_body.json").read_text())["joints"]
    joints = list(robot.joint_names)

    def ratio(values, key):
        ref = torch.tensor([nominal[n][key] for n in joints], device=env.device)
        keep = ref > 0
        return _range(values[:, keep] / ref[keep])

    com = data.body_com_pos_b.torch[:, torso]
    return {
        "torso_mass_kg": _range(mass[:, torso]),
        "palm_mass_kg": [_range(mass[:, i]) for i in palms],
        "link_mass_ratio": _range(link_ratio),
        "torso_com_offset_from_mean_m": [_range(com[:, k] - com[:, k].mean()) for k in range(3)],
        "stiffness_ratio": ratio(data.joint_stiffness.torch, "stiffness"),
        "damping_ratio": ratio(data.joint_damping.torch, "damping"),
        "armature_ratio": ratio(data.joint_armature.torch, "armature"),
    }


def stand_test(env, robot, zero) -> dict:
    """10 s of zero actions: the state stays finite (the centre-of-mass change can destabilize
    Newton). Zero actions don't balance, so robots fall either way; falls are only counted."""
    import torch

    falls = 0
    for _ in range(round(10.0 / env.step_dt)):
        env.step(zero)
        falls += int(env.termination_manager.terminated.sum())
        if not torch.isfinite(robot.data.root_pos_w.torch).all():
            return {"finite": False}
    return {"finite": True, "falls_in_10s": falls}


def latency_test(env, robot, zero) -> dict:
    """PD targets at each physics step after the action jumps from zero to one."""
    import torch

    term = env.action_manager.get_term("joint_pos")
    recorded = []
    original = term.apply_actions

    def recording():
        original()
        recorded.append(robot.data.joint_pos_target.torch[:, term._joint_ids].clone())

    term.apply_actions = recording
    env.step(zero)
    recorded.clear()
    env.step(torch.ones_like(zero))
    term.apply_actions = original
    new = term._processed_actions
    # robots that fell and reset during these two steps drew a new delay: left out
    kept = (env.episode_length_buf >= 2).nonzero().flatten().tolist()
    agree = 0
    for e in kept:
        hits = [k for k, t in enumerate(recorded) if torch.allclose(t[e], new[e], atol=1e-5)]
        switch = hits[0] if hits else len(recorded)
        agree += int(switch == min(int(term._delay[e]), len(recorded)))
    return {
        "physics_steps_per_action": len(recorded),
        "delays_drawn": sorted(set(term._delay.tolist())),
        "targets_switch_after_the_drawn_delay": f"{agree}/{len(kept)} (robots not reset)",
    }


def noise_test(env) -> dict:
    """Two reads of the same state differ only by the observation noise."""
    a = env.observation_manager.compute()["policy"]
    b = env.observation_manager.compute()["policy"]
    diff = (a - b).abs()
    return {
        "corruption_on": bool(env.cfg.observations.policy.enable_corruption),
        "max_read_difference": round(float(diff.max()), 4),
    }


def push_test(env, robot, zero) -> dict:
    """The velocity change each push sets, recorded at the push itself.

    With zero actions the robots fall every second or so, and every reset restarts the 4-6 s
    push timer, so the test sets the timers to zero: every robot is pushed at the next step.
    """
    manager = env.event_manager
    term = manager.get_term_cfg("push_robot")
    original, kicks = term.func, []

    def recording(env_, env_ids, **params):
        before = robot.data.root_lin_vel_w.torch[env_ids, :2].clone()
        original(env_, env_ids, **params)
        after = robot.data.root_lin_vel_w.torch[env_ids, :2]
        kicks.append((after - before).abs())

    term.func = recording
    index = manager._mode_term_names["interval"].index("push_robot")
    manager._interval_term_time_left[index][:] = 0.0
    env.step(zero)
    term.func = original
    speed = term.params["velocity_range"]["x"][1]
    if not kicks:
        return {"set_top_speed": speed, "pushes": 0}
    import torch

    k = torch.cat(kicks)
    return {
        "set_top_speed": speed,
        "pushes": int(k.shape[0]),
        "largest_kick_per_axis_mps": round(float(k.max()), 3),
        "mean_kick_per_axis_mps": round(float(k.mean()), 3),
    }


def slide_test(env, robot, zero) -> dict:
    """Stand 1 s, kick the pelvis sideways at 1 m/s, measure foot travel in the next 0.1 s.

    The window ends before a robot with zero actions tips over, and robots that reset in it
    (their feet jump to a new spawn point) are left out.
    """
    for _ in range(round(1.0 / env.step_dt)):
        env.step(zero)
    feet = [i for i, n in enumerate(robot.body_names) if n.endswith("ankle_roll_link")]
    start = robot.data.body_pos_w.torch[:, feet, :2].clone()
    vel = robot.data.root_vel_w.torch.clone()
    vel[:, 1] = 1.0
    robot.write_root_velocity_to_sim(vel)
    steps = round(0.1 / env.step_dt)
    for _ in range(steps):
        env.step(zero)
    kept = env.episode_length_buf > steps
    travel = (robot.data.body_pos_w.torch[:, feet, :2] - start).norm(dim=-1).mean(dim=-1)[kept]
    return {
        "robots_kept": int(kept.sum()),
        "foot_travel_cm_mean": round(100 * float(travel.mean()), 2),
        "foot_travel_cm_range": [round(100 * v, 2) for v in _range(travel)],
    }


if __name__ == "__main__":
    main()

"""Write policy_io.json: what another simulator needs to run a policy trained on an Isaac Lab task.

It is read from a live environment (one instance), not copied by hand, so joint order, default
pose, gains and observation layout are exactly what the policy was trained with:

    uv run pi-wbc-policy-io --task PiWbc-G1-Flat-Stage0 --out <run>/exported/policy_io.json

Isaac Lab's own IO-descriptor export is deprecated (removal planned for 3.2), hence this file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _plain(x: Any) -> Any:
    """Convert torch tensors, Warp arrays and Isaac Lab ProxyArrays to plain Python values."""
    if hasattr(x, "torch"):  # Isaac Lab ProxyArray
        x = x.torch
    if hasattr(x, "detach"):  # torch tensor
        return x.detach().cpu().tolist()
    if hasattr(x, "numpy"):  # warp array
        return x.numpy().tolist()
    return x


def _first_env(values: Any) -> list[float]:
    values = _plain(values)
    return values[0] if values and isinstance(values[0], list) else values


def collect(env: Any) -> dict[str, Any]:
    """Everything a sim-to-sim runner needs, from an unwrapped ManagerBasedRLEnv.

    Resets the environment first, so the PD targets of joints no action drives are the ones
    training uses (reset events may set them).
    """
    env.reset()
    robot = env.scene["robot"]
    joint_names = list(robot.joint_names)
    default_pos = _first_env(robot.data.default_joint_pos)

    # Per-joint PD gains and limits as the simulator holds them (not as configured), so any
    # default Isaac Lab applies on top of our config is captured too.
    data = robot.data
    fields = {
        "stiffness": data.joint_stiffness,
        "damping": data.joint_damping,
        "armature": data.joint_armature,
        "friction": data.joint_friction_coeff,
        "effort_limit": data.joint_effort_limits,
        "velocity_limit": data.joint_vel_limits,
        "pos_limits": data.joint_pos_limits,
    }
    values = {key: _first_env(v) for key, v in fields.items()}
    joints: dict[str, dict[str, Any]] = {}
    for i, name in enumerate(joint_names):
        joints[name] = {"default_pos": default_pos[i]}
        joints[name].update({key: vals[i] for key, vals in values.items()})

    action_term = env.action_manager.get_term("joint_pos")
    driven = set(action_term._joint_names)
    targets = _first_env(robot.data.joint_pos_target)
    hold_targets = {n: targets[i] for i, n in enumerate(joint_names) if n not in driven}
    obs_mgr = env.observation_manager
    terms = obs_mgr.active_terms["policy"]
    dims = obs_mgr.group_obs_term_dim["policy"]
    obs_layout, start = [], 0
    for name, dim in zip(terms, dims, strict=True):
        size = 1
        for d in dim:
            size *= d
        obs_layout.append({"name": name, "start": start, "size": size})
        start += size

    # Joint order of each joint observation, from its resolved asset config.
    obs_joint_names: dict[str, list[str]] = {}
    for name, term_cfg in zip(terms, obs_mgr._group_obs_term_cfgs["policy"], strict=True):
        asset_cfg = term_cfg.params.get("asset_cfg") if term_cfg.params else None
        if name in ("joint_pos", "joint_vel") and asset_cfg is not None:
            ids = asset_cfg.joint_ids
            ids = range(len(joint_names)) if isinstance(ids, slice) else ids
            obs_joint_names[name] = [joint_names[i] for i in ids]

    physics = env.cfg.sim.physics
    solver = physics.solver_cfg
    solver_settings = {
        "num_substeps": physics.num_substeps,
        "integrator": str(solver.integrator),
        "iterations": solver.iterations,
        "ls_iterations": solver.ls_iterations,
        "impratio": solver.impratio,
        "tolerance": solver.tolerance,
        "use_mujoco_contacts": solver.use_mujoco_contacts,
        "default_shape_mu": physics.default_shape_cfg.mu,
    }

    command_cfg = env.cfg.commands.base_velocity
    friction = env.cfg.events.physics_material.params
    gait = getattr(env.cfg, "gait", None)
    commands = env.command_manager.active_terms
    body_pose = upper_body = None
    if "body_pose" in commands:
        term = env.command_manager.get_term("body_pose")
        body_pose = {
            "order": ["height", "waist_yaw", "waist_roll", "waist_pitch"],
            "waist_joint_names": list(term.cfg.waist_joint_names),
            "nominal": _plain(term.nominal),
            "low": _plain(term.lo),
            "high": _plain(term.hi),
            "max_rates": _plain(term.max_rate),
            "resampling_time_range": list(term.cfg.resampling_time_range),
        }
    if "upper_body" in commands:
        term = env.command_manager.get_term("upper_body")
        upper_body = {
            "joint_names": list(term.joint_names),
            "low": _plain(term.lo),  # target range at curriculum level 1
            "high": _plain(term.hi),
            "resampling_time_range": list(term.cfg.resampling_time_range),
        }
    return {
        "task_cfg": type(env.cfg).__name__,
        "physics_dt": env.physics_dt,
        "decimation": env.cfg.decimation,
        "policy_dt": env.step_dt,
        "solver": solver_settings,
        "gravity": list(env.cfg.sim.gravity),
        "joint_names": joint_names,
        "body_names": list(robot.body_names),
        "joints": joints,
        "action": {
            "joint_names": list(action_term._joint_names),
            "scale": _plain(action_term._scale),
            "offset_is_default_pos": bool(action_term.cfg.use_default_offset),
            "clip": action_term.cfg.clip,
        },
        "observation": {"dim": start, "terms": obs_layout, "joint_names": obs_joint_names},
        "gait": None if gait is None else gait.to_dict(),
        # body-pose command (pelvis height, waist yaw, roll, pitch) and random arm and hand
        # motion (stage 1, step 2); None for tasks without them
        "body_pose": body_pose,
        "upper_body": upper_body,
        # PD targets of joints the policy does not drive, as held during training
        "hold_targets": hold_targets,
        "commands": {
            "lin_vel_x": list(command_cfg.ranges.lin_vel_x),
            "lin_vel_y": list(command_cfg.ranges.lin_vel_y),
            "ang_vel_z": list(command_cfg.ranges.ang_vel_z),
        },
        "friction": {
            "static": list(friction["static_friction_range"]),
            "dynamic": list(friction["dynamic_friction_range"]),
        },
        "root_init": {
            "pos": list(robot.cfg.init_state.pos),
            "rot_xyzw": list(robot.cfg.init_state.rot),
        },
    }


def main(argv: list[str] | None = None) -> None:
    from isaaclab.app import add_launcher_args, launch_simulation

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--task", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--save-mjcf", type=Path, default=None, help="also save the MuJoCo model Newton builds"
    )
    add_launcher_args(parser)
    # Anything left over (e.g. physics=newton_mjwarp) goes to Isaac Lab as a Hydra override.
    args, overrides = parser.parse_known_args(argv)

    import gymnasium as gym
    from isaaclab_tasks.utils import resolve_task_config

    import pi_wbc.tasks  # noqa: F401  (registers our tasks)

    env_cfg, _ = resolve_task_config(args.task, "rsl_rl_cfg_entry_point", overrides=overrides)
    env_cfg.scene.num_envs = 1
    if args.save_mjcf is not None:
        args.save_mjcf.parent.mkdir(parents=True, exist_ok=True)
        env_cfg.sim.physics.solver_cfg.save_to_mjcf = str(args.save_mjcf.resolve())
    with launch_simulation(env_cfg, args):
        env = gym.make(args.task, cfg=env_cfg)
        try:
            data = collect(env.unwrapped)
        finally:
            env.close()
    # Joint observations must name their joints so the MuJoCo runner can order them.
    terms = [t["name"] for t in data["observation"]["terms"]]
    for name in ("joint_pos", "joint_vel"):
        if name in terms and name not in data["observation"]["joint_names"]:
            raise SystemExit(f"[policy_io] {name} has no joint list")
    action_joints = data["action"]["joint_names"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=2) + "\n")
    print(
        f"[policy_io] wrote {args.out}: {len(data['joint_names'])} joints, "
        f"{len(action_joints)} driven by the policy, obs dim {data['observation']['dim']}"
    )


if __name__ == "__main__":
    main()

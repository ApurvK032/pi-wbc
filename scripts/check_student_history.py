"""Check the student's observation history layout in Isaac Lab against the MuJoCo bench's.

    uv run python scripts/check_student_history.py physics=newton_mjwarp

With noise off, each of the student's 6 frames must equal the teacher group's slice of the same
terms at that step, oldest first, and every frame must be the first one right after a reset:
this is how `pi_wbc.sim2sim.mujoco_g1.G1MujocoSim.observation` builds the history. Prints one
line per step and "ALL OK" if every step matches.
"""

from __future__ import annotations

import argparse

TASK = "PiWbc-G1-Student"
HISTORY, FRAME = 6, 88  # the student's frames; the teacher group has base velocity (3) first


def main() -> None:
    from isaaclab.app import add_launcher_args, launch_simulation

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_launcher_args(parser)
    args, overrides = parser.parse_known_args()

    import gymnasium as gym
    import torch
    from isaaclab_tasks.utils import resolve_task_config

    import pi_wbc.tasks  # noqa: F401

    overrides = [*overrides, "env.observations.policy.enable_corruption=false"]
    cfg, _ = resolve_task_config(TASK, "rsl_rl_cfg_entry_point", overrides=overrides)
    cfg.scene.num_envs = 8
    with launch_simulation(cfg, args):
        env = gym.make(TASK, cfg=cfg).unwrapped
        obs, _ = env.reset()
        frames, ok = [], True
        for k in range(9):
            frames.append(obs["teacher"][:, 3 : 3 + FRAME].clone())
            expect = torch.cat(
                [frames[max(0, len(frames) - HISTORY + i)] for i in range(HISTORY)], dim=-1
            )
            match = torch.allclose(obs["policy"], expect, atol=1e-5)
            ok &= match
            print(f"step {k}: student history == teacher frames (oldest first): {match}")
            action = 0.3 * torch.randn(
                env.num_envs, env.action_manager.total_action_dim, device=env.device
            )
            obs, *_ = env.step(action)
        print("ALL OK" if ok else "MISMATCH")
        env.close()


if __name__ == "__main__":
    main()

"""Check that the pick-place task applies exactly NVIDIA's joint targets under NvidiaLowLevel.

Steps a few robots for 2 s (standing, then walking at 0.4 m/s, with a lean command) and compares
the leg and waist PD targets the task applies with NVIDIA's (action x 0.25 + default angles).

    uv run python scripts/check_nvidia_targets.py --viz none
"""

import argparse
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--policy-dir", type=Path, default=Path("third_party/gr00t_wbc/policy"))
add_launcher_args(parser)
args, overrides = parser.parse_known_args()

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils import resolve_task_config  # noqa: E402

import pi_wbc.tasks  # noqa: E402, F401
from pi_wbc.stage3.nvidia_wbc import ACTION_SCALE, DEFAULT_ANGLES, NvidiaLowLevel  # noqa: E402
from pi_wbc.tasks.g1_pick_place import HighLevel  # noqa: E402

TASK = "PiWbc-G1-PickPlace"


def main() -> None:
    env_cfg, _ = resolve_task_config(
        TASK, "rsl_rl_cfg_entry_point", overrides=[*overrides, "physics=newton_mjwarp"]
    )
    env_cfg.scene.num_envs = 4
    with launch_simulation(env_cfg, args):
        env = gym.make(TASK, cfg=env_cfg)
        base = env.unwrapped
        base.reset()
        hl = HighLevel(env)
        nv = NvidiaLowLevel(env, args.policy_dir)
        robot = base.scene["robot"]
        worst = 0.0
        with torch.no_grad():
            for k in range(100):
                if k == 50:
                    hl.velocity[:, 0] = 0.4
                    hl.body_goal[:, 3] = 0.2  # lean
                hl.apply()
                raw = nv.act()
                base.step(raw)
                want = torch.as_tensor(
                    nv.action * ACTION_SCALE + DEFAULT_ANGLES, device=base.device
                )
                got = robot.data.joint_pos_target
                got = (got.torch if hasattr(got, "torch") else got)[:, nv.body_ids[:15]]
                worst = max(worst, float((got - want).abs().max()))
        print(f"[check] largest difference between applied and NVIDIA's targets: {worst:.2e} rad")
        assert worst < 1e-5, worst
        z = robot.data.root_pos_w
        print(
            "[check] pelvis heights after 2 s:",
            (z.torch if hasattr(z, "torch") else z)[:, 2].tolist(),
        )
        env.close()


if __name__ == "__main__":
    main()

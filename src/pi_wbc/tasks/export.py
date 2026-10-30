"""Export a trained checkpoint for the MuJoCo bench, and check the export.

    uv run pi-wbc-export --task PiWbc-G1-Flat-Stage0 --checkpoint <run>/model_1499.pt

writes, next to the checkpoint in `exported/`:

- `policy.onnx`: the actor, as rsl_rl exports it;
- `policy_io.json`: joint order, default pose, gains, observation layout, timing, solver
  settings (see `pi_wbc.tasks.policy_io`);
- `newton_model.xml`: the MuJoCo model Newton builds for this task (same robot description
  as training, for a simulator-only comparison);
- `export_check.json`: the largest difference between ONNX and PyTorch actions over a short
  rollout in Isaac Lab, with provenance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    from isaaclab.app import add_launcher_args, launch_simulation

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--task", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None, help="default: <run>/exported")
    parser.add_argument("--check-steps", type=int, default=200, help="rollout steps compared")
    add_launcher_args(parser)
    args, overrides = parser.parse_known_args(argv)

    import gymnasium as gym
    import numpy as np
    import onnxruntime as ort
    import torch
    from isaaclab_rl.rsl_rl import (
        RslRlVecEnvWrapper,
        check_rsl_rl_version,
        create_rsl_rl_runner,
        handle_deprecated_rsl_rl_cfg,
    )
    from isaaclab_tasks.utils import resolve_task_config

    import pi_wbc.tasks  # noqa: F401  (registers our tasks)
    from pi_wbc import provenance
    from pi_wbc.sim2sim.evaluate import single_thread_options
    from pi_wbc.tasks.policy_io import collect

    checkpoint = args.checkpoint.resolve()
    out_dir = (args.out_dir or checkpoint.parent / "exported").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    env_cfg, agent_cfg = resolve_task_config(
        args.task, "rsl_rl_cfg_entry_point", overrides=overrides
    )
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, check_rsl_rl_version())
    env_cfg.scene.num_envs = 1
    if hasattr(env_cfg, "make_nominal"):  # export nominal gains and masses, no noise or delay
        env_cfg.make_nominal()
    env_cfg.sim.physics.solver_cfg.save_to_mjcf = str(out_dir / "newton_model.xml")

    with launch_simulation(env_cfg, args):
        env = gym.make(args.task, cfg=env_cfg)
        try:
            io = collect(env.unwrapped)
            venv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
            runner = create_rsl_rl_runner(venv, agent_cfg)
            runner.load(str(checkpoint))
            policy = runner.get_inference_policy(device=venv.unwrapped.device)
            runner.export_policy_to_onnx(path=str(out_dir), filename="policy.onnx")

            session = ort.InferenceSession(
                str(out_dir / "policy.onnx"),
                sess_options=single_thread_options(),
                providers=["CPUExecutionProvider"],
            )
            input_name = session.get_inputs()[0].name
            obs = venv.get_observations()
            max_diff = 0.0
            with torch.inference_mode():
                for _ in range(args.check_steps):
                    actions = policy(obs)
                    flat_obs = obs["policy"].detach().cpu().numpy().astype(np.float32)
                    onnx_actions = session.run(None, {input_name: flat_obs})[0]
                    diff = np.abs(onnx_actions - actions.detach().cpu().numpy()).max()
                    max_diff = max(max_diff, float(diff))
                    obs, _, _, _ = venv.step(actions)
        finally:
            env.close()

    (out_dir / "policy_io.json").write_text(json.dumps(io, indent=2) + "\n")
    provenance.save_json(
        out_dir / "export_check.json",
        {
            "task": args.task,
            "checkpoint": checkpoint.name,
            "check_steps": args.check_steps,
            "max_abs_action_diff_onnx_vs_torch": max_diff,
        },
    )
    print(f"[export] wrote {out_dir}: policy.onnx, policy_io.json, newton_model.xml")
    print(
        f"[export] ONNX vs PyTorch, largest action difference over {args.check_steps} steps: "
        f"{max_diff:.2e}"
    )


if __name__ == "__main__":
    main()

"""Isaac Lab tasks for π-WBC.

Isaac Lab imports this module through the `isaaclab.tasks` entry point (see pyproject.toml), so
`isaaclab train --task <id>` finds these tasks. Registration only names config classes; nothing
from Isaac Lab is imported until a task is created.
"""

import gymnasium as gym

gym.register(
    id="PiWbc-G1-Flat-Stage0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_flat_stage0:G1FlatStage0EnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.g1_flat_stage0:G1FlatStage0PPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="PiWbc-G1-Gait",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_gait:G1GaitEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.g1_gait:G1GaitPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="PiWbc-G1-Body",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_body:G1BodyEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.g1_body:G1BodyPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="PiWbc-G1-Robust",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_robust:G1RobustEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.g1_robust:G1RobustPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="PiWbc-G1-Teacher",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_student:G1TeacherStudentEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.g1_student:G1TeacherPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="PiWbc-G1-Student",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_student:G1TeacherStudentEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.g1_student:G1StudentDistillationRunnerCfg",
        "default_agent": "rsl_rl",
    },
)

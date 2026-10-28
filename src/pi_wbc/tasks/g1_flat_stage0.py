"""Stage 0 bench task: Isaac Lab's G1 flat velocity task, moved onto the 29-DoF G1.

This is not the project's controller (that is stage 1, with our own rewards). It exists to
produce a walking policy for the MuJoCo sim-to-sim bench, so its choices favour a clean
comparison between simulators:

- The robot is Isaac Lab's `G1_29DOF_CFG` (29 body joints plus Dex3 hands, 43 joints in all),
  the same robot description as MuJoCo Menagerie's `g1_with_hands.xml`.
- The policy sees and drives only the 29 body joints. The 14 hand joints are held at a fixed
  pose; they add mass but take no part in walking.
- Every joint uses a PD controller inside the simulator (implicit actuator), which MuJoCo can
  reproduce with position actuators. Gains are moderate (Unitree-style), not the near-rigid
  arm and waist gains `G1_29DOF_CFG` uses for loco-manipulation.
- Default pose: Menagerie's standing pose for arms and hands, with slightly bent knees.

Observations, actions (scale 0.5 around the default pose), rewards, commands and randomization
otherwise follow the built-in `Isaac-Velocity-Flat-G1` task.
"""

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass, replace
from isaaclab_assets import G1_29DOF_CFG
from isaaclab_tasks.core.velocity.config.g1.agents.rsl_rl_ppo_cfg import G1FlatPPORunnerCfg
from isaaclab_tasks.core.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg

# The 29 body joints, as name patterns. Hand joints are `.*_hand_.*_joint`.
BODY_JOINTS = [
    ".*_hip_pitch_joint",
    ".*_hip_roll_joint",
    ".*_hip_yaw_joint",
    ".*_knee_joint",
    ".*_ankle_pitch_joint",
    ".*_ankle_roll_joint",
    "waist_.*_joint",
    ".*_shoulder_.*_joint",
    ".*_elbow_joint",
    ".*_wrist_.*_joint",
]
ARM_JOINTS = [".*_shoulder_.*_joint", ".*_elbow_joint", ".*_wrist_.*_joint"]

DEFAULT_JOINT_POS = {
    # legs: slightly bent knees
    ".*_hip_pitch_joint": -0.1,
    ".*_knee_joint": 0.3,
    ".*_ankle_pitch_joint": -0.2,
    # arms and hands: MuJoCo Menagerie's "stand" keyframe
    ".*_shoulder_pitch_joint": 0.2,
    "left_shoulder_roll_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    ".*_elbow_joint": 1.28,
    # Menagerie's keyframe uses ±1.05, just outside both models' thumb limit of ±1.047.
    "left_hand_thumb_1_joint": 1.0,
    "right_hand_thumb_1_joint": -1.0,
    # joints not listed start at 0 (patterns must not overlap, so no catch-all here)
}

ACTUATORS = {
    "legs": ImplicitActuatorCfg(
        joint_names_expr=[".*_hip_.*_joint", ".*_knee_joint"],
        stiffness={".*_hip_.*_joint": 100.0, ".*_knee_joint": 150.0},
        damping={".*_hip_.*_joint": 2.0, ".*_knee_joint": 4.0},
        joint_effort_limit={".*_hip_.*_joint": 88.0, ".*_knee_joint": 139.0},
        joint_velocity_limit={".*_hip_.*_joint": 32.0, ".*_knee_joint": 20.0},
        armature=0.03,
    ),
    "feet": ImplicitActuatorCfg(
        joint_names_expr=[".*_ankle_.*_joint"],
        stiffness=40.0,
        damping=2.0,
        joint_effort_limit=50.0,
        joint_velocity_limit=37.0,
        armature=0.03,
    ),
    "waist": ImplicitActuatorCfg(
        joint_names_expr=["waist_.*_joint"],
        stiffness=200.0,
        damping=5.0,
        joint_effort_limit={
            "waist_yaw_joint": 88.0,
            "waist_roll_joint": 50.0,
            "waist_pitch_joint": 50.0,
        },
        joint_velocity_limit={"waist_yaw_joint": 32.0, "waist_(roll|pitch)_joint": 37.0},
        armature=0.01,
    ),
    "arms": ImplicitActuatorCfg(
        joint_names_expr=[".*_shoulder_.*_joint", ".*_elbow_joint"],
        stiffness=40.0,
        damping=2.0,
        joint_effort_limit=25.0,
        joint_velocity_limit=37.0,
        armature=0.01,
    ),
    "wrists": ImplicitActuatorCfg(
        joint_names_expr=[".*_wrist_.*_joint"],
        stiffness=20.0,
        damping=1.0,
        joint_effort_limit={".*_wrist_roll_joint": 25.0, ".*_wrist_(pitch|yaw)_joint": 5.0},
        joint_velocity_limit=22.0,
        armature=0.01,
    ),
    "hands": ImplicitActuatorCfg(
        joint_names_expr=[".*_hand_.*_joint"],
        stiffness=20.0,
        damping=2.0,
        joint_effort_limit=5.0,
        joint_velocity_limit=10.0,
        armature=0.001,
    ),
}

G1_29DOF_STAGE0_CFG = replace(
    G1_29DOF_CFG,
    prim_path="{ENV_REGEX_NS}/Robot",
    # the velocity task's foot-contact rewards and fall termination need contact sensors
    spawn=replace(G1_29DOF_CFG.spawn, activate_contact_sensors=True),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.8),
        rot=(0.0, 0.0, 0.0, 1.0),  # (x, y, z, w): upright, facing +x
        joint_pos=DEFAULT_JOINT_POS,
        joint_vel={".*": 0.0},
    ),
    actuators=ACTUATORS,
)


@configclass
class G1FlatStage0EnvCfg(G1FlatEnvCfg):
    """The built-in G1 flat task on the 29-DoF G1, driving only the body joints."""

    def __post_init__(self):
        super().__post_init__()
        # The full-collision 29-DoF model makes more contacts than the built-in minimal G1.
        newton_mjwarp = self.sim.physics.newton_mjwarp
        newton_mjwarp.solver_cfg.njmax = 300
        newton_mjwarp.solver_cfg.nconmax = 40

        self.scene.robot = G1_29DOF_STAGE0_CFG

        # actions and joint observations: the 29 body joints only
        self.actions.joint_pos.joint_names = BODY_JOINTS
        # one SceneEntityCfg per term: Isaac Lab fills in joint ids when it resolves each one
        for term in (self.observations.policy.joint_pos, self.observations.policy.joint_vel):
            term.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=BODY_JOINTS)

        # rewards that named joints of the older 37-joint G1
        self.rewards.joint_deviation_arms.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=ARM_JOINTS
        )
        self.rewards.joint_deviation_torso.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names="waist_.*_joint"
        )
        self.rewards.joint_deviation_fingers = None


@configclass
class G1FlatStage0PPORunnerCfg(G1FlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "pi_wbc_g1_flat_stage0"

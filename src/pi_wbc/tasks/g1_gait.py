"""Stage 1, step 1: a G1 walking task with a gait clock (docs/STAGE1_DESIGN.md, section 12).

Changes from the stage 0 bench task:

- The robot uses the PD gains of NVIDIA's Decoupled WBC (D-012). Arms and hands are held at
  their default pose by their PD controllers; later steps give them commands.
- The policy drives the 12 leg joints and 3 waist joints, action scale 0.25.
- Velocity commands sample the yaw rate directly (no heading controller), with 10% standing,
  10% turning in place, and a deadband for very slow walking.
- The policy also sees a gait clock, and the rewards are Unitree's G1 recipe (unitree_rl_gym):
  a contact schedule, swing-foot height, no foot slip, hips near neutral, plus regularizers.
  The weights and reasons are listed in the design, section 5.

Not yet here: height, torso and arm commands, the new randomization, curriculum and the
teacher-student split (steps 2-4).
"""

import isaaclab_tasks.core.velocity.mdp as mdp
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass, replace

from pi_wbc.tasks import mdp as pi_mdp
from pi_wbc.tasks.g1_flat_stage0 import (
    G1_29DOF_STAGE0_CFG,
    G1FlatStage0EnvCfg,
    G1FlatStage0PPORunnerCfg,
)

LEG_JOINTS = [
    ".*_hip_pitch_joint",
    ".*_hip_roll_joint",
    ".*_hip_yaw_joint",
    ".*_knee_joint",
    ".*_ankle_pitch_joint",
    ".*_ankle_roll_joint",
]
WAIST_JOINTS = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
POLICY_JOINTS = LEG_JOINTS + WAIST_JOINTS
FEET = ".*_ankle_roll_link"

# NVIDIA Decoupled WBC gains (g1_29dof_gear_wbc.yaml, MOTOR_KP / MOTOR_KD). Effort and velocity
# limits and armature are unchanged from stage 0.
ACTUATORS = {
    "legs": ImplicitActuatorCfg(
        joint_names_expr=[".*_hip_.*_joint", ".*_knee_joint"],
        stiffness={".*_hip_.*_joint": 150.0, ".*_knee_joint": 200.0},
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
        stiffness=250.0,
        damping=5.0,
        joint_effort_limit={
            "waist_yaw_joint": 88.0,
            "waist_roll_joint": 50.0,
            "waist_pitch_joint": 50.0,
        },
        joint_velocity_limit={"waist_yaw_joint": 32.0, "waist_(roll|pitch)_joint": 37.0},
        armature=0.01,
    ),
    "shoulders": ImplicitActuatorCfg(
        joint_names_expr=[".*_shoulder_.*_joint"],
        stiffness={".*_shoulder_(pitch|roll)_joint": 100.0, ".*_shoulder_yaw_joint": 40.0},
        damping={".*_shoulder_(pitch|roll)_joint": 5.0, ".*_shoulder_yaw_joint": 2.0},
        joint_effort_limit=25.0,
        joint_velocity_limit=37.0,
        armature=0.01,
    ),
    "elbows": ImplicitActuatorCfg(
        joint_names_expr=[".*_elbow_joint"],
        stiffness=40.0,
        damping=2.0,
        joint_effort_limit=25.0,
        joint_velocity_limit=37.0,
        armature=0.01,
    ),
    "wrists": ImplicitActuatorCfg(
        joint_names_expr=[".*_wrist_.*_joint"],
        stiffness=20.0,
        damping=2.0,
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
G1_29DOF_GAIT_CFG = replace(G1_29DOF_STAGE0_CFG, actuators=ACTUATORS)


@configclass
class GaitCfg:
    """Gait clock and targets, shared by the observation, the rewards and the export."""

    period: float = 0.8
    """Seconds per gait cycle (two steps), as in unitree_rl_gym."""
    offset: float = 0.5
    """Right leg's lag behind the left, as a share of the period."""
    stance_fraction: float = 0.55
    """Share of each leg's cycle spent on the ground."""
    stand_threshold: float = 0.1
    """Commands below this (m/s and rad/s) mean "stand": both feet down."""
    swing_height: float = 0.08
    """Target height (m) of a swinging foot's ankle-roll link origin. The origin sits about
    3.6 cm above the sole, so this is about 4.5 cm of clearance (Unitree's value)."""
    base_height: float = 0.78
    """Target pelvis height (m), Unitree's value; a height command replaces it in step 2."""


@configclass
class GaitRewardsCfg:
    """Unitree's G1 reward recipe; weights and reasons in the design, section 5."""

    # task
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    base_height = RewTerm(func=mdp.base_height_l2, weight=-10.0, params={"target_height": 0.78})
    # gait
    gait_contact = RewTerm(
        func=pi_mdp.gait_contact,
        weight=0.18,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET),
            "command_name": "base_velocity",
            "period": 0.8,
            "offset": 0.5,
            "stance_fraction": 0.55,
            "stand_threshold": 0.1,
        },
    )
    feet_swing_height = RewTerm(
        func=pi_mdp.feet_swing_height,
        weight=-20.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET),
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET),
            "target_height": 0.08,
        },
    )
    feet_contact_velocity = RewTerm(
        func=pi_mdp.feet_contact_velocity,
        weight=-0.2,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET),
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET),
        },
    )
    hip_deviation = RewTerm(
        func=pi_mdp.joint_deviation_l2,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_(roll|yaw)_joint"])},
    )
    waist_deviation = RewTerm(
        func=pi_mdp.joint_deviation_l2,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=WAIST_JOINTS)},
    )
    # stability
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    alive = RewTerm(func=mdp.is_alive, weight=0.15)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    # effort and smoothness, on the joints the policy drives
    joint_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=POLICY_JOINTS)},
    )
    joint_vel_l2 = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1.0e-3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=POLICY_JOINTS)},
    )
    joint_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=POLICY_JOINTS)},
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=POLICY_JOINTS)},
    )


@configclass
class G1GaitEnvCfg(G1FlatStage0EnvCfg):
    """Flat-ground G1 walking with a gait clock; arms held at their default pose."""

    gait: GaitCfg = GaitCfg()

    def __post_init__(self):
        super().__post_init__()
        gait = self.gait
        self.scene.robot = G1_29DOF_GAIT_CFG

        # the policy drives legs and waist; arms and hands keep their default targets
        self.actions.joint_pos.joint_names = POLICY_JOINTS
        self.actions.joint_pos.scale = 0.25

        self.commands.base_velocity = pi_mdp.GaitVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(10.0, 10.0),
            rel_standing_envs=0.1,
            rel_turning_envs=0.1,
            deadband=gait.stand_threshold,
            heading_command=False,
            debug_vis=False,
            ranges=pi_mdp.GaitVelocityCommandCfg.Ranges(
                lin_vel_x=(-0.5, 1.0), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-1.0, 1.0)
            ),
        )

        self.observations.policy.gait_clock = ObsTerm(
            func=pi_mdp.gait_clock, params={"period": gait.period}
        )

        self.rewards = GaitRewardsCfg()
        clock = self.rewards.gait_contact.params
        clock.update(
            period=gait.period,
            offset=gait.offset,
            stance_fraction=gait.stance_fraction,
            stand_threshold=gait.stand_threshold,
        )
        self.rewards.feet_swing_height.params["target_height"] = gait.swing_height
        self.rewards.base_height.params["target_height"] = gait.base_height


@configclass
class G1GaitPPORunnerCfg(G1FlatStage0PPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "pi_wbc_g1_gait"
        self.max_iterations = 3000
        self.save_interval = 200
        self.actor.hidden_dims = [512, 256, 128]
        self.critic.hidden_dims = [512, 256, 128]

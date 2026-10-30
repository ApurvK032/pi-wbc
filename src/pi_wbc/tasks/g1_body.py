"""Stage 1, step 2: the gait task plus body-pose commands and moving arms (design, section 12).

Changes from `PiWbc-G1-Gait`:

- A body-pose command: pelvis height (0.50-0.78 m) and torso orientation as the three waist
  angles (yaw ±0.6, roll ±0.2, pitch 0-0.45 rad), moved smoothly towards a new random goal
  every 4-8 s. The policy sees it; height and waist rewards track it.
- The arms and hands move: every 2-4 s a new random pose, reached in a straight line, written
  straight to their PD targets (they are not part of the policy's action). 20% of episodes keep
  them still. (v6: was 1-3 s; arm swings of up to 2.5 rad/s cost turning in place.)
- A payload of 0-1 kg is added to each palm, drawn once per robot at startup: changing masses
  at every reset made Newton training 2.3 times slower (1.5 s more per iteration).
- A curriculum widens the body-pose and arm ranges from nothing (level 0) to full (level 1)
  while few episodes end in a fall and velocity tracking stays good, judged over windows of
  20 iterations.
- Standing still is enforced through joint speed rather than the offset from the default pose,
  so the robot may crouch while standing.
- The leg actions centre on a crouch pose that follows the commanded height (a table of hip,
  knee and ankle angles; zero offset at the nominal height). Without it, two runs never
  crouched: a 0.60 m crouch needs knee actions of about 5 at the action scale of 0.25.

The waist pitch range stops at 0.45 rad: the joint's limit is 0.52 rad and the joint-limit
penalty starts at 90% of it (the design's 0.6 rad was beyond the joint).
"""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_tasks.core.velocity import mdp

from pi_wbc.tasks import mdp as pi_mdp
from pi_wbc.tasks.g1_gait import (
    HELD_JOINTS,
    LEG_JOINTS,
    POLICY_JOINTS,
    WAIST_JOINTS,
    G1GaitEnvCfg,
    G1GaitPPORunnerCfg,
)

# Arm target ranges at curriculum level 1 (rad). They keep the arms out to the sides and in
# front of the body: training has no self-collisions (as in Isaac Lab's G1 asset), but MuJoCo
# Menagerie's model has them. Hands use their full joint limits.
ARM_RANGES = {
    ".*_shoulder_pitch_joint": (-2.0, 1.0),
    "left_shoulder_roll_joint": (-0.1, 1.5),
    "right_shoulder_roll_joint": (-1.5, 0.1),
    ".*_shoulder_yaw_joint": (-1.0, 1.0),
    ".*_elbow_joint": (-0.5, 2.0),
    ".*_wrist_roll_joint": (-1.5, 1.5),
    ".*_wrist_(pitch|yaw)_joint": (-1.0, 1.0),
}
PALMS = ".*_hand_palm_link"

# Crouch poses the leg actions centre on (v5): feet flat, pelvis level, centre of mass over the
# feet, from scripts/crouch_table.py on MuJoCo Menagerie's G1. Pelvis heights (m), ascending,
# and the hip pitch, knee and ankle pitch angles (rad) at each; the top row is the default pose.
CROUCH_HEIGHTS = [0.492, 0.550, 0.600, 0.643, 0.714, 0.784]
CROUCH_POSES = {
    ".*_hip_pitch_joint": [-1.13, -0.93, -0.75, -0.65, -0.45, -0.10],
    ".*_knee_joint": [2.00, 1.80, 1.60, 1.40, 1.00, 0.30],
    ".*_ankle_pitch_joint": [-0.87, -0.87, -0.85, -0.75, -0.55, -0.20],
}


@configclass
class G1BodyEnvCfg(G1GaitEnvCfg):
    """G1 walking with height and torso commands, moving arms and hand payloads."""

    def __post_init__(self):
        super().__post_init__()

        # leg actions centre on the crouch pose for the commanded height (v5)
        self.actions.joint_pos = pi_mdp.HeightPoseJointPositionActionCfg(
            asset_name="robot",
            joint_names=POLICY_JOINTS,
            scale=0.25,
            use_default_offset=True,
            command_name="body_pose",
            nominal_height=self.gait.base_height,
            heights=CROUCH_HEIGHTS,
            poses=CROUCH_POSES,
        )

        # commands
        self.commands.body_pose = pi_mdp.BodyPoseCommandCfg(
            asset_name="robot",
            resampling_time_range=(4.0, 8.0),
            nominal=(self.gait.base_height, 0.0, 0.0, 0.0),
        )
        self.commands.upper_body = pi_mdp.UpperBodyMotionCommandCfg(
            asset_name="robot",
            resampling_time_range=(2.0, 4.0),
            joint_names=HELD_JOINTS,
            ranges=ARM_RANGES,
        )
        self.observations.policy.body_pose_command = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "body_pose"}
        )

        # payload in each hand, one per robot for the whole run (see the module docstring)
        self.events.hand_payload = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=PALMS),
                "mass_distribution_params": (0.0, 1.0),
                "operation": "add",
            },
        )

        # rewards: height and waist follow the command instead of fixed values
        rewards = self.rewards
        rewards.base_height = RewTerm(
            func=pi_mdp.base_height_command_l2,
            weight=-10.0,
            params={"command_name": "body_pose", "asset_cfg": SceneEntityCfg("robot")},
        )
        # v4: the squared penalty alone left crouching untried (the robot stood at the nominal
        # height whatever the command); the waist, rewarded this way, was tracked
        rewards.height_tracking = RewTerm(
            func=pi_mdp.base_height_command_exp,
            weight=1.0,
            params={
                "command_name": "body_pose",
                "std": 0.1,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        rewards.waist_deviation = None
        rewards.waist_tracking = RewTerm(
            func=pi_mdp.waist_command_exp,
            weight=1.0,
            params={
                "command_name": "body_pose",
                "std": 0.25,
                "asset_cfg": SceneEntityCfg("robot", joint_names=WAIST_JOINTS, preserve_order=True),
            },
        )
        rewards.stand_still = RewTerm(
            func=pi_mdp.stand_still_joint_vel,
            weight=-0.1,
            params={
                "command_name": "base_velocity",
                "stand_threshold": self.gait.stand_threshold,
                "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS),
            },
        )

        # curriculum: widen body-pose and arm ranges while walking stays good
        self.curriculum.upper_body_level = CurrTerm(
            func=pi_mdp.upper_body_level,
            params={
                "command_names": ["body_pose", "upper_body"],
                # per-second reward over its weight; v7 adds turning (v3-v6 gave it up as the
                # ranges widened: 0.39 to 0.26 per second while speed tracking held)
                "reward_thresholds": {"track_lin_vel_xy_exp": 0.6, "track_ang_vel_z_exp": 0.35},
                "raise_below": 0.10,  # share of ended episodes that fell
                "lower_above": 0.25,
                "level_step": 0.05,
                "min_interval_steps": 480,  # 20 PPO iterations of 24 steps
            },
        )


@configclass
class G1BodyPPORunnerCfg(G1GaitPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "pi_wbc_g1_body"

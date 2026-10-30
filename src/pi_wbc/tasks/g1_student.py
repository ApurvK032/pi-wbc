"""Stage 1, step 4: teacher and student (design, section 7).

One environment, the robust task, with two observation groups:

- `policy`, the student's: what a real G1 measures. The IMU's turn rate and gravity direction,
  joint positions and speeds, the commands, the last action and the gait clock, with sensor
  noise, over the last 6 policy steps (0.12 s, oldest first, every term of a step together).
  No base velocity: a real robot cannot measure it, so the student must infer it.
- `teacher`: the same terms for the current step without noise, plus privileged state: base
  velocity, foot contacts and heights, the robot's friction, its torso and hand masses, its
  torso centre of mass, the scale of its leg gains and its action delay.

`PiWbc-G1-Teacher` trains the teacher with PPO on the teacher group (actor and critic).
`PiWbc-G1-Student` distills the student from a trained teacher with rsl_rl's `Distillation`:
the student drives the robot and the teacher labels what it should have done (DAgger), so the
student learns to recover from its own mistakes. Train it with `--checkpoint <teacher model>`.
"""

import copy

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlMLPModelCfg,
)

from pi_wbc.tasks import mdp as pi_mdp
from pi_wbc.tasks.g1_gait import FEET, LEG_JOINTS
from pi_wbc.tasks.g1_robust import G1RobustEnvCfg, G1RobustPPORunnerCfg

HISTORY_STEPS = 6


@configclass
class G1TeacherStudentEnvCfg(G1RobustEnvCfg):
    """The robust task with a student group (deployable, with history) and a teacher group."""

    def __post_init__(self):
        super().__post_init__()
        base = self.observations.policy

        student = copy.deepcopy(base)
        student.base_lin_vel = None
        student.history_length = HISTORY_STEPS
        student.history_order = "time"
        student.flatten_history_dim = True

        teacher = copy.deepcopy(base)
        teacher.enable_corruption = False
        teacher.feet_contact = ObsTerm(
            func=pi_mdp.feet_contact,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET)},
        )
        teacher.feet_height = ObsTerm(
            func=pi_mdp.feet_height, params={"asset_cfg": SceneEntityCfg("robot", body_names=FEET)}
        )
        teacher.friction = ObsTerm(
            func=pi_mdp.feet_friction,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=FEET)},
        )
        teacher.mass_offset = ObsTerm(
            func=pi_mdp.body_mass_offset,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=["torso_link", "left_hand_palm_link", "right_hand_palm_link"],
                    preserve_order=True,
                )
            },
        )
        teacher.com_offset = ObsTerm(
            func=pi_mdp.body_com_offset,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link")},
        )
        teacher.leg_stiffness = ObsTerm(
            func=pi_mdp.stiffness_scale,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS)},
        )
        teacher.action_delay = ObsTerm(func=pi_mdp.action_delay)

        self.observations.policy = student
        self.observations.teacher = teacher


@configclass
class G1TeacherPPORunnerCfg(G1RobustPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "pi_wbc_g1_teacher"
        self.obs_groups = {"actor": ["teacher"], "critic": ["teacher"]}


@configclass
class G1StudentDistillationRunnerCfg(RslRlDistillationRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 100
    experiment_name = "pi_wbc_g1_student"
    student = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.1),
    )
    # must match the PPO teacher's actor, whose weights are loaded into it
    teacher = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=2,
        learning_rate=1.0e-3,
        gradient_length=15,
        max_grad_norm=1.0,
    )

    def __post_init__(self):
        self.obs_groups = {"student": ["policy"], "teacher": ["teacher"]}

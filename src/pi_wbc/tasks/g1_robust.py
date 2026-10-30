"""Stage 1, step 3: the body task plus domain randomization in six groups (design, section 6).

- friction: robot shape friction 0.1-1.25, with the ground set to 0.05;
- mass: torso -1 to +3 kg, every link x0.9-1.1, torso centre of mass ±3 cm, hand payload
  0-1 kg (the payload comes from the body task);
- actuators: PD stiffness and damping, joint armature, x0.8-1.2;
- latency: 0-4 physics steps (0-20 ms) between a new action and its PD targets;
- sensor noise: the base task's noise on the policy's readings (Unitree's values);
- pushes: velocity kicks to the pelvis every 4-6 s, ±0.5 m/s growing to ±1.5 m/s with the
  curriculum.

Newton (MuJoCo-Warp) combines two touching surfaces' friction by taking the larger, so the
ground is set low and the robot's value decides (stage 0 finding; checked in a running
simulation, see `scripts/check_randomization.py`). Masses, centres of mass and gains are drawn
once per robot at startup: changing masses at every reset slowed Newton training 2.3 times
(D-017), and variety comes from 4096 robots.

`ABLATION_OVERRIDES` lists, per group, the config overrides that switch it off. They work on
the finished config (Isaac Lab applies overrides after `__post_init__`), so each removes event
terms or sets values the simulation reads at run time. `make_nominal` switches every group off,
for exporting a policy with nominal gains and masses.
"""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_tasks.core.velocity import mdp

from pi_wbc.tasks.g1_body import G1BodyEnvCfg, G1BodyPPORunnerCfg

GROUND_FRICTION = 0.05
ROBOT_FRICTION = (0.1, 1.25)
NOMINAL_FRICTION = 1.0  # Isaac Lab's ground; the bench's floor
PUSH_SPEEDS = (0.5, 1.5)  # m/s at curriculum level 0 and 1

ABLATION_OVERRIDES: dict[str, list[str]] = {
    "friction": [
        f"env.events.physics_material.params.static_friction_range=[{NOMINAL_FRICTION},{NOMINAL_FRICTION}]",
        f"env.events.physics_material.params.dynamic_friction_range=[{NOMINAL_FRICTION},{NOMINAL_FRICTION}]",
    ],
    "mass": [
        "env.events.add_base_mass=null",
        "env.events.link_masses=null",
        "env.events.torso_com=null",
        "env.events.hand_payload=null",
    ],
    "actuators": ["env.events.actuator_gains=null", "env.events.joint_armature=null"],
    "latency": ["env.actions.joint_pos.max_delay_steps=0"],
    "sensor_noise": ["env.observations.policy.enable_corruption=false"],
    "pushes": ["env.events.push_robot=null"],
}


@configclass
class G1RobustEnvCfg(G1BodyEnvCfg):
    """The body task with all six randomization groups on."""

    def __post_init__(self):
        super().__post_init__()
        events = self.events

        # friction
        ground = self.scene.terrain.physics_material
        ground.static_friction = ground.dynamic_friction = GROUND_FRICTION
        events.physics_material.params["static_friction_range"] = ROBOT_FRICTION
        events.physics_material.params["dynamic_friction_range"] = ROBOT_FRICTION

        # mass (the hand payload comes from the body task)
        events.add_base_mass.params.update(
            mass_distribution_params=(-1.0, 3.0), operation="add", distribution="uniform"
        )
        events.link_masses = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "mass_distribution_params": (0.9, 1.1),
                "operation": "scale",
            },
        )
        events.torso_com = EventTerm(
            func=mdp.randomize_rigid_body_com,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
                "com_range": {"x": (-0.03, 0.03), "y": (-0.03, 0.03), "z": (-0.03, 0.03)},
            },
        )

        # actuators (joint friction is 0 in this model, so scaling it would do nothing)
        events.actuator_gains = EventTerm(
            func=mdp.randomize_actuator_gains,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "stiffness_distribution_params": (0.8, 1.2),
                "damping_distribution_params": (0.8, 1.2),
                "operation": "scale",
            },
        )
        events.joint_armature = EventTerm(
            func=mdp.randomize_joint_parameters,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "armature_distribution_params": (0.8, 1.2),
                "operation": "scale",
            },
        )

        # latency
        self.actions.joint_pos.max_delay_steps = 4

        # sensor noise: the base task's (Unitree's values), on

        # pushes: every 4-6 s, growing with the curriculum
        events.push_robot.interval_range_s = (4.0, 6.0)
        self.curriculum.upper_body_level.params.update(
            push_event="push_robot", push_speeds=PUSH_SPEEDS
        )

    def make_nominal(self) -> None:
        """Switch every randomization group off (for exporting with nominal properties)."""
        events = self.events
        for name in ("physics_material",):
            getattr(events, name).params["static_friction_range"] = (NOMINAL_FRICTION,) * 2
            getattr(events, name).params["dynamic_friction_range"] = (NOMINAL_FRICTION,) * 2
        for name in (
            "add_base_mass",
            "link_masses",
            "torso_com",
            "hand_payload",
            "actuator_gains",
            "joint_armature",
            "push_robot",
        ):
            setattr(events, name, None)
        self.actions.joint_pos.max_delay_steps = 0
        self.observations.policy.enable_corruption = False


@configclass
class G1RobustPPORunnerCfg(G1BodyPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "pi_wbc_g1_robust"

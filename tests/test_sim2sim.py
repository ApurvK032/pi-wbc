"""Checks of the MuJoCo side of the sim-to-sim bench against Isaac Lab's conventions.

The fixture is a policy_io.json written by Isaac Lab for the stage 0 task. Tests that need the
G1 model skip when MuJoCo Menagerie has not been fetched (scripts/fetch_menagerie.sh).
"""

from pathlib import Path

import mujoco
import numpy as np
import pytest

from pi_wbc.sim2sim import evaluate
from pi_wbc.sim2sim.mujoco_g1 import (
    G1MujocoSim,
    PolicyIO,
    build_model,
    match_name,
    wxyz_to_xyzw,
    xyzw_to_wxyz,
)

ROOT = Path(__file__).resolve().parents[1]
IO_PATH = ROOT / "tests" / "data" / "policy_io_g1_stage0.json"
SCENE = ROOT / "third_party" / "mujoco_menagerie" / "unitree_g1" / "scene_with_hands.xml"
needs_model = pytest.mark.skipif(not SCENE.exists(), reason="run scripts/fetch_menagerie.sh")


@pytest.fixture(scope="module")
def io() -> PolicyIO:
    return PolicyIO.load(IO_PATH)


@pytest.fixture(scope="module")
def model(io: PolicyIO) -> mujoco.MjModel:
    if not SCENE.exists():
        pytest.skip("run scripts/fetch_menagerie.sh")
    return build_model(SCENE, io)


@pytest.fixture
def sim(model: mujoco.MjModel, io: PolicyIO) -> G1MujocoSim:
    s = G1MujocoSim(model, io)
    s.reset()
    return s


def _set_root(sim: G1MujocoSim, quat_wxyz, lin_w=(0, 0, 0), ang_local=(0, 0, 0)) -> None:
    a, v = sim.root_qpos_adr, sim.root_qvel_adr
    sim.data.qpos[a + 3 : a + 7] = quat_wxyz
    sim.data.qvel[v : v + 3] = lin_w  # free joint: linear velocity of the frame origin, world axes
    sim.data.qvel[v + 3 : v + 6] = ang_local  # free joint: angular velocity, body axes
    mujoco.mj_forward(sim.model, sim.data)


def _obs_term(sim: G1MujocoSim, obs: np.ndarray, name: str) -> np.ndarray:
    start = 0
    for term, size in sim.io.obs_terms:
        if term == name:
            return obs[start : start + size]
        start += size
    raise KeyError(name)


# -- conventions without a model --------------------------------------------------------------


def test_quaternion_order_round_trip() -> None:
    xyzw = np.array([0.1, -0.2, 0.3, 0.927])
    assert np.allclose(wxyz_to_xyzw(xyzw_to_wxyz(xyzw)), xyzw)
    assert np.allclose(xyzw_to_wxyz([0, 0, 0, 1]), [1, 0, 0, 0])


def test_match_name_handles_newton_prefixes() -> None:
    names = [
        "_World_envs_env_0_Robot_joints_left_hip_pitch_joint",
        "_World_envs_env_0_Robot_joints_left_hand_left_hand_index_0_joint",
        "right_hip_pitch_joint",
    ]
    assert match_name(names, "left_hip_pitch_joint") == 0
    assert match_name(names, "left_hand_index_0_joint") == 1
    assert match_name(names, "right_hip_pitch_joint") == 2
    with pytest.raises(KeyError):
        match_name(names, "knee_joint")  # no match
    with pytest.raises(KeyError):
        match_name(["a_hip_joint", "b_hip_joint"], "hip_joint")  # two matches


def test_policy_io_fixture(io: PolicyIO) -> None:
    assert len(io.joint_names) == 43
    assert len(io.action_joint_names) == 29
    assert not any("hand" in n for n in io.action_joint_names)
    assert io.obs_dim == 99
    assert io.action_scale == 0.5
    assert (io.physics_dt, io.decimation, io.num_substeps) == (0.005, 4, 2)
    assert np.allclose(io.root_quat_wxyz, [1, 0, 0, 0])


# -- the rebuilt MuJoCo model ------------------------------------------------------------------


@needs_model
def test_build_model_matches_training_settings(model: mujoco.MjModel, io: PolicyIO) -> None:
    assert model.nu == len(io.joint_names)
    assert model.opt.timestep == pytest.approx(0.0025)
    assert model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    assert model.nkey == 0
    for name in ("left_knee_joint", "waist_yaw_joint", "right_wrist_pitch_joint"):
        act = model.actuator(f"pd_{name}")
        assert act.gainprm[0] == pytest.approx(io.stiffness[name])
        assert act.biasprm[1] == pytest.approx(-io.stiffness[name])
        assert act.biasprm[2] == pytest.approx(-io.damping[name])
        joint = model.joint(name)
        assert joint.armature[0] == pytest.approx(io.armature[name])
        assert joint.frictionloss[0] == 0.0
        assert model.jnt_actfrcrange[joint.id][1] == pytest.approx(io.effort_limit[name])
    floor = [g for g in range(model.ngeom) if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE]
    assert model.geom_friction[floor[0]][0] == 1.0


@needs_model
def test_reset_gives_default_pose_and_upright_observation(sim: G1MujocoSim) -> None:
    obs = sim.observation((0.4, -0.1, 0.2))
    assert obs.shape == (99,) and obs.dtype == np.float32
    assert np.allclose(_obs_term(sim, obs, "projected_gravity"), [0, 0, -1], atol=1e-6)
    assert np.allclose(_obs_term(sim, obs, "velocity_commands"), [0.4, -0.1, 0.2])
    assert np.allclose(_obs_term(sim, obs, "joint_pos"), 0, atol=1e-6)
    assert np.allclose(_obs_term(sim, obs, "actions"), 0)
    assert sim.base_height() == pytest.approx(0.8)


@needs_model
def test_base_velocity_is_rotated_into_the_base_frame(sim: G1MujocoSim) -> None:
    yaw90 = [np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)]
    _set_root(sim, yaw90, lin_w=(1.0, 0.0, 0.0))
    obs = sim.observation((0, 0, 0))
    # facing +y and moving along world +x means moving to the robot's right (-y)
    assert np.allclose(_obs_term(sim, obs, "base_lin_vel"), [0, -1, 0], atol=1e-9)

    _set_root(sim, yaw90, ang_local=(0.0, 0.0, 1.0))
    obs = sim.observation((0, 0, 0))
    assert np.allclose(_obs_term(sim, obs, "base_ang_vel"), [0, 0, 1], atol=1e-9)


@needs_model
def test_projected_gravity_under_pitch(sim: G1MujocoSim) -> None:
    theta = 0.3
    _set_root(sim, [np.cos(theta / 2), 0, np.sin(theta / 2), 0])
    obs = sim.observation((0, 0, 0))
    expected = [np.sin(theta), 0, -np.cos(theta)]
    assert np.allclose(_obs_term(sim, obs, "projected_gravity"), expected, atol=1e-9)


@needs_model
def test_base_linear_velocity_is_the_centre_of_mass_velocity(sim: G1MujocoSim) -> None:
    """Isaac Lab's base_lin_vel is the COM velocity; check against a finite difference."""
    rng = np.random.default_rng(0)
    _set_root(sim, [1, 0, 0, 0], lin_w=(0.3, -0.2, 0.1), ang_local=rng.normal(size=3))
    m, d = sim.model, sim.data
    _, lin_w = sim.base_com_velocity_world()

    h = 1e-6
    com0 = d.xipos[sim.base_body].copy()
    qpos = d.qpos.copy()
    mujoco.mj_integratePos(m, d.qpos, d.qvel, h)
    mujoco.mj_kinematics(m, d)
    com1 = d.xipos[sim.base_body].copy()
    d.qpos[:] = qpos
    mujoco.mj_forward(m, d)
    assert np.allclose(lin_w, (com1 - com0) / h, atol=1e-4)
    # and it differs from the frame origin's velocity whenever the base rotates
    assert not np.allclose(lin_w, [0.3, -0.2, 0.1], atol=1e-3)


@needs_model
def test_action_sets_joint_targets_around_the_default_pose(sim: G1MujocoSim) -> None:
    action = np.zeros(len(sim.policy_ids))
    knee = sim.io.action_joint_names.index("left_knee_joint")
    action[knee] = 0.4
    sim.step(action)
    targets = dict(zip(sim.io.joint_names, sim.data.ctrl[sim.actuator_ids], strict=True))
    assert targets["left_knee_joint"] == pytest.approx(sim.io.default_pos["left_knee_joint"] + 0.2)
    assert targets["right_knee_joint"] == pytest.approx(sim.io.default_pos["right_knee_joint"])
    assert targets["left_hand_thumb_1_joint"] == pytest.approx(1.0)  # hands stay at default
    assert np.allclose(sim.last_action, action)
    assert sim.data.time == pytest.approx(0.02)  # one policy step = 8 MuJoCo steps of 2.5 ms


@needs_model
def test_episode_runs_with_a_zero_policy(sim: G1MujocoSim) -> None:
    fell, t_fall, _, _ = evaluate.run_episode(
        sim, lambda obs: np.zeros(29), (0.0, 0.0, 0.0), seed=0, duration_s=1.0
    )
    assert isinstance(fell, bool)
    assert t_fall is None or 0 < t_fall <= 1.0


def test_summarize_and_table() -> None:
    eps = [
        evaluate.Episode(
            "m",
            "stand",
            s,
            fell=(s == 0),
            fall_time_s=1.0 if s == 0 else None,
            duration_s=20.0,
            lin_err=0.1 * s,
            ang_err=0.05,
        )
        for s in range(4)
    ]
    (row,) = evaluate.summarize(eps)
    assert (row["falls"], row["episodes"]) == (1, 4)
    assert row["lin_err_mean"] == pytest.approx(0.2)  # fallen episodes are left out
    table = evaluate.markdown_table([row])
    assert "| m | stand | 1/4 | 25% (" in table

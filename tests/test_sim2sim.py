"""Checks of the MuJoCo side of the sim-to-sim bench against Isaac Lab's conventions.

The fixture is a policy_io.json written by Isaac Lab for the stage 0 task. Tests that need the
G1 model skip when MuJoCo Menagerie has not been fetched (scripts/fetch_menagerie.sh).
"""

import dataclasses
from pathlib import Path

import mujoco
import numpy as np
import pytest

from pi_wbc.sim2sim import compare, evaluate
from pi_wbc.sim2sim.gait import METRICS, GaitRecorder, gait_metrics
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
    # Joints no action drives keep the target Isaac Lab held them at in training. The stage 0
    # export records none, and Isaac Lab leaves such targets at 0 (not the default pose, 1.0).
    assert targets["left_hand_thumb_1_joint"] == pytest.approx(0.0)
    assert np.allclose(sim.last_action, action)
    assert sim.data.time == pytest.approx(0.02)  # one policy step = 8 MuJoCo steps of 2.5 ms


@needs_model
def test_episode_runs_with_a_zero_policy(sim: G1MujocoSim) -> None:
    fell, t_fall, _, _, gait, drift = evaluate.run_episode(
        sim, lambda obs: np.zeros(29), (0.0, 0.0, 0.0), seed=0, duration_s=1.0
    )
    assert isinstance(fell, bool)
    assert t_fall is None or 0 < t_fall <= 1.0
    assert gait is None or set(gait) == set(METRICS)
    assert drift is None  # 1 s ends before the 2 s settling time, or it fell


@needs_model
def test_inactive_gap_contacts_do_not_count_as_touching(io: PolicyIO) -> None:
    model = build_model(SCENE, io)
    model.geom_gap[:] = 0.01  # as in Newton's exported model
    sim = G1MujocoSim(model, io)
    sim.reset()
    a = sim.root_qpos_adr
    sim.data.qpos[a + 2] = 0.8  # lift the robot so the feet hover about 1 cm up
    mujoco.mj_forward(sim.model, sim.data)
    recorder = GaitRecorder(sim)
    listed = [c for c in sim.data.contact[: sim.data.ncon] if c.exclude == 1]
    assert listed  # MuJoCo lists the near-miss pairs ...
    assert recorder._foot_down() == [False, False]  # ... but they are not contact


@needs_model
def test_gait_recorder_sees_both_feet_down_when_standing(sim: G1MujocoSim) -> None:
    recorder = GaitRecorder(sim)
    for k in range(20):  # the robot spawns a little above the floor; sample after it lands
        sim.step(np.zeros(29))
        if k >= 15:
            recorder.sample()
    assert all(all(d) for d in recorder.down)
    g = recorder.metrics()
    assert g["step_rate_hz"] == 0 and g["double_support"] == 1.0


def _standing_rigid(sim: G1MujocoSim) -> GaitRecorder:
    """Land the robot on both feet, then stop every joint so it moves as one rigid body."""
    for _ in range(20):
        sim.step(np.zeros(29))
    sim.data.qvel[:] = 0
    return GaitRecorder(sim)


def _move_rigid(sim: G1MujocoSim, ang_w: np.ndarray, point_w: np.ndarray) -> None:
    """Give the whole robot angular velocity `ang_w` about a fixed point (world axes)."""
    a, v = sim.root_qpos_adr, sim.root_qvel_adr
    root_pos = sim.data.qpos[a : a + 3].copy()
    rot = sim.data.xmat[sim.base_body].reshape(3, 3)
    sim.data.qvel[v : v + 3] = np.cross(ang_w, root_pos - point_w)  # the point stays still
    sim.data.qvel[v + 3 : v + 6] = rot.T @ ang_w  # free joint: angular velocity, body axes
    mujoco.mj_forward(sim.model, sim.data)


@needs_model
def test_sole_slip_is_the_sliding_speed(sim: G1MujocoSim) -> None:
    recorder = _standing_rigid(sim)
    v = sim.root_qvel_adr
    sim.data.qvel[v] = 0.3  # the whole robot slides forward at 0.3 m/s
    mujoco.mj_forward(sim.model, sim.data)
    recorder.sample()
    assert recorder.down[-1] == [True, True]
    assert recorder.sole_slip[-1] == pytest.approx([0.3, 0.3], abs=1e-6)
    assert recorder.foot_speed[-1] == pytest.approx([0.3, 0.3], abs=1e-6)


@needs_model
def test_sole_slip_ignores_a_foot_pivoting_on_its_toe(sim: G1MujocoSim) -> None:
    recorder = _standing_rigid(sim)
    foot = recorder.feet[0]
    rot = sim.data.xmat[foot].reshape(3, 3)
    # Menagerie's toe spheres sit at x = 0.12 m, z = -0.03 m in the foot frame, radius 5 mm:
    # pitch the whole robot about the line under them, as at toe-off.
    toe_w = sim.data.xpos[foot] + rot @ np.array([0.12, 0.0, -0.035])
    _move_rigid(sim, 2.0 * rot[:, 1], toe_w)
    recorder.sample()
    assert recorder.down[-1][0]
    assert recorder.sole_slip[-1][0] < 0.01
    # The ankle, 3.5 cm above the sole, still moves sideways by about 2 rad/s x 3.5 cm.
    assert recorder.foot_speed[-1][0] == pytest.approx(0.07, abs=0.015)


def test_gait_metrics_sole_slip_skips_unloaded_samples() -> None:
    down, foot_z, foot_speed, base_speed, knee, shoulder, dt = _walk()
    slip = np.where(down, 0.02, np.nan)
    slip[50, :] = np.nan  # a down sample whose contact carried no load: left out
    g = gait_metrics(down, foot_z, foot_speed, base_speed, knee, shoulder, dt, sole_slip=slip)
    assert g["sole_slip_mps"] == pytest.approx(0.02)
    assert gait_metrics(*_walk())["sole_slip_mps"] is None  # not recorded


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
    assert table.splitlines()[-1].endswith("| — |")  # no heading drift recorded
    assert row["gait"]["step_rate_hz"] is None  # no gait recorded in these episodes
    assert "| m | stand | — |" in evaluate.gait_table([row])


def _walk(n: int = 200, period: int = 25, swing: int = 10, dt: float = 0.02):
    """Synthetic gait: each foot swings for `swing` of every `period` samples, out of phase."""
    t = np.arange(n)
    phase = np.stack([t % period, (t + period // 2) % period], axis=1)
    down = phase >= swing
    foot_z = np.where(down, 0.03, 0.03 + 0.08 * np.sin(np.pi * phase / swing))
    foot_speed = np.where(down, 0.05, 1.5)
    base_speed = np.full(n, 0.6)
    knee = np.stack([0.3 + 0.5 * np.sin(2 * np.pi * t / period)] * 2, axis=1)
    shoulder = np.zeros((n, 2))
    return down, foot_z, foot_speed, base_speed, knee, shoulder, dt


def test_gait_metrics_on_a_synthetic_walk() -> None:
    g = gait_metrics(*_walk())
    # 200 samples of 20 ms = 4 s. The left foot's first swing starts at sample 0, so it is cut
    # and left out: 7 left + 8 right whole swings.
    assert g["step_rate_hz"] == pytest.approx(15 / 4.0)
    assert g["step_length_m"] == pytest.approx(0.6 / 3.75)
    assert g["swing_s"] == pytest.approx(0.2)
    assert g["foot_lift_m"] == pytest.approx(0.08 * np.sin(np.pi * 5 / 10))
    assert g["stance_foot_speed_mps"] == pytest.approx(0.05)
    assert g["double_support"] + g["flight"] < 1
    assert 50 < g["knee_range_deg"] < 2 * np.degrees(0.5)
    assert g["arm_swing_deg"] == 0


def test_gait_metrics_ignore_short_lifts_and_cut_swings() -> None:
    down = np.ones((100, 2), dtype=bool)
    down[10:12, 0] = False  # 40 ms: contact flicker, not a step
    down[30:40, 0] = False  # a real step
    down[:20, 1] = False  # under way when recording starts
    down[90:, 1] = False  # not finished when recording stops
    zeros = np.zeros((100, 2))
    g = gait_metrics(down, zeros, zeros, np.zeros(100), zeros, zeros, dt=0.02)
    assert g["step_rate_hz"] == pytest.approx(1 / 2.0)
    assert g["swing_s"] == pytest.approx(0.2)


def _result(label: str, falls: int, lin: float, steps: float) -> dict:
    gait = dict.fromkeys(METRICS, 0.0) | {"step_rate_hz": steps}
    row = {"model": "menagerie", "scenario": "walk_1.0", "falls": falls, "episodes": 10}
    policy = f"logs/rsl_rl/x/slurm-{label}/exported/policy.onnx"
    row |= {"lin_err_mean": lin, "gait": gait}
    return {"label": label, "policy": policy, "summary": [row]}


def test_compare_groups_seeds_of_a_variant() -> None:
    results = [
        _result("base-s1", 0, 0.1, 11.0),
        _result("airtime-s2", 1, 0.2, 2.0),
        _result("base-s2", 2, 0.3, 12.0),
    ]
    base, air = compare.compare(results, "menagerie", "walk_1.0")
    assert (base["variant"], base["seeds"]) == ("base", [1, 2])
    assert (base["falls"], base["episodes"]) == (2, 20)
    assert base["lin_err_mean"] == pytest.approx(0.2)
    assert base["gait"]["step_rate_hz"] == pytest.approx(11.5)
    assert air["variant"] == "airtime"
    assert "| airtime | 1 | 2 | 1/10 (" in compare.markdown_table([air])
    assert "| 11.5 (11.0-12.0) |" in compare.markdown_table([base])
    per_run = compare.compare(results, "menagerie", "walk_1.0", per_run=True)
    assert len(per_run) == 3
    assert per_run[0]["variant"] == "base-s1 (slurm-base-s1)"


# -- stage 1 gait task: 15 driven joints, 29 observed, gait clock ------------------------------

GAIT_IO_PATH = ROOT / "tests" / "data" / "policy_io_g1_gait.json"


@needs_model
def test_gait_task_observation_layout_and_clock() -> None:
    io = PolicyIO.load(GAIT_IO_PATH)
    assert len(io.action_joint_names) == 15 and len(io.obs_joint_names["joint_pos"]) == 29
    sim = G1MujocoSim(build_model(SCENE, io), io)
    sim.reset()
    obs = sim.observation((0.5, 0.0, 0.0))
    assert obs.shape == (io.obs_dim,) == (87,)
    assert _obs_term(sim, obs, "joint_pos").shape == (29,)
    assert _obs_term(sim, obs, "actions").shape == (15,)
    assert _obs_term(sim, obs, "gait_clock") == pytest.approx([0.0, 1.0])

    # Only the 15 driven joints move their targets; arms keep the default pose.
    action = np.ones(15)
    for _ in range(10):  # 10 steps of 20 ms = a quarter of the 0.8 s period
        sim.step(action)
    obs = sim.observation((0.5, 0.0, 0.0))
    assert _obs_term(sim, obs, "gait_clock") == pytest.approx([1.0, 0.0], abs=1e-9)
    targets = dict(zip(io.joint_names, sim.data.ctrl[sim.actuator_ids], strict=True))
    assert targets["left_knee_joint"] == pytest.approx(io.default_pos["left_knee_joint"] + 0.25)
    assert targets["left_elbow_joint"] == pytest.approx(io.default_pos["left_elbow_joint"])


@needs_model
def test_gait_clock_reads_zero_when_told_to_stand() -> None:
    io = PolicyIO.load(GAIT_IO_PATH)
    io = dataclasses.replace(io, gait=io.gait | {"clock_off_when_standing": True})
    sim = G1MujocoSim(build_model(SCENE, io), io)
    sim.reset()
    assert _obs_term(sim, sim.observation((0.0, 0.0, 0.0)), "gait_clock") == pytest.approx([0, 0])
    assert _obs_term(sim, sim.observation((0.0, 0.0, 0.8)), "gait_clock") == pytest.approx([0, 1])
    assert _obs_term(sim, sim.observation((0.5, 0.0, 0.0)), "gait_clock") == pytest.approx([0, 1])

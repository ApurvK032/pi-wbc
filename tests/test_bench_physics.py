"""The bench scenes' physics (pi_wbc.stage3.bench_physics, D-051): the preset, not a default."""

import argparse
import sys
from pathlib import Path

import pytest

pytest.importorskip("isaaclab_tasks")

import isaaclab.sim as sim_utils
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.utils import configclass

from pi_wbc.stage3 import bench_physics

ROOT = Path(__file__).resolve().parents[1]
BENCH_SCRIPTS = ["scripts/grasp_test.py", "scripts/nvidia_wbc_test.py"]


def _resolved(physics_arg: str | None, substeps: int | None = None):
    """The physics config Isaac Lab's launcher hands a bench scene (no simulation started)."""

    @configclass
    class Scene:
        sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
            dt=0.005, render_interval=4, physics=bench_physics.newton(substeps)
        )

    parser = argparse.ArgumentParser()
    add_launcher_args(parser)
    args = vars(parser.parse_args([]))
    if physics_arg is not None:
        args["physics"] = physics_arg
    launcher = sys.modules[launch_simulation.__wrapped__.__module__]
    scan, _ = launcher.validate_launch(Scene(), args)
    return scan.resolved_physics_cfg


def _settings(cfg) -> dict:
    s, shape = cfg.solver_cfg, cfg.default_shape_cfg
    return {
        "num_substeps": cfg.num_substeps,
        "integrator": s.integrator,
        "use_mujoco_contacts": s.use_mujoco_contacts,
        "ke": shape.ke,
        "kd": shape.kd,
    }


def test_bench_scene_keeps_the_preset() -> None:
    got = _settings(_resolved(None, substeps=4))
    assert got == dict(bench_physics.PRESET, num_substeps=4)
    assert _resolved(None).num_substeps == 2  # the preset's own count


def test_the_physics_argument_would_replace_it() -> None:
    """Why the launcher argument stays unset: with it, the scene gets a default config."""
    got = _settings(_resolved("newton_mjwarp", substeps=4))
    assert got["num_substeps"] == 1 and got["ke"] != bench_physics.PRESET["ke"]


def test_bench_scripts_leave_the_argument_unset() -> None:
    for script in BENCH_SCRIPTS:
        text = (ROOT / script).read_text()
        assert "args_cli.physics =" not in text, script
        assert "bench_physics.newton(" in text and "bench_physics.check(" in text, script


def test_check_refuses_the_default() -> None:
    rec = dict(bench_physics.PRESET, num_substeps=2, solver_substeps=2)
    bench_physics.check(rec)
    with pytest.raises(RuntimeError):
        bench_physics.check(dict(rec, ke=2500.0))
    with pytest.raises(RuntimeError):
        bench_physics.check(dict(rec, solver_substeps=1))
    bench_physics.check(dict(rec, use_mujoco_contacts=True), mujoco_contacts=True)


def test_matches_the_task_configs() -> None:
    """Training's and the task's physics are the same preset, apart from contact budgets and
    the task's 4 substeps."""
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    import pi_wbc.tasks  # noqa: F401  (registers the tasks)

    for task, substeps in (("PiWbc-G1-Student", 2), ("PiWbc-G1-PickPlace", 4)):
        cfg = parse_env_cfg(task, device="cpu", num_envs=1).sim.physics
        assert _settings(cfg) == _settings(bench_physics.newton(substeps)), task

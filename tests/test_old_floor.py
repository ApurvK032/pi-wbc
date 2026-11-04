"""scripts/sim2sim_old_floor.py rebuilds the bench's floor as it was before D-039."""

import importlib.util
from pathlib import Path

import mujoco
import pytest

from pi_wbc.sim2sim.mujoco_g1 import G1MujocoSim, PolicyIO, build_model

ROOT = Path(__file__).resolve().parents[1]
IO_PATH = ROOT / "tests" / "data" / "policy_io_g1_stage0.json"
SCENE = ROOT / "third_party" / "mujoco_menagerie" / "unitree_g1" / "scene_with_hands.xml"
pytestmark = pytest.mark.skipif(not SCENE.exists(), reason="run scripts/fetch_menagerie.sh")


def _old_build():
    spec = importlib.util.spec_from_file_location(
        "sim2sim_old_floor", ROOT / "scripts" / "sim2sim_old_floor.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # defines old_build; patches nothing outside __main__
    return module.old_build


def _foot_floor_friction(model: mujoco.MjModel, io: PolicyIO) -> set[float]:
    sim = G1MujocoSim(model, io)
    sim.reset()
    for _ in range(50):  # let the feet settle on the floor
        mujoco.mj_step(sim.model, sim.data)
    plane = mujoco.mjtGeom.mjGEOM_PLANE
    return {
        round(float(c.friction[0]), 3)
        for c in sim.data.contact[: sim.data.ncon]
        if plane in (model.geom_type[c.geom1], model.geom_type[c.geom2]) and c.exclude == 0
    }


def test_old_floor_puts_menageries_feet_on_their_own_friction() -> None:
    """Before D-039 Menagerie's foot spheres (priority 1, friction 0.6) decided the friction;
    the corrected bench gives the floor's 1.0."""
    io = PolicyIO.load(IO_PATH)
    assert _foot_floor_friction(_old_build()(SCENE, io), io) == {0.6}
    assert _foot_floor_friction(build_model(SCENE, io), io) == {1.0}


def test_old_floor_leaves_the_bench_unpatched() -> None:
    """Importing the script must not change the bench for anything else in the process."""
    import pi_wbc.sim2sim.evaluate as ev
    from pi_wbc.sim2sim import mujoco_g1

    _old_build()
    assert ev.build_model is build_model
    assert mujoco_g1.build_model is build_model

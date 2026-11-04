"""The MuJoCo bench with every geom's contact priority as its model file sets it, as before D-039.

Takes the same arguments as `pi-wbc-sim2sim`. On Menagerie's model the foot spheres (priority 1,
friction 0.6) then decide the foot-floor friction, as in every stage 0 and stage 1 Menagerie run
before the fix; used only to compare against the corrected bench on the same seeds
(results/stage1/foot_friction/). Since 5ea97c6 the bench gives every geom priority 0, so this
copies the file's own priorities back rather than only lowering the floor's.

    uv run python scripts/sim2sim_old_floor.py --policy ... --policy-io ... --model menagerie=...
"""

import sys

import mujoco

import pi_wbc.sim2sim.evaluate as ev
from pi_wbc.sim2sim import mujoco_g1

_build = mujoco_g1.build_model


def old_build(xml_path, *args, **kwargs) -> mujoco.MjModel:
    """The bench's `build_model`, then every geom's contact priority back to the file's own."""
    model = _build(xml_path, *args, **kwargs)
    raw = mujoco.MjModel.from_xml_path(str(xml_path))
    if raw.ngeom != model.ngeom or not (raw.geom_type == model.geom_type).all():
        raise RuntimeError("the bench changed the model's geoms; their priorities can't be mapped")
    model.geom_priority[:] = raw.geom_priority
    return model


if __name__ == "__main__":
    ev.build_model = old_build
    mujoco_g1.build_model = old_build
    sys.exit(ev.main())

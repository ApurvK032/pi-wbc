"""The MuJoCo bench with the floor's contact priority as it was before D-039 (0).

Takes the same arguments as `pi-wbc-sim2sim`. On Menagerie's model the feet then stand on their
own friction (0.6), as in every stage 0 and stage 1 Menagerie run; used only to compare against
the corrected bench on the same seeds (results/stage1/foot_friction/).

    uv run python scripts/sim2sim_old_floor.py --policy ... --policy-io ... --model menagerie=...
"""

import sys

import mujoco

import pi_wbc.sim2sim.evaluate as ev
from pi_wbc.sim2sim import mujoco_g1

_build = mujoco_g1.build_model


def old_build(*a, **k):
    m = _build(*a, **k)
    for g in range(m.ngeom):
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
            m.geom_priority[g] = 0
    return m


ev.build_model = old_build
mujoco_g1.build_model = old_build
sys.exit(ev.main())

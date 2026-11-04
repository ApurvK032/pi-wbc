"""Physics for the hand-built bench scenes (the grasp test and NVIDIA's controller on our G1).

These scenes are built in the script, not through a task config, so they name their physics
themselves: the Newton settings training and the task use (Isaac Lab's locomotion preset,
`RoughPhysicsCfg().newton_mjwarp`), with the substep count as an option.

The launcher's `physics` argument must stay unset. When it is set (as these scripts did before
D-051, to "newton_mjwarp"), Isaac Lab replaces every physics config in the scene with a fresh
default one for that backend: 1 substep, Euler integration, MuJoCo's contacts and a contact
stiffness of 2,500 against the preset's 160,000. `record()` stores what the solver actually
runs with, and `check()` refuses a run whose settings are not the preset's.
"""

from typing import Any

from isaaclab_tasks.core.velocity.velocity_env_cfg import RoughPhysicsCfg

# the preset's values that the launcher's default config does not share
PRESET = {"integrator": "implicitfast", "use_mujoco_contacts": False, "ke": 160000.0, "kd": 1100.0}


def newton(substeps: int | None = None) -> Any:
    """The preset's Newton config (a new instance), at `substeps` if given (preset: 2)."""
    cfg = RoughPhysicsCfg().newton_mjwarp
    if substeps is not None:
        cfg.num_substeps = substeps
    return cfg


def record(physics_cfg: Any) -> dict[str, Any]:
    """The physics a run uses, for its result file. Call after the simulation is created; the
    substep count is read from Newton's manager, the solver's own value, not only the config's."""
    from isaaclab_newton.physics.newton_manager import NewtonManager

    s, shape = physics_cfg.solver_cfg, physics_cfg.default_shape_cfg
    return {
        "config": type(physics_cfg).__name__,
        "solver": type(s).__name__,
        "num_substeps": physics_cfg.num_substeps,
        "solver_substeps": getattr(NewtonManager, "_num_substeps", None),
        "solver_dt": getattr(NewtonManager, "_solver_dt", None),
        "integrator": s.integrator,
        "use_mujoco_contacts": s.use_mujoco_contacts,
        "cone": s.cone,
        "impratio": s.impratio,
        "njmax": s.njmax,
        "nconmax": s.nconmax,
        "ke": shape.ke,
        "kd": shape.kd,
        "margin": shape.margin,
    }


def check(rec: dict[str, Any], mujoco_contacts: bool = False) -> None:
    """Raise unless `rec` (from `record`) has the preset's settings and the solver runs the
    configured substeps. `mujoco_contacts` allows the grasp test's MuJoCo-contacts option."""
    want = dict(PRESET, use_mujoco_contacts=mujoco_contacts)
    bad = {k: (rec[k], v) for k, v in want.items() if rec[k] != v}
    if rec["solver_substeps"] not in (None, rec["num_substeps"]):
        bad["solver_substeps"] = (rec["solver_substeps"], rec["num_substeps"])
    if bad:
        raise RuntimeError(f"bench physics differs from the preset (got, want): {bad}")

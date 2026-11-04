"""Foot contact geometry of the G1 in Newton.

Each foot touches the floor through four small spheres on its ankle-roll link: two at the toe
(x = 0.12 m) and two at the heel. Isaac Lab's G1 asset puts the heel pair at x = -0.02 m; Unitree's
URDF (as used by NVIDIA's controller and by MuJoCo Menagerie's model) puts it at -0.05 m, a 3 cm
longer support. `set_heel_x` moves the heel spheres of every robot in the running Newton model.
"""

from __future__ import annotations

ASSET_HEEL_X = -0.02  # Isaac Lab's G1 asset
URDF_HEEL_X = -0.05  # Unitree's URDF, NVIDIA's MJCF, MuJoCo Menagerie


def set_heel_x(heel_x: float, tol: float = 0.005) -> int:
    """Move the heel contact spheres of all ankle-roll links to `heel_x` (m, link frame).

    Call after the simulation is reset. Returns how many spheres were moved (4 per robot).
    """
    import numpy as np
    import warp as wp
    from isaaclab_newton.physics.newton_manager import NewtonManager
    from newton import GeoType, ModelFlags

    model = NewtonManager.get_model()
    keys = list(model.body_key) if hasattr(model, "body_key") else list(model.body_label)
    body = model.shape_body.numpy()
    types = model.shape_type.numpy()
    xf = model.shape_transform.numpy()  # (shapes, 7): position, then quaternion
    moved = 0
    for i in range(model.shape_count):
        b = body[i]
        if b < 0 or not keys[b].endswith("ankle_roll_link") or types[i] != int(GeoType.SPHERE):
            continue
        if xf[i, 0] < 0.0 and abs(xf[i, 0] - heel_x) > 1e-6:
            if min(abs(xf[i, 0] - ASSET_HEEL_X), abs(xf[i, 0] - URDF_HEEL_X)) > tol:
                raise ValueError(f"unexpected heel sphere at x={xf[i, 0]:.3f} on {keys[b]}")
            xf[i, 0] = heel_x
            moved += 1
    model.shape_transform.assign(wp.array(xf.astype(np.float32), dtype=wp.transform))
    NewtonManager.add_model_change(ModelFlags.SHAPE_PROPERTIES)
    return moved

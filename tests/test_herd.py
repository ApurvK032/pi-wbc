"""One camera that sees every world (pi_wbc.stage3.herd): the BVH's top node."""

import numpy as np
import pytest

wp = pytest.importorskip("warp")

from pi_wbc.stage3 import herd  # noqa: E402


@wp.kernel
def _hits(bvh: wp.uint64, root: wp.int32, out: wp.array(dtype=wp.int32)):
    query = wp.bvh_query_ray(bvh, wp.vec3(0.0, 0.0, 0.0), wp.vec3(1.0, 0.0, 0.0), root)
    i = wp.int32(0)
    n = wp.int32(0)
    while wp.bvh_query_next(query, i):
        out[n] = i
        n += 1


def test_top_node_reaches_every_group() -> None:
    """Boxes in a row along x, in four groups (worlds): a ray from a group's root meets only that
    group's boxes, from the top node all of them."""
    n = 16
    x = 2.0 * np.arange(n) + 1.0
    lo = np.stack([x, np.full(n, -0.5), np.full(n, -0.5)], 1).astype(np.float32)
    groups = (np.arange(n) % 4).astype(np.int32)
    with wp.ScopedDevice("cpu"):
        bvh = wp.Bvh(wp.array(lo, dtype=wp.vec3), wp.array(lo + 1.0, dtype=wp.vec3),
                     groups=wp.array(groups, dtype=wp.int32))  # fmt: skip
        top = herd.top_node(bvh, n)
        out = wp.full(n, -1, dtype=wp.int32)
        wp.launch(_hits, 1, inputs=[bvh.id, top, out])
        assert sorted(int(i) for i in out.numpy() if i >= 0) == list(range(n))

"""One camera that sees every robot: front-page pictures of many robots at once.

Newton's ray tracer shows each world's camera only its own world's shapes and the global ones:
the shape BVH has one group per world, and a camera's rays start at its world's group root.
`see_all_worlds` points world 0's group root at the BVH's top node, so world 0's camera sees
every world. Rendering only: physics and collisions don't use this BVH, the other worlds'
cameras are unchanged, and the BVH is refit each frame, never rebuilt, so the change lasts. The
root is written into the existing array, so a renderer that already captured its arrays in a
CUDA graph sees it too.
"""

import numpy as np
import warp as wp


@wp.kernel
def _leaves_under(
    bvh: wp.uint64, lower: wp.vec3, upper: wp.vec3, counts: wp.array(dtype=wp.int32)
):  # fmt: skip
    node = wp.tid()
    query = wp.bvh_query_aabb(bvh, lower, upper, node)
    i = wp.int32(0)
    n = wp.int32(0)
    while wp.bvh_query_next(query, i):
        n += 1
    counts[node] = n


def top_node(bvh: wp.Bvh, num_leaves: int) -> int:
    """The BVH node whose subtree holds every leaf (a BVH of n leaves has 2n - 1 nodes)."""
    nodes = 2 * num_leaves - 1
    counts = wp.zeros(nodes, dtype=wp.int32, device=bvh.device)
    lower, upper = wp.vec3(-1.0e6, -1.0e6, -1.0e6), wp.vec3(1.0e6, 1.0e6, 1.0e6)
    wp.launch(_leaves_under, nodes, inputs=[bvh.id, lower, upper, counts], device=bvh.device)
    tops = np.flatnonzero(counts.numpy() == num_leaves)
    if len(tops) != 1:
        raise RuntimeError(f"expected one node holding all {num_leaves} leaves, found {len(tops)}")
    return int(tops[0])


def see_all_worlds(model) -> int:
    """Let world 0's camera see every world of a Newton `model`; returns the top node."""
    root = top_node(model.bvh_shapes, model.bvh_shape_count_enabled)
    roots = model.bvh_shapes_group_roots.numpy()
    roots[0] = root
    model.bvh_shapes_group_roots.assign(roots)
    return root

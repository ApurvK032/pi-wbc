"""Stage 3 layouts: deterministic, inside the table, objects apart, instructions consistent."""

import math

import pytest

from pi_wbc.stage3 import layout as L


def test_same_seed_same_layout():
    assert L.sample(123) == L.sample(123)
    assert L.sample(123) != L.sample(124)


@pytest.mark.parametrize("seed", [0, 1, 7, 49_999, 50_000, 60_999])
def test_layout_is_valid(seed):
    lay = L.sample(seed)
    assert len(set(lay.objects)) == 3 and set(lay.objects) <= set(L.OBJECT_NAMES)
    for p in lay.object_xy:
        assert L.OBJECT_X[0] <= p[0] <= L.OBJECT_X[1]
        assert L.OBJECT_Y[0] <= p[1] <= L.OBJECT_Y[1]
    for i in range(3):
        for j in range(i + 1, 3):
            assert math.dist(lay.object_xy[i], lay.object_xy[j]) >= L.OBJECT_SPACING
    assert sorted(lay.bins) == sorted(L.BIN_NAMES)
    assert lay.target_object in lay.objects and lay.target_bin in lay.bins
    assert L.describe(lay.target_object) in lay.instruction
    assert L.describe(lay.target_bin) in lay.instruction
    assert -L.START_DISTANCE[1] <= lay.robot_xy[0] <= -L.START_DISTANCE[0]
    assert abs(lay.robot_yaw) <= math.pi / 2


def test_bins_do_not_overlap_objects():
    for seed in range(200):
        lay = L.sample(seed)
        for name in lay.bins:
            bx, by = lay.bin_xy(name)
            for (x, y), obj in zip(lay.object_xy, lay.objects, strict=True):
                r = L.OBJECTS[obj]["radius"]
                assert abs(y - by) > L.BIN_SIZE[1] / 2 + r or abs(x - bx) > L.BIN_SIZE[0] / 2 + r


def test_splits_are_disjoint_and_named():
    assert L.split_of(0) == "train" and L.split_of(50_500) == "dev" and L.split_of(60_000) == "test"
    with pytest.raises(ValueError):
        L.split_of(55_000)

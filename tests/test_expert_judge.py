"""The expert's placement judge (`placement_label`), shared with the replay check."""

import numpy as np

from pi_wbc.stage3 import layout as L
from pi_wbc.stage3.expert import placement_label


def _resting_z(lay: L.Layout) -> float:  # on the bin's 1 cm floor
    return L.TABLE_TOP + 0.01 + L.OBJECTS[lay.target_object]["height"] / 2


def test_inside_the_target_bin_is_success() -> None:
    lay = L.sample(0)
    bx, by = lay.bin_xy(lay.target_bin)
    assert placement_label(lay, np.array([bx, by, _resting_z(lay)])) == "success"


def test_the_other_bin_and_the_table_are_not() -> None:
    lay = L.sample(0)
    other = next(b for b in lay.bins if b != lay.target_bin)
    ox, oy = lay.bin_xy(other)
    assert placement_label(lay, np.array([ox, oy, _resting_z(lay)])) == "wrong_bin"
    x, y, z = lay.object_pos(lay.target_object)  # still where it started
    assert placement_label(lay, np.array([x, y, z])) == "dropped"


def test_held_above_the_bin_is_not_placed() -> None:
    lay = L.sample(0)
    bx, by = lay.bin_xy(lay.target_bin)
    assert placement_label(lay, np.array([bx, by, _resting_z(lay) + 0.10])) == "dropped"

"""The shared outcome judge (pi_wbc.stage3.judge, D-052), on synthetic episodes."""

import json

import numpy as np
import pytest

from pi_wbc.stage3 import layout as L
from pi_wbc.stage3.judge import Judge, JudgeParams, up_z_of

DT = 0.02
LAY = L.sample(10000)  # white bottle into the purple bin; yellow and green cans beside it
T = L.OBJECT_NAMES.index(LAY.target_object)
OTHER = L.OBJECT_NAMES.index(next(n for n in LAY.objects if n != LAY.target_object))
H = L.OBJECTS[LAY.target_object]["height"]
IN_HAND = np.array([0.0, -0.09, 0.0])  # palm frame origin beside a held object
AWAY = np.array([0.0, -0.30, 0.20])  # a hand that let go and withdrew
OTHER_HAND = np.array([0.0, 5.0, 0.0])
LIFT = np.array([0.0, 0.0, 0.10])


def _start() -> np.ndarray:
    pos = np.full((5, 3), 5.0)  # objects not in the scene are parked far away
    for name in LAY.objects:
        pos[L.OBJECT_NAMES.index(name)] = LAY.object_pos(name)
    return pos


def _in_bin(name: str) -> np.ndarray:
    bx, by = LAY.bin_xy(name)
    return np.array([bx, by, L.TABLE_TOP + 0.01 + H / 2])


class Run:
    """Drives one episode through the judge, one control step at a time."""

    def __init__(self, params: JudgeParams | None = None):
        self.objects = _start()
        self.judge = Judge([LAY], self.objects[None], params)
        self.t = 0.0
        self.palm = np.array([0.0, 0.0, 1.0])  # far from everything
        self.z, self.up = 0.78, 1.0

    def hold(self, seconds: float) -> None:
        for _ in range(round(seconds / DT)):
            self.t += DT
            palms = np.stack([self.palm, self.palm + OTHER_HAND])[None]
            self.judge.step(self.t, np.array([self.z]), np.array([self.up]),
                            self.objects[None].copy(), palms)  # fmt: skip
            if self.judge.done[0]:
                return

    def carry_to(self, where: np.ndarray, release: bool = True) -> None:
        start = self.objects[T].copy()
        self.palm = start + IN_HAND  # in the hand
        self.hold(0.5)
        self.objects[T] = start + LIFT  # lifted
        self.palm = self.objects[T] + IN_HAND
        self.hold(0.5)
        self.objects[T] = where
        self.palm = where + IN_HAND
        self.hold(0.5)
        if release:
            self.palm = where + AWAY  # let go and withdraw

    @property
    def result(self) -> dict:
        return self.judge.result(0)


def test_success_is_judged_after_the_arm_withdraws() -> None:
    r = Run()
    r.carry_to(_in_bin(LAY.target_bin))
    placed = r.t
    r.hold(10.0)
    assert r.result["label"] == "success" and r.result["ended_by"] == "settled"
    # let go at the next step; judged settle_s later (to within a step of rounding)
    assert r.result["time_s"] == pytest.approx(placed + DT + JudgeParams().settle_s, abs=1.5 * DT)


def test_a_push_while_withdrawing_fails_the_episode() -> None:
    r = Run()
    r.carry_to(_in_bin(LAY.target_bin))
    r.hold(2.0)
    r.objects[OTHER, 0] += 0.03
    r.hold(5.0)
    assert r.result["label"] == "knocked_other"


def test_knocked_out_of_the_bin_is_dropped() -> None:
    r = Run()
    r.carry_to(_in_bin(LAY.target_bin))
    r.hold(1.0)
    r.objects[T] = [0.35, 0.0, L.TABLE_TOP + H / 2]  # back on the table, out of reach
    r.hold(5.0)
    assert r.result["label"] == "dropped"


def test_held_low_over_the_bin_is_not_placed() -> None:
    r = Run()
    r.carry_to(_in_bin(LAY.target_bin), release=False)
    r.hold(10.0)
    assert not r.judge.done[0]  # still in the hand: no window started
    r.palm = r.objects[T] + AWAY
    released = r.t
    r.hold(10.0)
    assert r.result["label"] == "success"
    assert r.result["placed_s"] == pytest.approx(released + DT, abs=DT)


def test_wrong_bin() -> None:
    other_bin = next(b for b in LAY.bins if b != LAY.target_bin)
    r = Run()
    r.carry_to(_in_bin(other_bin))
    r.hold(10.0)
    assert r.result["label"] == "wrong_bin"


@pytest.mark.parametrize(("z", "up"), [(0.40, 1.0), (0.78, 0.4)])
def test_falls(z: float, up: float) -> None:
    r = Run()
    r.hold(1.0)
    r.z, r.up = z, up
    r.hold(1.0)
    assert r.result["label"] == "fell" and r.result["ended_by"] == "fall"


def test_never_lifted() -> None:
    pushed = Run()
    pushed.objects[T, 0] += 0.04
    pushed.hold(1.0)
    pushed.judge.finish(pushed.t)
    assert pushed.result["label"] == "knocked"
    tried = Run()
    tried.palm = tried.objects[T] + IN_HAND
    tried.hold(1.0)
    tried.palm = np.array([0.0, 0.0, 1.0])
    tried.hold(1.0)
    tried.judge.finish(tried.t)
    assert tried.result["label"] == "missed_grasp"
    pushed_in_hand = Run()  # reached, then pushed without lifting: a missed grasp
    pushed_in_hand.palm = pushed_in_hand.objects[T] + IN_HAND
    pushed_in_hand.hold(0.5)
    pushed_in_hand.objects[T, 0] += 0.10
    pushed_in_hand.palm = np.array([0.0, 0.0, 1.0])
    pushed_in_hand.hold(0.5)
    pushed_in_hand.judge.finish(pushed_in_hand.t)
    assert pushed_in_hand.result["label"] == "missed_grasp"
    idle = Run()
    idle.hold(1.0)
    idle.judge.finish(idle.t)
    assert idle.result["label"] == "timed_out"


def test_still_holding_at_the_time_limit() -> None:
    r = Run(JudgeParams(max_s=3.0))
    r.carry_to(r.objects[T] + 2 * LIFT, release=False)
    r.hold(5.0)
    assert r.result["label"] == "timed_out" and r.result["ended_by"] == "time limit"
    json.dumps(r.result)  # results go into JSON files


def test_lifted_and_let_go_outside_the_bins_is_dropped() -> None:
    r = Run()
    r.carry_to(np.array([0.35, 0.0, L.TABLE_TOP + H / 2]))
    r.hold(1.0)
    r.judge.finish(r.t)
    assert r.result["label"] == "dropped"


def test_picked_up_again_restarts_the_window() -> None:
    r = Run()
    r.carry_to(_in_bin(LAY.target_bin))
    r.hold(2.0)
    r.palm = r.objects[T] + IN_HAND  # takes it again
    r.hold(1.0)
    r.palm = r.objects[T] + AWAY
    again = r.t
    r.hold(10.0)
    assert r.result["label"] == "success"
    assert r.result["time_s"] == pytest.approx(again + DT + JudgeParams().settle_s, abs=1.5 * DT)


def test_without_palms_the_hand_counts_as_open() -> None:
    j = Judge([LAY], _start()[None])
    objs = _start()
    objs[T] = _in_bin(LAY.target_bin)
    for k in range(1, 300):
        j.step(k * DT, np.array([0.78]), np.array([1.0]), objs[None])
    assert j.result(0)["label"] == "success"


def test_up_axis() -> None:
    assert up_z_of(np.array([0.0, 0.0, 0.0, 1.0])) == pytest.approx(1.0)
    s = np.sin(np.pi / 4)  # 90 degrees about x
    assert up_z_of(np.array([s, 0.0, 0.0, s])) == pytest.approx(0.0, abs=1e-9)
    s, c = np.sin(np.pi / 6), np.cos(np.pi / 6)  # 60 degrees about y: the fall threshold
    assert up_z_of(np.array([0.0, s, 0.0, c])) == pytest.approx(0.5)

"""One judge of an episode's outcome for every high level (D-052): the scripted expert, the replay
check of its recordings and the closed-loop evaluation of a VLA.

It sees only simulator state, never the policy's intentions: the pelvis height and tilt, the
objects' positions and, when given, both palms' positions. Each control step `step()` updates
it; an episode ends

- at once when the robot falls (pelvis below 0.45 m, or tilted more than 60 degrees) or another
  object has moved more than 2 cm from where it started (`knocked_other`);
- `settle_s` after the target object was first put down in a bin and let go: long enough for the
  expert to withdraw its arm (it takes 3.5-4.3 s from letting go to the end of its retract), so
  objects pushed while the arm withdraws count, as they do for the expert;
- otherwise at the time limit or when a recording ends (`finish()`).

When it ends, its label comes from the state at that moment (no other object moved, robot up):

- `success`: the target rests inside the named bin, out of the hand;
- `wrong_bin`: it rests inside the other bin, out of the hand;
- the target was never lifted: `knocked` if it was pushed more than 3 cm before a hand reached
  it, `missed_grasp` if a hand reached it, else `timed_out` (never reached);
- it was lifted: `timed_out` while a hand still holds it, else `dropped` (it lies outside the
  bins, or was knocked out of one).

"Out of the hand" and "a hand holds it" use the expert's own test: the target's centre within
12 cm of a palm (a held can is 7-10 cm from the palm frame's origin). Without palm positions the
hand is assumed open, and `missed_grasp` cannot be told from `timed_out`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pi_wbc.stage3 import layout as L
from pi_wbc.stage3.expert import placement_label

FAILURES = ("fell", "knocked_other", "wrong_bin", "dropped", "knocked", "missed_grasp", "timed_out")
LABELS = ("success", *FAILURES)


@dataclass(frozen=True)
class JudgeParams:
    fall_z: float = 0.45  # pelvis below this (m): fell
    fall_up: float = 0.5  # the pelvis's up axis has less than this vertical: tilted over 60 deg
    other_knock: float = 0.02  # another object moved this far horizontally (m): knocked_other
    target_knock: float = 0.03  # the target moved this far without being lifted: knocked
    lifted_dz: float = 0.03  # the target counts as lifted once this far above its start
    held_dist: float = 0.12  # target centre within this of a palm: in the hand (expert's test)
    settle_s: float = 4.5  # judged this long after the target is first in a bin and let go
    max_s: float = 60.0


@dataclass
class _Episode:
    layout: L.Layout
    target: int
    others: list[int]
    start: np.ndarray  # (5, 3) object positions at the start
    lifted: bool = False
    reached: bool = False
    max_shift: float = 0.0  # how far the target moved from its start, while never lifted
    knocked_early: bool = False  # pushed past target_knock before a hand reached it
    placed_at: float | None = None  # when it was first put in a bin and let go
    label: str | None = None
    end_s: float | None = None
    ended_by: str | None = None
    notes: dict = field(default_factory=dict)


def up_z_of(quat_xyzw: np.ndarray) -> np.ndarray:
    """Vertical component of the body's up axis from (..., 4) quaternions (x, y, z, w)."""
    x, y = quat_xyzw[..., 0], quat_xyzw[..., 1]
    return 1.0 - 2.0 * (x * x + y * y)


class Judge:
    """Judges a batch of episodes; positions are in each environment's own frame (m)."""

    def __init__(
        self,
        layouts: list[L.Layout],
        start_objects: np.ndarray,
        params: JudgeParams | None = None,
    ):
        self.p = params or JudgeParams()
        self.eps = []
        for lay, start in zip(layouts, np.asarray(start_objects, dtype=float), strict=True):
            target = L.OBJECT_NAMES.index(lay.target_object)
            others = [k for k, name in enumerate(L.OBJECT_NAMES)
                      if name in lay.objects and name != lay.target_object]  # fmt: skip
            self.eps.append(_Episode(lay, target, others, start.copy()))

    # -- results ------------------------------------------------------------------------------
    @property
    def done(self) -> np.ndarray:
        return np.array([e.label is not None for e in self.eps])

    @property
    def labels(self) -> list[str | None]:
        return [e.label for e in self.eps]

    def result(self, i: int) -> dict:
        e = self.eps[i]
        return {"label": e.label, "time_s": e.end_s, "ended_by": e.ended_by, **e.notes}

    # -- updates ------------------------------------------------------------------------------
    def step(
        self,
        t: float,
        pelvis_z: np.ndarray,
        up_z: np.ndarray,
        objects: np.ndarray,
        palms: np.ndarray | None = None,
    ) -> None:
        """One control step at time `t` (s since the episode began): pelvis heights (n,), up-axis
        verticals (n,), object positions (n, 5, 3), palm positions (n, 2, 3) or None."""
        p = self.p
        for i, e in enumerate(self.eps):
            if e.label is not None:
                continue
            if pelvis_z[i] < p.fall_z or up_z[i] < p.fall_up:
                self._end(e, "fell", t, "fall")
                continue
            if e.others:
                moved = np.linalg.norm(objects[i, e.others, :2] - e.start[e.others, :2], axis=1)
                if moved.max() > p.other_knock:
                    self._end(e, "knocked_other", t, "knock")
                    continue
            obj = objects[i, e.target]
            held = False
            if palms is not None:
                near = float(np.linalg.norm(palms[i] - obj, axis=1).min())
                held = near < p.held_dist
                e.reached |= held
            if obj[2] - e.start[e.target, 2] > p.lifted_dz:
                e.lifted = True
            if not e.lifted:
                e.max_shift = max(e.max_shift, float(np.linalg.norm(obj - e.start[e.target])))
                e.knocked_early |= e.max_shift > p.target_knock and not e.reached
            place = placement_label(e.layout, obj)
            in_bin = place in ("success", "wrong_bin") and not held
            if in_bin and e.placed_at is None:
                e.placed_at = t
            elif held and e.placed_at is not None:  # picked up again: start over
                e.placed_at = None
            if e.placed_at is not None and t - e.placed_at >= p.settle_s:
                self._end(e, self._state_label(e, obj, held), t, "settled")
                continue
            if t >= p.max_s:
                self._end(e, self._state_label(e, obj, held), t, "time limit")
                continue
            e.notes["last"] = (obj.copy(), held)

    def finish(self, t: float, ended_by: str = "end") -> None:
        """End every episode still running (time limit, or the end of a recording), labelled by
        its last state."""
        for e in self.eps:
            if e.label is None:
                obj, held = e.notes.get("last", (e.start[e.target], False))
                self._end(e, self._state_label(e, obj, held), t, ended_by)

    # -- internals ----------------------------------------------------------------------------
    def _state_label(self, e: _Episode, obj: np.ndarray, held: bool) -> str:
        place = placement_label(e.layout, obj)
        if place in ("success", "wrong_bin") and not held:
            return place
        if not e.lifted:
            if e.knocked_early:
                return "knocked"
            return "missed_grasp" if e.reached else "timed_out"
        return "timed_out" if held else "dropped"

    @staticmethod
    def _end(e: _Episode, label: str, t: float, ended_by: str) -> None:
        e.label, e.end_s, e.ended_by = label, round(float(t), 3), ended_by
        e.notes.pop("last", None)
        e.notes["placed_s"] = None if e.placed_at is None else round(float(e.placed_at), 3)

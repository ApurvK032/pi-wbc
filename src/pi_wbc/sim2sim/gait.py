"""Gait measurements for a G1 walking in MuJoCo.

Sampled once per policy step. A foot is down when its ankle-roll link presses on the floor
(a contact that carries force). Slip is the speed of the sole at those contacts; the ankle's
own speed while the foot is down is kept too, but it also counts heel-to-toe rolling. A step
is a touchdown after at least `MIN_AIR_S` in the air, so contact flicker while shuffling does
not count as a step; swings already under way when recording starts, or not finished when it
stops, are left out. Ranges of joint motion are 5th-95th percentile, so a single stumble does
not dominate them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import numpy as np

from pi_wbc.sim2sim.mujoco_g1 import match_name

if TYPE_CHECKING:
    from pi_wbc.sim2sim.mujoco_g1 import G1MujocoSim

MIN_AIR_S = 0.06
FOOT_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")
KNEE_JOINTS = ("left_knee_joint", "right_knee_joint")
SHOULDER_JOINTS = ("left_shoulder_pitch_joint", "right_shoulder_pitch_joint")

METRICS = (
    "step_rate_hz",  # touchdowns per second, both feet together
    "step_length_m",  # mean planar base speed / step rate
    "swing_s",  # mean time a foot spends in the air per step
    "foot_lift_m",  # mean over steps of peak swing height above the foot's stance height
    "stance_foot_speed_mps",  # mean planar speed of the ankle while its foot is down
    "sole_slip_mps",  # mean planar speed of the sole at its floor contacts while down (slip)
    "double_support",  # fraction of time with both feet down
    "flight",  # fraction of time with neither foot down
    "knee_range_deg",  # knee motion range, mean of both legs
    "arm_swing_deg",  # shoulder pitch motion range, mean of both arms
)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """(start, end) index pairs of the True runs in a 1-D boolean array; end is exclusive."""
    padded = np.concatenate([[False], mask, [False]]).astype(np.int8)
    edges = np.flatnonzero(np.diff(padded))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist(), strict=True))


def _range(x: np.ndarray) -> float:
    lo, hi = np.percentile(x, [5, 95])
    return float(hi - lo)


def gait_metrics(
    down: np.ndarray,
    foot_z: np.ndarray,
    foot_speed: np.ndarray,
    base_speed: np.ndarray,
    knee: np.ndarray,
    shoulder: np.ndarray,
    dt: float,
    min_air_s: float = MIN_AIR_S,
    sole_slip: np.ndarray | None = None,
) -> dict[str, float | None]:
    """Gait summary from per-step samples.

    Arrays have one row per sample; two-column arrays have one column per side (left, right):
    down (bool), foot_z (m), foot_speed (planar ankle speed, m/s), knee and shoulder (rad),
    and optionally sole_slip (planar speed at the floor contacts, m/s; NaN where the foot
    carries no load). base_speed is the planar base speed (m/s), one value per sample.
    """
    down = np.asarray(down, dtype=bool)
    n = len(down)
    if n == 0:
        return dict.fromkeys(METRICS)
    duration = n * dt
    min_air = max(1, round(min_air_s / dt))

    steps, swing_lengths, lifts = 0, [], []
    for side in range(2):
        stance_z = foot_z[down[:, side], side]
        ground = float(np.median(stance_z)) if stance_z.size else None
        for start, end in _runs(~down[:, side]):
            if start == 0 or end == n or end - start < min_air:
                continue
            steps += 1
            swing_lengths.append((end - start) * dt)
            if ground is not None:
                lifts.append(float(foot_z[start:end, side].max()) - ground)

    step_rate = steps / duration
    stance_speed = foot_speed[down]
    slip = np.asarray(sole_slip)[down] if sole_slip is not None else np.empty(0)
    slip = slip[np.isfinite(slip)]
    return {
        "step_rate_hz": step_rate,
        "step_length_m": float(np.mean(base_speed)) / step_rate if steps else None,
        "swing_s": float(np.mean(swing_lengths)) if swing_lengths else None,
        "foot_lift_m": float(np.mean(lifts)) if lifts else None,
        "stance_foot_speed_mps": float(np.mean(stance_speed)) if stance_speed.size else None,
        "sole_slip_mps": float(np.mean(slip)) if slip.size else None,
        "double_support": float(np.mean(down.all(axis=1))),
        "flight": float(np.mean(~down.any(axis=1))),
        "knee_range_deg": float(np.degrees(np.mean([_range(knee[:, i]) for i in range(2)]))),
        "arm_swing_deg": float(np.degrees(np.mean([_range(shoulder[:, i]) for i in range(2)]))),
    }


class GaitRecorder:
    """Collects the samples `gait_metrics` needs from a running `G1MujocoSim`."""

    def __init__(self, sim: G1MujocoSim) -> None:
        self.sim = sim
        m = sim.model
        body_names = [m.body(i).name for i in range(m.nbody)]
        self.feet = [match_name(body_names, b) for b in FOOT_BODIES]
        names = sim.io.joint_names  # Isaac Lab's names; sim.qpos_adr follows the same order
        self.knee_adr = [int(sim.qpos_adr[names.index(j)]) for j in KNEE_JOINTS]
        self.shoulder_adr = [int(sim.qpos_adr[names.index(j)]) for j in SHOULDER_JOINTS]
        self.dt = sim.io.physics_dt * sim.io.decimation
        self._side = {f: i for i, f in enumerate(self.feet)}
        self._vel6 = np.zeros((2, 6))
        self._force6 = np.zeros(6)
        self.clear()

    def clear(self) -> None:
        self.down, self.foot_z, self.foot_speed, self.sole_slip = [], [], [], []
        self.base_speed, self.knee, self.shoulder = [], [], []

    def _foot_down(self) -> list[bool]:
        bodies = self.sim.model.geom_bodyid
        touching = {int(bodies[g]) for c in self.sim.floor_contacts() for g in (c.geom1, c.geom2)}
        return [f in touching for f in self.feet]

    def _sole_slip(self) -> list[float]:
        """Planar speed of each sole where it touches the floor, weighted by contact force.

        The speed of the foot's own material at each force-carrying floor contact, averaged
        over that foot's contacts with their normal forces as weights; NaN for a foot with no
        such contact. Unlike the ankle's speed, it is near zero when the foot rolls from heel
        to toe or pivots on its toe without sliding. Uses the body velocities of `self._vel6`.
        """
        m, d = self.sim.model, self.sim.data
        floor = self.sim.floor_geoms
        speed_sum, weight_sum, plain_sum, count = [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0, 0]
        for i in range(d.ncon):
            c = d.contact[i]
            if c.exclude != 0:
                continue
            if c.geom1 in floor:
                other = c.geom2
            elif c.geom2 in floor:
                other = c.geom1
            else:
                continue
            body = int(m.geom_bodyid[other])
            side = self._side.get(body)
            if side is None:
                continue
            ang, lin = self._vel6[side, :3], self._vel6[side, 3:]
            v = lin + np.cross(ang, c.pos - d.xpos[body])  # velocity of the foot at the contact
            speed = float(np.hypot(v[0], v[1]))
            mujoco.mj_contactForce(m, d, i, self._force6)
            weight = max(float(self._force6[0]), 0.0)  # normal force, contact frame
            speed_sum[side] += weight * speed
            weight_sum[side] += weight
            plain_sum[side] += speed
            count[side] += 1
        out = []
        for side in range(2):
            if weight_sum[side] > 0:
                out.append(speed_sum[side] / weight_sum[side])
            elif count[side]:
                out.append(plain_sum[side] / count[side])  # touching, no force yet this step
            else:
                out.append(float("nan"))
        return out

    def sample(self) -> None:
        m, d = self.sim.model, self.sim.data
        self.down.append(self._foot_down())
        self.foot_z.append([float(d.xpos[f][2]) for f in self.feet])
        for side, f in enumerate(self.feet):
            # mjOBJ_XBODY: angular and linear velocity of the body frame origin (the ankle),
            # world axes, from the same state as the contacts.
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_XBODY, f, self._vel6[side], 0)
        self.foot_speed.append([float(np.linalg.norm(v[3:5])) for v in self._vel6])
        self.sole_slip.append(self._sole_slip())
        v_xy, _ = self.sim.velocity_yaw_frame()
        self.base_speed.append(float(np.linalg.norm(v_xy)))
        self.knee.append([float(d.qpos[a]) for a in self.knee_adr])
        self.shoulder.append([float(d.qpos[a]) for a in self.shoulder_adr])

    def metrics(self) -> dict[str, float | None]:
        return gait_metrics(
            np.array(self.down, dtype=bool).reshape(-1, 2),
            np.array(self.foot_z).reshape(-1, 2),
            np.array(self.foot_speed).reshape(-1, 2),
            np.array(self.base_speed),
            np.array(self.knee).reshape(-1, 2),
            np.array(self.shoulder).reshape(-1, 2),
            self.dt,
            sole_slip=np.array(self.sole_slip, dtype=float).reshape(-1, 2),
        )

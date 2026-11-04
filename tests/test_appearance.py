"""Looks for the head camera (pi_wbc.stage3.appearance): sampling, image light, scene turns."""

import colorsys
import math

import numpy as np
import pytest
import torch

from pi_wbc.stage3 import appearance as A
from pi_wbc.stage3 import layout as L


def test_looks_repeat_and_differ() -> None:
    assert A.sample_look(10_000) == A.sample_look(10_000)
    assert A.sample_look(10_000) != A.sample_look(10_001)
    assert A.sample_look(10_000) != A.sample_look(10_000, render_seed=1)


def test_looks_stay_in_their_ranges() -> None:
    looks = [A.sample_look(s) for s in range(10_000, 10_500)]
    for look in looks:
        for surface in ("floor", "table", "wall"):
            _, s, v = colorsys.rgb_to_hsv(*getattr(look, f"{surface}_rgb"))
            c = A.SURFACE_COLOURS[surface]
            assert c["s"][0] - 1e-9 <= s <= c["s"][1] + 1e-9
            assert c["v"][0] - 1e-9 <= v <= c["v"][1] + 1e-9
            tex = getattr(look, f"{surface}_texture")
            assert tex == A.PLAIN or A.TEXTURES[tex][0] in A.SURFACE_PATTERNS[surface]
        assert A.GAIN[0] <= look.gain <= A.GAIN[1]
        assert max(abs(x) for x in look.cam_shift) <= A.CAM_SHIFT_M
        assert max(abs(x) for x in look.cam_tilt) <= math.radians(A.CAM_TILT_DEG) + 1e-12
        assert -math.pi <= look.scene_yaw < math.pi
    # every sun height and every pattern is used
    assert {x.sun_elevation for x in looks} == set(range(len(A.SUN_ELEVATIONS_DEG)))
    used = {x.floor_texture for x in looks} | {x.table_texture for x in looks}
    used |= {x.wall_texture for x in looks}
    assert used == {A.PLAIN, *range(len(A.TEXTURES))}


def test_look_json_round_trips_the_numbers() -> None:
    look = A.sample_look(10_000)
    j = look.to_json()
    assert j["sun_elevation"] == look.sun_elevation
    assert j["gain"] == pytest.approx(look.gain, abs=1e-5)
    assert j["floor_rgb"] == pytest.approx(look.floor_rgb, abs=1e-5)


def test_textures() -> None:
    for name in {n for n, _ in A.TEXTURES}:
        tex = A.make_texture(name)
        assert tex.shape == (256, 256, 4) and tex.dtype == np.uint8
        assert (tex[..., 3] == 255).all()
        assert (tex[..., 0] == tex[..., 1]).all() and (tex[..., 1] == tex[..., 2]).all()
        assert tex[..., 0].min() >= 100 and tex[..., 0].max() <= 255
        assert np.array_equal(tex, A.make_texture(name))  # the same every run


def test_light_gain() -> None:
    rgb = (
        torch.arange(256, dtype=torch.uint8).repeat(4, 1).reshape(4, 8, 32, 1).expand(-1, -1, -1, 3)
    )
    one = A.apply_light(rgb, torch.ones(4), torch.ones(4, 3))
    assert torch.equal(one, rgb)  # no change at gain 1
    bright = A.apply_light(rgb, torch.full((4,), 1.3), torch.ones(4, 3)).int()
    assert (bright >= rgb.int()).all() and (bright > rgb.int()).float().mean() > 0.7
    warm = A.apply_light(rgb, torch.ones(4), torch.tensor([[1.1, 1.0, 0.9]] * 4)).int()
    mid = rgb.int()[..., 0] == 128
    assert (warm[..., 0][mid] > 128).all() and (warm[..., 2][mid] < 128).all()
    assert (warm[..., 1][mid] == 128).all()
    # linear 2x on a mid grey: sRGB 128 is linear 0.216, doubled 0.432, sRGB 175.5
    assert (
        abs(int(A.apply_light(rgb, torch.full((4,), 2.0), torch.ones(4, 3))[mid][0, 0]) - 175.5)
        <= 1
    )


def test_sun_direction() -> None:
    for e in A.SUN_ELEVATIONS_DEG:
        d = A.sun_direction(e)
        assert np.linalg.norm(d) == pytest.approx(1.0, abs=1e-6)
        assert math.degrees(math.asin(-d[2])) == pytest.approx(e, abs=1e-4)  # travels downwards


def test_walls_clear_the_table_and_the_start() -> None:
    far_edge = L.TABLE_CENTER[0] + L.TABLE_SIZE[0] / 2
    for s in range(10_000, 10_100):
        look = A.sample_look(s)
        walls = A.wall_poses(look)
        assert walls["wall_back"][0] - A.WALL_THICKNESS / 2 >= far_edge + A.BACK_GAP[0] - 1e-9
        # side walls outside every robot start and the table
        inner = min(abs(walls["wall_left"][1]), abs(walls["wall_right"][1])) - A.WALL_THICKNESS / 2
        assert inner > max(abs(y) for y in L.START_Y) + 0.5 and inner > L.TABLE_SIZE[1] / 2


def test_turning_the_scene_turns_the_camera_with_it() -> None:
    """The camera pose from a turned torso is the turned camera pose of the unturned torso."""
    pytest.importorskip("isaaclab.utils.math")
    g = torch.Generator().manual_seed(0)
    n = 16
    torso = torch.cat([torch.randn(n, 3, generator=g), torch.randn(n, 4, generator=g)], dim=-1)
    torso[:, 3:] /= torso[:, 3:].norm(dim=-1, keepdim=True)
    yaw = torch.rand(n, generator=g) * 2 * math.pi - math.pi
    shift = (torch.rand(n, 3, generator=g) - 0.5) * 0.02
    tilt = (torch.rand(n, 3, generator=g) - 0.5) * 0.03
    pos, quat = A.camera_poses(torso[:, :3], torso[:, 3:], shift, tilt)
    turned = A.turn_poses(torso.clone(), yaw)
    pos_t, quat_t = A.camera_poses(turned[:, :3], turned[:, 3:], shift, tilt)
    expect = A.turn_poses(torch.cat([pos, quat], dim=-1), yaw)
    assert torch.allclose(pos_t, expect[:, :3], atol=1e-5)
    same = torch.minimum(
        (quat_t - expect[:, 3:]).abs().amax(-1), (quat_t + expect[:, 3:]).abs().amax(-1)
    )
    assert (same < 1e-5).all()
    # and the turn keeps heights and distances from the origin
    assert torch.allclose(turned[:, 2], torso[:, 2])
    assert torch.allclose(turned[:, :2].norm(dim=-1), torso[:, :2].norm(dim=-1), atol=1e-5)


def test_camera_without_mount_error_matches_the_attached_camera() -> None:
    """Zero shift and tilt give the D435 offset composed with the torso pose."""
    pytest.importorskip("isaaclab.utils.math")
    torso_p = torch.tensor([[0.0, 0.0, 0.8]])
    torso_q = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    pos, quat = A.camera_poses(torso_p, torso_q, torch.zeros(1, 3), torch.zeros(1, 3))
    assert torch.allclose(pos, torso_p + torch.tensor([A.HEAD_CAM_POS]))
    assert torch.allclose(quat, torch.tensor([A.HEAD_CAM_ROT]), atol=1e-6)
    # 47.6 degrees down: the camera's forward axis (+x) points below the horizon
    from isaaclab.utils.math import quat_apply

    fwd = quat_apply(quat, torch.tensor([[1.0, 0.0, 0.0]]))[0]
    assert math.degrees(math.asin(-fwd[2] / fwd.norm())) == pytest.approx(47.6, abs=0.1)

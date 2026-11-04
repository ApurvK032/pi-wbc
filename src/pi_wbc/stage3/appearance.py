"""Randomized looks for the head camera (STAGE3_DESIGN.md section 8, D-043).

Every episode gets a look drawn from its layout seed: floor, table and wall colours and
patterns, the room's size, the sun's height and its direction relative to the scene, the
overall brightness and colour of the light, and a small error in the camera's mount. Object and
bin colours never change, since the instructions name them.

Newton's renderer has one global sun (a direction, white, of fixed strength) and a fixed
ambient term, so:

- the sun's direction relative to a scene comes from turning the whole scene (robot, table,
  objects, bins, floor, walls) by `scene_yaw` about its origin, which changes nothing else in
  the image; the sun's height is one of `SUN_ELEVATIONS_DEG`, and episodes with the same height
  are rendered together;
- the light's brightness and colour are a gain on the rendered image (`apply_light`), applied
  in linear colour, which is what a brighter or warmer light does to diffuse surfaces.

The parts that need the simulator (`add_look_assets`, `LookRenderer`) import it lazily, so the
sampling and the image maths are testable without it.
"""

from __future__ import annotations

import colorsys
import math
from dataclasses import asdict, dataclass

import numpy as np

from pi_wbc.stage3 import layout as L

# the head camera: Unitree's D435 mount on the torso link moved 4 cm forward (at the mount itself
# the camera sits inside the head mesh), tilted 47.6 degrees down; quaternion (x, y, z, w),
# camera looking along +x ("world" convention)
HEAD_CAM_POS = (0.0976, 0.0175, 0.4299)
HEAD_CAM_ROT = (0.0, 0.4035, 0.0, 0.9150)
HEAD_CAM_SIZE = (640, 480)
HEAD_CAM_FOCAL = 18.0  # mm, with a 20.955 mm horizontal aperture: 60 x 47 degrees
HEAD_CAM_APERTURE = 20.955

SUN_ELEVATIONS_DEG = (30.0, 40.0, 50.0, 60.0, 70.0, 80.0)
CAM_SHIFT_M = 0.01  # largest mount error per axis
CAM_TILT_DEG = 1.0
GAIN = (0.7, 1.3)  # light brightness, linear
WARMTH = 0.1  # red up and blue down by up to this fraction (or the reverse)
GREEN_TINT = 0.03

# floor, walls: one big tile and three walls per environment, visual only
FLOOR_SIZE = (16.0, 16.0, 0.01)
FLOOR_CENTER = (-1.0, 0.0)
FLOOR_TOP = 0.0005  # just above the ground plane (z = 0), so it covers it
WALL_THICKNESS, WALL_HEIGHT, WALL_LENGTH = 0.05, 2.5, 16.0
BACK_GAP = (0.3, 1.5)  # wall behind the table: distance from the table's far edge (m)
SIDE_Y = (2.0, 3.5)  # side walls at y = +-this (m)

# patterns: (name, metres per texture repeat). A texture holds 8 checker squares, 4 tiles,
# 8 planks or 16 stripes across, so the spans give 0.2 and 0.4 m squares, 0.3 and 0.6 m tiles,
# 0.1 and 0.2 m planks, 5 and 15 cm stripes.
TEXTURES = (
    ("checker", 1.6), ("checker", 3.2),
    ("tiles", 1.2), ("tiles", 2.4),
    ("planks", 0.8), ("planks", 1.6),
    ("speckle", 0.5), ("speckle", 2.0),
    ("stripes", 0.8), ("stripes", 2.4),
)  # fmt: skip
PLAIN = -1
SURFACE_PATTERNS = {
    "floor": ("checker", "tiles", "planks", "speckle"),
    "table": ("planks", "speckle"),
    "wall": ("stripes", "speckle", "tiles"),
}
PLAIN_SHARE = 0.25
# colours (HSV, sRGB values): muted, so none looks like an object's or a bin's colour; the table
# keeps clear of white and black, so the white bottle and the black bin stand out on it
SURFACE_COLOURS = {
    "floor": {"s": (0.0, 0.2), "v": (0.2, 0.85)},
    "table": {"s": (0.0, 0.2), "v": (0.3, 0.8)},
    "wall": {"s": (0.0, 0.2), "v": (0.45, 0.95)},
}
LOOK_STREAM = 43  # rng stream id, so looks never share random numbers with layouts


@dataclass(frozen=True)
class Look:
    floor_rgb: tuple[float, float, float]
    floor_texture: int  # index into TEXTURES, or PLAIN
    table_rgb: tuple[float, float, float]
    table_texture: int
    wall_rgb: tuple[float, float, float]
    wall_texture: int
    back_gap: float
    side_y: float
    sun_elevation: int  # index into SUN_ELEVATIONS_DEG
    scene_yaw: float  # rad
    gain: float
    tint: tuple[float, float, float]
    cam_shift: tuple[float, float, float]  # m, torso frame
    cam_tilt: tuple[float, float, float]  # rotation vector (rad), camera frame

    def to_json(self) -> dict:
        return {k: (round(v, 5) if isinstance(v, float) else
                    [round(x, 5) for x in v] if isinstance(v, tuple) else v)
                for k, v in asdict(self).items()}  # fmt: skip


def _colour(rng: np.random.Generator, surface: str) -> tuple[float, float, float]:
    c = SURFACE_COLOURS[surface]
    h, s, v = rng.uniform(0, 1), rng.uniform(*c["s"]), rng.uniform(*c["v"])
    return tuple(float(x) for x in colorsys.hsv_to_rgb(h, s, v))


def _texture(rng: np.random.Generator, surface: str) -> int:
    if rng.uniform() < PLAIN_SHARE:
        return PLAIN
    allowed = [k for k, (name, _) in enumerate(TEXTURES) if name in SURFACE_PATTERNS[surface]]
    return int(rng.choice(allowed))


def sample_look(seed: int, render_seed: int = 0) -> Look:
    """The look of the episode with layout seed `seed`; `render_seed` selects another set."""
    rng = np.random.default_rng([LOOK_STREAM, render_seed, seed])
    warm = rng.uniform(-WARMTH, WARMTH)
    tint = (1.0 + warm, 1.0 + rng.uniform(-GREEN_TINT, GREEN_TINT), 1.0 - warm)
    return Look(
        floor_rgb=_colour(rng, "floor"), floor_texture=_texture(rng, "floor"),
        table_rgb=_colour(rng, "table"), table_texture=_texture(rng, "table"),
        wall_rgb=_colour(rng, "wall"), wall_texture=_texture(rng, "wall"),
        back_gap=float(rng.uniform(*BACK_GAP)), side_y=float(rng.uniform(*SIDE_Y)),
        sun_elevation=int(rng.integers(len(SUN_ELEVATIONS_DEG))),
        scene_yaw=float(rng.uniform(-math.pi, math.pi)),
        gain=float(rng.uniform(*GAIN)), tint=tuple(float(x) for x in tint),
        cam_shift=tuple(float(x) for x in rng.uniform(-CAM_SHIFT_M, CAM_SHIFT_M, 3)),
        cam_tilt=tuple(float(x) for x in np.radians(rng.uniform(-CAM_TILT_DEG, CAM_TILT_DEG, 3))),
    )  # fmt: skip


def sun_direction(elevation_deg: float) -> np.ndarray:
    """Newton's light direction (the way the light travels) for a sun at this height, from -x."""
    e = math.radians(elevation_deg)
    return np.array([math.cos(e), 0.0, -math.sin(e)], dtype=np.float32)


# --- static poses in the environment frame (before the scene turn) --------------------------


def wall_poses(look: Look) -> dict[str, tuple[float, float, float]]:
    """Centres of the three walls: behind the table, and at both sides."""
    far_edge = L.TABLE_CENTER[0] + L.TABLE_SIZE[0] / 2
    z = WALL_HEIGHT / 2
    return {
        "wall_back": (far_edge + look.back_gap + WALL_THICKNESS / 2, 0.0, z),
        "wall_left": (FLOOR_CENTER[0], look.side_y + WALL_THICKNESS / 2, z),
        "wall_right": (FLOOR_CENTER[0], -look.side_y - WALL_THICKNESS / 2, z),
    }


# --- textures ---------------------------------------------------------------------------------


def make_texture(name: str, res: int = 256) -> np.ndarray:
    """A grey pattern (res x res RGBA, uint8) that multiplies a surface's colour."""
    rng = np.random.default_rng(_NAME_SEED[name])
    u = np.arange(res)
    if name == "checker":
        cell = res // 8
        grey = np.where(((u[:, None] // cell) + (u[None, :] // cell)) % 2 == 0, 1.0, 0.78)
    elif name == "tiles":
        cell = res // 4
        shade = rng.uniform(0.86, 1.0, (4, 4))
        grey = shade[u[:, None] // cell, u[None, :] // cell]
        grout = ((u % cell) < 3)[:, None] | ((u % cell) < 3)[None, :]
        grey = np.where(grout, 0.55, grey)
    elif name == "planks":
        width = res // 8
        plank = u // width  # planks run along the first texture axis
        shade = rng.uniform(0.75, 1.0, 8)
        seam = rng.integers(0, res, 8)
        grain = 0.06 * np.sin(u[None, :] / res * 2 * np.pi * rng.uniform(3, 9, 8)[plank][:, None]
                              + rng.uniform(0, 6.3, 8)[plank][:, None])  # fmt: skip
        grey = shade[plank][:, None] + grain
        joint = np.abs(u[None, :] - seam[plank][:, None]) < 2
        edge = (u % width) < 2
        grey = np.where(joint | edge[:, None], 0.55, grey)
    elif name == "speckle":
        noise = rng.uniform(0, 1, (res, res))
        for _ in range(3):  # blur with wrap-around, so the texture tiles without seams
            noise = (noise + np.roll(noise, 1, 0) + np.roll(noise, -1, 0)
                     + np.roll(noise, 1, 1) + np.roll(noise, -1, 1)) / 5  # fmt: skip
        noise = (noise - noise.min()) / (noise.max() - noise.min())
        grey = 0.72 + 0.28 * noise
    elif name == "stripes":
        grey = np.where((u // (res // 16)) % 2 == 0, 1.0, 0.85)[:, None] * np.ones((1, res))
    else:
        raise ValueError(f"unknown pattern {name}")
    v = np.clip(np.round(np.broadcast_to(grey, (res, res)) * 255), 0, 255).astype(np.uint8)
    return np.stack([v, v, v, np.full_like(v, 255)], axis=-1)


_NAME_SEED = {"checker": 1, "tiles": 2, "planks": 3, "speckle": 4, "stripes": 5}


# --- image and pose maths (torch) ------------------------------------------------------------


def srgb_to_linear(x):
    """sRGB values in [0, 1] to linear (numpy or torch)."""
    if isinstance(x, np.ndarray):
        return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    import torch

    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x):
    """Linear values (torch) to sRGB, clipped to [0, 1]."""
    import torch

    x = x.clamp(0.0, 1.0)
    return torch.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)


def apply_light(rgb, gain, tint):
    """Brightness and colour of the light on rendered sRGB images.

    rgb: (n, h, w, 3) uint8; gain: (n,); tint: (n, 3). Returns (n, h, w, 3) uint8.
    """
    import torch

    lut = srgb_to_linear(torch.arange(256, device=rgb.device, dtype=torch.float32) / 255)
    lin = lut[rgb.long()] * (gain[:, None] * tint)[:, None, None, :]
    return (linear_to_srgb(lin) * 255 + 0.5).to(torch.uint8)


def yaw_quat(yaw):
    """(x, y, z, w) quaternions of rotations by `yaw` about z."""
    import torch

    z = torch.zeros_like(yaw)
    return torch.stack([z, z, torch.sin(yaw / 2), torch.cos(yaw / 2)], dim=-1)


def turn_poses(pose, yaw):
    """Turn poses (n, ..., 7: position, quaternion x y z w) about z through the origin by
    `yaw` (n,)."""
    import torch
    from isaaclab.utils.math import quat_apply, quat_mul

    shape = pose.shape
    q = yaw_quat(yaw).reshape(shape[0], *([1] * (len(shape) - 2)), 4).expand(*shape[:-1], 4)
    q = q.reshape(-1, 4)
    flat = pose.reshape(-1, 7)
    return torch.cat([quat_apply(q, flat[:, :3]), quat_mul(q, flat[:, 3:])], dim=-1).reshape(shape)


def camera_poses(torso_pos, torso_quat, shift, tilt):
    """World poses of the head camera ("world" convention) from torso link poses, with each
    environment's mount error: `shift` (n, 3) m in the torso frame, `tilt` (n, 3) rotation
    vectors in the camera frame."""
    import torch
    from isaaclab.utils.math import quat_apply, quat_from_angle_axis, quat_mul

    dev = torso_pos.device
    off_p = torch.tensor(HEAD_CAM_POS, device=dev).expand_as(torso_pos)
    off_q = torch.tensor(HEAD_CAM_ROT, device=dev).expand_as(torso_quat)
    angle = tilt.norm(dim=-1)
    axis = torch.where(angle[:, None] > 0, tilt / angle.clamp_min(1e-12)[:, None],
                       torch.tensor([1.0, 0.0, 0.0], device=dev))  # fmt: skip
    pos = torso_pos + quat_apply(torso_quat, off_p + shift)
    quat = quat_mul(quat_mul(torso_quat, off_q), quat_from_angle_axis(angle, axis))
    return pos, quat


# --- the simulator side ------------------------------------------------------------------------


def add_look_assets(scene_cfg) -> None:
    """Add the head camera's floor tile and walls (kinematic, no collisions) to a scene config."""
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg

    def box(name, size):
        return RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/{name}",
            spawn=sim_utils.CuboidCfg(
                size=size,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -5.0)),
        )

    scene_cfg.look_floor = box("LookFloor", FLOOR_SIZE)
    for name in ("wall_back", "wall_left", "wall_right"):
        size = ((WALL_THICKNESS, WALL_LENGTH, WALL_HEIGHT) if name == "wall_back"
                else (WALL_LENGTH, WALL_THICKNESS, WALL_HEIGHT))  # fmt: skip
        setattr(scene_cfg, f"look_{name}", box(f"Look_{name}", size))


def head_camera_cfg(renderer_cfg):
    """The head camera as a free camera; its pose is set every frame (`camera_poses`)."""
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import CameraCfg

    return CameraCfg(
        prim_path="{ENV_REGEX_NS}/head_cam",
        update_period=0.0,
        width=HEAD_CAM_SIZE[0],
        height=HEAD_CAM_SIZE[1],
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=HEAD_CAM_FOCAL,
            horizontal_aperture=HEAD_CAM_APERTURE,
            clipping_range=(0.05, 20.0),
        ),
        renderer_cfg=renderer_cfg,
    )


class LookRenderer:
    """Writes looks into Newton's renderer: surface colours and patterns per environment, and
    the sun's height for all of them. Call `install_textures` before the first render (the
    renderer captures its arrays in a CUDA graph then; later changes go into the same arrays).
    """

    def __init__(self, camera, num_envs: int):
        import warp as wp

        self.wp = wp
        sensor = camera._renderer.newton_sensor
        self.model = sensor.model
        self.ctx = sensor._SensorTiledCamera__render_context  # no public setter for these arrays
        labels = list(self.model.shape_label)
        self.shapes = {"floor": [], "table": [], "wall": []}
        for i in range(num_envs):
            prefix = f"/World/envs/env_{i}/"
            idx = {"floor": [], "table": [], "wall": []}
            for k, label in enumerate(labels):
                if not label.startswith(prefix):
                    continue
                part = label[len(prefix) :].split("/")[0]
                if part == "LookFloor":
                    idx["floor"].append(k)
                elif part == "Table":
                    idx["table"].append(k)
                elif part.startswith("Look_wall"):
                    idx["wall"].append(k)
            for s in idx:
                if not idx[s]:
                    raise RuntimeError(f"env {i}: no {s} shapes found")
                self.shapes[s].append(idx[s])
        self.colours = self.model.shape_color.numpy().copy()
        self.texture_ids = np.full(len(labels), PLAIN, dtype=np.int32)

    def install_textures(self) -> None:
        from newton._src.sensors.warp_raytrace.types import TextureData

        wp = self.wp
        cache, data = {}, []
        for name, span in TEXTURES:
            if name not in cache:
                cache[name] = wp.Texture2D(
                    make_texture(name), filter_mode=wp.TextureFilterMode.LINEAR,
                    address_mode=wp.TextureAddressMode.WRAP, normalized_coords=True,
                    dtype=wp.uint8, num_channels=4, device=self.ctx.device,
                )  # fmt: skip
            d = TextureData()
            d.texture = cache[name]
            d.repeat = wp.vec2f(1.0 / span, 1.0 / span)
            data.append(d)
        self._textures = cache  # keep the textures alive
        self.ctx.texture_data = wp.array(data, dtype=TextureData, device=self.ctx.device)
        self.ctx.shape_texture_ids = wp.array(self.texture_ids, dtype=wp.int32,
                                              device=self.ctx.device)  # fmt: skip

    def set_looks(self, looks: list[Look]) -> None:
        for i, look in enumerate(looks):
            for surface in ("floor", "table", "wall"):
                for k in self.shapes[surface][i]:
                    self.colours[k] = getattr(look, f"{surface}_rgb")
                    self.texture_ids[k] = getattr(look, f"{surface}_texture")
        self.model.shape_color.assign(self.colours.astype(np.float32))
        self.ctx.shape_texture_ids.assign(self.texture_ids)

    def set_sun(self, elevation_deg: float) -> None:
        d = sun_direction(elevation_deg)
        self.ctx.lights_orientation.assign(np.array([d], dtype=np.float32))

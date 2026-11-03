"""Seeded layouts and instructions of the stage 3 pick-and-place task (D-035).

All positions are in the environment's frame: the table's front edge runs along x = 0, the
table lies in x 0..0.6 m and y -0.6..0.6 m, its top 0.75 m up. A layout is drawn from one
integer seed and fixes the robot's start, the three objects on the table (of five), the two bins
at the table's ends, the target object and bin, and the instruction.

Seed ranges are frozen (D-035): train 0-49,999, development 50,000-50,999, test 60,000-60,999.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

SPLITS = {"train": (0, 50_000), "dev": (50_000, 51_000), "test": (60_000, 61_000)}

TABLE_TOP = 0.75
TABLE_SIZE = (0.6, 1.2, TABLE_TOP)  # depth (x), width (y), height
TABLE_CENTER = (0.3, 0.0)

# Round objects only (D-034): a can and a bottle were tested in the grasp test.
OBJECTS = {
    "red_can": {"kind": "can", "color": "red", "rgb": (0.80, 0.08, 0.08), "radius": 0.026,
                "height": 0.12},
    "yellow_can": {"kind": "can", "color": "yellow", "rgb": (0.95, 0.78, 0.10), "radius": 0.026,
                   "height": 0.12},
    "green_can": {"kind": "can", "color": "green", "rgb": (0.10, 0.55, 0.20), "radius": 0.026,
                  "height": 0.12},
    "blue_bottle": {"kind": "bottle", "color": "blue", "rgb": (0.15, 0.30, 0.85),
                    "radius": 0.022, "height": 0.14},
    "white_bottle": {"kind": "bottle", "color": "white", "rgb": (0.92, 0.92, 0.92),
                     "radius": 0.022, "height": 0.14},
}  # fmt: skip
OBJECT_NAMES = list(OBJECTS)

BINS = {
    "black_bin": {"color": "black", "rgb": (0.10, 0.10, 0.10)},
    "purple_bin": {"color": "purple", "rgb": (0.50, 0.20, 0.60)},
}
BIN_NAMES = list(BINS)
BIN_SIZE = (0.18, 0.22)  # outer x, y (m)
BIN_WALL = (0.01, 0.05)  # wall thickness, wall height (m); the floor is 1 cm thick
BIN_X, BIN_Y = 0.20, 0.45  # bin centres at the table's two ends

OBJECT_X = (0.12, 0.20)  # object centres, distance behind the table's front edge (m)
OBJECT_Y = (-0.30, 0.30)
OBJECT_SPACING = 0.12  # minimum distance between object centres (m)
OBJECT_MASS = (0.1, 0.5)  # kg

START_DISTANCE = (1.5, 3.0)  # pelvis to the table's front edge (m)
START_Y = (-1.0, 1.0)
START_YAW = (-math.pi / 2, math.pi / 2)  # 0 faces the table (+x)

TEMPLATES = (
    "put the {obj} in the {bin}",
    "place the {obj} into the {bin}",
    "pick up the {obj} and drop it in the {bin}",
    "move the {obj} to the {bin}",
    "take the {obj} and put it in the {bin}",
    "put the {obj} into the {bin}",
    "the {obj} goes in the {bin}",
    "grab the {obj} and place it in the {bin}",
    "drop the {obj} into the {bin}",
    "pick the {obj} up and put it in the {bin}",
)


@dataclass(frozen=True)
class Layout:
    seed: int
    robot_xy: tuple[float, float]
    robot_yaw: float
    objects: tuple[str, str, str]  # names of the three objects on the table
    object_xy: tuple[tuple[float, float], ...]
    object_mass: tuple[float, ...]
    bins: tuple[str, str]  # bin at y = -BIN_Y, bin at y = +BIN_Y
    target_object: str
    target_bin: str
    instruction: str

    def object_pos(self, name: str) -> tuple[float, float, float]:
        """Centre of an object on the table (env frame)."""
        x, y = self.object_xy[self.objects.index(name)]
        return (x, y, TABLE_TOP + OBJECTS[name]["height"] / 2)

    def bin_xy(self, name: str) -> tuple[float, float]:
        return (BIN_X, -BIN_Y if self.bins.index(name) == 0 else BIN_Y)

    def to_dict(self) -> dict:
        return asdict(self)


def describe(name: str) -> str:
    """How instructions name an object or a bin: "red can", "purple bin"."""
    if name in OBJECTS:
        return f"{OBJECTS[name]['color']} {OBJECTS[name]['kind']}"
    return f"{BINS[name]['color']} bin"


def split_of(seed: int) -> str:
    for name, (lo, hi) in SPLITS.items():
        if lo <= seed < hi:
            return name
    raise ValueError(f"seed {seed} is in no split")


def sample(seed: int) -> Layout:
    """The layout of one seed (deterministic, independent of numpy's global state)."""
    rng = np.random.default_rng([0x5EED_3, seed])
    d = rng.uniform(*START_DISTANCE)
    robot_xy = (-d, float(rng.uniform(*START_Y)))
    robot_yaw = float(rng.uniform(*START_YAW))
    objects = tuple(str(n) for n in rng.choice(OBJECT_NAMES, size=3, replace=False))
    xy: list[tuple[float, float]] = []
    while len(xy) < 3:  # rejection sampling; three objects fit easily in 0.6 m of table
        p = (float(rng.uniform(*OBJECT_X)), float(rng.uniform(*OBJECT_Y)))
        if all(math.dist(p, q) >= OBJECT_SPACING for q in xy):
            xy.append(p)
    mass = tuple(float(m) for m in rng.uniform(*OBJECT_MASS, size=3))
    bins = tuple(str(n) for n in rng.permutation(BIN_NAMES))
    target_object = objects[int(rng.integers(3))]
    target_bin = bins[int(rng.integers(2))]
    template = TEMPLATES[int(rng.integers(len(TEMPLATES)))]
    instruction = template.format(obj=describe(target_object), bin=describe(target_bin))
    return Layout(
        seed=seed,
        robot_xy=(float(robot_xy[0]), robot_xy[1]),
        robot_yaw=robot_yaw,
        objects=objects,  # type: ignore[arg-type]
        object_xy=tuple(xy),
        object_mass=mass,
        bins=bins,  # type: ignore[arg-type]
        target_object=target_object,
        target_bin=target_bin,
        instruction=instruction,
    )

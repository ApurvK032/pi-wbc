"""Record a GIF of a policy in MuJoCo, with a camera that follows the pelvis.

Off-screen rendering needs a GL backend chosen before MuJoCo is imported; on a headless
NVIDIA machine run with `MUJOCO_GL=egl`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import mujoco
import numpy as np

from pi_wbc.sim2sim.mujoco_g1 import G1MujocoSim


def record_gif(
    sim: G1MujocoSim,
    policy: Callable[[np.ndarray], np.ndarray],
    command: Sequence[float],
    path: str | Path,
    duration_s: float = 6.0,
    seed: int = 0,
    fps: int = 20,
    size: tuple[int, int] = (400, 264),
    colors: int = 64,
) -> Path:
    """Run one episode from the default pose and save every frame at `fps` as a GIF."""
    from PIL import Image

    rng = np.random.default_rng(seed)
    sim.reset(joint_offsets=rng.uniform(-0.05, 0.05, len(sim.policy_ids)))
    width, height = size
    renderer = mujoco.Renderer(sim.model, height=height, width=width)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = sim.base_body
    camera.distance, camera.elevation, camera.azimuth = 2.8, -12.0, 140.0

    policy_dt = sim.io.physics_dt * sim.io.decimation
    frames = []
    next_frame_t = 0.0  # capture on exact 1/fps marks so playback runs at real time
    for k in range(round(duration_s / policy_dt)):
        sim.step(policy(sim.observation(command)))
        if (k + 1) * policy_dt >= next_frame_t - 1e-9:
            renderer.update_scene(sim.data, camera=camera)
            frames.append(Image.fromarray(renderer.render()))
            next_frame_t += 1.0 / fps
    renderer.close()

    # One shared palette keeps the GIF small and stops colours flickering between frames.
    palette = frames[len(frames) // 2].quantize(colors=colors)
    frames = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=round(1000 / fps),
        loop=0,
        optimize=True,
    )
    return path

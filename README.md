<div align="center">

# π-WBC

**PPO whole-body control for the Unitree G1, trained in Isaac Lab, with a GR00T / π0.5 VLA on top.**

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![Isaac Lab 3.0](https://img.shields.io/badge/Isaac%20Lab-3.0-76B900.svg)](https://github.com/isaac-sim/IsaacLab)
[![MuJoCo](https://img.shields.io/badge/sim--to--sim-MuJoCo-orange.svg)](https://mujoco.org/)

</div>

> **Status: stage 1 (whole-body controller) in progress.** The G1 now walks like a person: a
> gait clock and Unitree's G1 rewards turn stage 0's fast shuffle into about 2.5 steps per
> second of 39 cm at 1 m/s, with no falls in 360 MuJoCo episodes across 3 training runs
> ([stage 1 results](results/stage1/README.md)). Step 2 adds height, torso and arm commands:
> the robot crouches, leans and keeps its balance while its arms move, but its walking and
> turning have regressed and are being worked on. Stage 0 (setup, sim-to-sim bench, cluster
> runs) is done ([stage 0 results](results/stage0/README.md)).

<p align="center">
  <img src="results/stage1/gait_v3_s1_walk_1.0.gif" alt="G1 walking at 1 m/s in MuJoCo" width="400"><br>
  <em>Stage 1 gait policy, trained in Isaac Lab, walking at 1 m/s in plain MuJoCo on MuJoCo
  Menagerie's G1 model.</em>
</p>

## What this is

A simulated Unitree G1 (29 DoF) walks to a table, picks up the named object and puts it in the
named bin, from a language instruction.

- **Low level: our whole-body controller.** A policy trained with PPO in Isaac Lab tracks base
  velocity and upper-body commands and keeps the robot balanced. It is trained with domain
  randomization, a curriculum and teacher-student distillation, then checked sim-to-sim in
  MuJoCo.
- **High level: a vision-language-action model.** GR00T N1.6, and π0.5 as a second model, reads
  the head camera and the instruction and outputs base velocity commands plus arm and hand
  targets for the controller.

**Research question.** When the robot fails, which layer is at fault: did the VLA choose wrong,
or did the controller slip, fall or miss the grasp? And does a more robust controller raise
whole-task success? NVIDIA's released G1 whole-body controller is the baseline: it goes under
the same VLA, through the same command interface.

## Roadmap

| Stage | What | Status |
|---|---|---|
| 0 | Setup: Isaac Lab training, MuJoCo sim-to-sim bench, cluster jobs | done |
| 1 | Whole-body controller: PPO, rewards, curriculum, domain randomization, teacher-student | in progress: gait done; height, torso and arm commands under way |
| 2 | Human motion imitation (retargeted motion capture) | planned |
| 3 | Pick-and-place task, scripted expert, rendered demonstrations | planned |
| 4 | VLA fine-tuning (GR00T N1.6, π0.5) and closed-loop evaluation | planned |
| 5 | Failure attribution study, controller swap, release | planned |

The full plan is in [docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md); design decisions and their
costs are in [docs/DECISIONS.md](docs/DECISIONS.md).

## Development setup

Needs [uv](https://docs.astral.sh/uv/), Python 3.12, Linux x86_64 and an NVIDIA GPU with
driver 580.65.06 or newer (CUDA 13).

```bash
uv sync                    # dev tools + Isaac Lab 3.0 (Newton backend); about 3.8 GB download
uv run pre-commit install
uv run pytest
```

For lint and unit tests only (no GPU, small download): `uv sync --no-group sim`.

## License

MIT, see [LICENSE](LICENSE). Upstream projects, models and datasets keep their own licenses;
they are credited here as they are used.

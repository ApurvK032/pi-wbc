# Decision log

Each entry says what was decided, why, and what it costs. Entries are numbered D-NNN and never
deleted; a later entry that replaces one says so, and the old one is marked superseded.

## D-001: Robot is the Unitree G1, 29 DoF (2026-10-28)

- **What:** a Unitree G1 with 29 body joints. Hands default to Dex3 (three fingers); the final
  hand choice is made before stage 3 (see open items).
- **Why:** the best-supported humanoid in open tooling: Isaac Lab assets and tasks, MuJoCo
  models, GR00T N1.6 support, NVIDIA's G1 whole-body controller (the baseline) and retargeted
  human motion data.
- **Rejected:** Apptronik Apollo. Almost no public tooling, no GR00T support and no retargeted
  motion data.
- **Cost:** none beyond being one robot among many; results are G1-specific.

## D-002: Train in Isaac Lab 3.0 (Newton backend), check transfer in MuJoCo (2026-10-28)

- **What:** train in Isaac Lab 3.0 on the Newton physics backend (NVIDIA Warp). Check transfer
  by running the exported ONNX policy in plain MuJoCo at the same control rate and conventions.
- **Why:** headless Newton simulation needs neither Isaac Sim nor RTX hardware, so the same code
  runs on a 12 GB RTX 4070 and on cluster GPUs. MuJoCo is the common open evaluation simulator
  for G1 policies.
- **Cost:** Newton support is new in 3.0. Both built-in G1 velocity tasks
  (`Isaac-Velocity-Flat-G1`, `Isaac-Velocity-Rough-G1`) default to Newton, so rough terrain
  doesn't need PhysX (checked in the v3.0.0 source). Newton's default solver is MuJoCo-Warp,
  which shares its contact model with MuJoCo, so a Newton → MuJoCo check is a smaller gap than
  PhysX → MuJoCo. Stage 1 must state which solver trained each policy, and a PhysX-trained run
  is a candidate for a harder transfer test. All transfer claims are sim-to-sim, never
  sim-to-real.

## D-003: Tooling is uv, Python 3.12, ruff, pytest, pre-commit (2026-10-28)

- **What:** one uv-managed project on Python 3.12; ruff for lint and format; pytest; pre-commit
  runs ruff from the project environment so there is one ruff version (from `uv.lock`).
- **Why:** Python 3.12 is what Isaac Lab 3.0 targets; uv gives a lockfile and fast installs on
  both the workstation and the cluster.
- **Cost:** none notable.

## D-004: NVIDIA's GR00T whole-body controller is the baseline (2026-10-28)

- **What:** NVIDIA GR00T-WholeBodyControl ("Decoupled WBC": RL lower body, IK upper body) is the
  controller our own is compared against, under the same VLA and command interface.
- **Why:** it is the controller GR00T N1.5 / N1.6 is used with on the G1, so it is the natural
  reference.
- **Cost:** some of its training scripts are not released, so it is compared as a released
  checkpoint and can't be retrained under our randomizations. Its weights are under the NVIDIA
  Open Model License, which must be checked before anything built on them is redistributed.

## D-005: Every result file carries provenance (2026-10-28)

- **What:** result files are written through `pi_wbc.provenance.save_json`, which adds the git
  commit and dirty flag, Python and package versions, host, and GPU name, driver and memory.
- **Why:** any number in the README must trace back to the code and machine that produced it.
- **Cost:** a few lines per result script.

## D-006: Isaac Lab is installed as a wheel into the project environment (2026-10-28)

- **What:** `isaaclab==3.0.0` from NVIDIA's package index and torch 2.11.0 built for CUDA 13.0
  from PyTorch's index, in the `sim` dependency group of this project's uv environment (on by
  default). No Isaac Lab source clone and no Isaac Sim.
- **Why:** one environment and one lockfile for our code and Isaac Lab, identical on the
  workstation and the cluster. The 3.0.0 wheel bundles the task packages, rsl-rl and the
  `isaaclab train` / `isaaclab play` commands. Both extra indexes are explicit, so each serves
  only the packages pinned to it and everything else comes from PyPI.
- **Cost:** about 3.8 GB to download. Isaac Lab's `scripts/` folder is not in the wheel, so we
  use the `isaaclab` command and our own scripts. Rendering with Isaac Sim 6.2 needs a separate
  install with Isaac Lab's dependency overrides (later, on the cluster).

## Open items

- Hands: Dex3 (default, recommended) or a five-finger hand such as Inspire. NVIDIA's controller
  ships G1 29-DoF + Dex3 scenes; Isaac Lab 3.0 has fixed-base G1 configs for both Dex3 and
  Inspire, but neither is set up for walking. Decide before stage 3.
- The built-in G1 velocity tasks use an older 37-joint G1 (`torso_joint`, two elbow joints,
  seven hand joints per side), not the 29-DoF robot. Stage 1 uses a 29-DoF G1 whose joint
  order matches the MuJoCo model (Isaac Lab's `G1_29DOF_CFG` or an MJCF imported into both).
- Isaac Lab 3.0 stores quaternions as (x, y, z, w); MuJoCo uses (w, x, y, z). The MuJoCo bench
  must convert and test this.
- GR00T N1.7 is now the main Isaac-GR00T release (N1.6 is on branch `n1d6`). N1.7 inference
  is listed at 16 GB+, above the RTX 4070. Choose the version before stage 4.
- Motion data: the LAFAN1 retargeting mirror used by BeyondMimic is CC BY-NC-ND 4.0 (no
  derivatives), which limits what processed clips can be released. Check before stage 2.

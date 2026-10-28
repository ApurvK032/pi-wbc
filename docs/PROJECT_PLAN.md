# Project plan

π-WBC (π is the reinforcement-learning symbol for a policy): PPO whole-body control for the
Unitree G1, trained in Isaac Lab, with a GR00T / π0.5 VLA on top.

## 1. The project

The final system: a simulated Unitree G1 walks to a table, picks up the named object and puts it
in the named bin, from a language instruction.

- **High level:** a VLA (GR00T N1.6, and π0.5 as a second model) reads the head camera and the
  instruction and outputs base velocity commands plus arm and hand targets.
- **Low level:** our own RL whole-body controller (PPO) tracks those commands and keeps the
  body balanced.

**Research question:** when the robot fails, which layer is at fault (the VLA chose wrong, or
the controller slipped, fell or missed the grasp)? And does a more robust controller (domain
randomization, curriculum, push recovery) raise whole-task success? NVIDIA's released G1
controller is the baseline: swap it in under the same VLA, through the same command interface,
and compare.

**Robot:** Unitree G1, 29 DoF. Hands: Dex3 three-finger hands by default. Before stage 3, check
which hands GR00T N1.6's G1 setup and NVIDIA's controller expect, and record the choice in
[DECISIONS.md](DECISIONS.md).

**Compute:** a workstation with an RTX 4070 (12 GB) for code, debugging, short training runs,
MuJoCo tests and VLA inference; the Minnesota Supercomputing Institute (MSI) cluster (A40, L40S,
A100, H100) for long PPO runs, motion imitation, rendering and VLA fine-tuning.

**Claims:** transfer results are sim-to-sim (Isaac Lab → MuJoCo). Nothing here is sim-to-real
unless a policy has run on real hardware.

## 2. Stages

Each stage ends with a result that stands on its own. Times are rough; about 6-7 weeks in all.

### Stage 0: Setup (3-4 days)

- Repository: uv, Python 3.12, ruff, pytest, pre-commit, MIT license, README, this plan and
  the decision log.
- Isaac Lab 3.0 on the workstation. Smoke test: a built-in G1 flat-terrain velocity task trains
  headless on the Newton backend for a few minutes; record steps/s and VRAM at several
  environment counts.
- MuJoCo sim-to-sim bench: the G1 in plain MuJoCo running an exported ONNX policy at the same
  control rate with the same conventions (joint order, default pose, PD gains, action scale,
  observation layout). Test the mapping.
- MSI: an Apptainer container for Isaac Lab (plus Isaac Sim 6.2 for rendering) and Slurm job
  scripts.
- **Done when** a G1 policy trains on the workstation and on MSI from the same commit, and the
  MuJoCo bench runs a policy.

### Stage 1: Whole-body controller (~1.5 weeks)

- Our own environment on Isaac Lab's G1 assets, with our own rewards (each term documented with
  its reason). Commands: base velocity (vx, vy, yaw rate), optionally body height, and
  upper-body targets (arm joints or wrist poses) so the VLA can drive it later.
- Curriculum: command ranges, push strength and terrain difficulty grow with performance.
- Domain randomization: friction, link masses and centre of mass, motor strength and PD gains,
  action delay, observation noise, pushes.
- Teacher-student: a teacher with privileged information (true velocities, friction, etc.)
  trained with PPO; a student that sees only proprioception and its history, trained by
  distillation (DAgger-style).
- Sim-to-sim: export the student to ONNX and run it in MuJoCo. Ablation: train with and without
  each randomization group and measure MuJoCo fall rate and command-tracking error on fixed
  seeds.
- **Done when** the student walks and tracks commands in Isaac Lab and in MuJoCo, with an
  ablation table (95% intervals) and GIFs.

### Stage 2: Human motion imitation (~1 week, can overlap stage 3)

- Retarget human motion capture to the G1. Unitree has published LAFAN1 retargeted to its
  robots on Hugging Face (to verify). AMASS requires registration and is non-commercial; users
  download it themselves.
- Train a BeyondMimic-style tracking policy (dance, kicks, walking styles) on MSI; check it in
  MuJoCo.
- **Done when** several clips are tracked in both simulators, with GIFs.

### Stage 3: Task and demonstrations (~1 week)

- Scene: G1, a table, objects, bins, head camera (and wrist cameras if the hands have them).
  Language instructions; seeded layouts split into frozen train, development and test ranges.
- Scripted expert: privileged state → base velocity and arm/hand targets → our controller. It
  should succeed on nearly all held-out layouts.
- Render demonstrations with randomized lighting and textures on MSI (Isaac Sim needs RT cores:
  L40S and A40 have them, A100 and H100 don't). Export in the formats GR00T N1.6 fine-tuning
  and π0.5 (openpi or LeRobot) expect.
- **Done when** the dataset passes a replay check (every episode replays), has a manifest, and
  the expert's success rate is reported.

### Stage 4: VLA (~1.5 weeks)

- Fine-tune GR00T N1.6 on a 48 GB card. Then π0.5 (LoRA needs >22.5 GB; a full fine-tune needs
  >70 GB, so an A100 80 GB or H100).
- Closed-loop evaluation on held-out layouts: on the workstation if inference fits in 12 GB,
  else on MSI.
- **Done when** success rates with Wilson intervals on frozen test layouts, failure labels and
  GIFs exist.

### Stage 5: Study and release (~1 week)

- Failure attribution from simulator ground truth: fell, slipped or missed the grasp →
  controller; wrong object, wrong bin or wandered → VLA.
- Controller swap: the same VLA on our controller vs NVIDIA's; our controller with vs without
  randomization and curriculum. Conditions: heavier objects, floor friction, pushes, sensor
  delay; MuJoCo transfer for the controller.
- Report, README with GIFs, checkpoints and dataset on Hugging Face, decision log.

### Optional later

- A world-model stage: learn the G1's dynamics from rollouts and predict falls ahead of time.
- Real sim-to-real on two Trossen WidowX AI arms, with a person at the e-stop for every real
  trial.

## 3. Ground rules for results

- Frozen train, development and test seed ranges.
- Success rates with 95% Wilson intervals; no rankings from too few runs.
- Every result file records the commit, package versions and GPU (`pi_wbc.provenance`).
- Decisions go in [DECISIONS.md](DECISIONS.md) as D-NNN entries: what, why, cost.

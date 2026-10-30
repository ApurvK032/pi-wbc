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

## D-007: Stage 0 bench policy uses the 29-DoF G1 with Dex3 hands (2026-10-28)

- **What:** task `PiWbc-G1-Flat-Stage0` (`src/pi_wbc/tasks/g1_flat_stage0.py`): Isaac Lab's
  built-in G1 flat velocity task moved onto `G1_29DOF_CFG` (29 body joints + 14 Dex3 hand
  joints). The policy sees and drives the 29 body joints; the hands are held at a fixed pose.
  Every joint uses a simulator-side PD controller with moderate, Unitree-style gains (legs
  100-150, ankles 40, waist 200, arms 40, wrists 20 N·m/rad). Default pose: Menagerie's
  standing pose for arms and hands, knees slightly bent. Rewards, commands and randomization
  stay as in the built-in task.
- **Why:** the built-in G1 tasks use an older 37-joint G1 that no MuJoCo model matches.
  Simulator-side PD is what MuJoCo position actuators reproduce exactly, which keeps the
  sim-to-sim comparison clean. `G1_29DOF_CFG`'s own gains (3000-5000 on arms and waist) are
  meant for an upper body steered by IK, not for walking.
- **Cost:** this is a bench policy, not the project's controller: stage 1 replaces its rewards,
  actuator model and randomization.

## D-008: Sim-to-sim protocol (2026-10-28)

- **What:** `pi-wbc-export` writes, from the live Isaac Lab environment, the ONNX policy,
  `policy_io.json` (joint order, default pose, gains, armature, effort limits, observation
  layout, timing, solver settings) and the MuJoCo model Newton builds for the task. It also
  checks that ONNX and PyTorch give the same actions on a rollout. `pi-wbc-sim2sim` then runs
  the policy in plain MuJoCo on two models:
  - **newton**: Newton's own exported model, i.e. the same robot description as training. Any
    gap here comes from the simulator (MuJoCo's collision detection and CPU solver instead of
    Newton's collision pipeline and MuJoCo-Warp).
  - **menagerie**: MuJoCo Menagerie's `g1_with_hands.xml`, an independently maintained
    description of the same robot. The extra gap here comes from the robot description.

  Both get the training's PD gains, armature, effort limits, 2.5 ms step (Newton runs 2
  substeps per 5 ms physics step), implicit-fast integrator, solver iterations and floor
  friction 1.0. Evaluation: 6 fixed velocity commands inside the training ranges, seeded
  starts (joint offsets ±0.05 rad, random heading, base velocity ±0.1 m/s), 20 s episodes. A
  fall is the torso touching the floor (Isaac Lab's termination) or the pelvis below 0.35 m.
  Tracking error is measured after 2 s, as the training rewards measure it. Fall rates carry
  95% Wilson intervals.
- **Why:** reading conventions from the live environment avoids hand-copied joint orders and
  gains, the usual source of sim-to-sim bugs. The two targets separate simulator effects from
  model-description effects. Unit tests pin down the frame conventions (centre-of-mass base
  velocity in the pelvis frame, quaternion order, projected gravity, action mapping).
- **Cost:** MuJoCo has no joint velocity limits, so those are not enforced on the MuJoCo side.
  No sensor noise, delay or pushes yet; stage 1 adds them.

## Findings (stage 0)

- **Friction on Newton:** Newton's MuJoCo-Warp solver uses the larger of the two touching
  surfaces' friction (like MuJoCo) and ignores Isaac Lab's `friction_combine_mode`. Isaac Lab's
  ground has friction 1.0, so randomizing the robot's friction below 1.0 has no effect. Stage 1
  must randomize the ground's friction instead. (Read from Newton's code; not yet confirmed in
  a running simulation.)
- **Isaac's G1 hands:** in Isaac Lab's 29-DoF G1 model, the left palm weighs 0.161 kg and the
  right 0.373 kg. Menagerie's hands are symmetric. This accounts for the whole 0.21 kg
  total-mass difference between the two models (34.18 vs 34.39 kg).
- **Foot contact:** both models touch the ground with four 5 mm spheres per foot, but
  Menagerie's heel spheres sit 3 cm further back (x = −0.05 vs −0.02 m), giving a longer
  support area.
- Joint limits agree to 1e-5 rad. Joint order differs only within the hands; joints are matched
  by name.
- **Bench policy gait:** with the built-in rewards, PPO learns a fast, low shuffle: about 11
  steps/s of 8 cm, 2-3 cm foot lift, feet sliding at about 0.35 m/s in stance, nearly straight
  knees, no arm swing (measured in MuJoCo at 1 m/s; all 13 runs of the setup do this, see
  `results/stage0/README.md` part 4). Stage 1's rewards
  must shape the gait: a stepping rhythm of about 1.5-2 steps/s, swing-foot clearance, stronger
  slip and energy penalties, arm swing. Its evaluation reports step rate, step length, foot
  lift and slip.
- **Cluster queue:** the interactive GPU partition used so far runs one job per user at a time
  (`AssocMaxJobsLimit`).
- **Training is not bit-reproducible.** Runs with the same seed, even on the same node, export
  different policies. Across 13 runs of the stage 0 setup, some measures spread widely: time
  with both feet down 11-57%, turn-in-place error 0.12-0.80 rad/s.
- **Newton's exported MJCF differs from Menagerie in contact settings:** every geom has a 1 cm
  `gap` (MuJoCo then lists force-free contacts up to 2 cm apart), and contacts are stiffer
  (time constant 1.8 ms requested, raised by MuJoCo to 5 ms, vs Menagerie's default 20 ms).
- **Cluster jobs must not write tracked files.** The stage 0 benchmark wrote a results file into
  the cluster clone; once that file was committed on the workstation, `git pull` on the cluster
  failed and the next 12 jobs ran old code (the planned reward pilot became 12 default runs).

## D-009: Cluster runs use an Apptainer container and the workstation's lockfile (2026-10-28)

- **What:** jobs on MSI run in an Apptainer image built from `msi/pi-wbc.def` (Ubuntu 24.04,
  Python 3.12, uv). The Python environment is created in the clone by `uv sync --frozen` inside
  the container, from the same `uv.lock` as the workstation. Code arrives by `git pull`; Slurm
  account and partition are given at submission, never stored in the repository; results come
  home by `rsync`. Commands are run by hand from short labelled blocks (docs/MSI.md).
- **Why:** MSI's Rocky Linux 8 has glibc 2.28, too old for several Isaac Lab 3.0 wheels (open3d,
  usd-exchange, omniverseclient need 2.34-2.35). The container fixes the OS; the shared lockfile
  fixes the packages, so the same commit trains the same way on both machines (stage 0 part 3).
  The image builds without admin rights and is small (151 MB) because the environment is not
  baked in.
- **Cost:** about 8 GB of environment per clone; the image must be rebuilt when the recipe
  changes. Jobs have no checkpoint-and-requeue logic yet (stage 0 runs take under 25 min).

## D-010: Gait measures in the MuJoCo evaluation (2026-10-28)

- **What:** `pi-wbc-sim2sim` reports, after the 2 s settling time and for episodes that did
  not fall: step rate, step length, swing time, foot lift, foot speed while in stance, time
  with both feet down and with none, knee range and arm swing (`src/pi_wbc/sim2sim/gait.py`).
  A foot is down when one of its contacts with the floor carries force; a step is a touchdown
  after at least 60 ms in the air; swings cut off by the start or end of recording are left
  out; joint ranges are 5th-95th percentile. Training changes are compared on the Menagerie
  model at a 1 m/s command.
- **Why:** stage 0's policies walk without falling but shuffle, and falls and tracking error
  don't show that. Stage 1's reward design needs a number for "walks like a person". Contact
  is defined by force so that a model's inactive contacts (Newton's `gap`) don't count.
- **Cost:** contact is sampled at the 50 Hz policy rate, so very short swings and contacts are
  approximate. Step length divides speed by step rate, so it is meaningless when standing.

## D-011: Training changes are compared against the run-to-run spread (2026-10-28)

- **What:** every training change is run at least twice, and its results are shown next to the
  lowest and highest run of the unchanged setup. A change counts as having an effect only if it
  moves a measure clearly outside that range. Rankings between changes need more runs.
- **Why:** GPU training is not bit-reproducible, and 13 runs of one setup differ by a factor of
  5 or more in some measures (see Findings). Single-run comparisons would mostly measure noise.
- **Cost:** two to three times the cluster time per change. The 13 default runs give the
  baseline range at no extra cost.

## Open items

- The stage 1 design is a draft under review: [STAGE1_DESIGN.md](STAGE1_DESIGN.md). Its
  section 15 lists seven decisions; the two items below are among them.
- Hands: Dex3 (default, recommended) or a five-finger hand such as Inspire. NVIDIA's controller
  ships G1 29-DoF + Dex3 scenes; Isaac Lab 3.0 has fixed-base G1 configs for both Dex3 and
  Inspire, but neither is set up for walking. NVIDIA's Decoupled WBC and IsaacLab-Arena both
  use three-finger hands with Dex3's joint layout (checked 2026-10-29). Decide before stage 3.
- The built-in G1 velocity tasks use an older 37-joint G1 (`torso_joint`, two elbow joints,
  seven hand joints per side), not the 29-DoF robot. Stage 0's bench already uses
  `G1_29DOF_CFG` (D-007); stage 1 keeps it.
- GR00T N1.7 is now the main Isaac-GR00T release (N1.6 is on branch `n1d6`). N1.7 inference
  is listed at 16 GB+, above the RTX 4070. N1.7's G1 path uses the SONIC controller, which takes
  latent motion tokens rather than velocity commands, so it can't be swapped under our command
  interface (checked 2026-10-29). Choose the version before stage 4.
- Motion data: the LAFAN1 retargeting mirror used by BeyondMimic is CC BY-NC-ND 4.0 (no
  derivatives), which limits what processed clips can be released. Check before stage 2.

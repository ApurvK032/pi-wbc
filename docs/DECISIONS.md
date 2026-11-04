# Decision log

Each entry says what was decided, why, and what it costs. Entries are numbered D-NNN and never
deleted; a later entry that replaces one says so, and the old one is marked superseded.

**Index** (stage 0: D-001 to D-011; stage 1: D-012 to D-029; stage 3: D-030 onwards, with
D-039 and D-040 correcting stage 1's MuJoCo bench; open items at the [end](#open-items)):

| Decision | What | Date |
|---|---|---|
| [D-001](#d-001-robot-is-the-unitree-g1-29-dof-2026-10-28) | Robot is the Unitree G1, 29 DoF | 2026-10-28 |
| [D-002](#d-002-train-in-isaac-lab-30-newton-backend-check-transfer-in-mujoco-2026-10-28) | Train in Isaac Lab 3.0 (Newton backend), check transfer in MuJoCo | 2026-10-28 |
| [D-003](#d-003-tooling-is-uv-python-312-ruff-pytest-pre-commit-2026-10-28) | Tooling is uv, Python 3.12, ruff, pytest, pre-commit | 2026-10-28 |
| [D-004](#d-004-nvidias-gr00t-whole-body-controller-is-the-baseline-2026-10-28) | NVIDIA's GR00T whole-body controller is the baseline | 2026-10-28 |
| [D-005](#d-005-every-result-file-carries-provenance-2026-10-28) | Every result file carries provenance | 2026-10-28 |
| [D-006](#d-006-isaac-lab-is-installed-as-a-wheel-into-the-project-environment-2026-10-28) | Isaac Lab is installed as a wheel into the project environment | 2026-10-28 |
| [D-007](#d-007-stage-0-bench-policy-uses-the-29-dof-g1-with-dex3-hands-2026-10-28) | Stage 0 bench policy uses the 29-DoF G1 with Dex3 hands | 2026-10-28 |
| [D-008](#d-008-sim-to-sim-protocol-2026-10-28) | Sim-to-sim protocol (amended by D-040) | 2026-10-28 |
| [D-009](#d-009-cluster-runs-use-an-apptainer-container-and-the-workstations-lockfile-2026-10-28) | Cluster runs use an Apptainer container and the workstation's lockfile | 2026-10-28 |
| [D-010](#d-010-gait-measures-in-the-mujoco-evaluation-2026-10-28) | Gait measures in the MuJoCo evaluation | 2026-10-28 |
| [D-011](#d-011-training-changes-are-compared-against-the-run-to-run-spread-2026-10-28) | Training changes are compared against the run-to-run spread | 2026-10-28 |
| [D-012](#d-012-stage-1-controller-interface-follows-nvidias-decoupled-wbc-2026-10-29) | Stage 1 controller interface follows NVIDIA's Decoupled WBC | 2026-10-29 |
| [D-013](#d-013-hands-are-dex3-2026-10-29) | Hands are Dex3 | 2026-10-29 |
| [D-014](#d-014-gr00t-n16-for-stage-4-2026-10-29) | GR00T N1.6 for stage 4 | 2026-10-29 |
| [D-015](#d-015-stage-1-scope-and-compute-2026-10-29) | Stage 1 scope and compute | 2026-10-29 |
| [D-016](#d-016-slip-is-measured-at-the-soles-floor-contacts-2026-10-29) | Slip is measured at the sole's floor contacts | 2026-10-29 |
| [D-017](#d-017-how-step-2-carries-out-the-body-and-arm-commands-2026-10-30) | How step 2 carries out the body and arm commands | 2026-10-30 |
| [D-018](#d-018-how-step-3-randomizes-and-how-its-groups-are-switched-off-2026-10-30) | How step 3 randomizes, and how its groups are switched off | 2026-10-30 |
| [D-019](#d-019-the-student-trains-at-its-teachers-curriculum-level-held-fixed-2026-10-31) | The student trains at its teacher's curriculum level, held fixed (warm-up added by D-021) | 2026-10-31 |
| [D-020](#d-020-the-two-ablation-configurations-that-are-not-randomization-2026-10-31) | The two ablation configurations that are not randomization | 2026-10-31 |
| [D-021](#d-021-the-student-warms-up-to-its-teachers-level-over-300-iterations-2026-10-31) | The student warms up to its teacher's level over 300 iterations | 2026-10-31 |
| [D-022](#d-022-a-narrow-speed-tracking-reward-at-weight-05-fixes-the-speed-miss-2026-11-01) | A narrow speed-tracking reward at weight 0.5 fixes the speed miss | 2026-11-01 |
| [D-023](#d-023-test-the-controller-on-isaac-sims-physx-as-a-second-physics-engine-2026-11-01) | Test the controller on Isaac Sim's PhysX as a second physics engine | 2026-11-01 |
| [D-024](#d-024-take-the-b2-teachers-on-to-the-full-command-ranges-before-distilling-2026-11-01) | Take the B2 teachers on to the full command ranges before distilling (lengthened by D-028) | 2026-11-01 |
| [D-025](#d-025-all-rights-reserved-and-a-map-for-the-old-commit-hashes-2026-11-01) | All rights reserved, and a map for the old commit hashes | 2026-11-01 |
| [D-026](#d-026-the-student-sees-the-last-25-control-steps-05-s-2026-11-02) | The student sees the last 25 control steps (0.5 s) | 2026-11-02 |
| [D-027](#d-027-the-narrow-turning-reward-stays-off-2026-11-02) | The narrow turning reward stays off | 2026-11-02 |
| [D-028](#d-028-train-the-full-range-teachers-7000-iterations-instead-of-1000-2026-11-03) | Train the full-range teachers 7000 iterations instead of 1000 | 2026-11-03 |
| [D-029](#d-029-stage-1-closes-with-the-turning-target-met-on-one-seed-of-three-2026-11-03) | Stage 1 closes with the turning target met on one seed of three (amended by D-040: met on none) | 2026-11-03 |
| [D-030](#d-030-stage-3-before-stage-2-2026-11-03) | Stage 3 before stage 2 | 2026-11-03 |
| [D-031](#d-031-newtons-renderer-for-the-demonstrations-recorded-as-states-and-rendered-afterwards-2026-11-03) | Newton's renderer for the demonstrations, recorded as states and rendered afterwards | 2026-11-03 |
| [D-032](#d-032-the-perfect-layer-grid-for-stage-5-and-demonstrations-recorded-with-our-controller-2026-11-03) | The perfect-layer grid for stage 5, and demonstrations recorded with our controller | 2026-11-03 |
| [D-033](#d-033-dex3-confirmed-for-gr00t-n16-hands-move-between-open-and-closed-poses-2026-11-03) | Dex3 confirmed for GR00T N1.6; hands move between open and closed poses | 2026-11-03 |
| [D-034](#d-034-the-grasp-a-side-grasp-from-above-round-objects-4-physics-substeps-2026-11-03) | The grasp: a side grasp from above, round objects, 4 physics substeps | 2026-11-03 |
| [D-035](#d-035-the-task-one-table-three-round-objects-two-bins-frozen-seed-ranges-2026-11-03) | The task: one table, three round objects, two bins, frozen seed ranges | 2026-11-03 |
| [D-036](#d-036-the-scripted-expert-a-state-machine-over-the-true-state-no-retries-2026-11-03) | The scripted expert: a state machine over the true state, no retries | 2026-11-03 |
| [D-037](#d-037-the-dataset-1000-episodes-per-controller-head-camera-replay-check-2026-11-03) | The dataset: 1,000 episodes per controller, head camera, replay check | 2026-11-03 |
| [D-038](#d-038-the-g1s-heel-contact-spheres-at-unitrees-urdf-position-from-stage-3-on-2026-11-03) | The G1's heel contact spheres at Unitree's URDF position from stage 3 on | 2026-11-03 |
| [D-039](#d-039-erratum-on-menageries-model-the-mujoco-benchs-feet-stood-on-friction-06-2026-11-03) | Erratum: on Menagerie's model the MuJoCo bench's feet stood on friction 0.6 (amended by D-040) | 2026-11-03 |
| [D-040](#d-040-seven-more-mujoco-bench-fixes-and-stage-1s-bench-numbers-rerun-on-the-fixed-bench-2026-11-04) | Seven more MuJoCo-bench fixes, and stage 1's bench numbers rerun on the fixed bench | 2026-11-04 |
| [D-041](#d-041-the-experts-hand-choice-retract-and-success-rule-and-its-official-test-range-2026-11-04) | The expert's hand choice, retract and success rule, and its official test range | 2026-11-04 |
| [D-042](#d-042-the-replay-check-tests-each-episodes-first-second-and-the-training-sets-seeds-2026-11-04) | The replay check tests each episode's first second, and the training set's seeds | 2026-11-04 |
| [D-043](#d-043-how-the-head-camera-video-is-rendered-and-randomized-2026-11-04) | How the head-camera video is rendered and randomized (lens amended by D-044) | 2026-11-04 |
| [D-044](#d-044-the-head-cameras-lens-is-87-degrees-wide-2026-11-04) | The head camera's lens is 87 degrees wide | 2026-11-04 |
| [D-045](#d-045-the-datasets-export-format-gr00t-flavoured-lerobot-v21-and-lerobot-v30-2026-11-04) | The dataset's export format: GR00T-flavoured LeRobot v2.1, and LeRobot v3.0 | 2026-11-04 |
| [D-046](#d-046-nvidias-controller-as-the-tasks-low-level-2026-11-04) | NVIDIA's controller as the task's low level, and its matched set | 2026-11-04 |
| [D-047](#d-047-training-set-v2-and-a-validation-set-2026-11-04) | Training set v2 and a validation set | 2026-11-04 |
| [D-048](#d-048-50-frames-per-second-versions-of-the-datasets-2026-11-04) | 50 frames-per-second versions of the datasets | 2026-11-04 |
| [D-049](#d-049-results-record-the-code-their-run-started-with-2026-11-04) | Results record the code their run started with | 2026-11-04 |
| [D-050](#d-050-stage-3-closes-2026-11-04) | Stage 3 closes | 2026-11-04 |

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
  friction 1.0 (on Menagerie's model the feet had 0.6 until D-039 corrected it). Evaluation: 6 fixed velocity commands inside the training ranges, seeded
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
- **Amended by D-040 (2026-11-04):** the bench reads the end of each policy step (`mj_forward`
  after the physics steps), every geom has contact priority 0 as in Newton's training model,
  episodes with kicks run 23 s, and the 20 ms delay holds only the joints the policy drives.

## Findings (stage 0)

- **Friction on Newton:** Newton's MuJoCo-Warp solver uses the larger of the two touching
  surfaces' friction (like MuJoCo) and ignores Isaac Lab's `friction_combine_mode`. Isaac Lab's
  ground has friction 1.0, so randomizing the robot's friction below 1.0 has no effect. Stage 1
  sets the ground low and randomizes the robot's friction instead. (Read from Newton's code;
  seen in a running simulation in stage 1, D-018.)
- **Isaac's G1 hands:** in Isaac Lab's 29-DoF G1 model, the left palm weighs 0.161 kg and the
  right 0.373 kg. Menagerie's hands are symmetric. This accounts for the whole 0.21 kg
  total-mass difference between the two models (34.18 vs 34.39 kg).
- **Foot contact:** both models touch the ground with four 5 mm spheres per foot, but
  Menagerie's heel spheres sit 3 cm further back (x = −0.05 vs −0.02 m), giving a longer
  support area.
- Joint limits agree to 1e-5 rad. Joint order differs only within the hands; joints are matched
  by name.
- **Bench policy gait:** with the built-in rewards, PPO learns a fast, low shuffle: about 11
  steps/s of 8 cm, 2-3 cm foot lift, ankles moving at about 0.3 m/s in stance (0.14 m/s of
  slip at the sole, D-016), nearly straight
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
- **Isaac Lab holds undriven joints at 0.** Joints that no action term drives keep a PD
  target of 0 unless the task sets one, not the default pose. Stage 0's hands and the first
  stage 1 gait policies' arms were held at 0 in training but at the default pose in the
  MuJoCo bench; the bench now uses the targets the export records, and the stage 1 task holds
  arms and hands at the default pose with a reset event. Stage 0's numbers were re-run (small
  changes, same conclusions).
- **Runs must not pick up code edited while they go on.** `train_eval_local.sh` exported and
  evaluated each run with whatever code was on disk when training finished. One export
  (step 2, v4) picked up a half-written task change and was tested with leg offsets its
  policy never trained with (10 of 10 falls when crouching; re-exported from the training
  commit: 0 falls). The script now runs every step from a copy of the starting commit.
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

- **What:** `pi-wbc-sim2sim` reports, after the 2 s settling time and for episodes that did not
  fall: step rate, step length, swing time, foot lift, foot speed while in stance (for slip,
  see D-016), time with both feet down and with none, knee range and arm swing
  (`src/pi_wbc/sim2sim/gait.py`). A foot is down when one of its contacts with the floor
  carries force; a step is a touchdown after at least 60 ms in the air; swings cut off by the
  start or end of recording are left out; joint ranges are 5th-95th percentile. Training
  changes are compared on the Menagerie model at a 1 m/s command.
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

## D-012: Stage 1 controller interface follows NVIDIA's Decoupled WBC (2026-10-29)

- **What:** the controller takes NVIDIA's Decoupled WBC commands: navigate (forward and
  sideways speed, yaw rate), pelvis height, torso roll, pitch and yaw, and 7 + 7 arm and
  7 + 7 hand joint targets. The RL policy drives the 12 leg and 3 waist joints; arm and hand
  targets go straight to their PD controllers. The simulated robot uses NVIDIA's PD gains and
  default leg pose. Details: [STAGE1_DESIGN.md](STAGE1_DESIGN.md), sections 3-4.
- **Why:** this is the interface GR00T N1.6 uses with NVIDIA's controller on the G1, so either
  controller can sit under the same VLA, and swapping them changes only the lower body. HOMIE,
  AMO and NVIDIA's WBC-AGILE split the robot the same way. Arm accuracy does not depend on RL.
- **Cost:** the RL policy never moves the arms, so "whole-body" here means legs and waist
  balancing against whatever the arms do, not learned arm motion (FALCON's approach was
  considered and not chosen). The stage 0 gains are replaced.

## D-013: Hands are Dex3 (2026-10-29)

- **What:** the G1 keeps Unitree's Dex3 three-finger hands (7 joints each) for all stages.
  Settles the open item from D-001.
- **Why:** NVIDIA's Decoupled WBC and IsaacLab-Arena's G1 both use three-finger hands with
  Dex3's joint layout, and the GR00T N1.6 G1 action space has 7 values per hand.
- **Cost:** no five-finger dexterity; grasps stay simple.

## D-014: GR00T N1.6 for stage 4 (2026-10-29)

- **What:** the VLA is GR00T N1.6 (Isaac-GR00T's `n1.6.1-release`), not N1.7. Settles the open
  item on the GR00T version.
- **Why:** N1.7's G1 path drives NVIDIA's SONIC controller with latent motion tokens, which
  has no command interface to swap another controller under. N1.6 sends the D-012 commands.
- **Cost:** not the newest model. Since NVIDIA's pretrained G1 data comes from its own
  controller, fine-tuning uses the `NEW_EMBODIMENT` tag on our own demonstrations.

## D-015: Stage 1 scope and compute (2026-10-29)

- **What:** stage 1 trains on flat ground only. Before the ablation (about 45 GPU-hours), the
  cluster's partition limits are checked with read-only commands to decide between queueing
  on the current partition, a partition with parallel jobs, or longer workstation runs.
- **Why:** the stage 3 task is indoors on a flat floor, and Newton's rough-terrain support is
  newer. The ablation is the largest compute item of the project so far.
- **Cost:** no rough-terrain robustness claims from stage 1.

## D-016: Slip is measured at the sole's floor contacts (2026-10-29)

- **What:** the MuJoCo evaluation's slip measure (`sole_slip_mps`, "sole slip") is the planar
  speed of the foot itself at each of its floor contacts that carry force, averaged over that
  foot's contacts with their normal forces as weights, over the samples where the foot is
  down. The stage 1 slip target (at most 0.1 m/s) applies to it. D-010's measure, the planar
  speed of the ankle-roll link's origin while the foot is down (`stance_foot_speed_mps`,
  "stance foot speed"), is still reported for comparison with earlier results.
- **Why:** the ankle sits 3.5 cm above the sole, so it moves whenever the foot rolls from heel
  to toe or pivots, without any sliding. With the robot pivoting on its toe at 2 rad/s, the
  ankle measure reads 0.07 m/s and the sole measure 0.0002 m/s (a unit test). On the stage 1
  gait at 1 m/s (Menagerie model), the ankle measure gives 0.15-0.21 m/s and the sole measure
  0.04-0.10 m/s: more than half of what had been read as slip was rolling. Re-running all 26
  evaluated policies with the new measure changed no other number.
- **Cost:** the stage 1 target is now easier to meet than it was as first measured, so both
  numbers are reported side by side. The training penalty `feet_contact_velocity` also uses
  the ankle's velocity (including its vertical part), so it discourages rolling as well as
  sliding; whether that shapes the gait badly is open for the later reward work.

## D-017: How step 2 carries out the body and arm commands (2026-10-30)

- **What:** `PiWbc-G1-Body` (`src/pi_wbc/tasks/g1_body.py`) adds to the gait task:
  - a body-pose command of pelvis height (0.50-0.78 m) and the three waist angles (yaw ±0.6,
    roll ±0.2, pitch 0-0.45 rad), moving at most 0.25 m/s and 1 rad/s towards a new goal every
    4-8 s;
  - random arm and hand motion written straight to those joints' PD targets: a new pose every
    2-4 s, reached in a straight line;
  - 0-1 kg added to each palm, drawn once per robot at startup;
  - height rewarded like velocity: exp(-error² / 0.1²), weight 1, on top of the squared
    penalty (weight -10) that step 1 used for its fixed height;
  - a curriculum that widens both from nothing to the full ranges, in steps of 5%, judged
    over 20-iteration windows (raise when under 10% of ended episodes fell, speed tracking is
    over 60% of its reward and turn-rate tracking over 35%; lower when over 25% fell).
- **Why, where it departs from the design:**
  - The torso command is the waist joint angles: the G1's waist turns in the order yaw, roll,
    pitch, so these angles are the torso's orientation relative to the pelvis.
  - Waist pitch stops at 0.45 rad, not 0.6: the joint's limit is 0.52 rad, and the joint-limit
    penalty starts at 90% of it.
  - "Stand still" penalizes leg joint speed instead of the offset from the default pose,
    which would have penalized crouching on command.
  - Arm ranges keep the arms beside and in front of the body. Training has no self-collisions
    (Isaac Lab's G1 asset turns them off, on PhysX and Newton); MuJoCo Menagerie's model has
    them.
  - Payload once per robot: changing masses at every reset made Newton training 2.3 times
    slower (2.7 instead of 1.2 s per iteration on the workstation).
  - Curriculum windows: the first version judged each step's few resetting robots on their
    own, so the level moved in fits and started late (iteration 750 of 1000).
  - Leg actions centre on a crouch pose that follows the commanded height (hip, knee and
    ankle angles from `scripts/crouch_table.py`: feet flat, pelvis level, centre of mass over
    the feet; zero offset at the nominal height). Without it, two runs never held a crouch: a
    0.60 m crouch needs knee actions of about 5 at the action scale of 0.25. With the height
    bonus alone the robot still stood at its nominal height (17 cm short of a 0.60 m command
    in MuJoCo).
  - Arm poses every 2-4 s rather than 1-3 s: the faster motion (shoulders up to 2.5 rad/s,
    with up to 1 kg in the hand) went with turning in place being unlearned as the curriculum
    widened (training yaw-rate error 0.65 to 1.05 rad/s).
  - Turning in the curriculum's criteria: in runs v3-v6 the ranges widened while turning in
    place was being unlearned (turn-rate tracking fell from 0.39 to 0.26 per second, speed
    tracking held).
  - Height bonus: with the squared penalty alone, a 1400-iteration run never crouched (its
    training height error, 10 cm, was what standing still gives), while the waist, rewarded
    with the exp form, was tracked.
- **Cost:** a robot keeps its payload for the whole run (variety comes from 4096 robots, not
  from episodes). The torso command must be converted if a VLA sends torso roll, pitch and yaw
  in another convention (stage 3).

## D-018: How step 3 randomizes, and how its groups are switched off (2026-10-30)

- **What:** `PiWbc-G1-Robust` (`src/pi_wbc/tasks/g1_robust.py`) is the body task plus six
  randomization groups: friction (0.1-1.25, one value per robot for all its shapes, on a 0.05
  ground), mass (torso -1 to
  +3 kg, links ×0.9-1.1, torso centre of mass ±3 cm, the hand payload), actuators (stiffness,
  damping, armature ×0.8-1.2), latency (0-4 physics steps between a new action and its
  targets, drawn per robot at each reset), sensor noise (the base task's, Unitree's values),
  and pushes (every 4-6 s, ±0.5 m/s growing to ±1.5 m/s with the upper-body curriculum's
  level). Masses, centre of mass and gains are drawn once per robot at startup.
  `ABLATION_OVERRIDES` gives, per group, the config overrides that switch it off;
  `make_nominal` switches all of them off, and the export uses it.
- **Why:**
  - Overrides, not switches read in `__post_init__`: Isaac Lab applies config overrides after
    `__post_init__`, so each group is removed by deleting its event terms or setting a value
    the simulation reads at run time.
  - Startup, not reset, for masses and gains: per-reset mass changes made Newton 2.3 times
    slower (D-017); 4096 robots give the variety.
  - Joint friction is not randomized: it is 0 in this model, so scaling it changes nothing.
  - One friction per robot (`randomize_robot_friction`): Isaac Lab's material event draws one
    value per collision shape, and a G1 foot touches the ground through four spheres, so each
    foot mixed four draws: over 16 robots the mean foot friction only spanned 0.43-0.94 of the
    0.1-1.25 range. Per robot, 256 robots span 0.10-1.24.
  - Nominal export: the export builds one robot, and with the groups on it would record that
    robot's randomized gains and masses as the policy's robot.
- **Checked in a running simulation** (`scripts/check_randomization.py`, results in
  `results/stage1/randomization_check/`): masses, centre of mass and gains read back inside
  their ranges; PD targets switch after exactly the drawn delay; pushes set at most the
  configured speed; the centre-of-mass change keeps the state finite. Friction: kicked
  sideways, feet slide 8.7-9.2 cm in 0.1 s at robot friction 0.1 on the 0.05 ground against
  3.7-4.8 cm at 1.0 or 1.25, and a 0.1 robot on a 1.0 ground slides like a 1.0 robot on the
  0.05 ground, as Newton taking the larger of the two values predicts (the stage 0 finding,
  now seen in simulation).
- **Cost:** a robot keeps its masses and gains for the whole run. Latency and noise are not
  yet in the MuJoCo bench (step 5's test conditions).

## D-019: The student trains at its teacher's curriculum level, held fixed (2026-10-31)

- **What:** distillation (`PiWbc-G1-Student`) starts at the teacher's last curriculum level, as
  before, and now keeps it: the upper-body curriculum's thresholds are set out of reach
  (`raise_below=-1`, `lower_above=2`), so the body-pose and arm ranges and the push speed stay
  at the teacher's. `scripts/teacher_student_local.sh` and `msi/jobs/teacher_student.sbatch`
  pass both overrides to the student. The local script also reuses a teacher whose last
  checkpoint is already there.
- **Why:** a distilled student starts as an untrained network, so in its first iterations
  nearly every episode ends in a fall, and the curriculum, built for the teacher, lowered the
  ranges in response. Whether they came back depended on how fast distillation went. The full
  student (0.85 at the start) and the one trained without randomization (1.0) recovered by
  iteration 200. The one trained without pushes (1.0) learned more slowly, its level reached 0
  at iteration 500 and stayed there with 17-24% falls, and it never trained a crouch or a
  moving arm: in Isaac Lab and in both MuJoCo models it fell in every crouch, low-walk and
  arm-motion episode, while its teacher did them (`results/stage1/ablation/isaac/`). An
  ablation needs every configuration's student to learn the same commands, and the usual
  teacher-student setup trains the student on the teacher's command distribution.
- **Cost:** students trained before this (the seed 1 full student, the MSI seeds 2 and 3, the
  first two ablation runs) are redone from their teachers: about 35 min each on the
  workstation, teachers unchanged. Results from the first protocol are kept in
  `results/stage1/ablation/first_protocol/`.

## D-020: The two ablation configurations that are not randomization (2026-10-31)

- **What:** "no curriculum" (`g1_robust.TRAINING_ABLATIONS["curriculum"]`) trains the teacher
  at full body-pose and arm ranges and the full 1.5 m/s push speed from the first step, with
  the curriculum held there (the D-019 mechanism); its student follows as usual. "No teacher"
  (`PiWbc-G1-Student-PPO`) trains the student's network ([512, 256, 128]) on the student's
  inputs with PPO for 4000 iterations, with the curriculum as the teacher has it, and a critic
  that sees the teacher group.
- **Why:** level 1 from the start is what the curriculum would otherwise reach, so the
  comparison isolates the ordering. 4000 iterations is the teacher's 3000 plus the student's
  1000, the same number of training iterations. The privileged critic (an asymmetric
  actor-critic) is the strongest common way to train a policy on limited inputs without a
  teacher, so "teacher-student is worth it" has to beat it, not a plain PPO that also limits
  the critic.
- **Cost:** a distillation iteration is cheaper than a PPO one, so the PPO student uses
  somewhat more compute than teacher plus student; reported with the results.

## D-021: The student warms up to its teacher's level over 300 iterations (2026-10-31)

- **What:** distillation starts at curriculum level 0 and raises it on a fixed schedule to the
  teacher's last level over the first 7200 environment steps (300 of the 1000 iterations),
  then holds it there (`mdp.upper_body_level(..., ramp_steps=7200)`; the push speed follows the
  level). The thresholds stay out of reach as in D-019, so the student's own falls never move
  the level. `scripts/teacher_student_local.sh` (`RAMP_STEPS`, default 7200; 0 gives D-019's
  held level), `msi/jobs/teacher_student.sbatch` and `msi/jobs/ablation.sbatch` pass it. The
  no-curriculum ablation keeps `ramp_steps=0`, since a warm-up is itself a curriculum.
- **Why:** with the level held from the start (D-019), the students of the same teachers walked
  worse than under the first protocol and varied more by seed. Speed error at 1 m/s on the
  development seeds (mean of both MuJoCo models; training seeds 1, 2, 3): first protocol 0.24,
  0.21, 0.18; held 0.27, 0.35, 0.15; warm-up 0.17, 0.23, 0.21. Turning: 0.20, 0.23, 0.18;
  0.18, 0.32, 0.17; 0.20, 0.23, 0.22. The warm-up matches the first protocol on average (0.21
  at 1 m/s; 0.12 against 0.14 at 0.5 m/s; 0.15 sideways for both) without its flaw: the level
  no longer depends on how fast the student learns, which is what left the no-pushes student
  at level 0. Falls are rare under all three (without kicks: first protocol 1, held 0,
  warm-up 3 of 3300 episodes; 2 m/s kicks: 33, 30 and 35 of 660). Three seeds don't separate
  the recipes statistically (held minus warm-up, per seed: +0.10, +0.12, -0.06 at 1 m/s); the
  choice rests on the warm-up matching the first protocol, having no seed as bad as the held
  recipe's seed 2, and being the same for every configuration. None of the three meets the
  1 m/s target (0.15 or less), so the recipe does not fix the speed miss.
- **Test seeds (added the same day; numbers on the fixed bench of D-040):** on test seeds
  1000-1199 (4400 episodes per seed and condition) every recipe's walking repeats its
  development seeds within 0.02 m/s, but the warm-up students fall more to kicks on seed 1, and
  at 1 m/s on seed 2. Falls with 1 m/s kicks, seeds 1 / 2 / 3: first protocol 40 / 113 / 93,
  held 14 / 39 / 116, warm-up 90 / 146 / 85; with 2 m/s kicks: 80 / 342 / 227, 49 / 236 / 263,
  254 / 317 / 235 (pooled: 1.9%, 1.3% and 2.4% of episodes at 1 m/s; 4.9%, 4.2% and 6.1% at
  2 m/s). Seed 3's warm-up student falls least of its three at 1 m/s, so three seeds don't
  settle this either. A likely cause, not checked: the kicks' top speed follows the level, so
  the warm-up student meets its teacher's full kicks (1.30-1.35 m/s) only from iteration 300
  and trains 700 iterations at them instead of 1000. Without kicks the recipes fall 92, 16 and
  70 times in 66000 episodes, nearly all with the arms moving, and most of them on seed 2 under
  the first protocol and the warm-up. The ablation keeps the warm-up, so its configurations
  stay comparable to each other.
- **Kicks at full size from the start (tried the same day, seed 1, not adopted):**
  `upper_body_level(..., ramp_pushes=False)` warms up the command ranges only and sets the kicks
  to the teacher's size from the first step (off by default). Seed 1's student, distilled again
  from the same teacher, on the test seeds (fixed bench): falls with 1 m/s kicks 62 (warm-up 90,
  held 14), with 2 m/s kicks 147 (254, 49), without kicks 12 of 22000 (6, 5). So the kick ramp
  explains part of the warm-up's kick weakness, not all of it. It walked worst of the four
  recipes on seed 1: 0.36 m/s off at 1 m/s (warm-up 0.16, held 0.27), 0.19 at 0.5 m/s,
  0.34 rad/s turning (0.22, 0.19), with the same final level and distillation loss as the
  warm-up student. One seed can't show that it
  walks worse in general (the held recipe's seed 2 was as far off), but it gives no reason to
  redo the students; the warm-up stays. Cost: 21 min of training and 1.5 h of MuJoCo tests on
  the workstation. Results: `results/stage1/student_ramp_kick/`,
  `results/stage1/student_test_ramp_kick/`.
- **Cost:** the ablation students trained with the held level (no pushes and no randomization
  on the workstation; friction, mass, actuators and sensor noise on MSI) are redone from their
  teachers, about 35 min each on the workstation; their held-level results are kept in
  `results/stage1/ablation/fixed_level/`. The three-seed test on seeds 1000-1199 is run again
  for the warm-up students. Results: `results/stage1/student_ramp/` (development seeds),
  `results/stage1/student_fixed/` (held level).

## D-022: A narrow speed-tracking reward at weight 0.5 fixes the speed miss (2026-11-01)

- **What:** the final controller adds a second speed-tracking term, `track_lin_vel_xy_fine`:
  the same function as `track_lin_vel_xy_exp` (exp of minus the squared error over std²) with
  std 0.25 m/s instead of 0.5, at weight 0.5. The term is in the code at weight 0 by default and
  switched on with `env.rewards.track_lin_vel_xy_fine.weight=0.5`, so every earlier run is
  unchanged. Teacher (3000 iterations) and student (1000, warm-up of D-021) are both trained
  with it. Called B2 below.
- **Why:** every student so far walked at about 86% of the commanded speed (0.86 m/s when asked
  for 1.0, 1.1-1.16 when asked for 1.4; MuJoCo, nominal robot), so the 1 m/s target (error 0.15
  or less) was missed on every seed. The robot is not at a limit: the gait-only policy of step 1
  reached about 93%. The wide kernel pays 92% of its reward at 0.14 m/s off, too little against
  the effort and smoothness penalties to close the gap; the narrow one pays 73% there. A
  one-day timebox on training seed 1 tried three candidates, each judged on the development
  seeds against the warm-up student of the same seed (mean of both MuJoCo models):

  | Candidate | 1 m/s | 0.5 m/s | Sideways | Turn | Steps/s at 1 m/s | Falls without kicks | Falls, 1 / 2 m/s kicks |
  |---|---|---|---|---|---|---|---|
  | warm-up student (D-021) | 0.175 | 0.100 | 0.119 | 0.205 | 2.48 | 0 / 1100 | 1 / 14 of 220 |
  | A: wide term at weight 2.0 | 0.185 | 0.074 | 0.047 | 0.295 | 3.02 | 22 / 1100 | 8 / 19 |
  | B: narrow term at weight 1.0 | 0.083 | 0.046 | 0.074 | 0.239 | 2.66 | 2 / 1100 | 6 / 66 |
  | B2: narrow term at weight 0.5 | 0.107 | 0.043 | 0.128 | 0.167 | 2.47 | 2 / 1100 | 1 / 11 |

  A did not reach the speed, turned worse and fell without kicks (its teacher's level reached
  only 0.60). B reached the speed (1.04-1.07 m/s when asked for 1.0) but fell five times as often
  to 2 m/s kicks, which its level (0.70, the same as B2's) does not explain. B2 met every
  finish-line target on seed 1 and was then trained on seeds 2 (workstation) and 3 (MSI, L40S).
- **Three seeds, B2 against the warm-up students** (development seeds 0-9, mean of both models;
  seeds 1 / 2 / 3 (mean)):

  | Student | 1 m/s | 0.5 m/s | Sideways | Turn | Crouch, cm | Falls without kicks | Falls, 2 m/s kicks |
  |---|---|---|---|---|---|---|---|
  | warm-up (D-021) | 0.17 / 0.23 / 0.21 (0.21) | 0.10 / 0.16 / 0.09 (0.12) | 0.12 / 0.19 / 0.15 (0.15) | 0.20 / 0.23 / 0.22 (0.22) | 3.8 / 2.7 / 3.8 (3.4) | 3 / 3300 | 35 / 660 |
  | B2 | 0.11 / 0.08 / 0.13 (0.10) | 0.04 / 0.04 / 0.07 (0.05) | 0.13 / 0.07 / 0.09 (0.10) | 0.17 / 0.24 / 0.23 (0.21) | 2.4 / 1.7 / 2.9 (2.3) | 2 / 3300 | 26 / 660 |

  Lower on all three seeds: speed error at 1 m/s and at 0.5 m/s, and crouch error. Turning moves
  in no consistent direction (better on seed 1, slightly worse on 2 and 3) and still misses its
  target (0.2 or less) on seeds 2 and 3, as before. Kick falls are no higher overall. The
  teachers' last levels were 0.70, 0.80 and 0.75, against 0.85, 0.80 and 0.80 for the warm-up's
  teachers, so the lower crouch error is not a higher level.
- **Caveat, seed 2:** at 1 m/s seed 2's student steps faster than the gait clock (3.17 steps/s
  on the Menagerie model, 2.85 on Newton; the clock gives 2.5 and the target allows 1.5-3) and
  its soles slide 0.18 m/s on Menagerie (target 0.1 or less), in every episode. Seeds 1 and 3
  stay on the clock (2.5 steps/s, 36-37 cm steps, slip 0.06 m/s or less). At 0.5 m/s all three
  are on the clock. Candidate A went faster the same way. The clock's fixed 0.8 s period asks
  for 40 cm steps at 1 m/s; a period that shortens with the commanded speed would remove the
  conflict. It was not tried: one seed in three can't show that it fixes anything (D-011), and
  stage 3's walking is mostly at 0.5 m/s or less. Asked for 1.0 m/s, seeds 1 and 3 walk at
  0.89-0.93 m/s, inside the target, and seed 2 at 1.01-1.04.
- **Test seeds (added the same day; numbers on the fixed bench of D-040):** on seeds 1000-1199
  (4400 episodes per seed and condition) walking repeats the development seeds within 0.01 m/s,
  and all three seeds meet the speed targets at 1 and 0.5 m/s; sideways, seed 1 is just over on
  Menagerie's model (0.152; 0.11 before D-040). Kick falls are lower than the warm-up students'
  at 1 m/s on every seed (244 against 321 pooled) and at 2 m/s on seeds 1 and 3 (738 against 806
  pooled). Falls without kicks are higher on seed 1, nearly all while standing with the arms
  moving: it falls in 38 of 200 nominal episodes of that command on the Newton model (19%;
  warm-up: none); seeds 2 and 3 fall less than their warm-up students. Pooled, nominal falls are
  54 of 13,200 (0.31%-0.53%), within the target, but stage 3 is standing at a table and moving the
  arms, so this was checked (same day): the falls come from arm motions wider than training
  covered. The tests move the arms over the full training ranges (`ARM_RANGES`, curriculum level
  1); B2's
  teachers stopped at levels 0.70, 0.80 and 0.75. With the arm range set to its training level,
  seed 1 falls in 0 of 300 episodes of that command (Isaac Lab, Newton), against 36 of 300 at the
  full range; seeds 2 and 3 fall 1 and 0 of 50 at the full range. The warm-up students (levels
  0.85, 0.80, 0.80) don't fall there at either range. Results: `results/stage1/speed_fix_test/`,
  `results/stage1/arm_level/`.
- **Not redone:** the ablation (step 6) keeps its students, trained without the narrow term;
  its configurations compare with each other, not with B2.

## D-023: Test the controller on Isaac Sim's PhysX as a second physics engine (2026-11-01)

- **What:** the trained students are run, without retraining, on Isaac Sim 6.2's PhysX through
  the same Isaac Lab task (`pi-wbc-isaac-eval ... physics=isaacsim_physx`), next to Newton and
  the MuJoCo bench. Isaac Sim lives in a second environment, `.venv-isaacsim`
  (`scripts/isaacsim_env.sh`: `isaaclab[isaacsim,ov]==3.0.0` with Isaac Lab's tested
  overrides; about 9.5 GB to download, 27 GB on disk), so the training environment and
  `uv.lock` don't change. Its Newton, Warp, MuJoCo, PyTorch and NumPy versions are the
  training environment's. The NVIDIA Omniverse License Agreement was accepted for these runs.
- **Why:** Newton's solver is MuJoCo-Warp, so the Newton → MuJoCo check (D-002) is the easy
  direction of transfer. PhysX has its own contact model and solver; with the Menagerie model
  (another robot description in MuJoCo) the two checks cover both a new engine and a new model.
  Isaac Sim 6.2 lists an RTX 4080 with 16 GB as its minimum; it runs on the workstation's
  12 GB 4070 for this task (driver 595.99, newer than the 595.58 NVIDIA tested).
- **What a fair test needed** (in `pi_wbc.isaac_eval`, only for engines other than Newton):
  - *Joint order.* Isaac Lab lists joints in the engine's order: Newton walks each limb in turn,
    PhysX goes level by level down the tree. Terms that pick joints by name keep that order, so
    unchanged the policy's actions went to the wrong joints and every robot fell (110 of 110).
    The action and the joint observations are now pinned, with `preserve_order`, to the lists
    in the run's `exported/policy_io.json`, which match Newton's order exactly (checked).
  - *Friction.* Training sets the ground to 0.05 and each robot to its draw (1.0 when nominal)
    through Newton, which uses the larger of two touching frictions. PhysX would multiply them.
    The Newton-only event is dropped and the ground gets 1.0 with PhysX's `max` rule: an
    effective 1.0, as in training and in MuJoCo.
  - *The teacher's friction input* reads Newton's model; it becomes a constant. The student
    never reads it.
  - *Saving results.* With Isaac Sim, closing the simulation ends the process, so the script
    now writes its results before closing (before, nothing was saved and the exit code was 0).
- **Kicks:** `--push-speed` adds the bench's kicks (same times, speed and random horizontal
  direction) to the Isaac Lab evaluation, so both engines and the bench are compared alike.
- **Result** (stage 1 README, step 7): speed and crouch errors on PhysX within 0.03 of Newton's
  on every seed; turning slightly worse on every seed (0.26 against 0.22 rad/s); falls without
  kicks 15 against 8 of 1650; with 2 m/s kicks 246 against 76 of 1650. Walking transfers; kick
  recovery is about three times weaker on PhysX.
- **Also checked, for stage 3:** a camera scene (G1 at a table with objects, head and side
  cameras) renders on the workstation with all three Isaac Lab 3.0 renderers: Newton's Warp
  renderer, OVRTX (RTX, with Newton physics) and Isaac Sim RTX (with PhysX)
  (`scripts/render_test.py`). Which one records the demonstrations is decided before stage 3.
- **Cost:** about 3 hours, the 27 GB environment, and no change to training. Results:
  `results/stage1/physx/`.
- **Cost:** five teacher-student runs of 3000 + 1000 iterations (A, B and B2 seeds 1-2 on the
  workstation, about 1 h 45 min each; B2 seed 3 on MSI, 1 h 05 min) and MuJoCo tests. Results:
  `results/stage1/speed_fix/` (development seeds; labels `speed-w2`, `speed-fine`,
  `speed-fine05`).

## D-024: Take the B2 teachers on to the full command ranges before distilling (2026-11-01)

- **What:** each B2 teacher (D-022) is resumed from its last checkpoint for 1000 more PPO
  iterations. Its curriculum level ramps on a fixed schedule from where it stopped to 1 (the
  full body-pose and arm-motion ranges, and the full kick size, 1.5 m/s) over the first 300
  iterations, then stays at 1, whatever the falls and rewards (`upper_body_level(...,
  ramp_from=...)`; `scripts/teacher_student_local.sh` with `CONTINUE_FROM`). A new student is
  then distilled from it with the warm-up of D-021, which now climbs to level 1.
- **Why:** the tests move the arms over the full training ranges (`ARM_RANGES`), but the
  curriculum waits for the turning reward and B2's teachers stopped at levels 0.70, 0.80 and 0.75.
  Seed 1's student falls in 36 of 300 episodes while standing with the arms moving over the full
  range, and in none of 300 within the range it trained on (D-022). Stage 3 stands the robot at a
  table and moves its arms; a controller with an untrained part of the arm range is a hidden risk
  there and in the controller-swap study (stage 5).
- **Cost:** per seed about 25 min of teacher training, 20 min of distillation and the usual
  tests. Seed 1 first, on the workstation; seeds 2 and 3 after its result.
- **Result (three seeds):** the teachers hardly fall with the arms over the full range (3 of
  9900 test episodes) and walk at 0.91-0.98 m/s when asked for 1.0, but turn worse: seed 2's
  teacher went from 0.78 to 0.40 rad/s when asked for 0.8, and seed 1's from 0.45 to 0.26 when
  asked to turn while walking. The first students distilled from them fell more than B2's and
  walked slower; D-026 fixes that. Details in
  [results/stage1/README.md](../results/stage1/README.md), "Full command ranges and a 25-step
  student".

## D-025: All rights reserved, and a map for the old commit hashes (2026-11-01)

- **What:** the repository is public but not open source: [LICENSE](../LICENSE) reserves all
  rights. The git history was rewritten so that every commit carries this license, which gave
  every commit a new hash. Result files written on the old history (up to 2026-11-02) record
  old hashes; [commit_map.tsv](commit_map.tsv) maps each old hash to its new one, whose code is
  otherwise the same.
- **Why:** the project is shared to be read, not yet for reuse.
- **Cost:** a lookup in the map for any commit hash recorded on the old history.

## D-026: The student sees the last 25 control steps (0.5 s) (2026-11-02)

- **What:** the student is distilled from a full-range teacher (D-024) for 3000 iterations
  with 25 steps of observation history instead of 6 (88 numbers per step, 2200 inputs; same
  512-256-128 network), set with `env.observations.policy.history_length=25`. The code's
  default stays 6, so earlier runs are unchanged; the export records the history length and the
  MuJoCo bench reads it from there.
- **Why:** with 6 steps (0.12 s) the full-range students either fell more than B2's (1000
  iterations: 55 of 3300 test episodes) or, trained for 3000 iterations, walked at 0.71 m/s
  when asked for 1.0, with the randomized robot too. With 25 steps, on three seeds: 0.91-1.00
  m/s, 6 falls in 9900 nominal test episodes (B2: 53) and 84 with the randomized robot (B2: 285),
  and 4 falls to 2 m/s kicks in 660 MuJoCo episodes (B2: 26). [STAGE1_DESIGN.md](STAGE1_DESIGN.md)
  section 4 named a longer history as the fallback if the student could not match the teacher.
- **Cost:** about 64 min of distillation per seed on the workstation (20 min before), and a
  four times larger input layer. Turning stays worse than B2's (0.31-0.47 rad/s when asked for
  0.8, B2 0.59-0.67), mostly lost in the full-range teachers; a narrow turning reward did not
  fix it (D-027).

## D-027: The narrow turning reward stays off (2026-11-02)

- **What:** `track_ang_vel_z_fine` (an exponential turn-rate reward with std 0.25 rad/s) keeps
  weight 0 in the controller. It was tried at weight 0.5 in the full-range teacher training
  (D-024) with 25-step students (D-026) on three seeds, and the code keeps it for later tests.
- **Why:** it did not make turning better on average. Asked to turn at 0.8 rad/s in place, the
  students turned at 0.37 / 0.44 / 0.31 rad/s (seeds 1 / 2 / 3) against 0.47 / 0.31 / 0.45
  without it, and the teachers at 0.58 / 0.48 / 0.41 against 0.60 / 0.40 / 0.59: better on seed
  2, worse on seeds 1 and 3. In MuJoCo the students fell 14 times in 660 episodes with 2 m/s
  kicks, against 4 without it. The wide turning reward still pays 85% of its maximum at 0.2
  rad/s off, but a narrow one barely pays at the 0.3-0.5 rad/s shortfall in place, so it
  rewards turning faster only once the robot is already close to the command. The turn rate in place follows the gait instead:
  students that step around while turning reach 0.44-0.49 rad/s, students that twist on their
  feet with fewer steps 0.31-0.37, and which gait a seed learns did not follow the reward.
- **Cost:** about 2 h per seed of training and tests (seeds 1 and 2 on the workstation, seed 3
  on MSI); the full-range recipe of D-024 and D-026 stays the controller. Next for turning: a
  longer teacher training at the full ranges (is the teacher undertrained there?), then more
  robots turning in place during training (10% now).

## D-028: Train the full-range teachers 7000 iterations instead of 1000 (2026-11-03)

- **What:** each B2 teacher is resumed at the full command ranges as in D-024, but for 7000 PPO
  iterations instead of 1000 (to iteration 9998); the 25-step student (D-026) is then distilled
  from its last checkpoint for 3000 iterations. Trained on MSI, one L40S per seed. Seed 1's
  first job stopped at iteration 5432 when the simulation returned NaN observations; it was
  resumed from its iteration-5000 checkpoint for the remaining 4999 iterations, so it ends at
  the same iteration.
- **Why:** the full-range teachers had lost turning (0.60 / 0.40 / 0.59 rad/s when asked for 0.8;
  the B2 teachers they started from, 0.64 and 0.78 on seeds 1 and 2), and neither a longer history
  (D-026) nor a narrow turning reward (D-027) brought it back. The training log's turn-rate error
  was still falling after 1000 iterations: on seed 2 it rose from 0.77 to 0.97 as the ranges grew
  to full, was 0.94 at 1000 iterations and 0.71 at 7000. Asked to turn at 0.8 rad/s in place, the
  students now turn at 0.70 / 0.57 / 0.64 rad/s in Isaac Lab (seeds 1 / 2 / 3; 0.47 / 0.31 / 0.45
  after 1000 iterations, B2 0.64 / 0.59 / 0.67) and 0.72 / 0.56 / 0.65 in MuJoCo (on the fixed
  bench, D-040; 0.73 / 0.59 / 0.65 before), and all three step around instead of twisting on their
  feet. They never fell on the nominal robot in Isaac Lab (6 of 9900 before) or in MuJoCo without
  kicks, and fell less under MuJoCo kicks (3 against 13 of 1320 on the fixed bench); on the
  randomized robot in Isaac Lab a few more (110 against 84 of 9900, intervals overlapping). Seed 1
  walks a little slower (speed error at 1 m/s 0.10 against 0.07 m/s in MuJoCo). These students are
  the stage 1 controller. On the test seeds (1000-1199), on the fixed bench (D-040), they fell in
  1 of 66,000 episodes without kicks and in 38 of 13,200 with 2 m/s kicks (B2: 203 and 738).
- **Cost:** 1 h 43 min of teacher training per seed on an L40S instead of about 17 min. The whole
  recipe took 3.2 h for seed 3 on one L40S (B2 teacher 43 min, this teacher 1 h 44 min, student 42
  min, from the job logs); seeds 1 and 2 trained their first 3000 teacher iterations on the
  workstation. Turning in place swings between checkpoints 1000 iterations apart (seed 3: 0.28 to
  1.00 rad/s, and one of them fell in 51 of the 300 episodes standing with the arms moving), so
  the last checkpoint is used, never a chosen one.

## D-029: Stage 1 closes with the turning target met on one seed of three (2026-11-03)

- **What:** stage 1 is done with the controller of D-028. Of the design's finish line
  ([STAGE1_DESIGN.md](STAGE1_DESIGN.md), section 1), every target is met on every training seed
  on the test seeds except the turn-rate error turning in place at 0.8 rad/s (at most
  0.2 rad/s): 0.19 on seed 1, 0.24-0.25 on seed 2 and 0.22 on seed 3. It is recorded as a known
  limit, and stage 3 starts.
- **Why:** the miss is 0.02-0.05 rad/s, and three changes aimed at turning have been tried: a
  longer history (D-026), a narrow turning reward (D-027, no gain) and longer teacher training
  (D-028, which brought turning in place from 0.31-0.47 to 0.57-0.73 rad/s). Everything else
  holds with margin: no falls in 66,000 test episodes without kicks, kicks of 2.5 m/s survived
  98-99% of the time, and speed errors at most 0.12 m/s against 0.15. The pick-and-place task
  of stage 3 turns the robot under a closed loop (the expert's or the VLA's commands), which
  corrects a turn rate that falls short. The next thing to try, if stage 3 or 5 needs sharper
  turning, is a larger share of training robots asked to turn in place (10% now).
- **Cost:** in place, the controller turns at 70-90% of the commanded rate, so stage 3's
  expert and the VLA see a slower turn than they ask for; the stage 5 comparison with NVIDIA's
  controller reports turning as it is.
- **Amended by D-040 (2026-11-04):** on the fixed bench no seed meets the turning target on both
  robot models: 0.21 / 0.19 rad/s on seed 1 (Menagerie / Newton's model), 0.29 / 0.26 on seed 2
  and 0.24 / 0.22 on seed 3, a miss of 0.01-0.09 rad/s; in place the students turn at 0.72 /
  0.56 / 0.65 rad/s (MuJoCo, 0.8 asked). The other numbers above become 1 fall in 66,000 test
  episodes without kicks, kicks of 2.5 m/s survived 98.6%-99.4% of the time and speed errors at
  most 0.10 m/s. The decision stands: stage 1 closes with turning as a known limit.

## D-030: Stage 3 before stage 2 (2026-11-03)

- **What:** stage 3 (the pick-and-place task, the scripted expert and the demonstrations) comes
  next; stage 2 (human motion imitation) later, running on MSI alongside it.
- **Why:** stage 3 is on the way to the VLA (stage 4) and the study (stage 5), which hold the
  project's research question; stage 2 stands on its own and needs mostly cluster time.
- **Cost:** the motion-imitation results and the many-robot training GIF come later.

## D-031: Newton's renderer for the demonstrations, recorded as states and rendered afterwards (2026-11-03)

- **What:** the dataset's images and the closed-loop tests of stages 4-5 use Newton's own
  renderer (with Newton physics). Demonstrations are recorded as simulator states and rendered
  in a separate pass. OVRTX re-renders a few episodes for the front page.
- **Why:** the VLA must be tested with the renderer it was trained on, over thousands of
  closed-loop episodes. Newton's renderer needs 0.4 GB of GPU memory (leaving room for GR00T on
  the workstation's 12 GB card), runs on any GPU, including MSI's A100 and H100 and its
  current container, and is about ten times faster than OVRTX (1,558 against 162 frames per
  second for 16 robots, 640 x 480). OVRTX and Isaac Sim's RTX renderer need RT cores and are
  not set up on MSI; Isaac Sim's also brings PhysX, where the controller recovers three times
  worse from kicks (D-023). Recording states first keeps the choice reversible: the same
  episodes can be rendered again with OVRTX.
- **Cost:** plainer images, further from those GR00T was pretrained on; one extra pass.
  Design: [STAGE3_DESIGN.md](STAGE3_DESIGN.md), section 8.

## D-032: The perfect-layer grid for stage 5, and demonstrations recorded with our controller (2026-11-03)

- **What:** stage 5 crosses the high level (the VLA, the scripted expert) with the low level
  (an ideal body that does exactly what it is told, our controller, ours without randomization
  and curriculum, NVIDIA's); the stage 3 task environment takes any of them under the same
  commands. Demonstrations are recorded with the stage 1 controller, seed 1. NVIDIA's controller
  records a second set over the same layouts only if its standing drift gets fixed.
- **Why:** the grid gives each layer's share of the failures from success rates, not only from
  failure labels. A VLA trained only on demonstrations through our controller may do worse on
  NVIDIA's for that reason alone; recording through both would let stage 4 measure or remove
  that confound. NVIDIA's controller was therefore tried in Newton first
  ([nvidia_wbc_test.py](../scripts/nvidia_wbc_test.py)): it walks (0.44 m/s for 0.5) and turns
  in place better than ours (0.77-0.79 rad/s for 0.8), but standing still its "Balance" policy
  drifts backwards at 0.25-0.4 m/s and, with some arm poses and height commands, falls (9 of 10
  with stage 1's hanging arms at its nominal height); the task is mostly standing at a table.
- **Cost:** an ideal-body mode to build. If NVIDIA's standing is not fixed, the comparison
  carries the confound, and only the expert's rows of the grid show its size.
- **Update (2026-11-03):** the standing drift came from Isaac Lab's G1 asset, whose heel
  contact spheres sit 3 cm further forward than in Unitree's URDF (x = -0.02 against -0.05 m).
  With the URDF's heel, NVIDIA's controller never fell in the 11 test commands and stands still
  ([results/stage3](../results/stage3/README.md)), so the second demonstration set is planned
  (D-038, with the foot geometry; [STAGE3_DESIGN.md](STAGE3_DESIGN.md), section 6).

## D-033: Dex3 confirmed for GR00T N1.6; hands move between open and closed poses (2026-11-03)

- **What:** D-013 holds. The hands move between an open and a closed pose; the dataset stores
  their 7 joint targets each.
- **Why:** GR00T N1.6's G1 setup in IsaacLab-Arena (`gr00t_43dof_joint_space.yaml`) is 29 body
  joints plus 7 per hand with Dex3's joint names; Isaac-GR00T's `unitree_g1` configuration
  says the hand "is controlled by binary signals like a gripper"; NVIDIA's teleoperation also
  switches between fixed open and closed poses.
- **Cost:** no finger-level dexterity.

## D-034: The grasp: a side grasp from above, round objects, 4 physics substeps (2026-11-03)

- **What:** the right hand's palm faces the object from the side, fingers forward, and comes
  straight down beside it; fingers and thumb close together. The grasp point is 8.5 cm along the
  fingers and 6.2 cm out from the palm frame's origin; the torso leans 0.25 rad to reach the
  0.75 m table. The arm's path is solved on Newton's model of the robot, inside the controller's
  trained arm ranges. Stage 3 scenes run Newton at 4 substeps (800 Hz), if step 1's check of
  the controller passes.
- **Why:** in the grasp test ([results/stage3](../results/stage3/README.md)) a can and a bottle
  are held 27 of 27 times with the object up to 1 cm off, at 0.1-0.5 kg. Square objects are
  pinched off-centre (3/27, 1/27); closing the fingers first fails (0/27); at 2 substeps 24/27
  hold and the object slides 4.4 cm in the hand instead of 1.7. Upright, the grasp is out of
  reach.
- **Cost:** round objects only, about 1 cm of tolerance, and physics twice as slow as
  training's, at a substep count the controller was not trained with.
- **Check (2026-11-03, step 1):** at 4 substeps with the URDF's heel the controller is unchanged:
  3 falls against 2 in 4,950 robots over nominal, randomized and kicked conditions and three
  seeds, speed and turning within 0.02 ([results/stage3](../results/stage3/README.md), step 1).

## D-035: The task: one table, three round objects, two bins, frozen seed ranges (2026-11-03)

- **What:** the scene of [STAGE3_DESIGN.md](STAGE3_DESIGN.md), section 3: a 0.75 m table, three
  of five round objects told apart by colour and kind, two bins at the table's ends, the robot
  starting 1.5-3 m away within 90 degrees of facing it, about ten instruction templates, right
  hand only in version 1. Layout seeds: train 0-49,999, development 50,000-50,999, test
  60,000-60,999 (used once, for the reported numbers).
- **Why:** it is the project's task (walk to a table, pick the named object, put it in the named
  bin) within the reach and grasp the tests allow.
- **Cost:** no left-hand grasps, no square objects, one table.

## D-036: The scripted expert: a state machine over the true state, no retries (2026-11-03)

- **What:** walk to the stance, settle, reach (lean, raise the hand, over the object), grasp
  (correcting the hand on the measured palm-object offset), lift, carry (straighten, side-step
  or turn to the bin), release, retract. Seven failure labels from simulator state: fell,
  knocked, missed grasp, dropped, wrong object, wrong bin, timed out. The ideal body separates
  the expert's own failures from the controller's.
- **Why:** an expert without retries gives an honest success rate through each controller, and
  its labels are stage 5's failure attribution.
- **Cost:** the demonstrations contain no recoveries, so the VLA never sees one.

## D-037: The dataset: 1,000 episodes per controller, head camera, replay check (2026-11-03)

- **What:** 1,000 training episodes through our controller (seed 1) and a matched set through
  NVIDIA's over the same layouts (D-032); head camera, 640 x 480 at 25 frames per second;
  states recorded every control step and rendered afterwards (D-031). Replay check: written-back
  states render exactly, and re-running the recorded commands must give the same outcome.
  Exported to GR00T N1.6's format (LeRobot v2 with `modality.json`, `NEW_EMBODIMENT`) and
  LeRobot's current format for π0.5, with a manifest.
- **Why:** a size in line with GR00T and π0.5 fine-tuning practice, to be revised in stage 4;
  Newton on the GPU does not repeat runs exactly, so a re-run can only reproduce outcomes.
- **Cost:** two data loaders to install for the load tests (asked first).

## D-038: The G1's heel contact spheres at Unitree's URDF position from stage 3 on (2026-11-03)

- **What:** in every stage 3 scene and later evaluation, the feet's heel contact spheres sit at
  x = -0.05 m on the ankle-roll link, where Unitree's URDF has them, instead of Isaac Lab's
  asset's -0.02 m, for every controller (`pi_wbc.stage3.feet.set_heel_x`).
- **Why:** Unitree's URDF (NVIDIA's copy), NVIDIA's MuJoCo models and MuJoCo Menagerie agree on
  -0.05 m. On the asset's shorter heel NVIDIA's controller drifts backwards at 0.25-0.4 m/s
  standing and falls; on the URDF's it never fell in 220 robots and stands still. Ours is
  unchanged there: no falls in 1,100 robots, none in 550 with 1 m/s kicks.
- **Cost:** our controller runs on a foot it was not trained on: it turns in place a little
  more slowly (0.64 against 0.70 rad/s for 0.8). Stage 1's Newton numbers were taken on the
  asset's heel. Which position matches the real robot's sole was not checked.

## D-039: Erratum: on Menagerie's model the MuJoCo bench's feet stood on friction 0.6 (2026-11-03)

- **What:** every MuJoCo-bench run on Menagerie's G1 in stages 0 and 1, except under the
  `friction_0.4` condition, had the feet touching the floor with friction 0.6, not the 1.0 that
  D-008 states. Menagerie's foot spheres have `priority="1"` and friction 0.6, and MuJoCo takes
  the higher-priority geom's friction instead of the larger of the two. The bench now gives the
  floor the highest priority of the model's geoms, so the larger friction applies, as in Newton
  (`build_model`, with a test that reads the friction of the actual foot-floor contacts). Runs
  on Newton's exported model were never affected (all its geoms have priority 0 and friction
  1.0), nor was the `friction_0.4` condition (it sets the robot's friction too).
- **What changes** (seed 1's stage 1 controller, Menagerie, development seeds,
  [results/stage1/foot_friction](../results/stage1/foot_friction/)):
  - Nominal and with 1 m/s kicks (110 episodes each): no falls before or after; speed and
    turn-rate errors move by at most 0.03; sole slip a little lower (walking at 1 m/s 0.043
    against 0.045 m/s nominal, 0.069 against 0.082 with kicks).
  - The weakness stage 1 reported on the test seeds (walking at 0.5 m/s, seed 1 fell after a
    0.5 m/s kick in 30 of 200 episodes on Menagerie's model, never on Newton's) came from the
    0.6 floor: on the same 100 development seeds it falls in 16 with the old floor
    (`scripts/sim2sim_old_floor.py`; 95% CI 10-24%) and in none with the corrected one (0-4%).
  - Not explained by it: Menagerie's model still slips more than Newton's (about 0.04 against
    0.005 m/s at 1 m/s), so that gap comes from the models' other contact settings.
  - Stage 1's conclusions stand: every finish-line target was met on the more slippery floor.
- **Cost:** stage 0 and stage 1 Menagerie results are not re-run (the test seeds take several
  hours); their result files record `floor_friction: 1.0` although the feet had 0.6, and later
  Menagerie results are not directly comparable with them.
- **Amended by D-040 (2026-11-04):** the numbers above come from the bench before D-040's
  fixes. Two of its statements do not hold:
  - "Every finish-line target was met" was wrong before this erratum too: turning in place was
    missed by seeds 2 and 3 (D-029). With the corrected floor seed 1 missed it as well on
    Menagerie's model (0.214 rad/s on the development seeds), and on the fixed bench every seed
    misses it.
  - The 0.5 m/s kick trips did not come from the floor alone: on the fixed bench, with the
    corrected floor, seed 1 falls in 5 of 100 of those development episodes (2%-11%) and in 10
    of 200 on the test seeds (30 before); with the old floor on the fixed bench (this entry's
    script, restoring the file's own contact priorities), in 14 of 100 and 23 of 200. The floor
    accounts for about two thirds of the drop from 30, the other fixes for the rest.
  - Stage 1's published bench numbers are now rerun on the fixed bench (D-040); stage 0's are
    not.

## D-040: Seven more MuJoCo-bench fixes, and stage 1's bench numbers rerun on the fixed bench (2026-11-04)

- **What:** an audit of the MuJoCo bench after D-039 found seven more faults. Each is fixed in
  commit `5ea97c6`, with a test that fails on the code before it:
  1. **Stale readings.** After `mj_step`, MuJoCo's body orientations, velocities and contacts
     describe the start of the last 2.5 ms physics step, so the policy's IMU readings were one
     step old. The bench now calls `mj_forward` after each policy step and reads the end of the
     step, as Newton's state does; the physics is unchanged.
  2. **Kick episodes too short.** Episodes ran 20 s with the last kick at 19 s, and a kicked
     robot falls up to about 2.4 s after a kick (90th percentile), so falls from that kick were
     mostly missed. Kick episodes now run 23 s. The Isaac Lab test had the same fault (fixed in
     `a4a9da6`).
  3. **Delay on the wrong joints.** `delay_20ms` held every PD target, arms and hands included;
     training's delay holds only the joints the policy drives. Now only those.
  4. **A kick at the moment of a fall.** The kick count behind "kicks survived" counted a kick
     due at the fall's own time as received; it no longer does (no published number changed).
  5. **Contact priority.** Every geom now has priority 0, as every shape in Newton's training
     model, so MuJoCo combines two touching geoms as Newton does (the larger friction; solref
     and solimp mixed by solmix). For Menagerie's feet on the floor this is D-039's fix; it also
     covers other geoms touching the floor.
  6. **Shared random stream.** The arm-motion targets drew from the same random stream as the
     starts and kicks; they now have their own, so a change to the arm motion cannot change the
     starts or the kicks.
  7. **Provenance.** Result files now record each model's contact and solver settings, the
     Isaac Sim version and a SHA-256 of the source code.
- **Rerun:** every MuJoCo-bench result the stage 1 pages give as a final or comparison number
  was run again from a worktree of `5ea97c6`, both robot models:
  - on MSI's CPUs (309 one-CPU jobs, about 430 CPU-hours; 16 of them also on the workstation,
    with the same falls and fall times): the final controller's test and development seeds and
    kick sweep, the warm-up and B2 students' test seeds and B2's kick sweep, and the three-seed
    ablation;
  - a second batch of 81 jobs: the older test-seed sets of step 4 (first protocol, held level,
    full-size kicks) and the first-protocol and held-level ablation runs (79 on MSI; the 2 that
    stopped there with a bus error ran on the workstation);
  - on the workstation: the turn sweeps (now with provenance, `9416351`), the development seeds
    of D-028's comparison and of step 7's B2 columns, seed 1's corrected-floor checks, the
    tracking traces and the front page's GIFs; and the 24 Isaac Lab and PhysX kick runs at 23 s
    (`a4a9da6`).

  The development-seed tables of the earlier steps are kept as measured, as the record of the
  choices made then, and the stage 1 page says which numbers are which.
- **What changes** (final controller, test seeds, Menagerie / Newton's model):
  - **Turning in place now misses on every seed:** turn-rate error at 0.8 rad/s 0.21 / 0.19 rad/s
    on seed 1 (0.19 / 0.19 before), 0.29 / 0.26 on seed 2 (0.25 / 0.24), 0.24 / 0.22 on seed 3
    (0.22 / 0.22). D-029 is amended. In place the students turn at 0.72 / 0.56 / 0.65 rad/s (0.73
    / 0.59 / 0.65).
  - **Falls without kicks: 1 in 66,000** (0 before): seed 1 under sensor noise, walking while
    turning on Menagerie's model.
  - **Kicks:** 21 and 38 falls in 13,200 episodes with 1 and 2 m/s kicks (11 and 74 before;
    21 and 33 in the first 20 s, so the longer watch adds 0-5 and the other fixes move the rest
    both ways). The largest kick survived 95% of the time stays 2.5 m/s on every seed
    (98.6%-99.4% of kicks).
  - **Seed 1's trips over 0.5 m/s kicks** while walking at 0.5 m/s on Menagerie's model: 10 of
    200 test episodes (30 before) and 5 of 100 development episodes on the corrected floor (none
    on the old bench); with the old floor on the fixed bench 23 of 200 and 14 of 100. D-039 is
    amended.
  - Other finish-line measures move by at most about 0.02 m/s and 1 cm; the speed error at
    1 m/s drops a little (seed 1: 0.12 / 0.11 to 0.10 / 0.10).
  - Earlier recipes on the test seeds: falls without kicks 70 for the warm-up students (65) and
    203 for B2 (199); with 1 and 2 m/s kicks 321 and 806 (256 and 793) and 244 and 738 (142 and
    654). B2's seed 1 now just misses the sideways target (0.152) and its seed 2 the step length
    (28 cm on Menagerie's model).
  - Step 4's older recipes on the test seeds (first protocol, held level, and seed 1's full-size
    kicks): falls without kicks 92, 16 and 12 (60, 14 and 3 before); with 1 m/s kicks 246, 169
    and 62 (154, 131 and 42); with 2 m/s kicks 649, 548 and 147 (656, 533 and 147). The
    first-protocol students no longer meet the turning target on any seed (seeds 1 and 3 did),
    and their kick sweep still puts every seed's 95% point between 2.0 and 2.5 m/s. D-021's
    findings hold: the warm-up students fall most to kicks on seed 1, three seeds don't rank
    the recipes, and on seed 1 the full-size kicks cut the warm-up's kick falls without
    reaching the held level's.
  - The first-protocol student trained without randomization (seed 1) still falls to sensor
    noise and kicks: 749 and 1790 of 4400 episodes (789 and 1650 before), and now 42 with a
    20 ms delay (15).
  - Ablation: the claims made on all three seeds hold with new numbers, except two the audit
    had questioned: PPO alone does not turn worst on every seed (it turns about twice as badly
    as the full student, as does the run without a curriculum), and the student trained without
    sensor noise falls more to noise than the full students (48 against 6 of 13,200). New on
    every seed: without actuator randomization, more falls to 2 m/s kicks.
  - Isaac Lab and PhysX, kicks at 23 s: the final controller falls 7 and 88 times in 1650
    episodes with 2 m/s kicks (4 and 85 at 20 s), B2 99 and 265 (76 and 246). The engine gap
    stands.
- **Why:** D-039 and this audit showed the bench was not the one D-008 describes, so every
  published bench number is redone on the bench as described.
- **Also corrected on the pages:** the test seeds 1000-1199 were run for every candidate recipe
  from step 4 on, and one problem first seen there led to D-024; "full range" for the arms means
  the full training ranges (`g1_body.ARM_RANGES`), narrower than the joints' limits; Isaac
  Lab's fall test is the torso link touching the ground.
- **Cost:** about 430 CPU-hours on MSI for the first batch and 110 for the second, about half
  an hour of the workstation's GPU and an hour of its CPUs, and about an hour of one CPU for the
  second batch's two jobs. The
  development record of steps 1-4 stays on the old bench, so its numbers and the final result's
  are not directly comparable.

## D-041: The expert's hand choice, retract and success rule, and its official test range (2026-11-04)

- **What:**
  - **Which hand:** the left hand when the target's right side has under 12 cm of free space
    (to another object's edge or the bin's inner wall) and its left side has more; the grasp,
    stances, approach and release mirror. The right hand alone succeeded on 50-64% of layouts
    with under 10 cm and 95-99% with more (1015 episodes): its hand met the right-hand bin's
    wall or the neighbour. On 32 training layouts next to that bin: right hand 20, left 31.
  - **Retract:** after the release the raised hand is pulled back toward the table's front edge
    (x = -0.15) before the arm is lowered; lowering from over the table swung the hand through
    other objects.
  - **Success is judged when the episode ends,** not when the hand opens: the old retract
    knocked 26 of 176 placed objects back out of the bin on development layouts, so the expert's
    earlier success rates were about 13 points too high.
  - **A new failure label, `knocked_other`:** an episode fails when another object moves more
    than 2 cm. A demonstration must not push the scene around.
  - **Test layouts:** 60000-60383 were run twice (judged at release, then at the end after a
    fix found on training and development layouts) and are retired. The official number comes
    from 60384-60767, run once at f1dd315: 364 of 384 succeed (94.8%, 95% interval
    92.1-96.6%; [results](../results/stage3/expert/test-60384-60767.json)). Fresh training
    layouts 1000-1383: 361 of 384 (94.0%). Tuning now uses training layouts; 960 of the 1000
    development seeds are spent.
- **Why:** the second audit (2026-11-03) found that most failures were the right hand meeting
  the bin wall, not the pelvis tipping back (that came with it), and that pushed neighbours went
  unnoticed.
- **Cost:** the left hand needs the left arm's IK and mirrored commands, no new training. The
  stricter rule lowers success relative to the old counting. The interval's lower end clears the
  design's 90%; the 95% goal is met only to rounding. Of the 20 test failures 16 are pushed
  neighbours: 7 while the hand comes down, 6 in the retract, 3 while carrying.

## D-042: The replay check tests each episode's first second, and the training set's seeds (2026-11-04)

- **What:**
  - **Replay check** (`scripts/replay_check.py`): each recorded episode's commands go through
    the controller and the physics again, without the expert. An episode passes when the replay
    stays within 1 cm of its recording (pelvis and target object) over the first second. How
    often replays also end the same way is reported over the whole set and drops nothing. This
    replaces design section 8's "episodes that fail it are dropped".
  - **Every pick-place reset now samples the zero velocity command.** It drew a random one,
    which the expert overwrote before the first step but which sat in the controller's first
    observation and 25-step history, so no two runs of a layout started alike.
  - **The training set:** the first 1,000 successful episodes, in seed order, from training seed
    10,000 on. Every recorded episode, failures included, is kept with its label for stage 5.
- **Why:** Newton on the GPU doesn't repeat a run exactly. The expert corrects small differences
  as it goes; a replay of its commands can't. Of 31 recorded successes, 24-27 succeeded again on
  each of three replays; 11 failed at least once but only 2 every time, and those that failed
  looked no harder (hand, room beside the target, tilt). Dropping on the outcome would remove
  about one good demonstration in five, by chance. The first second separates recordings that
  belong to their commands (at most 2.7 mm of drift, median 0.6) from mismatched ones (17-124 mm).
- **Cost:** the dataset can't claim that every episode reproduces end to end; it reports the
  share that does. The reset change came after D-041's official number; the expert's logic is
  unchanged (31 of 32 training episodes succeeded after it), and the next official number comes
  from fresh test seeds.

## D-043: How the head-camera video is rendered and randomized (2026-11-04)

- **What:** `scripts/render_episodes.py` renders each recorded episode with Newton's renderer
  (D-031), in a copy of the pick-place scene that runs no physics: every second control step,
  the recorded pelvis pose, joint positions and object poses are written in, giving 25 frames
  per second (frame k shows the state control step 2k's command was applied to). The head
  camera is design section 3's, 640 x 480 (its lens: D-044). Each episode's look comes from its layout seed
  (`pi_wbc/stage3/appearance.py`, render seed 0), so any episode re-renders with the same look:
  - floor, table and walls: muted colours (saturation at most 0.2; the table's brightness
    0.3-0.8, so the white bottle and the black bin stand out on it) and patterns (plain a
    quarter of the time, else checker, tiles, planks, speckle or stripes at two scales each);
    a wall 0.3-1.5 m behind the table and walls 2-3.5 m to each side;
  - the sun: one of six heights (30-80 degrees), from any direction relative to the scene;
    shadows on;
  - the light's brightness (x0.7-1.3) and colour (red and blue up to 10% apart, green 3%);
  - the camera's mount: up to 1 cm and 1 degree off on each axis.

  Object and bin colours never change. Videos are H.264 (libx264, quality 20, a key frame every
  25 frames, yuv420p), one per episode, with a manifest of each episode's look, frame count
  and file hash.
- **Why:** Newton's renderer has one white sun of fixed strength for all environments and a
  fixed ambient term. So the sun's direction relative to a scene comes from turning the whole
  scene (robot, table, objects, bins, floor, walls) about its origin, which changes nothing
  else in the image. Episodes with the same sun height render together. Brightness and colour
  are a gain on the image in linear colour, which is what a brighter or warmer light does to
  diffuse surfaces. H.264 because every video decoder reads it; the export step can re-encode.
- **Amended (2026-11-04, after D-047):** the same look, not the same pixels. Rendering the 1,000
  v1 episodes again inside training set v2 gave byte-identical videos for 16 of them; the frames
  of others differ by 0.02-0.84 grey levels on average, with 0.2-10% of pixels more than 2 levels
  apart, at edges (GPU ray tracing in other batch slots is not bit-repeatable). Each dataset's
  videos are hashed in its own render manifest.
- **Cost:** the lighting is plainer than RTX: one light, no coloured shadows, no reflections.
  Patterns are generated greys, not photographs. The renderer's speed on the RTX 4070 is about
  660 frames per second for 32 episodes at once.

## D-044: The head camera's lens is 87 degrees wide (2026-11-04)

- **What:** the head camera sees 87 degrees across (11.04 mm focal length on Isaac Lab's
  20.955 mm aperture), 71 degrees up and down at 640 x 480, instead of Isaac Lab's default
  60.4 degrees (18 mm). The 1,000 training videos are rendered again with it; the rest of D-043
  is unchanged.
- **Why:** [scripts/camera_view_check.py](../scripts/camera_view_check.py) projects the target
  object's centre and the working palm into the camera for every frame of the 1,000 training
  episodes, from the recorded states and each episode's mount error. The expert holds the
  object beside the body, so a narrow lens loses it. Share of frames with the object's centre
  at least 2 degrees inside the picture:

  | Lens (across) | Reach and grasp | Carry | Place | Palm, while carrying |
  |---|---:|---:|---:|---:|
  | 60.4 (Isaac Lab's default) | 88.0% | 12.9% | 51.1% | 0.0% |
  | 69.4 (the D435's colour camera) | 97.5% | 57.6% | 84.5% | 0.6% |
  | 80 | 100.0% | 98.0% | 98.5% | 23.3% |
  | 87 (the D435's depth cameras) | 100.0% | 99.9% | 99.8% | 67.5% |
  | 90 | 100.0% | 100.0% | 99.9% | 84.7% |

  From 80 degrees on the object stays in view; wider lenses make it smaller (a can at the
  grasp is about 49 pixels wide at 87 degrees, 46 at 90, 80 at 60.4). 87 is a real width of
  the camera the G1 carries.
- **Cost:** objects look smaller than with the default lens. The colour image is wider than
  the real D435's colour camera (69 degrees); that would matter on the real robot, which this
  project doesn't claim. The first render, at 60.4 degrees, is superseded.

## D-045: The dataset's export format: GR00T-flavoured LeRobot v2.1, and LeRobot v3.0 (2026-11-04)

- **What:** [scripts/export_dataset.py](../scripts/export_dataset.py)
  ([src/pi_wbc/stage3/export.py](../src/pi_wbc/stage3/export.py)) writes each dataset in the
  layout GR00T N1.6 reads (LeRobot v2.1 plus `meta/modality.json`), with the names of GR00T
  N1.6's own `unitree_g1` data config:
  - one row per video frame, 25 per second: frame k shows the state control step 2k's command
    was applied to (D-043);
  - `observation.state`: the 43 joint positions (legs, waist, left arm, left hand, right arm,
    right hand), in the same order as NVIDIA's G1 datasets;
  - `action`, 35 values: left and right arm and hand joint targets (28), waist yaw, roll and
    pitch goals, pelvis height goal, and `navigate_command` (forward and sideways speed, turn
    rate): the expert's commands at that control step, which the VLA will send;
  - the instruction as `task_index` and `annotation.human.task_description`;
  - GR00T's own `gr00t/data/stats.py` (branch `n1d6`, commit 9b37aa1) adds `stats.json` and the
    relative arm statistics of its G1 config.

  LeRobot's own converter (`convert_dataset_v21_to_v30`, LeRobot 0.6.1) makes the LeRobot v3.0
  copy, run offline on a hard-linked copy, with LeRobot's quantile statistics added for π0.5
  ([scripts/check_lerobot_dataset.py](../scripts/check_lerobot_dataset.py): LeRobot's own
  functions on the same rows and sampled frames as its script, each episode decoded in one
  pass; identical to its script's numbers on the episodes compared, which its random-access
  decoding needs about 6 s each for).
- **Checks:** GR00T's `LeRobotEpisodeLoader` with its `unitree_g1` config loads all 1,000
  training episodes ([scripts/check_groot_dataset.py](../scripts/check_groot_dataset.py)):
  every state and action group has its size and equals the table, the instruction is the
  episode's, the decoded frames equal ffmpeg's (difference 0.00) and change between frames,
  and its training-sample builder returns 30-step action chunks. LeRobot's `LeRobotDataset`
  loads the v3.0 copy: on 12 episodes the rows equal the v2.1 tables, π0.5's 50-step chunks
  follow the rows and are marked as padding past the end, and the images equal the source
  frames.
- **Why:** GR00T reads LeRobot v2 with `modality.json`; its G1 names let stage 4 fine-tune
  under either its `UNITREE_G1` head (pretrained on NVIDIA's G1 data, same joint order) or
  `NEW_EMBODIMENT`. LeRobot's current format is v3.0, and its `pi05` policy normalizes state
  and action by quantiles.
- **Cost:** 25 frames per second where NVIDIA's G1 data has 50, so stage 4's action horizon is
  chosen for 25. The commands at odd control steps are not in the data; the VLA's command is
  held for two control steps. LeRobot's `pi05` pads state and action to 32 values, and ours are
  43 and 35; stage 4 must cut them for π0.5. The working hand's 7 targets are exactly open +
  s x (closed - open), s from 0 to 1 (largest error 6e-8 rad over all 1,088 recordings), and
  the idle hand holds its default pose, so 14 arm + 2 hand + 3 waist + 1 height + 3 velocity =
  23 command values lose nothing if the idle hand is held at its default; the state without
  the legs has 31 values.

## D-046: NVIDIA's controller as the task's low level (2026-11-04)

- **What:** `--low-level nvidia` (scripts/expert_rollout.py, scripts/replay_check.py;
  [src/pi_wbc/stage3/nvidia_wbc.py](../src/pi_wbc/stage3/nvidia_wbc.py)) runs NVIDIA's released
  Balance and Walk policies (D-004) under the same commands as ours (design sections 6-7). Its
  observation is built as in the Newton test from the task's commands: velocity as given;
  height = our pelvis-height command minus 0.04 m (the two controllers' nominal heights, 0.78
  and 0.74 m: the command's height is taken relative to each controller's nominal); torso
  roll, pitch and yaw = our waist roll, pitch and yaw goals. Arms and hands are set by the task,
  as for ours. Its leg and waist targets are converted into our action term's units, and
  [scripts/check_nvidia_targets.py](../scripts/check_nvidia_targets.py) checks that the task
  applies exactly them (largest difference 0.0 rad over 2 s of standing, walking and leaning).
- **Result:** the expert through NVIDIA's controller on training layouts 1000-1383, the ones
  ours ran on for D-041: 367 of 384 (95.6%, 95% interval 93.0-97.2%) against 361 (94.0%,
  91.2-96.0%). On the same layouts 352 succeed with both, 15 only with NVIDIA's, 9 only with
  ours, 8 with neither (exact paired test p = 0.31): no difference shown. NVIDIA's episodes are
  slower (median 37.7 s against 31.4 s); its failures are other objects pushed while the hand
  comes down (8) or reaches over the bin (4), where ours pushes them mostly in the retract (10).
  NVIDIA's own results file: logs/expert/nvidia/train-1000-1383.json.
- **Why:** the stage 5 grid needs NVIDIA's controller under the same command interface, and
  the matched demonstration set (design section 6) needs the expert to run through it.
- **Cost:** the height offset and the waist-as-torso mapping are this project's adapter
  choices, not NVIDIA's; its released checkpoints have no training code here to check them
  against.
- **The matched set (added the same day).** The expert through NVIDIA's controller on the
  training set's own layouts, seeds 10,000-11,087: 1,049 of 1,088 succeed (96.4%, 95% interval
  95.1-97.4%), against 1,010 (92.8%, 91.1-94.2%) through ours. On the same layouts 986 succeed
  with both, 63 only with NVIDIA's, 24 only with ours, 15 with neither (p = 3.5e-5). The gap is
  other objects pushed during the retract: 45 episodes with ours, 4 with NVIDIA's (the cause is
  not traced yet; a stage 5 lead). Run again through ours, 8 of those layouts (ones NVIDIA's
  controller passed) gave 6 successes and 2 pushes in the retract: these failures are mostly
  chance, which fits ours repeating runs less well (below). Every recording passes the replay check's first second, and
  974 of the 1,000 chosen episodes also succeed when their commands are replayed open loop
  (97.4%), against 695 of 1,000 with ours (D-042): small differences between runs grow far less
  under NVIDIA's controller. **NVIDIA's training set v1** = the first 1,000 qualifying episodes
  (seeds 10,000-11,036; 936 layouts shared with ours), by D-042's rule:
  results/stage3/dataset/train-v1-nvidia-manifest.json.

## D-047: Training set v2 and a validation set (2026-11-04)

- **What:** the expert recorded training seeds 11,088-12,239 (1,076 of 1,152 succeed, 93.4%,
  95% interval 91.8-94.7%) and validation seeds 40,000-40,255, unused training-range seeds
  kept apart (241 of 256, 94.1%). By D-042's rule
  ([scripts/build_dataset_manifest.py](../scripts/build_dataset_manifest.py), which rebuilds
  v1's manifest exactly): **training set v2** = the first 2,000 qualifying episodes from seed
  10,000 (seeds 10,000-12,155; it contains v1; 6 successes dropped by the replay check; 1,484
  right-hand and 516 left-hand), and **validation set v1** = all 241 qualifying episodes.
  Manifests: results/stage3/dataset/train-v2-manifest.json, val-v1-manifest.json.
- **Why:** stage 4 can then measure what the second 1,000 episodes add, and track a validation
  loss on layouts it never trains on; test layouts stay for closed-loop success only.
- **Cost:** 71 minutes of GPU time for the recording and replay checks (05:17-06:28). The new runs' result files say commit
  eae88f1, but provenance stamps the commit when a file is saved: the runs started at 5d1eecd,
  and the commits between them added only export code they never import (the package's source
  hash is the same, e354b5f5d565). Two replay files say "dirty" for the same reason: two
  scripts they don't use were uncommitted.

## D-048: 50 frames-per-second versions of the datasets (2026-11-04)

- **What:** besides the 25 fps sets of D-043 and D-045, each dataset is also rendered and
  exported at 50 frames per second: one row and one video frame per control step, with that
  step's commands, as NVIDIA's own G1 datasets are. Stage 4 chooses between them.
- **Why:** the expert's recorded commands, replayed open loop through the closed-loop evaluation
  loop ([scripts/vla_eval.py](../scripts/vla_eval.py), the stage 1 controller, the 241
  validation episodes, success held 1 s within 60 s):

  | Commands sent | Success | 95% interval |
  |---|---:|---|
  | every control step's (50 Hz) | 171 of 241 (71.0%) | 65-76% |
  | the 25 fps dataset's, each held for two control steps | 120 (49.8%) | 44-56% |
  | the 25 fps dataset's, the second step taking the mean of it and the next | 143 (59.3%) | 53-65% |

  Paired on the same episodes, 50 Hz beats 25 Hz held (70 against 19 episodes, p = 5e-8) and
  averaged (51 against 23, p = 0.0015); averaging beats holding (56 against 33, p = 0.019).
  The expert corrects small errors at every control step, and a 25 Hz command stream loses much
  of that; a VLA re-plans every few frames, so its loss is smaller, but it has the same handicap.
  At 50 fps GR00T's own `unitree_g1` config (30-step chunks, 0.6 s) fits as NVIDIA trained it.
  The 50 Hz replay here (71%) agrees with the replay check's 162 of 241 (67%, D-042), so the loop
  itself adds no loss. The loss is not our controller's alone: NVIDIA's recorded commands (D-046), on
  241 of its matched set's episodes, succeed on 233 (96.7%, 93.6-98.3%) at 50 Hz and 173 (71.8%,
  65.8-77.1%) at 25 Hz held (66 against 6 episodes, p = 7e-14).
- **Cost:** twice the frames to render (training set v1: 1,579,583, in 66 min with other jobs
  on the GPU; its 25 fps render took 21 min alone), but only 1.3 times the video storage (3.8 GiB
  against 3.0 GiB: H.264 compresses the extra frames well); twice the policy calls in closed loop
  if a 50 fps model re-plans as often in frames.

## D-049: Results record the code their run started with (2026-11-04)

- **What:** `pi_wbc.provenance` takes the git state and the package's source digest once, when it
  is first imported (each script imports it at start), and every result the process saves
  records those, marked `"code_recorded_at": "process start"`. Package versions, GPU and time
  are still read at save time.
- **Why:** they used to be read when each result was saved. Overnight runs of several hours
  saved results stamped with commits made after they started (D-047): the code they ran was
  unchanged, which the source digest showed, but the stamp misled, and a merge into the
  checkout during a run would have recorded code the run never used.
- **Cost:** results saved before this change record the state at save time; the runs of
  2026-11-04 that crossed a commit are listed where they are used (D-047).

## D-050: Stage 3 closes (2026-11-04)

- **What:** stage 3's finish line ([STAGE3_DESIGN.md](STAGE3_DESIGN.md) section 1) is met:
  - **The expert:** 364 of 384 unused test layouts (94.8%, 95% interval 92.1-96.6%), a table of
    failure causes and GIFs (D-041; [results](../results/stage3/README.md)). The interval's lower
    end clears 90%; the 95% target is met only to rounding.
  - **The dataset:** manifests with the layout seeds, split, commits, package versions, GPU and
    checksums (SHA-256 of every recording, and of every video in its render manifest); the
    replay check on every recorded episode (D-042); training set v1 (1,000 episodes), v2 (2,000),
    a validation set (241) and NVIDIA's matched set (1,000), at 25 frames per second, and v1 and
    the validation set also at 50 (D-047, D-048).
  - **The loaders:** GR00T N1.6's `LeRobotEpisodeLoader` loads every episode of training set v1
    (25 and 50 fps), v2, the validation set (25 and 50 fps) and NVIDIA's set, 5,482 episodes;
    LeRobot's `LeRobotDataset` loads the v3.0 copies of v1 and the validation set (25 and 50 fps)
    and NVIDIA's set (12 episodes each, compared row by row). v2's LeRobot copy was still being
    checked when this was written (D-045).
- **Beyond the finish line:** NVIDIA's controller as the task's low level and the matched set
  (D-046), the closed-loop evaluation loop and the command-rate test (D-048), a GR00T fine-tune
  launcher, configs and cluster job for stage 4 (draft: [STAGE4_DESIGN.md](STAGE4_DESIGN.md)).
- **Left open:** the ideal-body low level (design section 7); why our controller pushes objects
  during the retract where NVIDIA's does not (D-046); OVRTX renders for the front page.
- **Why close now:** what stage 4 needs is in place and checked; the open items belong to stage 5.

## Open items

- The stage 1 design ([STAGE1_DESIGN.md](STAGE1_DESIGN.md)) was accepted on 2026-10-29
  (D-012 to D-015).
- The built-in G1 velocity tasks use an older 37-joint G1 (`torso_joint`, two elbow joints,
  seven hand joints per side), not the 29-DoF robot. Stage 0's bench already uses
  `G1_29DOF_CFG` (D-007); stage 1 keeps it.
- Motion data: the LAFAN1 retargeting mirror used by BeyondMimic is CC BY-NC-ND 4.0 (no
  derivatives), which limits what processed clips can be released. Check before stage 2.

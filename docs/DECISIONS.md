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
    over 20-iteration windows (raise when under 10% of ended episodes fell and velocity
    tracking is over 60% of its reward; lower when over 25% fell).
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
    0.60 m crouch needs knee actions of about 5 at the action scale of 0.25; with the height
    bonus alone the robot tried and fell (10 of 10 MuJoCo episodes).
  - Arm poses every 2-4 s rather than 1-3 s: the faster motion (shoulders up to 2.5 rad/s,
    with up to 1 kg in the hand) went with turning in place being unlearned as the curriculum
    widened (training yaw-rate error 0.65 to 1.05 rad/s).
  - Height bonus: with the squared penalty alone, a 1400-iteration run never crouched (its
    training height error, 10 cm, was what standing still gives), while the waist, rewarded
    with the exp form, was tracked.
- **Cost:** a robot keeps its payload for the whole run (variety comes from 4096 robots, not
  from episodes). The torso command must be converted if a VLA sends torso roll, pitch and yaw
  in another convention (stage 3).

## Open items

- The stage 1 design ([STAGE1_DESIGN.md](STAGE1_DESIGN.md)) was accepted on 2026-10-29
  (D-012 to D-015).
- The built-in G1 velocity tasks use an older 37-joint G1 (`torso_joint`, two elbow joints,
  seven hand joints per side), not the 29-DoF robot. Stage 0's bench already uses
  `G1_29DOF_CFG` (D-007); stage 1 keeps it.
- Motion data: the LAFAN1 retargeting mirror used by BeyondMimic is CC BY-NC-ND 4.0 (no
  derivatives), which limits what processed clips can be released. Check before stage 2.

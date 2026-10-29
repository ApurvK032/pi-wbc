# Stage 1 design: the π-WBC whole-body controller

**Status: accepted (2026-10-29).** Apurv accepted all seven recommendations of section 15;
they are recorded as D-012 to D-015 in [DECISIONS.md](DECISIONS.md). Details below may still
change as the work shows what is needed; changes are logged there.

Stage 0 left us with a working pipeline (Isaac Lab training on the workstation and on MSI, a
MuJoCo sim-to-sim bench, gait measurements) and a bench policy that walks without falling but
shuffles: about 11 steps per second of 8 cm, in all 13 runs of that setup
([stage 0 results](../results/stage0/README.md), part 4). Stage 1 builds the project's own
controller.

## 1. Goal and finish line

A policy that a VLA can drive: it walks with a natural gait, follows base-velocity and
upper-body commands, recovers from pushes, and still works in plain MuJoCo. It sees only what a
real G1 could measure (joint encoders and the IMU), learned from a teacher that sees more.

Stage 1 is done when all of these hold on frozen test seeds, with 95% Wilson intervals for
fall rates and at least 3 training runs per configuration (D-011):

| Measure | Target | Stage 0 bench policy (13 runs: mean, or range) |
|---|---|---|
| Falls, all MuJoCo test commands, both models | upper 95% bound below 5% | 1.0% (0.6-1.6%) |
| Speed error while walking | at most 0.15 m/s | 0.11 m/s |
| Turn-rate error, turning in place at 0.8 rad/s | at most 0.2 rad/s | 0.12-0.80 rad/s |
| Step rate at 1 m/s | 1.5-3 steps/s | 10.7 |
| Step length at 1 m/s | at least 0.3 m | 0.08 m |
| Foot lift | at least 5 cm | 2.8 cm |
| Foot speed in stance (slip) | at most 0.1 m/s | 0.31 m/s |
| Push recovery | report the largest push survived 95% of the time | not tested |
| Arm tracking while walking | report joint error | not applicable |

Plus an ablation table (section 9) and GIFs for the README.

## 2. What changes from stage 0, in one table

| Area | Stage 0 bench | Stage 1 |
|---|---|---|
| Commands | velocity from a heading controller | NVIDIA's controller interface: velocity (yaw rate sampled directly), pelvis height, torso orientation; arm and hand joint targets (section 3) |
| What the policy moves | all 29 body joints | legs and waist (15 joints); arms and hands follow their targets directly |
| PD gains | moderate, chosen in stage 0 | NVIDIA's controller's gains, so both controllers run on the same simulated robot |
| Gait | no rhythm, weak air-time reward | a gait clock and foot-lift, slip and contact-schedule rewards (section 5) |
| Policy inputs | include base linear velocity (not measurable on a real robot) | student: encoders and IMU only, with history; teacher: privileged state (section 4) |
| Training | one PPO policy | PPO teacher, then DAgger-style student (section 7) |
| Randomization | robot friction (no effect on Newton), torso mass, pushes | six groups that can be switched off one at a time (section 6) |
| Curriculum | none | command ranges, pushes and arm motion grow with performance (section 8) |
| Evaluation | MuJoCo only | the same fixed protocol in Isaac Lab and MuJoCo, plus pushes, payload, delay, friction (section 9) |

## 3. Command interface (what the VLA sends)

**Proposal: use the interface of NVIDIA's controller.** NVIDIA's "Decoupled WBC" is the
controller GR00T N1.6 is used with on the G1, and the baseline of our study (D-004). Using its
commands means either controller can sit under the same VLA. Sources are in appendix A
(checked 2026-10-29).

| Command | Size | Meaning | Who carries it out | Stage 1 training range |
|---|---|---|---|---|
| navigate | 3 | forward and sideways speed (m/s), yaw rate (rad/s) | our RL policy | forward -0.5 to 1.0, sideways ±0.5, yaw rate ±1.0 |
| base height | 1 | pelvis height (m) | our RL policy | 0.50-0.78 (NVIDIA's default 0.74; our default pose stands at about 0.785) |
| torso orientation | 3 | torso roll, pitch, yaw relative to the pelvis (rad) | our RL policy, through the waist | roll ±0.2, pitch 0 to 0.6 (leaning forward), yaw ±0.6 |
| arms | 14 | 7 joint targets per arm | PD controllers directly | smooth random targets within the joint limits |
| hands | 14 | 7 Dex3 joint targets per hand | PD controllers directly | random open and close |

**The policy drives the legs and waist (15 joints); the arms and hands follow the VLA's joint
targets directly.** The legs learn to keep balance whatever the arms do, and arm accuracy does
not depend on RL. NVIDIA's controller, HOMIE, AMO and NVIDIA's WBC-AGILE (an Isaac Lab
G1 controller) all split the robot this way. The alternative, RL that also moves the arms
(as FALCON does), is decision 1 in section 15.

**What this gives the comparison (stage 5).** Swapping controllers changes only the 15-joint
lower body. Arms, hands, commands, PD gains and the robot model stay identical, so differences
in task success come from the lower-body controller. To make that true, stage 1 adopts NVIDIA's
PD gains and default leg pose (appendix A), so the simulated robot is the same in both tests.

**Consequences for later stages:**

- GR00T N1.6's pretrained G1 data was collected with NVIDIA's controller, and NVIDIA advises
  fine-tuning under a new embodiment tag when data comes from a different controller. Stage 3
  will collect demonstrations through one controller and fine-tune with `NEW_EMBODIMENT`;
  which controller collects them is a stage 3 decision.
- NVIDIA's newest path, GR00T N1.7 with the SONIC controller, has the VLA output latent motion
  tokens instead of commands, so it cannot be swapped under this interface. This settles the
  open N1.6 vs N1.7 item in favour of N1.6 (decision 4).
- NVIDIA's controller has no Isaac Lab version in its own repository; IsaacLab-Arena
  re-implements it on PhysX. It runs natively in MuJoCo, which is another reason to keep the
  MuJoCo bench central: both controllers can be compared there without porting either.

## 4. Policy inputs and outputs

**Rate.** 50 Hz policy, 200 Hz physics (5 ms, 2 Newton substeps), as in stage 0. Joint
position targets go to simulator-side PD controllers, so MuJoCo reproduces them exactly
(D-007, D-008).

**Student observations** (what a real G1 measures):

- angular velocity and projected gravity from the pelvis IMU;
- joint positions and velocities of the 29 body joints, relative to the default pose;
- the commands: navigate, base height, torso orientation (section 3). The arm targets are not
  inputs; the policy sees where the arms are through the joint readings;
- the last action (15 values);
- the gait clock as sin and cos of the phase (section 5);
- a history of the last 6 steps (0.12 s) of all of the above, as in NVIDIA's controller and
  HOMIE.

**Actions:** 15 joint position targets (12 leg joints, 3 waist joints), target = default pose
+ 0.25 × action, as in NVIDIA's controller and Unitree's training code.

Base linear velocity is dropped: a real robot cannot measure it directly, and a policy that
needs it can't be deployed. The student has to infer it from the history.

**Teacher observations:** the student's, plus privileged state: base linear velocity, contact
states and foot heights, the episode's randomized friction, added masses and centre-of-mass
shift, motor strength and gain scales, and the current push. With this the teacher learns
fast and well; the student then learns to act like it without the extra information.

**Networks.** Teacher MLP 512-256-128 (ELU). Student: an MLP over the 6-step history first,
since it exports to ONNX as a single matrix pipeline and the MuJoCo bench only needs a history
buffer. If the student can't match the teacher, try a longer history or an LSTM (rsl_rl
supports both).

## 5. Gait: how the robot is told to walk like a person

Stage 0 showed why the bench policy shuffles: its rewards ask for speed and stability, and
tiny fast steps are the cheapest way to get both. The one reward for real steps (feet air
time) earned 50-300 times less than speed tracking. Stage 1 describes the walk directly,
following the recipe of Unitree's own G1 training code (`unitree_rl_gym`), which is known to
produce a normal gait on this robot:

- **A gait clock.** A phase variable runs from 0 to 1 every gait period, with the right foot
  half a period behind the left. Each foot should be on the ground in the first part of its
  cycle and in the air in the rest. The policy sees the phase (as sin and cos), and a reward
  pays each foot for matching the schedule. When the command is near zero, the clock switches
  to standing: both feet down.
- **Foot lift.** A penalty on the swing foot's height away from a target: the ankle at 8 cm,
  which is about 4.5 cm of clearance under the sole (Unitree's value; raised if the measured
  lift falls short of the section 1 target).
- **No slipping.** A penalty on foot speed while the foot is on the ground.
- **Posture.** Penalties on hip yaw and roll away from neutral (no splayed or crossed legs), on
  body tilt, and on pelvis height away from the target height.
- **Smoothness and effort.** Small penalties on joint acceleration, velocity, torque and
  action changes, plus joint-limit penalties.

Starting weights. Isaac Lab, like Unitree's code, multiplies every weight by the time step, so
Unitree's G1 weights carry over directly. Changes from Unitree's values are marked and
explained.

| Group | Term | Weight | Source and reason |
|---|---|---:|---|
| Task | speed tracking, exp(-error²/0.25) | 1.0 | Unitree, Isaac Lab |
| Task | yaw-rate tracking | 1.0 | Unitree uses 0.5; raised because turning in place is a stage 0 weakness |
| Task | pelvis height to the commanded height | -10 | Unitree tracks a fixed 0.78 m; here it tracks the height command |
| Task | torso orientation to the command | 0.5 (exp) | new: the torso command of section 3 |
| Gait | contact matches the gait clock | 0.18 | Unitree; clock period 0.8 s (2.5 steps/s), right foot half a period behind, stance in the first 55% |
| Gait | swing-foot height, (ankle height - 0.08 m)² | -20 | Unitree; the ankle sits about 3.6 cm above the sole |
| Gait | foot speed while in contact | -0.2 | Unitree (`contact_no_vel`) |
| Gait | hip roll and yaw away from zero | -1.0 | Unitree (`hip_pos`) |
| Gait | feet air time | 0 | dropped: the clock replaces it |
| Stability | vertical pelvis speed | -2.0 | Unitree |
| Stability | pelvis roll and pitch rate | -0.05 | Unitree |
| Stability | pelvis tilt | -1.0 | Unitree |
| Stability | alive bonus | 0.15 | Unitree |
| Stability | fall (episode ends) | -200 | Isaac Lab's G1 task |
| Effort | joint acceleration | -2.5e-7 | Unitree |
| Effort | joint velocity | -1e-3 | Unitree |
| Effort | torque | -1e-5 | Unitree |
| Effort | action change | -0.01 | Unitree |
| Effort | joint near its limit | -5.0 | Unitree |

Unitree's code also clips the total reward at zero (`only_positive_rewards`); Isaac Lab does
not. If early training stalls because penalties dominate, clipping is the first thing to try.

Every term, its weight and its reason go into the task file and into `docs/DECISIONS.md`. Gait
is measured with the stage 0 gait measures (D-010) in both simulators.

**Arm swing.** With the arms under VLA command (section 3), they don't swing while walking, as
for a person carrying something. A natural arm swing is not a stage 1 goal.

## 6. Domain randomization

Six groups, each with one switch, so the ablation (section 9) can remove one at a time. Every
randomization was checked to exist and to support the Newton backend in Isaac Lab 3.0, but
each one will also be checked in a running simulation before it is trusted (for example,
by measuring that sampled friction changes how far a pushed foot slides).

| Group | What varies | Range (start) | Isaac Lab 3.0 mechanism |
|---|---|---|---|
| Friction | ground-foot friction | 0.1-1.25 (Unitree's range) | ground set to 0.05, robot shapes randomized: Newton uses the larger of two surfaces' friction (stage 0 finding), so the robot's value decides |
| Mass | torso mass, link masses, torso centre of mass, payload in each hand | torso -1 to +3 kg (Unitree's added-mass range); links ×0.9-1.1; CoM ±3 cm; payload 0-1 kg | `randomize_rigid_body_mass`, `randomize_rigid_body_com` |
| Actuators | PD stiffness and damping, joint friction and armature | ×0.8-1.2 | `randomize_actuator_gains`, `randomize_joint_parameters` |
| Latency | delay between the policy's action and the PD target | 0-20 ms (0-4 physics steps) | our action term: actions are applied every physics step, so a short buffer delays them |
| Sensor noise | noise on IMU and joint readings | Unitree's: joint position 0.01 rad, joint velocity 1.5 rad/s, gyro 0.2 rad/s, gravity 0.05 | observation noise settings |
| Pushes | velocity kicks to the pelvis, steady forces on the torso | up to 1.5 m/s every 5 s (Unitree), reached through the curriculum | `push_by_setting_velocity`, `apply_external_force_torque` |

## 7. Teacher and student

1. **Teacher:** PPO with privileged observations, all randomization and the curriculum.
2. **Student:** DAgger-style distillation with rsl_rl's `Distillation` (the student drives the
   robot and the teacher labels what it should have done, so the student learns to recover
   from its own mistakes). Same environment, same randomization.
3. Report the teacher-student gap on the evaluation protocol. The student is the policy that
   goes to MuJoCo, the ablation and the VLA.

## 8. Commands and curriculum

**Command sampling.** Base velocity is sampled directly, including yaw rate; there is no
heading controller (the cause of stage 0's weak turning in place). Each 10 s window: 10%
standing still, 10% turning in place, the rest uniform within the current ranges. Height and
torso commands are resampled every few seconds and approached smoothly. Arm targets follow
HOMIE's recipe: a new random arm pose every second, reached by interpolation, with the share
of arm motion growing as velocity tracking improves (AGILE does the same with
trapezoidal speed profiles).

**Curriculum** (each stage unlocks when tracking and survival are good enough):

- velocity ranges start at ±0.5 m/s forward, ±0.3 sideways, ±0.5 rad/s and grow to the full
  ranges of section 3;
- pushes start at zero and grow to 1.5 m/s;
- arm motion starts small and slow and grows to the full range and speed;
- flat ground only. The stage 3 task is indoors on a flat floor, and Newton's rough-terrain
  support is newer; rough terrain is a later extension.

Isaac Lab 3.0 provides the building blocks (`modify_env_param`, `DifficultyScheduler`).

## 9. Evaluation and ablation

**One protocol, two simulators.** The fixed commands and seeded starts of the MuJoCo bench
(D-008) also run in Isaac Lab, so Isaac Lab and MuJoCo numbers are directly comparable. Stage 0
only had Isaac Lab's training statistics, which mix random commands, noise and pushes.

**Seed ranges** (frozen before training): evaluation seeds 0-99 are for development, 1000-1199
for the final test; training seeds 1-3 for every configuration.

**Conditions:** nominal; floor friction 0.4; 1 kg in each hand; 20 ms extra delay; sensor
noise; pushes of increasing size at fixed times (gives the largest push survived); arms
following a scripted reach while walking. All in Isaac Lab, both MuJoCo models where the
condition exists there.

**Metrics:** falls (Wilson 95%), speed, turn-rate and height tracking, gait measures (D-010),
arm joint tracking error, energy per metre walked.

**Ablation** (each with 3 training runs):

| Configuration | Question it answers |
|---|---|
| full | the reference |
| without each randomization group (6) | which group matters for transfer and robustness |
| no randomization | how much randomization matters in total |
| no curriculum | whether the curriculum helps or only speeds training |
| PPO student without a teacher | whether teacher-student training is worth it |

That is 10 configurations × 3 runs = 30 training runs, each a teacher and a student (the
teacher depends on the randomization, so it is retrained per configuration).

## 10. Compute

Rough cost, to be measured on the first runs: stage 0 trained 1500 iterations in 18 min on an
L40S. A teacher needs about twice that many iterations and larger inputs, so about 1 h; a
student about 0.5 h. The ablation is then about 45 GPU-hours: about 2 days of back-to-back jobs
on the cluster partition used so far, which runs one job at a time. The workstation trained the
stage 0 policy only 1.5× slower than the L40S (27 vs 18 min), so it can take part of the load
if the caps are lifted for long runs.

## 11. Sim-to-sim bench changes

The MuJoCo bench gains: the gait clock, the observation history, the command interface of
section 3 (including arm targets), action delay, sensor noise, pushes and payloads. The export
writes these into `policy_io.json`; new unit tests check each against Isaac Lab, as in stage 0.

## 12. Order of work

1. **Gait first.** The 15-joint action, NVIDIA's gains, the new velocity commands, the gait
   clock and the new rewards, with the arms held still and no teacher. Short runs on the
   workstation, then 3 runs on the cluster. Gate: the gait targets of section 1 in MuJoCo. This
   is the riskiest part and the most visible, so it comes first.
2. **Upper body.** Arm and height commands, random arm motion, payloads.
3. **Randomization and curriculum,** each group checked in a running simulation.
4. **Teacher and student,** export, MuJoCo bench updates.
5. **Isaac Lab evaluation protocol** and push tests.
6. **Ablation runs,** tables, GIFs, README.

Estimate: 2-3 weeks (the plan's 1.5 weeks was optimistic).

## 13. Risks

- **The gait clock fixes the step rhythm.** The robot will step at the clock's rate at every
  speed. That is normal for this recipe; a speed-dependent period is a later refinement.
- **Newton is beta.** A randomization may run without doing what it says; hence the check in
  a running simulation for each group.
- **No base velocity for the student.** Speed tracking may drift; the history and the
  teacher's labels usually fix this. Measured, not assumed.
- **Arms and balance.** Fast arm motion with a payload can tip the robot; the arm curriculum
  and payload randomization are there for this.

## 14. Not in stage 1

Rough terrain; a realistic motor model (torque-speed limits); a Unitree SDK deployment path;
a third simulator (PhysX); natural arm swing. These are hardware-readiness and polish items
to revisit after stage 1.

## 15. Decisions (all accepted 2026-10-29)

Each was accepted as recommended.

1. **Who moves the arms.** Recommended: arm and hand targets go straight to PD controllers and
   the policy moves legs and waist (what NVIDIA, HOMIE, AMO and AGILE do). Alternative: the
   policy also moves the arms toward their targets (FALCON). That is closer to "whole-body"
   in name, but makes arm accuracy depend on RL and breaks the like-for-like swap with
   NVIDIA's controller.
2. **The command interface of section 3** (navigate, base height, torso orientation, arm and
   hand joint targets). Recommended: yes, it is NVIDIA's.
3. **Hands: Dex3.** Recommended: keep Dex3 and close the open item. NVIDIA's G1 setup and
   IsaacLab-Arena both use three-finger hands with Dex3's joint layout.
4. **GR00T N1.6 for stage 4,** not N1.7, whose G1 path (SONIC) has no command interface to
   swap under. Recommended: N1.6.
5. **PD gains and default pose: NVIDIA's,** so both controllers run on the same simulated
   robot. Recommended: yes.
6. **Compute for the ablation:** about 45 GPU-hours. Options: queue it on the current cluster
   partition (about 2 days back to back), find out whether another partition allows parallel
   jobs, or lift the workstation caps for long runs. Recommended: check MSI's partition
   limits with a read-only command first.
7. **Flat ground only in stage 1.** Recommended: yes; rough terrain later if time allows.

## Appendix A: reference controllers

Read from source code on 2026-10-29 (commits: GR00T-WholeBodyControl `1fa5f7f`, Isaac-GR00T
`n1.6.1-release` `5dc80c4`, IsaacLab-Arena `fd5566c`, unitree_rl_gym `276801e`).

**NVIDIA Decoupled WBC** ([NVlabs/GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl), code Apache-2.0, weights NVIDIA Open Model License)

- Lower body: two released ONNX networks, "Balance" (used when the velocity command is below
  0.05) and "Walk". Input: 6 frames of 86 values (velocity command scaled by 2, 2, 0.5;
  height; torso roll, pitch, yaw; angular velocity × 0.5; projected gravity; 29 joint positions;
  29 joint velocities × 0.05; last 15 actions). Output: 15 targets (12 legs, 3 waist), scale
  0.25. 50 Hz control, 200 Hz simulation. The network includes an estimator, probably of base
  velocity (not confirmed).
- Upper body: arm and hand joint targets go straight to PD control; IK is used only for
  teleoperation.
- PD gains: legs 150, 150, 150, 200, 40, 40 (damping 2, 2, 2, 4, 2, 2); waist 250 (5); arms
  100, 100, 40, 40, 20, 20, 20 (damping 5, 5, 2, 2, 2, 2, 2).
- Default leg pose: hip pitch -0.1, knee 0.3, ankle pitch -0.2 (the same as ours); waist 0;
  height command 0.74 m.
- Hands: 7 joints each, with Dex3-1's joint names (the repository does not say "Dex3").
- Training code, rewards and randomization for this controller are not released, and its
  training command ranges are not documented. NVIDIA's G1 simulation dataset uses velocity
  commands within ±0.5 m/s and ±1 rad/s and a constant height of 0.74 m.
- IsaacLab-Arena's G1 controller `homie_v2` has the same configuration, so NVIDIA's lower
  body appears to derive from HOMIE.

**GR00T N1.6, G1 embodiment** ([Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T)): actions
per step are left and right arm (7 each), left and right hand (7 each), waist (3), base height
(1) and navigate command (3), predicted 30 steps ahead; the state is the 43 joint positions.
NVIDIA reports 58% success on its simulated G1 apple-to-plate task.

**unitree_rl_gym G1** ([unitreerobotics/unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym), BSD-3-Clause): 12 leg joints, 50 Hz,
action scale 0.25, gains hip 100, knee 150, ankle 40. Gait clock period 0.8 s with the right
leg offset by half a period, stance below phase 0.55; the policy sees sin and cos of the
phase. Reward weights as in the table of section 5. Randomization: friction 0.1-1.25, added
mass -1 to +3 kg, pushes up to 1.5 m/s every 5 s. LSTM policy, 10000 iterations.

**Other G1 controllers:** HOMIE ([arXiv 2502.13013](https://arxiv.org/abs/2502.13013)):
RL legs, arm targets set directly, random-arm-pose curriculum, 6-step history. AMO
([arXiv 2505.03738](https://arxiv.org/abs/2505.03738)): RL legs and waist with velocity,
torso orientation and height commands. FALCON ([arXiv 2505.06776](https://arxiv.org/abs/2505.06776)):
separate RL agents for upper and lower body. NVIDIA WBC-AGILE
([nvidia-isaac/WBC-AGILE](https://github.com/nvidia-isaac/WBC-AGILE)): Isaac Lab 3.0, RL legs
with velocity and height commands, random arm and waist motion, teacher-student with a
recurrent student; the closest open reference for this stage.

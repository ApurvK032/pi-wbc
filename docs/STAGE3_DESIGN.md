# Stage 3 design: the pick-and-place task, the scripted expert and the demonstrations

**Status: proposed (2026-11-03).** Sections 1-2 and the decisions marked accepted in section 13
were agreed with Apurv on 2026-11-03 (D-030 to D-033 in [DECISIONS.md](DECISIONS.md)); the
rest waits for his OK. Results so far: [results/stage3](../results/stage3/README.md).

Stage 1 left a controller that walks, turns, crouches, leans and keeps its balance while the
arms move, driven through NVIDIA's command interface (D-012, D-028, D-029). Stage 3 builds the
task the VLA will learn in stage 4, an expert that solves it from simulator state through that
controller, and the demonstrations.

## 1. Goal and finish line

A G1 starts up to a few metres from a table, walks to it, picks up the object named in a
language instruction and puts it in the named bin. The expert reads the true state of the
simulation (object and robot poses) and sends exactly what the VLA will send: base velocity,
pelvis height, waist angles, and arm and hand joint targets.

Stage 3 is done when:

- the expert's success rate on frozen test layouts is reported with a 95% Wilson interval
  (target: at least 95%, with the interval's lower end at or above 90%, from at least 300
  layouts), with a table of failure causes and GIFs;
- the dataset of expert demonstrations has a manifest (episodes, layout seeds, split, commit,
  package versions, GPU, checksums), passes the replay check (section 8) for every episode, and
  loads with GR00T N1.6's and LeRobot's own data loaders.

## 2. What the two first tests settled

Both ran on 2026-11-03 before this design was written; details in
[results/stage3](../results/stage3/README.md).

**The Dex3 hand can pick up objects in Newton**, with physical contacts, no attachment trick:
a can and a bottle are held 27 of 27 times with the object up to 1 cm off the spot the hand
aims for (0.1-0.5 kg). What it takes:

- a side ("handshake") grasp: the palm faces the object from the side, fingers forward, and the
  hand comes straight down beside it; fingers and thumb close together;
- round objects: square ones are pinched off-centre and spin or tip (3/27 and 1/27);
- a torso lean of 0.25 rad to reach a 0.75 m table (upright, the grasp is 5 cm out of reach
  within the arm ranges the controller was trained with);
- 4 physics substeps (800 Hz) rather than training's 2: less slip in the hand, more margin;
- an arm path planned and solved by inverse kinematics on Newton's own model of the robot
  ([arm_ik.py](../src/pi_wbc/stage3/arm_ik.py)), with a slow correction of gravity sag.

**NVIDIA's controller runs on our G1 in Newton** through the same commands
([nvidia_wbc_test.py](../scripts/nvidia_wbc_test.py)): it walks (0.5 m/s gives 0.44-0.45),
turns in place better than ours (0.75-0.79 rad/s of 0.8) and crouches to 0.58 m of 0.6. On
Isaac Lab's G1 asset it could not stand still: its "Balance" policy drifted backwards at
0.25-0.4 m/s and often fell. The cause is the asset's foot contact geometry: its heel contact
spheres sit 3 cm further forward than in Unitree's URDF, NVIDIA's models and MuJoCo Menagerie's
(x = -0.02 against -0.05 m). With the heel moved to -0.05 m, NVIDIA's controller never fell in
any of the 11 test commands and stands still; ours is unchanged (no falls in 1,100 robots, 550
more with kicks) except for turning in place a little more slowly (0.64 against 0.70 rad/s).
So the scenes use the URDF's heel for every controller (D-038, proposed).

## 3. The scene

- **Table**: 0.75 m high, 0.6 m deep, 1.2 m wide, plain top (its colour is randomized for
  rendering only).
- **Objects**: three round objects at a time, drawn from a set of five that differ in colour
  and size (for example a red can, 5.2 cm across and 12 cm tall; a blue bottle, 4.4 x 14 cm; a
  green cup; a yellow can; a white bottle), 0.1-0.5 kg. Instructions name them by colour and
  kind.
- **Bins**: two open trays (about 18 x 22 cm, 6 cm walls), one at each end of the table, in
  two colours that are never an object's colour.
- **Robot start**: 1.5-3 m from the table edge, any heading within 90 degrees of facing it.
- **Cameras**: the head camera at Unitree's D435 mount (torso link, tilted 47.6 degrees down;
  as in the renderer test), 640 x 480; a fixed side camera for GIFs only, not in the dataset.
  Dex3 has no wrist cameras, and GR00T's G1 setup uses the head camera only (`ego_view`).
- **Physics**: Newton with training's settings except 4 substeps (to be checked with the
  controller first, section 11, step 1), and the feet's heel contact spheres at Unitree's
  URDF position (D-038).

**Layouts** are drawn from a seed: robot start, which three objects, their places on the table
(spread along its front 25 cm, at least 12 cm apart), which bin is where. **Seed ranges, frozen
now:** train 0-49,999; development 50,000-50,999; test 60,000-60,999. The test range is used
once, for the reported numbers.

**Instructions**: templates such as "put the red can in the blue bin", "place the white bottle
into the grey tray", "pick up the green cup and drop it in the blue bin", about ten in all, the
template drawn from the seed. One instruction per episode.

## 4. The command interface (what the expert and the VLA send)

Unchanged from stage 1 (D-012), and the same keys as GR00T N1.6's G1 setup in IsaacLab-Arena
(`navigate_command`, `base_height_command`, `waist`, `left_arm`, `right_arm`, `left_hand`,
`right_hand`), all absolute:

| Group | Values | Goes to |
|---|---|---|
| Navigate | forward and sideways speed, yaw rate (3) | controller |
| Body | pelvis height, waist yaw, roll, pitch (4) | controller |
| Arms | 7 joint targets per arm (14) | arm PD controllers directly |
| Hands | 7 joint targets per hand (14) | hand PD controllers directly |

The hands move between an open and a closed pose (D-033); the dataset stores their 7 joint
targets, so a VLA can be trained on either the joint values or the open/close amount.

## 5. The scripted expert

A state machine over the true state, sending commands every control step (50 Hz):

1. **Walk to the stance**: a point in front of the table from which the target object sits in
   front of the right shoulder at the grasp test's reach. Base velocity from a proportional
   controller on the stance's position and heading (turn first if facing away; walk; side-step
   the last few centimetres), arms hanging. The controller turns at 70-90% of the asked rate
   in place (D-029); the closed loop makes up for it.
2. **Settle**: zero velocity until the body has stopped moving.
3. **Reach**: lean the waist to 0.25 rad, raise the hand in front of the chest, move it above
   the object (arm path solved as in the grasp test, from the object's true pose).
4. **Grasp**: lower beside the object, correct the hand's position on the measured palm-object
   offset (privileged), close fingers and thumb together.
5. **Lift** 15 cm; check the object moved with the hand.
6. **Carry**: straighten the waist, side-step or turn until the bin is in reach, move the hand
   above the bin.
7. **Release**: open the hand; the object drops into the bin. Retract the arm, stand.

The expert retries nothing in version 1: a failed step ends the episode with a failure label.
Its own success rate is then honest about how hard the task is for the controller.

**Failure labels** (also stage 5's), from simulator state, the first that applies:

| Label | Test | Layer |
|---|---|---|
| fell | pelvis below 0.45 m or tilted past 60 degrees | controller |
| knocked | the target object moved over 3 cm or tipped before the grasp | controller (or expert) |
| missed grasp | hand closed, object not lifted with it | controller (or expert) |
| dropped | object left the hand before the release | controller |
| wrong object | another object lifted | high level |
| wrong bin | released over the other bin | high level |
| timed out | no success within 60 s | high level |

For the expert, "controller (or expert)" failures are separated by replaying the same layout
with the ideal body (section 7): a failure there is the expert's.

## 6. Which controller records the demonstrations

Both, over the same layouts. The main set goes through stage 1's controller, seed 1 (the only
seed that met the turning target, D-029); a matched set goes through NVIDIA's controller, now
that it stands with the URDF's heel (section 2). Recording is cheap, since rendering comes
afterwards (section 8). Stage 4 then chooses what the VLA trains on: our demonstrations only,
both mixed, or one VLA per controller, which measures the confound of D-032 directly. The
expert is the same for both; if it succeeds much less often through NVIDIA's controller, that
is itself a stage 5 result. The other two seeds of ours and the ablation controllers are used
in stage 5 for evaluation, not for recording.

## 7. The low-level layers for the stage 5 grid

The task environment takes any of these under the same commands (D-032):

- **ours**: the stage 1 student, run from its export (any seed);
- **ours without randomization and curriculum**: the stage 1 ablation controllers;
- **NVIDIA's**: Balance and Walk, as in the Newton test;
- **ideal body**: the pelvis follows the commanded velocity, height and waist exactly (its pose
  written every step, legs held), arms, hands and objects simulated as usual. It separates the
  expert's own mistakes from the controller's, and is the "perfect controller" row of the grid.

## 8. Recording, replay and rendering

**Record states first, render afterwards** (D-031). Each episode stores, every control step:
the commands (the actions), robot root pose and velocity, all joint positions and velocities,
object and bin poses, contact flags of the hands, the phase of the expert, plus the layout
seed, instruction and outcome. About 1 MB per 30 s episode before compression.

**Replay check**: (1) every recorded state is written back into the simulator and rendered,
which is exact; (2) every episode's commands are re-run through the physics and must give the
same outcome. Newton on the GPU does not repeat runs bit for bit (seen in the grasp test), so
(2) checks the outcome and how far the trajectory drifts, not equality; episodes that fail it
are dropped and counted.

**Rendering** with Newton's renderer (D-031): head camera, 640 x 480 at 25 frames per second,
randomized per episode: light direction, intensity and colour; floor, table and wall colours
and textures; camera pose jitter (1 cm, 1 degree). Object and bin colours stay fixed, since
the instructions name them. OVRTX re-renders a few episodes for the front page.

## 9. Export

- **GR00T N1.6**: LeRobot v2 layout with GR00T's `meta/modality.json`, embodiment tag
  `NEW_EMBODIMENT` (D-014); state = arm, hand and waist joint positions (plus legs, as in
  Isaac-GR00T's `unitree_g1`); action = the command groups of section 4; video = `ego_view`.
- **π0.5**: LeRobot's current dataset format, for LeRobot's `pi05` policy or openpi.
- **Size**: 1,000 training episodes to start (stage 4 decides whether more help), from train
  seeds; development and test layouts are never in the dataset. Few large files (one video per
  episode), since the MSI group's file count is its tight quota.
- Both loaders are installed in their own environments, after Apurv's OK (sizes stated first).

## 10. Compute

Expert rollouts are physics only, in batches of robots on the workstation's RTX 4070: 1,000
episodes of about 30 s are 1.5 million control steps. Their speed with the controller and the
objects in the loop is measured in step 2; a rough guess is one to two hours. Rendering the
replays with Newton's renderer: about 750,000 head-camera frames, roughly 10-20 minutes at the
renderer test's rate. Everything fits on the
workstation; MSI is needed only for stage 4's fine-tuning (and possibly the stage 2 runs in
parallel). Long unattended runs on the workstation still need Apurv's OK.

## 11. Order of work

1. Check the controller at 4 substeps (Isaac Lab nominal and kick tests, seed 1, against its
   2-substep numbers).
2. The task environment: scene, layouts, instructions, controller in the loop (ours first),
   ideal body; success and failure labels.
3. The expert, step by step on development layouts: walk to the stance; reach and grasp with
   the controller balancing; carry and release.
4. Expert success on the test layouts, failure table, GIFs.
5. Recorder and replay check.
6. Rendering with randomization.
7. Export to GR00T's and LeRobot's formats, load tests, manifest, dataset card.
8. Write-up, decision log, front page, PR.

## 12. Risks

- **Grasping while the controller balances.** The grasp test fixed the pelvis. Leaning with up
  to 0.5 kg in one hand is inside the controller's training (lean to 0.45 rad, 0-1 kg per
  palm), but the 1 cm grasp tolerance may be tight with the body swaying. Fallbacks: correct the
  hand on the measured palm-object offset just before closing (planned); less lean with a lower
  pelvis; a slower close.
- **The controller at 4 substeps** behaves a little differently from training. If it gets worse,
  use 4 substeps only near the grasp (not possible within one run) or tune the grasp for 2
  substeps (24/27 held there).
- **Tight grasp tolerance for the VLA.** About 1 cm of tolerance is demanding for a learned
  policy; stage 4 may see grasp misses dominate. That is a finding for stage 5, and the failure
  labels will show it.
- **NVIDIA's controller under load.** It stands and walks with the URDF's heel, but it has not
  been tested with kicks, leaning to reach, or an object in hand; the expert may need more
  margin with it.
- **Plain images**: Newton's renderer is simpler than RTX; if the VLA's vision struggles, the
  recorded states can be re-rendered with OVRTX without re-running anything.

## 13. Decisions

Accepted by Apurv on 2026-11-03:

- **D-030**: stage 3 now, stage 2 (motion imitation) later, alongside it on MSI.
- **D-031**: Newton's renderer for the dataset and for closed-loop testing; states recorded
  first and rendered afterwards; OVRTX for front-page GIFs.
- **D-032**: the perfect-layer grid for stage 5; NVIDIA's controller tried in Newton first (done:
  section 2); demonstrations recorded with ours (seed 1), and a second set with NVIDIA's once
  it stands (it does with the URDF's heel, D-038).
- **D-033**: Dex3 hands confirmed for GR00T N1.6 (its G1 joint space has 7 joints per hand with
  Dex3's names); the hands move between open and closed poses.

Proposed, for Apurv's OK:

- **D-034** (grasp): side grasp from above, round objects only, fingers and thumb closing
  together, torso lean 0.25 rad at a 0.75 m table, 4 physics substeps (after step 1's check).
- **D-035** (task): the scene, objects, bins, instructions and frozen seed ranges of section 3;
  right hand only in version 1.
- **D-036** (expert): the state machine and failure labels of section 5, no retries.
- **D-037** (dataset): 1,000 training episodes, head camera 640 x 480 at 25 frames per second,
  the replay check of section 8, GR00T and LeRobot formats; a matched set through NVIDIA's
  controller (section 6).
- **D-038** (feet): stage 3 onwards, the G1's heel contact spheres sit where Unitree's URDF has
  them (x = -0.05 m), for every controller; stage 1's Newton numbers were taken with the
  asset's -0.02 m.

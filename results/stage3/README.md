# Stage 3 results: task, scripted expert and demonstrations

Stage 3 builds the pick-and-place task, a scripted expert that solves it from simulator state
through the stage 1 controller, and the demonstrations the VLA learns from in stage 4. The plan
is in [PROJECT_PLAN.md](../../docs/PROJECT_PLAN.md), stage 3. Every result file in this folder
records the commit, package versions and GPU it came from.

**Contents**

- [Grasp test: can the Dex3 hand hold the task's objects in Newton?](#grasp-test-can-the-dex3-hand-hold-the-tasks-objects-in-newton)

## Grasp test: can the Dex3 hand hold the task's objects in Newton?

The biggest risk of the stage was whether Dex3's three fingers can physically pick up and hold
an object in Newton, so it was tested first, on a G1 with its base fixed
([scripts/grasp_test.py](../../scripts/grasp_test.py)). The answer is yes, for round objects:

![The right hand picks up a can](grasp/can_grasp.gif)

| Object (all 12-14 cm tall) | Placement error | Held | File |
|---|---|---:|---|
| Can, 5.2 cm across | up to 1 cm, 9 directions | 27/27 | [can-1cm.json](grasp/can-1cm.json) |
| Bottle, 4.4 cm across | up to 1 cm | 27/27 | [bottle-1cm.json](grasp/bottle-1cm.json) |
| Can | 1.5 cm, 8 directions | 20/24 | [can-1.5cm.json](grasp/can-1.5cm.json) |
| Square block, 4.5 cm | up to 1 cm | 3/27 | [block-1cm.json](grasp/block-1cm.json) |
| Box, 5 x 6.5 cm | up to 1 cm | 1/27 | [box-1cm.json](grasp/box-1cm.json) |
| Can, training's 2 physics substeps | up to 1 cm | 24/27 | [can-1cm-2substeps.json](grasp/can-1cm-2substeps.json) |
| Can, fingers closed before the thumb | up to 1 cm | 0/27 | [can-1cm-fingers-first.json](grasp/can-1cm-fingers-first.json) |

Each row is 3 object masses (0.1, 0.3 and 0.5 kg) times the placement errors: the object sits
that far off the spot the hand aims for, as when the walk to the table stops short. Held means
the object ends at least 10 cm above where it started and within 10 cm of the palm, after 3 s
of holding. The 1.5 cm misses are all with the can both farther forward and farther from the
palm (1.1 cm each way, 0/3) and one 1.5 cm forward. In the hold the can slides at most 1.7 cm in
the hand (4.4 cm with 2 substeps).

How the grasp works (the scripted expert will do the same):

- **Side grasp, coming straight down.** The palm faces the object from the side, fingers
  forward ("handshake"). Open, the fingers lie flat in the palm's plane and the thumb stands
  behind the object, so the hand comes down beside the object without touching it, 2-3 cm from
  the palm face. Closing curls the two fingers round the front and sweeps the thumb in behind,
  pressing the object against the palm (the palm carries about 90 N, each digit 40-100 N).
- **Fingers and thumb close together.** On a can both reach it about half-way through their
  sweep. Closing the fingers first pushes the object back before the thumb arrives: 0/27.
- **Round objects.** On a square block the thumb meets the back face near its far edge and the
  fingers the front face nearer the palm, so the pinch is off-centre and spins or tips the
  block. On a cylinder every contact pushes towards the axis. The task's objects are therefore
  cylinders (cans, bottles, cups) told apart by colour and size.
- **Where the object sits in the hand.** The grasp point is 8.5 cm along the fingers from the
  palm frame's origin and 6.2 cm out from it (2 cm from the palm face for the can), found with
  a kinematic contact sweep and then centred on the placement errors that still hold.
- **Reach.** Table top 0.75 m, pelvis at the controller's walking height (0.78 m), torso leaned
  forward 0.25 rad (the controller is trained for up to 0.45). Upright, a side grasp at that
  height is 5 cm out of reach within the arm ranges the controller was trained with.
- **Arm path.** The palm's path (rise in front of the chest, over the object, down, up) is
  solved for joint angles on Newton's own model of the robot
  ([src/pi_wbc/stage3/arm_ik.py](../../src/pi_wbc/stage3/arm_ik.py)), backwards from the grasp
  so that the arm stays on one smooth solution, and clamped to the controller's trained arm
  ranges. A slow, bounded correction on the measured palm error removes the 1-2 cm the arm
  sags under gravity; the palm arrives within 5 mm.
- **4 physics substeps** (800 Hz) instead of training's 2: the grasp holds with less slip (1.7
  instead of 4.4 cm) and more margin. With the controller in the loop this has to be checked
  (stage 1 trained at 2).

Not yet tested: the controller in the loop (here the pelvis is fixed), the left hand, and
releasing the object into a bin.

Also seen: Newton on the GPU does not repeat runs exactly. The block row gave 6/27 in an earlier
run of the same commit and settings, the box 2/27. The demonstration replay check (stage 3's
finish line) must allow for this: it can replay recorded states exactly for rendering, but a
physics re-run of recorded commands can only be expected to reproduce the outcome, not every
state.

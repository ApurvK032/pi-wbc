# Stage 1 results

Design: [docs/STAGE1_DESIGN.md](../../docs/STAGE1_DESIGN.md). This page grows as the steps of
its section 12 are done.

## Step 1: gait (in progress)

**In short:** a gait clock and Unitree's G1 rewards turn the stage 0 shuffle (about 11 steps
per second of 8 cm) into a walk of about 2.5 steps per second of about 30 cm, with no falls in
MuJoCo. These are development runs: one training run each, 1000 iterations on the workstation.
The step 1 result proper (3 runs per version, 3000 iterations, frozen test seeds) runs on the
cluster next.

Task: `PiWbc-G1-Gait` (`src/pi_wbc/tasks/g1_gait.py`): the policy drives the 12 leg and 3
waist joints with NVIDIA's Decoupled WBC gains; arms and hands are held still; velocity
commands sample the yaw rate directly; the policy sees a gait clock (period 0.8 s).

### Three versions

| Version | Change | Falls (all commands, both models) | Steps/s | Step length | Foot lift | Foot speed in stance | Speed error | Heading drift | Turn-in-place error | Stands still |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| stage 0 bench (13 runs) | built-in rewards | 17/1560 | 10.2 | 9 cm | 3.0 cm | 0.31 m/s | 0.12 m/s | — | 0.11-0.80 rad/s | yes |
| v1 | gait clock, Unitree's rewards | 0/120 | 2.5 | 37 cm | 5.8 cm | 0.15 m/s | 0.11 m/s | 0.08 rad/s | 0.18 rad/s | no: steps in place |
| v2 | + heading reward, stand signal | 0/120 | 2.6 | 34 cm | 5.1 cm | 0.14 m/s | 0.16 m/s | 0.05 rad/s | 0.09 rad/s | yes |
| v3 | + arms held in the default pose | 0/120 | 2.7 | 28 cm | 4.6 cm | 0.13 m/s | 0.27 m/s | 0.11 rad/s | 0.07 rad/s | yes |

Gait, speed error and heading drift: MuJoCo Menagerie model, 1 m/s command; turning: 0.8
rad/s in place; same protocol as stage 0 (10 seeded starts per command, 20 s). Heading drift
is the average turn rate over an episode minus the commanded one, which the per-step
turn-rate error hides under the rocking of each step. Raw data in [`gait_dev/`](gait_dev/)
(commit `746fc4c`, clean tree). Single training runs: differences between versions smaller
than stage 0's run-to-run spread (D-011) are not evidence.

### What happened on the way

- **The shuffle is fixed by describing the walk.** With a clock that says when each foot
  should be down and rewards for following it, lifting the swing foot and not slipping, every
  version steps at the clock's 2.5 steps per second with 4-6 cm of foot lift.
- **A bench mismatch looked like a transfer failure.** The first MuJoCo tests of v1 and v2
  fell often (up to 10 of 10 episodes per command) and walked in circles. The cause was the
  arms: no action drives them, Isaac Lab leaves such joints' targets at 0, and the bench held
  them in the default pose with elbows bent. The robot was being tested with a different
  centre of mass from training. The bench now holds such joints at the targets the export
  records; with that, both versions walk without falling (table above). Stage 0's hands had the
  same mismatch on a much smaller scale (re-run in its results, part 2).
- **Direct turn-rate commands need a heading term.** v1's turn-rate reward let a steady turn
  bias through. v2 adds a reward for staying on the heading the commanded turn rates add up to;
  it also gives the clock a "stand" reading (zero) when the command is zero, so the robot stops
  stepping when told to stand.
- **v3 holds the arms as designed** (default pose, elbows bent) instead of hanging. It walks
  without falling but falls short of 1 m/s at 1000 iterations (speed error 0.27 m/s, 28 cm
  steps), and its training curves were still improving. The cluster runs train for 3000
  iterations.

### Remaining gaps against the stage 1 targets (design, section 1)

At 1 m/s, v3 is short on speed (0.27 vs at most 0.15 m/s) and slip (0.13 vs at most
0.1 m/s), and just short on foot lift (4.6 vs at least 5 cm). Longer training comes first;
the swing-height target (an ankle height of 8 cm, about 4.5 cm of clearance) can be raised if
lift stays short.

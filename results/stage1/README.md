# Stage 1 results

Design: [docs/STAGE1_DESIGN.md](../../docs/STAGE1_DESIGN.md). This page grows as the steps of
its section 12 are done.

## Step 1: gait

**In short:** a gait clock and Unitree's G1 rewards turn the stage 0 shuffle (about 10 steps
per second of 9 cm) into a walk. Three training runs of 3000 iterations each take 2.5 steps per
second of 39 cm at 1 m/s and lift their feet 6.5 cm. Their speed error at 1 m/s is 0.06 m/s,
their turn-rate error when turning in place at most 0.08 rad/s, and none fell in 360 MuJoCo
episodes. Six of the seven gait targets of the
design are met. Slip is not: the foot moves at 0.15-0.21 m/s while it is down (target at most
0.1 m/s), and part of that may be how it is measured (see below).

<p align="center">
  <img src="gait_v3_s1_walk_1.0.gif" alt="G1 walking at 1 m/s in MuJoCo" width="400"><br>
  <em>Gait v3, training seed 1 (the first run, not picked), walking at 1 m/s in plain MuJoCo on
  MuJoCo Menagerie's G1 model.</em>
</p>

Task: `PiWbc-G1-Gait` (`src/pi_wbc/tasks/g1_gait.py`): the policy drives the 12 leg and 3
waist joints with NVIDIA's Decoupled WBC gains; arms and hands are held in the default pose;
velocity commands sample the yaw rate directly; the policy sees a gait clock (period 0.8 s).

### Result: 3 runs of v3, 3000 iterations

Training seeds 1-3, 3000 iterations, 4096 environments, on the workstation's RTX 4070 (about
58 min per run) at commit `3ea445b`, 2026-10-29. The design planned these runs for the cluster;
the workstation was free overnight and runs the same code. MuJoCo protocol as in stage 0: both
models, 6 commands, 10 seeded starts each, 20 s. The starts use evaluation seeds 0-9, inside
the development range (0-99); the test range (1000-1199) is kept for the final stage 1 result.

Gait on the Menagerie model at 1 m/s; mean over the 3 runs, with the lowest and highest run in
brackets:

| Measure (design, section 1) | Target | Stage 0 default (13 runs) | v3 (3 runs) | Met |
|---|---|---|---|---|
| Falls, all MuJoCo test commands, both models | upper 95% bound below 5% | 17/1560 (1%-2%) | 0/360 (0%-1%) | yes |
| Speed error while walking | at most 0.15 m/s | 0.12 m/s | 0.06 m/s (0.06-0.07); at most 0.09 m/s on any walking command and model | yes |
| Turn-rate error, turning in place at 0.8 rad/s | at most 0.2 rad/s | 0.11-0.80 rad/s | 0.06-0.08 rad/s | yes |
| Step rate at 1 m/s | 1.5-3 steps/s | 10.2 | 2.5 (2.4-2.5) | yes |
| Step length at 1 m/s | at least 0.3 m | 9 cm | 39 cm (38-39) | yes |
| Foot lift | at least 5 cm | 3.0 cm | 6.5 cm (6.1-6.9) | yes |
| Foot speed in stance (slip) | at most 0.1 m/s | 0.31 m/s | 0.17 m/s (0.15-0.21) | no |
| Push recovery, arm tracking | report | — | — | later steps |

Other measures at 1 m/s: both feet down 6% of the time (4-10%), knee range 23° (14-30°), heading
drift 0.02-0.08 rad/s. Isaac Lab's training statistics (last 50 iterations, over the random
training commands): speed error 0.14 m/s, turn-rate error 0.34-0.36 rad/s, 1.0-1.3% of episodes
ending in a fall. Stage 1's training commands differ from stage 0's, so these don't compare
directly with stage 0's.

Raw data: one file per run in [`gait_step1/`](gait_step1/) (commit `3ea445b`, clean tree), the
summary in [`gait_step1_compare.json`](gait_step1_compare.json), the training-side numbers in
[`gait_step1_training.json`](gait_step1_training.json). Rerun with
`scripts/train_eval_local.sh PiWbc-G1-Gait 3000 results/stage1/gait_step1 gait-v3 1 2 3`.

What the numbers show:

- **Longer training closed the speed gap.** The 1000-iteration development run of v3 (below)
  fell 0.27 m/s short of 1 m/s with 28 cm steps; all three 3000-iteration runs track within
  0.07 m/s with 38-39 cm steps.
- **The three runs agree** on step rate, step length, foot lift and tracking. They differ in
  how much they bend the knee: seed 3 uses 14° of knee range against 25-30° for the others, and
  has the most slip (0.21 m/s) and time with both feet down (10%).
- **Foot lift drops at slower commands:** 3.9-4.6 cm at 0.5 m/s and 1.6-2.9 cm when stepping
  sideways or turning in place. The gait targets are measured at 1 m/s (D-010).
- **Slip rose as the steps grew longer:** 0.13 m/s for the 1000-iteration run, 0.15-0.21 m/s
  now (Newton's model: 0.11-0.15 m/s).

### Slip: the one target not met, and a question about the measure

The bench measures slip as the planar speed of each ankle-roll link's origin while that foot
touches the floor. That point sits above the sole, so it also moves when the foot rolls from
heel to toe or pivots without sliding. Part of the 0.15-0.21 m/s may therefore be rolling. The
next check measures the speed of the contact points themselves (or only of feet flat on the
floor) on these same runs, before any reward is changed. Stage 0's numbers were measured the
same way, so the comparisons on this page hold either way.

### Development runs: three versions, 1000 iterations, one run each

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
  version steps at the clock's 2.5 steps per second with 4-6 cm of foot lift. Stage 0's reward
  pilot ([stage 0 results, part 5](../stage0/README.md#5-reward-pilot-four-reward-changes-two-runs-each))
  reaches a similar step rate by raising the foot-air-time reward alone, but none of its
  variants turns or holds its heading as well, and one of its two best runs sways.
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
- **v3 holds the arms as designed** (default pose, elbows bent) instead of hanging. At 1000
  iterations it fell short of 1 m/s, with its training curves still improving; 3000 iterations
  closed the gap (result above).

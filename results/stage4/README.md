# Stage 4 results: the VLA on top

Stage 4 fine-tunes a vision-language-action model on stage 3's demonstrations and tests it in
closed loop (draft plan: [STAGE4_DESIGN.md](../../docs/STAGE4_DESIGN.md)). Every result file in
this folder records the commit, package versions and GPU it came from.

## One judge for every high level (D-052)

Before any VLA is scored, the scripted expert, the replay check and the closed-loop evaluation
loop now judge an episode the same way (they did not: D-051). One judge
([src/pi_wbc/stage3/judge.py](../../src/pi_wbc/stage3/judge.py)) decides every episode from
simulator state, at every control step:

- a fall (pelvis below 0.45 m, or tilted more than 60 degrees) or another object moved more
  than 2 cm ends the episode at once (`fell`, `knocked_other`);
- otherwise it ends 4.5 s after the target is first in a bin and out of the hand, or at 60 s,
  and is labelled by its state then: `success` (resting in the named bin), `wrong_bin`,
  `dropped` (lifted, then let go outside the bins or knocked out of one), `knocked` (pushed more
  than 3 cm before a hand reached it), `missed_grasp` (reached but never lifted), `timed_out`
  (never reached, or still in the hand).

"Out of the hand" is the expert's own test: the object's centre more than 12 cm from both palms
(a held can is 7-10 cm from the palm frame). It is needed because the expert carries the object
low over the bin: it passes the "resting in the bin" test up to 3.3 s before the hand opens.
The 4.5 s cover the expert's withdrawal: from letting go to the end of its retract it takes
3.5-4.3 s (median 4.24 s over training set v1's 1,010 successes), so objects pushed while the
arm withdraws count against any policy, as they do against the expert.

**Against the expert's own labels**, judging the recordings again
([scripts/judge_recordings.py](../../scripts/judge_recordings.py); the palms are not recorded
and are computed from the recorded pelvis pose and joint angles on the controller's MuJoCo
model):

| | Training set v1 | Validation set v1 |
|---|---:|---:|
| Recordings | 1,088 | 256 |
| The expert's successes judged successes | 1,010 of 1,010 | 241 of 241 |
| The expert's failures judged successes | 45 | 4 |
| Same label | 1,015 | 244 |

Every expert failure judged a success is one the expert labelled `knocked_other`, and so are
most other differences: of its 66 (validation: 10) such episodes the judge calls 45 (4)
successes, 20 (5) timed out with the object still in the hand and 1 (1) a missed grasp. The
push happened in the control step at which the expert stopped, which the recorder does not
store; the largest push left in those recordings is 2.0 cm. In live runs the judge sees that
step. The remaining differences are names for the same grasp failures (training 7 of 12,
validation 2 of 5), each the expert's "missed grasp" unless said: its palm never came within
12 cm of the object (the judge: timed out, never reached; training 2), the object rose more
than 3 cm and was still against the hand when the episode stopped (timed out, still held; 1 and
1) or slipped out of it (dropped; 1 and 1), it was pushed away before a palm got near (knocked;
1); and twice the expert's "knocked" where the push crossed 3 cm in that unstored last step
(missed grasp). Files:
[judge/train-v1.json](judge/train-v1.json), [judge/val-v1.json](judge/val-v1.json).

## The command-rate test again, under the judge (D-048)

The expert's recorded commands replayed open loop through the evaluation loop
([scripts/vla_eval.py](../../scripts/vla_eval.py)), and through the replay check
([scripts/replay_check.py](../../scripts/replay_check.py)), on the 241 validation episodes:

| Commands sent | Success | 95% CI | Stage 3's rule (D-048) |
|---|---:|---|---:|
| every control step's (50 Hz) | 166 (68.9%) | 63-74% | 171 |
| the 25 fps dataset's, each held for two control steps | 119 (49.4%) | 43-56% | 120 |
| the 25 fps dataset's, the second step the mean of it and the next | 150 (62.2%) | 56-68% | 143 |
| the replay check, every control step's | 168 (69.7%) | 64-75% | 162 |

Paired on the same episodes: 50 Hz beats 25 Hz held (73 episodes against 26, p = 2.5e-6) and
averaging beats holding (59 against 28, p = 0.0012); 50 Hz against averaging is 48 against 32
(p = 0.09), no difference shown. So holding a 25 Hz command stream costs about 20 points, as
D-048 found, and averaging recovers most of that. The evaluation loop and the replay check,
judged alike, agree within noise (33 episodes against 35, p = 0.90): the loop adds no loss. Two
runs of the same commands still end differently on many single episodes (68 of 241 between the
loop and the check; Newton on the GPU does not repeat runs), so only paired totals are
compared. The stage 3 column comes from separate runs, not only another rule. Not rerun yet:
D-048's NVIDIA replays.

Files: [judge/replay50-val-v1.json](judge/replay50-val-v1.json),
[judge/replay25-val-v1.json](judge/replay25-val-v1.json),
[judge/replay25mean-val-v1.json](judge/replay25mean-val-v1.json),
[judge/replay-check-val-v1.json](judge/replay-check-val-v1.json) (the three loop runs record one
uncommitted file, docs/STAGE4_DESIGN.md, text only).

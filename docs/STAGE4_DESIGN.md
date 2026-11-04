# Stage 4 design (draft for review): the VLA on top

Drafted 2026-11-04 from what stage 3 left ready. Nothing here is decided until it is
accepted; each choice below becomes a D-NNN entry then.

## 1. Goal and finish line

A VLA reads the head camera and the instruction and sends the task's commands (arm and hand
joint targets, waist goals, pelvis height, walking velocity) to the stage 1 controller, which
keeps the robot balanced. Stage 4 is done when:

- GR00T N1.6, fine-tuned on the stage 3 demonstrations, has a closed-loop success rate on frozen
  test layouts with a 95% Wilson interval, a table of failure causes and GIFs;
- π0.5 has the same, or a written reason why it was dropped.

## 2. What stage 3 left ready

| | |
|---|---|
| Training data | v1: 1,000 episodes (790,046 frames at 25 fps); v2: 2,000 (contains v1); validation: 241, from layouts never trained on (D-047); v1, v2 and validation also at 50 fps (D-048) |
| Evaluation loop | [scripts/vla_eval.py](../scripts/vla_eval.py): live head camera, a GR00T policy server client (written to the server's message format, not yet run against a server); replaying the expert's own commands open loop succeeds on 71% of validation episodes at 50 Hz, 50% at the 25 fps dataset's rate (D-048) |
| Formats | GR00T-flavoured LeRobot v2.1 with GR00T's `unitree_g1` names; LeRobot v3.0 with quantile statistics (D-045) |
| Loader checks | GR00T's loader on every episode of each export (v2 at 50 fps: a 12-episode sample); LeRobot's `LeRobotDataset` on 12-episode samples (D-045, D-050) |
| GR00T on this PC | N1.6 environment (`third_party/Isaac-GR00T-n1d6`, branch n1d6); NVIDIA's G1 checkpoint runs on our observations: 6,830 MiB of GPU memory at batch 1 (7,154 at batch 8), 79 ms per call at batch 1, 377 ms at batch 8 (GPU shared with other jobs while measured) |
| Matched set | the training layouts through NVIDIA's controller (936 of its 1,000 episodes share a layout with ours; D-046), for the controller swap of stage 5 |
| Expert reference | 94.8% on test layouts 60384-60767 (D-041) |

## 3. Model: GR00T N1.6 first

- **N1.6** (open weights, `nvidia/GR00T-N1.6-3B`, 6.6 GB): the plan's model; its `UNITREE_G1`
  embodiment uses the same 43-joint state order as our data, and NVIDIA's own G1
  pick-and-place fine-tune reaches 58% (±15%) in their MuJoCo benchmark, a reference point.
  The base checkpoint has a G1 embodiment slot but no G1 statistics; fine-tuning computes them
  from our data (done: `meta/stats.json`).
- **N1.7** (GR00T's main branch since 2026): newer backbone (Cosmos-Reason2-2B, a gated model:
  access must be requested on Hugging Face), relative end-effector actions by default, which
  would need hand poses in the data. A later comparison, not the first run.

## 4. Fine-tuning (MSI, one L40S)

- **Frame rate and embodiment tag.** First run: the 50 fps export (D-048) under `UNITREE_G1` as
  GR00T defines it (arms as changes from the current angles, everything else absolute; 30-step
  chunks, 0.6 s, exactly as in NVIDIA's 50 fps G1 data). Second run: the 25 fps export under
  `NEW_EMBODIMENT` with a 16-step chunk (0.64 s), which the replay test predicts is worse (the
  25 Hz command stream alone costs about 21 points open loop). Compared on the validation set's action error and
  a short closed-loop pilot; the better one goes to the test layouts. The second run's config:
  [configs/groot/pi_wbc_g1_h16.py](../configs/groot/pi_wbc_g1_h16.py). GR00T stores the
  relative-arm statistics per chunk step (shape (30, 7) for `unitree_g1`) and keeps keys
  already in `meta/relative_stats.json`, so each chunk length gets its own dataset copy (data
  and videos hard-linked, `meta/` without that file); checked: (16, 7) statistics and 16-step
  training chunks on such a copy.
- **Settings.** GR00T's defaults: the projector, the action head and the language model's top
  4 layers train, vision and the rest of the language model stay frozen; learning rate 1e-4,
  colour jitter on. NVIDIA's G1 example uses 10,000 steps at batch 1,024 on 8 GPUs; one L40S
  takes about batch 64 (to be measured in the first job), so the first job measures steps per
  second and memory, then the run length is set.
- **Launcher.** [scripts/groot_finetune.py](../scripts/groot_finetune.py): GR00T's launch steps
  and defaults, with our arguments recorded next to the checkpoints. GR00T resumes from its last
  checkpoint in the output folder, so a 24 h job is resubmitted with `afterany` until done.
- **Not yet checked:** a fine-tune on this PC. It runs out of memory with 9.63 GiB allocated
  at batch 1 and 4, also with only the language model's top layers training (201 million
  parameters), while another job held 1.6 GB; GR00T asks for 40 GB or more to fine-tune. Data
  loading, the forward pass and our launcher all ran. The first MSI job is therefore a 50-step
  smoke run that saves a checkpoint, which `scripts/groot_inference_test.py` then loads here.

## 5. MSI setup

- GR00T N1.6's environment: Python 3.10, PyTorch 2.7.1 for CUDA 12.8, flash-attn 2.7.4
  (prebuilt wheel), installed with `uv sync --frozen` from the n1d6 checkout (MSI's driver
  580 supports CUDA 12.8). DeepSpeed is only used with more than one GPU; its import needs
  `CUDA_HOME` (module load cuda), or it can be removed as on this PC.
- Data: the first run's export (v1 at 50 fps) is 4.3 GiB in 2,008 files (one table and one
  video per episode, plus metadata); v1 at 25 fps is 3.2 GiB. The MSI
  group's file count is its tight quota; check it before uploading (`du --inodes`).
- Checkpoints: size with optimizer state to be measured in the smoke test; the launcher keeps
  the last 5.

## 6. Closed-loop evaluation (this PC)

- GR00T runs in its own environment (Python 3.10, PyTorch 2.7) and the simulator in ours
  (Python 3.12, PyTorch 2.11), so the evaluation uses GR00T's policy server (`run_gr00t_server.py`,
  ZeroMQ) and a client in our task: every few control steps the head camera is rendered live
  (Newton, each episode's look from its seed), the client sends image, joint state and
  instruction, and executes the returned chunk at the dataset's rate (50 Hz for the first run's
  50 fps data; with 25 fps data each command is held for two 50 Hz control steps).
- Lighting: the render-only trick of turning the whole scene for the sun's direction
  (D-043) can't be used with live physics; each batch gets a random global sun direction
  instead.
- Success and failure labels: one judge shared by the expert, the replay check and this loop
  (object resting in the named bin, no other object moved more than 2 cm until the arm has
  withdrawn, the robot's falls; 60 s limit), and timeouts labelled as timeouts. Not yet so: the
  loop stops 1 s after the object comes to rest, so it misses pushes during the retract, and it
  labels a timeout "dropped" (D-051). Fixed before any VLA is scored.
- Layouts: checkpoint choice on validation layouts (40000-40255, never trained on); the reported
  number on test layouts 60384-60767, the expert's official set, so every layout has the
  expert's outcome beside the VLA's. The expert's 94.8% there predates D-042's zero-velocity
  start and single layouts change outcome between runs, so the expert is run there again with
  the current code and the shared judge for that comparison.

## 7. π0.5 (after GR00T)

LeRobot's `pi05` pads state and action to 32 values; ours are 43 and 35. A π0.5 export drops the
legs from the state (31 values) and gives each hand one closing value (the working hand's
targets are exactly open + s x (closed - open), D-045): 23 action values. LoRA fine-tuning
needs more than 22.5 GB (fits an L40S); a full fine-tune needs an 80 GB card.

## 8. Risks

- **Success may be low.** NVIDIA's own G1 pick-and-place reaches 58% (±15%), a community N1.6 G1
  fine-tune 50%. Stage 5's attribution (VLA versus controller) is still meaningful at that level.
- **Rendering mismatch.** Training images were rendered after the fact with the scene turned;
  evaluation renders live without the turn. Same renderer, looks and camera; only the sun's
  direction is drawn differently.
- **MSI queue time** at low fair-share priority; each fine-tune is one to a few 24 h jobs.

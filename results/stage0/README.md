# Stage 0: Isaac Lab smoke test on the workstation

**Question:** does Isaac Lab 3.0 train a G1 policy on this PC with the Newton backend, and how
fast, at which environment counts?

**Answer:** yes. PPO on the built-in G1 flat-terrain task runs at 42,000 environment steps per
second with 512 environments and 122,000 with 8192, using at most 2.6 GB of the RTX 4070's
12 GB. At Isaac Lab's default settings (4096 environments, 1500 iterations) a full flat-terrain
run takes about 22 minutes here.

## Setup

| Item | Value |
|---|---|
| Task | `Isaac-Velocity-Flat-G1` (built-in), rsl_rl PPO, seed 1 |
| Robot | Isaac Lab's built-in G1 (`G1_MINIMAL_CFG`: older 37-joint model, reduced collision meshes), not the 29-DoF robot used from stage 1 |
| Physics | Newton 1.6.1, MuJoCo-Warp solver (`physics=newton_mjwarp`); 0.005 s step, 4 steps per policy step (50 Hz control) |
| Software | Isaac Lab 3.0.0, torch 2.11.0+cu130, Warp 1.17.0, rsl-rl-lib 5.5.1 |
| Hardware | RTX 4070 12 GB (driver 595.99.02), Ryzen 9 7900X; no CPU or GPU caps |
| Run | 50 PPO iterations of 24 steps per environment; the first 5 iterations are warm-up and left out |

Command (one run per environment count):

```bash
uv run pi-wbc-bench --num-envs 512 1024 2048 4096 8192 --warmup-iters 5 \
    --out results/stage0/bench_rtx4070_newton_g1flat.json -- \
    isaaclab train --rl_library rsl_rl --task Isaac-Velocity-Flat-G1 \
    --num_envs {num_envs} --max_iterations 50 --seed 1 physics=newton_mjwarp
uv run pi-wbc-bench-table results/stage0/bench_rtx4070_newton_g1flat.json
```

## Results

| Envs | Steps/s | Sim time / iter | Learn time / iter | Peak VRAM | Median power | Status |
|---:|---:|---:|---:|---:|---:|---|
| 512 | 42,016 | 0.253 s | 0.039 s | 0.5 GB | 83 W | ok |
| 1024 | 65,472 | 0.335 s | 0.040 s | 0.6 GB | 102 W | ok |
| 2048 | 91,147 | 0.493 s | 0.046 s | 0.9 GB | 125 W | ok |
| 4096 | 110,434 | 0.814 s | 0.077 s | 1.5 GB | 144 W | ok |
| 8192 | 122,181 | 1.461 s | 0.149 s | 2.6 GB | 158 W | ok |

- **Steps/s** is rsl_rl's figure: environment (policy) steps per second, including learning
  time; median over the 45 timed iterations. Physics steps are 4× higher.
- **Peak VRAM** is the whole GPU's memory in use (nvidia-smi), start-up included; the desktop
  used 15 MB before each run. **Median power** covers the timed iterations only; the card's
  limit is 200 W.
- Raw data: [`bench_rtx4070_newton_g1flat.json`](bench_rtx4070_newton_g1flat.json), with a
  provenance record (package versions, GPU, driver, time).

## What this shows

- Throughput is still rising at 8192 environments, but slowly: doubling from 4096 to 8192 adds
  11%. Power reaches 158 of 200 W, so the GPU is close to full use. Simulation takes 87-91% of
  each iteration; PPO updates are the rest.
- During the 1024-environment run the Isaac Lab process held one CPU core at 100% while the GPU
  drew about 100 W (one spot check), so small runs are probably limited by Python and
  kernel-launch overhead rather than the GPU.
- Memory is not a constraint for this task: 8192 environments use under a quarter of 12 GB.
- Start-up (imports, scene build, solver set-up) takes 12-16 s per run once GPU kernels are
  cached. The very first run on this machine spent 15 s on solver set-up and 23 s on the first
  iteration while Warp compiled its kernels; later runs reuse `~/.cache/warp`.

## Limits

- One run per environment count. Iteration-to-iteration variation is small (coefficient of
  variation 0.5-1.2%), but run-to-run variation was not measured.
- Throughput only: 50 iterations are far too few to learn to walk, and the mean reward was not
  meant to improve. Training quality is stage 1's job.
- Flat terrain and the 37-joint built-in G1. Rough terrain (height scanner, terrain mesh) and
  our 29-DoF robot will be slower; re-measure in stage 1.
- The result file was written before the repository's first commit, so its provenance has no
  commit hash. Re-run after the first commit to get one.

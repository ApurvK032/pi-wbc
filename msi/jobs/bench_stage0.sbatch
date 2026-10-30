#!/bin/bash
# Stage 0 throughput benchmark on a cluster GPU: the same sweep as the workstation's
# (built-in Isaac-Velocity-Flat-G1, Newton, 50 iterations per environment count), plus 16384
# environments since cluster GPUs have 48 GB or more.
#
#   mkdir -p logs/slurm
#   sbatch -A <account> -p <partition> msi/jobs/bench_stage0.sbatch
#
# Writes logs/results/stage0/bench_<gpu>_newton_g1flat.json (e.g. bench_l40s_...). Jobs write
# only under the ignored logs/ folder: a job that wrote a tracked file would make the next
# `git pull` on the cluster fail once that file is committed on the workstation.
#SBATCH --job-name=piwbc-bench
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --output=logs/slurm/%x-%j.out

set -euo pipefail
SIF=${PIWBC_SIF:-$HOME/containers/pi-wbc.sif}
cd "${SLURM_SUBMIT_DIR}"
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | awk '{print tolower($NF)}')

echo "commit $(git rev-parse --short HEAD), uncommitted changes: $(git status --porcelain | wc -l)"
echo "node $(hostname -s), GPU ${GPU}"

apptainer exec --nv "${SIF}" uv run --frozen pi-wbc-bench \
    --num-envs 512 1024 2048 4096 8192 16384 --warmup-iters 5 \
    --label "stage 0 smoke test on the cluster: built-in Isaac-Velocity-Flat-G1, Newton MJWarp, ${GPU}" \
    --out "logs/results/stage0/bench_${GPU}_newton_g1flat.json" -- \
    isaaclab train --rl_library rsl_rl --task Isaac-Velocity-Flat-G1 \
    --num_envs {num_envs} --max_iterations 50 --seed 1 physics=newton_mjwarp

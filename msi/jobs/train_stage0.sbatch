#!/bin/bash
# Train the stage 0 G1 bench policy on one GPU, then export it for the MuJoCo bench.
#
# Submit from the repository root; account and partition stay on the command line so they
# never enter the repository:
#   mkdir -p logs/slurm
#   sbatch -A <account> -p <partition> msi/jobs/train_stage0.sbatch [seed]
#
# Output: logs/rsl_rl/pi_wbc_g1_flat_stage0/slurm-<jobid>/ (checkpoints, exported/ bundle)
# and the job log logs/slurm/piwbc-stage0-<jobid>.out. The run is ~30 min, far below the
# 24 h limit, so it has no checkpoint-and-requeue logic (long stage 1 runs will).
#SBATCH --job-name=piwbc-stage0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=logs/slurm/%x-%j.out

set -euo pipefail
SEED=${1:-1}
SIF=${PIWBC_SIF:-$HOME/containers/pi-wbc.sif}
TASK=PiWbc-G1-Flat-Stage0
RUN=slurm-${SLURM_JOB_ID}
cd "${SLURM_SUBMIT_DIR}"

echo "commit $(git rev-parse --short HEAD), uncommitted changes: $(git status --porcelain | wc -l)"
echo "node $(hostname -s), GPU $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"
echo "seed ${SEED}, container ${SIF}"

run() { apptainer exec --nv "${SIF}" uv run --frozen "$@"; }

run isaaclab train --rl_library rsl_rl --task "${TASK}" --num_envs 4096 --seed "${SEED}" \
    --run_timestamp "${RUN}" physics=newton_mjwarp

RUN_DIR=logs/rsl_rl/pi_wbc_g1_flat_stage0/${RUN}
CHECKPOINT=$(ls "${RUN_DIR}"/model_*.pt | sort -V | tail -1)
run pi-wbc-export --task "${TASK}" --checkpoint "${CHECKPOINT}" physics=newton_mjwarp
echo "done: ${RUN_DIR}"

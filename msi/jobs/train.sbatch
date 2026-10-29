#!/bin/bash
# Train one of our Isaac Lab tasks on one GPU, then export the policy for the MuJoCo bench.
#
# Submit from the repository root; account and partition stay on the command line so they
# never enter the repository:
#   mkdir -p logs/slurm
#   sbatch -A <account> -p <partition> msi/jobs/train.sbatch <task> [seed] [tag] [overrides...]
#
#   task       registered task id, e.g. PiWbc-G1-Gait
#   seed       training seed (default 1)
#   tag        short name for this variant, added to the run folder (default: base)
#   overrides  Isaac Lab (Hydra) config overrides, e.g. env.rewards.gait_contact.weight=0.3
#
# Output: logs/rsl_rl/<experiment>/slurm-<jobid>-<tag>/ (checkpoints, params/, exported/) and
# the job log logs/slurm/piwbc-train-<jobid>.out. Everything is written under the ignored
# logs/ folder (docs/MSI.md, working rules). A 3000-iteration gait run takes about 1-1.5 h on
# an A40, inside the 24 h limit, so there is no checkpoint-and-requeue logic yet.
#SBATCH --job-name=piwbc-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=logs/slurm/%x-%j.out

set -euo pipefail
TASK=${1:?usage: train.sbatch <task> [seed] [tag] [overrides...]}
SEED=${2:-1}
TAG=${3:-base}
shift $(( $# < 3 ? $# : 3 ))
OVERRIDES=("$@")
SIF=${PIWBC_SIF:-$HOME/containers/pi-wbc.sif}
RUN=slurm-${SLURM_JOB_ID}-${TAG}
cd "${SLURM_SUBMIT_DIR}"

echo "commit $(git rev-parse --short HEAD), uncommitted changes: $(git status --porcelain | wc -l)"
echo "node $(hostname -s), GPU $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"
echo "task ${TASK}, seed ${SEED}, variant ${TAG}, overrides: ${OVERRIDES[*]:-none}"

run() { apptainer exec --nv "${SIF}" uv run --frozen "$@"; }

run isaaclab train --rl_library rsl_rl --task "${TASK}" --num_envs 4096 --seed "${SEED}" \
    --run_timestamp "${RUN}" physics=newton_mjwarp "${OVERRIDES[@]}"

RUN_DIR=$(ls -d logs/rsl_rl/*/"${RUN}")
CHECKPOINT=$(ls "${RUN_DIR}"/model_*.pt | sort -V | tail -1)
run pi-wbc-export --task "${TASK}" --checkpoint "${CHECKPOINT}" physics=newton_mjwarp \
    "${OVERRIDES[@]}"
echo "done: ${RUN_DIR}"

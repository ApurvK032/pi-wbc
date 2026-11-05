#!/bin/bash
# Load a GR00T N1.6 checkpoint and run it on observations from an exported dataset (stage 4):
# every action group comes back with its size and finite values, and the time and GPU memory per
# call are measured (scripts/groot_inference_test.py). GR00T's environment inside the cluster
# image, set up as in groot_finetune.sbatch. Arguments go to scripts/groot_inference_test.py.
#
# Give a checkpoint-<step> folder: it holds the processor files at its top level, which GR00T's
# policy reads (the run's final folder keeps them in processor/).
#
#   mkdir -p logs/slurm
#   sbatch -A <account> -p <gpu partition> --gres=gpu:1 msi/jobs/groot_check.sbatch \
#       --model logs/groot/<run>/<run>/checkpoint-<step> \
#       --dataset logs/export/pi_wbc_g1_pick_place_val_v1_50fps --out logs/groot/<run>/check.json
#
# Job log: logs/slurm/piwbc-groot-check-<job>.out
#SBATCH --job-name=piwbc-groot-check
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=logs/slurm/%x-%j.out

set -uo pipefail
SIF=${PIWBC_SIF:-$HOME/containers/pi-wbc.sif}
GROOT=third_party/Isaac-GR00T-n1d6
cd "${SLURM_SUBMIT_DIR}" || exit 1
export HF_HUB_OFFLINE=1
[ -x "${GROOT}/.venv/bin/python" ] || { echo "no GR00T environment at ${GROOT}/.venv"; exit 1; }
echo "code $(git rev-parse --short HEAD); GR00T $(git -C "${GROOT}" rev-parse --short HEAD);" \
    "node $(hostname -s); $(date '+%F %T')"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
apptainer exec --nv "${SIF}" "${GROOT}/.venv/bin/python" scripts/groot_inference_test.py "$@"

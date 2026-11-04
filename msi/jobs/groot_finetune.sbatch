#!/bin/bash
# Fine-tune GR00T N1.6 on one GPU (stage 4, draft: docs/STAGE4_DESIGN.md section 4).
#
# Runs scripts/groot_finetune.py with GR00T's own environment (third_party/Isaac-GR00T-n1d6,
# branch n1d6) inside the cluster image (msi/pi-wbc.def). Arguments after the script name go
# to scripts/groot_finetune.py. GR00T resumes from the last checkpoint in --output, so the same
# command can be resubmitted after a 24 h job ends (sbatch --dependency=afterany:<job>).
#
# One-time setup on a login node, from the repository root (GR00T N1.6 wants Python 3.10;
# the image pins 3.12 and turns uv's Python downloads off, so both are overridden here; the
# image has no CUDA toolkit, and DeepSpeed, used only with several GPUs, fails to import
# without one, so it is removed):
#   git clone --depth 1 -b n1d6 https://github.com/NVIDIA/Isaac-GR00T.git third_party/Isaac-GR00T-n1d6
#   (cd third_party/Isaac-GR00T-n1d6 && git checkout 9b37aa1)   # the commit checked locally
#   apptainer exec ~/containers/pi-wbc.sif env UV_PYTHON=3.10 UV_PYTHON_DOWNLOADS=automatic \
#       uv sync --frozen --project third_party/Isaac-GR00T-n1d6
#   apptainer exec ~/containers/pi-wbc.sif uv pip uninstall \
#       --python third_party/Isaac-GR00T-n1d6/.venv/bin/python deepspeed
#   apptainer exec ~/containers/pi-wbc.sif third_party/Isaac-GR00T-n1d6/.venv/bin/hf download \
#       nvidia/GR00T-N1.6-3B
#
# Submit from the repository root; account and partition stay on the command line:
#   mkdir -p logs/slurm
#   sbatch -A <account> -p <gpu partition> --gres=gpu:l40s:1 msi/jobs/groot_finetune.sbatch \
#       --base-model nvidia/GR00T-N1.6-3B \
#       --dataset logs/export/pi_wbc_g1_pick_place_train_v1_50fps \
#       --tag UNITREE_G1 --output logs/groot/<run name> --max-steps 10000 --batch 64
#
# Job log: logs/slurm/piwbc-groot-<job>.out
#SBATCH --job-name=piwbc-groot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/%x-%j.out

set -uo pipefail
SIF=${PIWBC_SIF:-$HOME/containers/pi-wbc.sif}
GROOT=third_party/Isaac-GR00T-n1d6
cd "${SLURM_SUBMIT_DIR}" || exit 1
export HF_HUB_OFFLINE=1  # the base model is downloaded on a login node first (see above)
[ -x "${GROOT}/.venv/bin/python" ] || { echo "no GR00T environment at ${GROOT}/.venv"; exit 1; }
echo "code $(git rev-parse --short HEAD); GR00T $(git -C "${GROOT}" rev-parse --short HEAD);" \
    "node $(hostname -s); $(date '+%F %T')"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
apptainer exec --nv "${SIF}" "${GROOT}/.venv/bin/python" scripts/groot_finetune.py "$@"

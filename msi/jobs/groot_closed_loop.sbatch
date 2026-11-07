#!/bin/bash
# Closed-loop evaluation of a GR00T N1.6 checkpoint in the pick-place task (stage 4): GR00T's
# policy server (GR00T's own environment, as in groot_finetune.sbatch) and scripts/vla_eval.py
# (the project's environment, which needs the `vla` dependency group: `uv sync --frozen`) run
# side by side on one GPU, both in the cluster image. Episodes are judged by the shared judge
# (D-052). The first argument is a checkpoint-<step> folder; the rest go to vla_eval.py.
#
#   mkdir -p logs/slurm
#   sbatch -A <account> -p <gpu partition> --gres=gpu:1 msi/jobs/groot_closed_loop.sbatch \
#       logs/groot/<run>/<run>/checkpoint-<step> --fps 50 --execute 15 --num-envs 32 \
#       --seeds 40000-40255 --out logs/vla/<name>.json
#
# Job log: logs/slurm/piwbc-closed-loop-<job>.out
#SBATCH --job-name=piwbc-closed-loop
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm/%x-%j.out

set -uo pipefail
MODEL=${1:?usage: groot_closed_loop.sbatch <checkpoint folder> [vla_eval.py arguments]}
shift
SIF=${PIWBC_SIF:-$HOME/containers/pi-wbc.sif}
GROOT=third_party/Isaac-GR00T-n1d6
PORT=$((20000 + SLURM_JOB_ID % 10000))
cd "${SLURM_SUBMIT_DIR}" || exit 1
export HF_HUB_OFFLINE=1
[ -d "${MODEL}" ] || { echo "no checkpoint at ${MODEL}"; exit 1; }
echo "code $(git rev-parse --short HEAD); GR00T $(git -C "${GROOT}" rev-parse --short HEAD);" \
    "model ${MODEL}; node $(hostname -s); $(date '+%F %T')"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
apptainer exec --nv "${SIF}" "${GROOT}/.venv/bin/python" "${GROOT}/gr00t/eval/run_gr00t_server.py" \
    --model-path "${MODEL}" --embodiment-tag UNITREE_G1 --host 127.0.0.1 --port "${PORT}" &
SERVER=$!
trap 'kill ${SERVER} 2>/dev/null' EXIT
for _ in $(seq 120); do  # up once its port answers (loading the model takes about a minute)
    (exec 3<>"/dev/tcp/127.0.0.1/${PORT}") 2>/dev/null && break
    kill -0 "${SERVER}" 2>/dev/null || { echo "the GR00T server exited"; exit 1; }
    sleep 5
done
apptainer exec --nv "${SIF}" .venv/bin/python scripts/vla_eval.py --policy groot \
    --host 127.0.0.1 --port "${PORT}" --viz none "$@"

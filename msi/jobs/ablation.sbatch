#!/bin/bash
# Stage 1 ablation on the cluster: for each configuration in turn, train a teacher with that
# randomization group switched off, distill a student from it at the teacher's last curriculum
# level, held fixed (D-019), and export the student; all in one job.
#
# Submit from the repository root; account and partition stay on the command line so they
# never enter the repository:
#   mkdir -p logs/slurm
#   sbatch -A <account> -p <partition> --gres=gpu:l40s:1 --time=06:00:00 \
#       --export=ALL,PIWBC_COMMIT=$(git rev-parse HEAD) \
#       msi/jobs/ablation.sbatch <seed> <config>...
#
#   seed      training seed
#   config    "none" (every group off) or a group of g1_robust.ABLATION_OVERRIDES: friction,
#             mass, actuators, latency, sensor_noise or pushes
#
# Each configuration is a teacher (3000 iterations) and a student (1000), as
# scripts/teacher_student_local.sh does on the workstation; about 65 min on an L40S. Output per
# configuration: logs/rsl_rl/pi_wbc_g1_teacher/slurm-<jobid>-abl-<config>/ (checkpoints),
# logs/rsl_rl/pi_wbc_g1_student/slurm-<jobid>-abl-<config>/ (checkpoints, the exported/ student),
# training logs logs/slurm/piwbc-abl-<jobid>-<config>-{teacher,student}.log, job log
# logs/slurm/piwbc-abl-<jobid>.out. A configuration that fails is reported and skipped. Runs
# from a copy of PIWBC_COMMIT made under logs/snapshots/, so a `git pull` cannot change its code.
#SBATCH --job-name=piwbc-abl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm/%x-%j.out

set -uo pipefail
SEED=${1:?usage: ablation.sbatch <seed> <config>...}
shift
CONFIGS=("$@")
[ ${#CONFIGS[@]} -gt 0 ] || { echo "no configurations given"; exit 1; }
SIF=${PIWBC_SIF:-$HOME/containers/pi-wbc.sif}
cd "${SLURM_SUBMIT_DIR}"

COMMIT=${PIWBC_COMMIT:-$(git rev-parse HEAD)}
SNAP=logs/snapshots/${SLURM_JOB_ID}
git worktree add --detach --quiet "${SNAP}" "${COMMIT}" || exit 1
trap 'git worktree remove --force "${SNAP}"' EXIT
export PYTHONPATH="${PWD}/${SNAP}/src${PYTHONPATH:+:${PYTHONPATH}}"
echo "commit ${COMMIT:0:7}, run from a copy at ${SNAP}"
echo "node $(hostname -s), GPU $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"
echo "ablation, seed ${SEED}, configurations: ${CONFIGS[*]}"

run() { apptainer exec --nv "${SIF}" uv run --frozen "$@"; }

for CONFIG in "${CONFIGS[@]}"; do
    TAG=abl-${CONFIG//_/-}
    RUN=slurm-${SLURM_JOB_ID}-${TAG}
    LOG=logs/slurm/piwbc-abl-${SLURM_JOB_ID}-${CONFIG}
    # shellcheck disable=SC2086  # the overrides are separate arguments
    if ! ABL=$(run python -c '
import sys
from pi_wbc.tasks.g1_robust import ABLATION_OVERRIDES as A
names = sys.argv[1:]
groups = list(A) if names == ["none"] else names
print(" ".join(o for g in groups for o in A[g]))
' "${CONFIG}"); then
        echo "$(date +%H:%M) ${CONFIG}: unknown configuration, skipped"
        continue
    fi
    read -r -a OVERRIDES <<< "${ABL}"
    echo "$(date +%H:%M) ${CONFIG}: teacher, overrides: ${ABL}"
    if ! run isaaclab train --rl_library rsl_rl --task PiWbc-G1-Teacher --num_envs 4096 \
        --seed "${SEED}" --max_iterations 3000 --run_timestamp "${RUN}" physics=newton_mjwarp \
        "${OVERRIDES[@]}" > "${LOG}-teacher.log" 2>&1; then
        echo "$(date +%H:%M) ${CONFIG}: teacher failed, see ${LOG}-teacher.log"
        continue
    fi
    T_CKPT=$(ls logs/rsl_rl/pi_wbc_g1_teacher/"${RUN}"/model_*.pt | sort -V | tail -1)
    LEVEL=$(tr '\r' '\n' < "${LOG}-teacher.log" | sed 's/\x1b\[[0-9;]*m//g' \
        | awk '/Curriculum\/upper_body_level/{v=$2} END{print (v == "" ? 0 : v)}')
    echo "$(date +%H:%M) ${CONFIG}: student from ${T_CKPT}, curriculum level ${LEVEL}, held"
    if ! run isaaclab train --rl_library rsl_rl --task PiWbc-G1-Student --num_envs 4096 \
        --seed "${SEED}" --max_iterations 1000 --run_timestamp "${RUN}" --checkpoint "${T_CKPT}" \
        physics=newton_mjwarp "${OVERRIDES[@]}" \
        "env.commands.body_pose.initial_level=${LEVEL}" "env.commands.upper_body.initial_level=${LEVEL}" \
        env.curriculum.upper_body_level.params.raise_below=-1.0 \
        env.curriculum.upper_body_level.params.lower_above=2.0 \
        > "${LOG}-student.log" 2>&1; then
        echo "$(date +%H:%M) ${CONFIG}: student failed, see ${LOG}-student.log"
        continue
    fi
    S_CKPT=$(ls logs/rsl_rl/pi_wbc_g1_student/"${RUN}"/model_*.pt | sort -V | tail -1)
    if ! run pi-wbc-export --task PiWbc-G1-Student --checkpoint "${S_CKPT}" physics=newton_mjwarp \
        "${OVERRIDES[@]}" > "${LOG}-export.log" 2>&1; then
        echo "$(date +%H:%M) ${CONFIG}: export failed, see ${LOG}-export.log"
        continue
    fi
    echo "$(date +%H:%M) ${CONFIG}: done, logs/rsl_rl/pi_wbc_g1_student/${RUN}"
done
echo "$(date +%H:%M) all configurations done"

#!/bin/bash
# Train a teacher, distill a student from it and export the student, in one job (stage 1, step 4).
#
# Submit from the repository root; account and partition stay on the command line so they
# never enter the repository:
#   mkdir -p logs/slurm
#   sbatch -A <account> -p <partition> --export=ALL,PIWBC_COMMIT=$(git rev-parse HEAD) \
#       msi/jobs/teacher_student.sbatch [seed] [tag] [teacher_iters] [student_iters] [overrides...]
#
#   seed            training seed (default 1)
#   tag             short name, added to both run folders (default: base)
#   teacher_iters   PPO iterations of the teacher (default 3000)
#   student_iters   distillation iterations of the student (default 1000)
#   overrides       Isaac Lab (Hydra) overrides for both trainings and the export, e.g. an
#                   ablation's from pi_wbc.tasks.g1_robust.ABLATION_OVERRIDES
#
# The student's curriculum level rises from 0 to the teacher's last level over its first 7200
# environment steps (300 iterations), then stays there (D-021). Output:
# logs/rsl_rl/pi_wbc_g1_teacher/slurm-<jobid>-<tag>/ and logs/rsl_rl/pi_wbc_g1_student/
# slurm-<jobid>-<tag>/ (checkpoints, params/, the student's exported/), job log
# logs/slurm/piwbc-ts-<jobid>.out. Runs from a copy of PIWBC_COMMIT made under
# logs/snapshots/, so a `git pull` while the job waits or runs cannot change its code.
# About 2 h on an A40 at the defaults.
#SBATCH --job-name=piwbc-ts
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/%x-%j.out

set -euo pipefail
SEED=${1:-1}
TAG=${2:-base}
T_ITERS=${3:-3000}
S_ITERS=${4:-1000}
shift $(( $# < 4 ? $# : 4 ))
OVERRIDES=("$@")
SIF=${PIWBC_SIF:-$HOME/containers/pi-wbc.sif}
RUN=slurm-${SLURM_JOB_ID}-${TAG}
cd "${SLURM_SUBMIT_DIR}"

COMMIT=${PIWBC_COMMIT:-$(git rev-parse HEAD)}
SNAP=logs/snapshots/${SLURM_JOB_ID}
git worktree add --detach --quiet "${SNAP}" "${COMMIT}"
trap 'git worktree remove --force "${SNAP}"' EXIT
export PYTHONPATH="${PWD}/${SNAP}/src${PYTHONPATH:+:${PYTHONPATH}}"
echo "commit ${COMMIT:0:7}, run from a copy at ${SNAP}"
echo "node $(hostname -s), GPU $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"
echo "teacher-student, seed ${SEED}, variant ${TAG}, ${T_ITERS}+${S_ITERS} iterations, overrides: ${OVERRIDES[*]:-none}"

run() { apptainer exec --nv "${SIF}" uv run --frozen "$@"; }

run isaaclab train --rl_library rsl_rl --task PiWbc-G1-Teacher --num_envs 4096 --seed "${SEED}" \
    --max_iterations "${T_ITERS}" --run_timestamp "${RUN}" physics=newton_mjwarp \
    "${OVERRIDES[@]}" > "logs/slurm/piwbc-ts-${SLURM_JOB_ID}-teacher.log" 2>&1
T_DIR=logs/rsl_rl/pi_wbc_g1_teacher/${RUN}
T_CKPT=$(ls "${T_DIR}"/model_*.pt | sort -V | tail -1)
LEVEL=$(tr '\r' '\n' < "logs/slurm/piwbc-ts-${SLURM_JOB_ID}-teacher.log" \
    | sed 's/\x1b\[[0-9;]*m//g' \
    | awk '/Curriculum\/upper_body_level/{v=$2} END{print (v == "" ? 0 : v)}')
echo "teacher done: ${T_CKPT}, curriculum level ${LEVEL}"

run isaaclab train --rl_library rsl_rl --task PiWbc-G1-Student --num_envs 4096 --seed "${SEED}" \
    --max_iterations "${S_ITERS}" --run_timestamp "${RUN}" --checkpoint "${T_CKPT}" \
    physics=newton_mjwarp "${OVERRIDES[@]}" \
    "env.commands.body_pose.initial_level=${LEVEL}" "env.commands.upper_body.initial_level=${LEVEL}" \
    env.curriculum.upper_body_level.params.raise_below=-1.0 \
    env.curriculum.upper_body_level.params.lower_above=2.0 \
    env.curriculum.upper_body_level.params.ramp_steps=7200 \
    > "logs/slurm/piwbc-ts-${SLURM_JOB_ID}-student.log" 2>&1
S_DIR=logs/rsl_rl/pi_wbc_g1_student/${RUN}
S_CKPT=$(ls "${S_DIR}"/model_*.pt | sort -V | tail -1)
echo "student done: ${S_CKPT}"

run pi-wbc-export --task PiWbc-G1-Student --checkpoint "${S_CKPT}" physics=newton_mjwarp \
    "${OVERRIDES[@]}"
echo "done: ${S_DIR}"

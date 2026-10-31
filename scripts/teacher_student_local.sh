#!/usr/bin/env bash
# Train a teacher, distill a student from it, export the student and test it in MuJoCo, for
# several seeds, one after another on the workstation (stage 1, step 4).
#
#   setsid nohup scripts/teacher_student_local.sh TEACHER_ITERS STUDENT_ITERS OUT_DIR LABEL SEED... \
#       > logs/<name>.log 2>&1 < /dev/null &
#
# Per seed: PiWbc-G1-Teacher for TEACHER_ITERS (PPO, privileged observations), then
# PiWbc-G1-Student for STUDENT_ITERS from the teacher's last checkpoint (distillation), then
# the student's export and the MuJoCo bench under every test condition in CONDITIONS (default:
# nominal and the five step 5 conditions), development seeds 0-9. Results:
# OUT_DIR/<LABEL>-s<seed>-<condition>.json.
#
# The student trains at the teacher's last curriculum level, held fixed (D-019). A teacher whose
# last checkpoint is already there is not trained again; set TEACHER_LABEL to use the teacher of
# another label (pc-<TEACHER_LABEL>-teacher-s<seed>-<TEACHER_ITERS>it).
# Set OVERRIDES for Isaac Lab (Hydra) overrides to both trainings and the export. Set ABLATION
# to switch randomization groups off by name (g1_robust.ABLATION_OVERRIDES; "none" for all of
# them, "full" for none of them) or training choices (g1_robust.TRAINING_ABLATIONS, such as
# "curriculum"), from that module of the commit being run. Set PPO_STUDENT_ITERS to train the
# student by PPO alone for that many iterations, with no teacher (PiWbc-G1-Student-PPO). Set
# PIWBC_COMMIT to run a given commit instead of HEAD. Set TEST_OUT to also test each student on
# the test seeds (scripts/test_seeds.sh, in the background on CPU threads 16 and up) there.
#
# Like train_eval_local.sh: every step runs the code of the starting commit from a temporary
# copy, at the lowest CPU priority on 16 of the 24 CPU threads, and a failed seed is skipped.
set -uo pipefail

T_ITERS=${1:?usage: $0 TEACHER_ITERS STUDENT_ITERS OUT_DIR LABEL SEED...}
S_ITERS=${2:?}
OUT=${3:?}
LABEL=${4:?}
shift 4
MENAGERIE=third_party/mujoco_menagerie/unitree_g1/scene_with_hands.xml
CONDITIONS=${CONDITIONS:-nominal friction_0.4 payload_1kg delay_20ms noise push_1.0}
# D-019: the student keeps its starting level (never raised, never lowered)
FIXED_LEVEL=(
    "env.curriculum.upper_body_level.params.raise_below=-1.0"
    "env.curriculum.upper_body_level.params.lower_above=2.0"
)
export PATH=$HOME/.local/bin:$PATH
run() { nice -n 19 taskset -c 0-15 uv run --frozen "$@"; }
read -r -a EXTRA <<< "${OVERRIDES:-}"
mkdir -p "${OUT}" logs

COMMIT=${PIWBC_COMMIT:-$(git rev-parse HEAD)}
SNAP=$(mktemp -d)/pi-wbc-${COMMIT:0:7}
git worktree add --detach --quiet "${SNAP}" "${COMMIT}" || exit 1
trap 'git worktree remove --force "${SNAP}"' EXIT
export PYTHONPATH="${SNAP}/src${PYTHONPATH:+:${PYTHONPATH}}"
if [ -n "${ABLATION:-}" ]; then
    # shellcheck disable=SC2086  # group names are separate arguments
    ABL=$(uv run --frozen python -c '
import sys
from pi_wbc.tasks.g1_robust import ABLATION_OVERRIDES as A, TRAINING_ABLATIONS as T
names = sys.argv[1:]
groups = list(A) if names == ["none"] else [] if names == ["full"] else names
print(" ".join(o for g in groups for o in {**A, **T}[g]))
' ${ABLATION}) || exit 1
    read -r -a ABL_ARGS <<< "${ABL}"
    EXTRA+=("${ABL_ARGS[@]}")
    OVERRIDES="${OVERRIDES:-} ${ABL}"
fi
echo "[$(date +%H:%M)] commit ${COMMIT:0:7}, run from a copy at ${SNAP}" \
    "(uncommitted changes, not used: $(git status --porcelain -- src | wc -l) in src/)"
echo "[$(date +%H:%M)] overrides: ${OVERRIDES:-none}; conditions: ${CONDITIONS}"

for SEED in "$@"; do
    if [ -n "${PPO_STUDENT_ITERS:-}" ]; then
        S_TASK=PiWbc-G1-Student-PPO
        S_RUN=pc-${LABEL}-student-ppo-s${SEED}-${PPO_STUDENT_ITERS}it
        echo "[$(date +%H:%M)] seed ${SEED}: PPO student, no teacher, ${PPO_STUDENT_ITERS} iterations (${S_RUN})"
        if ! run isaaclab train --rl_library rsl_rl --task "${S_TASK}" --num_envs 4096 \
            --max_iterations "${PPO_STUDENT_ITERS}" --seed "${SEED}" --run_timestamp "${S_RUN}" \
            physics=newton_mjwarp "${EXTRA[@]}" > "logs/train_${S_RUN}.log" 2>&1; then
            echo "[$(date +%H:%M)] seed ${SEED}: PPO student failed, see logs/train_${S_RUN}.log"
            continue
        fi
    else
        S_TASK=PiWbc-G1-Student
        T_RUN=pc-${TEACHER_LABEL:-${LABEL}}-teacher-s${SEED}-${T_ITERS}it
        S_RUN=pc-${LABEL}-student-fixed-s${SEED}-${S_ITERS}it
        T_LAST=logs/rsl_rl/pi_wbc_g1_teacher/${T_RUN}/model_$((T_ITERS - 1)).pt
        if [ -f "${T_LAST}" ] && [ -f "logs/train_${T_RUN}.log" ]; then
            echo "[$(date +%H:%M)] seed ${SEED}: teacher already trained, ${T_LAST}"
        else
            echo "[$(date +%H:%M)] seed ${SEED}: teacher, ${T_ITERS} iterations (${T_RUN})"
            if ! run isaaclab train --rl_library rsl_rl --task PiWbc-G1-Teacher --num_envs 4096 \
                --max_iterations "${T_ITERS}" --seed "${SEED}" --run_timestamp "${T_RUN}" \
                physics=newton_mjwarp "${EXTRA[@]}" > "logs/train_${T_RUN}.log" 2>&1; then
                echo "[$(date +%H:%M)] seed ${SEED}: teacher training failed, see logs/train_${T_RUN}.log"
                continue
            fi
        fi
        T_DIR=$(ls -d logs/rsl_rl/*/"${T_RUN}")
        T_CKPT=$(ls "${T_DIR}"/model_*.pt | sort -V | tail -1)
        # the student trains at the teacher's last level (it would restart at 0 otherwise)
        LEVEL=$(tr '\r' '\n' < "logs/train_${T_RUN}.log" | sed 's/\x1b\[[0-9;]*m//g' \
            | awk '/Curriculum\/upper_body_level/{v=$2} END{print (v == "" ? 0 : v)}')
        echo "[$(date +%H:%M)] seed ${SEED}: student, ${S_ITERS} iterations from ${T_CKPT}," \
            "curriculum level ${LEVEL}, held (${S_RUN})"
        if ! run isaaclab train --rl_library rsl_rl --task PiWbc-G1-Student --num_envs 4096 \
            --max_iterations "${S_ITERS}" --seed "${SEED}" --run_timestamp "${S_RUN}" \
            --checkpoint "${T_CKPT}" physics=newton_mjwarp "${EXTRA[@]}" \
            "env.commands.body_pose.initial_level=${LEVEL}" \
            "env.commands.upper_body.initial_level=${LEVEL}" "${FIXED_LEVEL[@]}" \
            > "logs/train_${S_RUN}.log" 2>&1; then
            echo "[$(date +%H:%M)] seed ${SEED}: student training failed, see logs/train_${S_RUN}.log"
            continue
        fi
    fi
    S_DIR=$(ls -d logs/rsl_rl/*/"${S_RUN}")
    S_CKPT=$(ls "${S_DIR}"/model_*.pt | sort -V | tail -1)
    echo "[$(date +%H:%M)] seed ${SEED}: exporting ${S_CKPT}"
    if ! run pi-wbc-export --task "${S_TASK}" --checkpoint "${S_CKPT}" physics=newton_mjwarp \
        "${EXTRA[@]}" > "logs/export_${S_RUN}.log" 2>&1; then
        echo "[$(date +%H:%M)] seed ${SEED}: export failed, see logs/export_${S_RUN}.log"
        continue
    fi
    if [ -n "${TEST_OUT:-}" ]; then
        echo "[$(date +%H:%M)] seed ${SEED}: test seeds in the background, into ${TEST_OUT}"
        setsid nohup scripts/test_seeds.sh "${LABEL}-s${SEED}" "${S_DIR}/exported" "${TEST_OUT}" \
            > "logs/test_seeds_${LABEL}-s${SEED}.log" 2>&1 < /dev/null &
    fi
    for COND in ${CONDITIONS}; do
        echo "[$(date +%H:%M)] seed ${SEED}: MuJoCo, condition ${COND}"
        run pi-wbc-sim2sim --label "${LABEL}-s${SEED}" --condition "${COND}" \
            --policy "${S_DIR}/exported/policy.onnx" --policy-io "${S_DIR}/exported/policy_io.json" \
            --model "newton=${S_DIR}/exported/newton_model.xml" --model "menagerie=${MENAGERIE}" \
            --seeds 10 --out "${OUT}/${LABEL}-s${SEED}-${COND}.json" \
            > "logs/sim2sim_${S_RUN}_${COND}.txt" 2>&1 \
            || echo "[$(date +%H:%M)] seed ${SEED}: MuJoCo ${COND} failed"
    done
done
echo "[$(date +%H:%M)] all seeds done"

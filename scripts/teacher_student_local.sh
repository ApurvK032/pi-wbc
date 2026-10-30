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
# nominal and the five step 5 conditions). Results: OUT_DIR/<LABEL>-s<seed>-<condition>.json.
# Set OVERRIDES for Isaac Lab (Hydra) overrides to both trainings and the export (e.g. an
# ablation's from g1_robust.ABLATION_OVERRIDES).
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
export PATH=$HOME/.local/bin:$PATH
run() { nice -n 19 taskset -c 0-15 uv run --frozen "$@"; }
read -r -a EXTRA <<< "${OVERRIDES:-}"
mkdir -p "${OUT}" logs

COMMIT=$(git rev-parse HEAD)
SNAP=$(mktemp -d)/pi-wbc-${COMMIT:0:7}
git worktree add --detach --quiet "${SNAP}" "${COMMIT}" || exit 1
trap 'git worktree remove --force "${SNAP}"' EXIT
export PYTHONPATH="${SNAP}/src${PYTHONPATH:+:${PYTHONPATH}}"
echo "[$(date +%H:%M)] commit ${COMMIT:0:7}, run from a copy at ${SNAP}" \
    "(uncommitted changes, not used: $(git status --porcelain -- src | wc -l) in src/)"
echo "[$(date +%H:%M)] overrides: ${OVERRIDES:-none}; conditions: ${CONDITIONS}"

for SEED in "$@"; do
    T_RUN=pc-${LABEL}-teacher-s${SEED}-${T_ITERS}it
    S_RUN=pc-${LABEL}-student-s${SEED}-${S_ITERS}it
    echo "[$(date +%H:%M)] seed ${SEED}: teacher, ${T_ITERS} iterations (${T_RUN})"
    if ! run isaaclab train --rl_library rsl_rl --task PiWbc-G1-Teacher --num_envs 4096 \
        --max_iterations "${T_ITERS}" --seed "${SEED}" --run_timestamp "${T_RUN}" \
        physics=newton_mjwarp "${EXTRA[@]}" > "logs/train_${T_RUN}.log" 2>&1; then
        echo "[$(date +%H:%M)] seed ${SEED}: teacher training failed, see logs/train_${T_RUN}.log"
        continue
    fi
    T_DIR=$(ls -d logs/rsl_rl/*/"${T_RUN}")
    T_CKPT=$(ls "${T_DIR}"/model_*.pt | sort -V | tail -1)
    echo "[$(date +%H:%M)] seed ${SEED}: student, ${S_ITERS} iterations from ${T_CKPT} (${S_RUN})"
    if ! run isaaclab train --rl_library rsl_rl --task PiWbc-G1-Student --num_envs 4096 \
        --max_iterations "${S_ITERS}" --seed "${SEED}" --run_timestamp "${S_RUN}" \
        --checkpoint "${T_CKPT}" physics=newton_mjwarp "${EXTRA[@]}" \
        > "logs/train_${S_RUN}.log" 2>&1; then
        echo "[$(date +%H:%M)] seed ${SEED}: student training failed, see logs/train_${S_RUN}.log"
        continue
    fi
    S_DIR=$(ls -d logs/rsl_rl/*/"${S_RUN}")
    S_CKPT=$(ls "${S_DIR}"/model_*.pt | sort -V | tail -1)
    echo "[$(date +%H:%M)] seed ${SEED}: exporting ${S_CKPT}"
    if ! run pi-wbc-export --task PiWbc-G1-Student --checkpoint "${S_CKPT}" physics=newton_mjwarp \
        "${EXTRA[@]}" > "logs/export_${S_RUN}.log" 2>&1; then
        echo "[$(date +%H:%M)] seed ${SEED}: export failed, see logs/export_${S_RUN}.log"
        continue
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

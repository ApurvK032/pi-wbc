#!/usr/bin/env bash
# Train a task on the workstation for several seeds, one after another; export each run and
# evaluate it in MuJoCo (both models, 10 seeded starts per command).
#
#   setsid nohup scripts/train_eval_local.sh TASK ITERATIONS OUT_DIR LABEL SEED... \
#       > logs/<name>.log 2>&1 < /dev/null &
#
# Started this way it does not depend on the login session, so a dropped connection doesn't
# stop it. It runs at the lowest CPU priority on 16 of the 24 CPU threads (the workstation's
# CPU cap). Each result is OUT_DIR/<LABEL>-s<seed>-<run>.json, labelled <LABEL>-s<seed> for
# pi-wbc-sim2sim-compare. A seed whose training fails is skipped; the next one still runs.
set -uo pipefail

TASK=${1:?usage: $0 TASK ITERATIONS OUT_DIR LABEL SEED...}
ITERS=${2:?}
OUT=${3:?}
LABEL=${4:?}
shift 4
MENAGERIE=third_party/mujoco_menagerie/unitree_g1/scene_with_hands.xml
export PATH=$HOME/.local/bin:$PATH
run() { nice -n 19 taskset -c 0-15 uv run --frozen "$@"; }
mkdir -p "${OUT}" logs

echo "[$(date +%H:%M)] commit $(git rev-parse --short HEAD), uncommitted: $(git status --porcelain | wc -l)"
for SEED in "$@"; do
    RUN=pc-${LABEL}-s${SEED}-${ITERS}it
    echo "[$(date +%H:%M)] seed ${SEED}: training ${TASK} for ${ITERS} iterations (${RUN})"
    if ! run isaaclab train --rl_library rsl_rl --task "${TASK}" --num_envs 4096 \
        --max_iterations "${ITERS}" --seed "${SEED}" --run_timestamp "${RUN}" \
        physics=newton_mjwarp > "logs/train_${RUN}.log" 2>&1; then
        echo "[$(date +%H:%M)] seed ${SEED}: training failed, see logs/train_${RUN}.log"
        continue
    fi
    DIR=$(ls -d logs/rsl_rl/*/"${RUN}")
    CKPT=$(ls "${DIR}"/model_*.pt | sort -V | tail -1)
    echo "[$(date +%H:%M)] seed ${SEED}: exporting ${CKPT}"
    if ! run pi-wbc-export --task "${TASK}" --checkpoint "${CKPT}" physics=newton_mjwarp \
        > "logs/export_${RUN}.log" 2>&1; then
        echo "[$(date +%H:%M)] seed ${SEED}: export failed, see logs/export_${RUN}.log"
        continue
    fi
    echo "[$(date +%H:%M)] seed ${SEED}: evaluating in MuJoCo"
    run pi-wbc-sim2sim --label "${LABEL}-s${SEED}" \
        --policy "${DIR}/exported/policy.onnx" --policy-io "${DIR}/exported/policy_io.json" \
        --model "newton=${DIR}/exported/newton_model.xml" --model "menagerie=${MENAGERIE}" \
        --seeds 10 --out "${OUT}/${LABEL}-s${SEED}-${RUN}.json" > "logs/sim2sim_${RUN}.txt" 2>&1 \
        || echo "[$(date +%H:%M)] seed ${SEED}: evaluation failed, see logs/sim2sim_${RUN}.txt"
done
echo "[$(date +%H:%M)] all seeds done"

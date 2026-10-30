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
#
# Set OVERRIDES to pass Isaac Lab (Hydra) config overrides to training and export, e.g.
#   OVERRIDES="env.commands.upper_body.rel_still_envs=1.0" scripts/train_eval_local.sh ...
#
# Every step runs the code of the commit checked out at the start, from a temporary copy of
# it, so edits made while the runs go on can't reach a later export or evaluation (one export
# once picked up a half-written task change). Uncommitted changes are not used.
set -uo pipefail

TASK=${1:?usage: $0 TASK ITERATIONS OUT_DIR LABEL SEED...}
ITERS=${2:?}
OUT=${3:?}
LABEL=${4:?}
shift 4
MENAGERIE=third_party/mujoco_menagerie/unitree_g1/scene_with_hands.xml
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
echo "[$(date +%H:%M)] overrides: ${OVERRIDES:-none}"
for SEED in "$@"; do
    RUN=pc-${LABEL}-s${SEED}-${ITERS}it
    echo "[$(date +%H:%M)] seed ${SEED}: training ${TASK} for ${ITERS} iterations (${RUN})"
    if ! run isaaclab train --rl_library rsl_rl --task "${TASK}" --num_envs 4096 \
        --max_iterations "${ITERS}" --seed "${SEED}" --run_timestamp "${RUN}" \
        physics=newton_mjwarp "${EXTRA[@]}" > "logs/train_${RUN}.log" 2>&1; then
        echo "[$(date +%H:%M)] seed ${SEED}: training failed, see logs/train_${RUN}.log"
        continue
    fi
    DIR=$(ls -d logs/rsl_rl/*/"${RUN}")
    CKPT=$(ls "${DIR}"/model_*.pt | sort -V | tail -1)
    echo "[$(date +%H:%M)] seed ${SEED}: exporting ${CKPT}"
    if ! run pi-wbc-export --task "${TASK}" --checkpoint "${CKPT}" physics=newton_mjwarp \
        "${EXTRA[@]}" > "logs/export_${RUN}.log" 2>&1; then
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

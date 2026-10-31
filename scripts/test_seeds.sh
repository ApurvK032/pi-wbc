#!/usr/bin/env bash
# Test an exported student on the frozen test seeds (1000-1199) in the MuJoCo bench, every
# condition at once, one CPU thread each (stage 1 final results).
#
#   scripts/test_seeds.sh LABEL EXPORT_DIR OUT_DIR
#
# EXPORT_DIR holds policy.onnx, policy_io.json and newton_model.xml (a run's exported/). Both
# models, all commands, 200 starts each; results OUT_DIR/<LABEL>-<condition>.json.gz
# (gzip-compressed: `pi_wbc.provenance.load_json` reads them), logs
# logs/sim2sim_test_<LABEL>_<condition>.txt. Set CONDITIONS to change the list and CPUS to the
# first CPU thread to use (default 16: threads 16 and up, beside training's 0-15). Waits for all.
set -uo pipefail

LABEL=${1:?usage: $0 LABEL EXPORT_DIR OUT_DIR}
EXP=${2:?}
OUT=${3:?}
CONDITIONS=${CONDITIONS:-nominal friction_0.4 payload_1kg delay_20ms noise push_1.0 push_2.0}
CPU=${CPUS:-16}
MENAGERIE=third_party/mujoco_menagerie/unitree_g1/scene_with_hands.xml
export PATH=$HOME/.local/bin:$PATH
mkdir -p "${OUT}" logs

pids=()
for COND in ${CONDITIONS}; do
    nice -n 19 taskset -c "${CPU}" uv run --frozen pi-wbc-sim2sim --label "${LABEL}" \
        --condition "${COND}" --policy "${EXP}/policy.onnx" --policy-io "${EXP}/policy_io.json" \
        --model "newton=${EXP}/newton_model.xml" --model "menagerie=${MENAGERIE}" \
        --seeds 200 --seed-start 1000 --out "${OUT}/${LABEL}-${COND}.json.gz" \
        > "logs/sim2sim_test_${LABEL}_${COND}.txt" 2>&1 &
    pids+=($!)
    CPU=$((CPU + 1))
done
failed=0
for pid in "${pids[@]}"; do
    wait "${pid}" || failed=$((failed + 1))
done
echo "[$(date +%H:%M)] ${LABEL}: test seeds done, ${failed} condition(s) failed"

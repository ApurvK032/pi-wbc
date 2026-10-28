#!/usr/bin/env bash
# Evaluate exported cluster runs in MuJoCo, a few at a time; one result file per run.
#
#   scripts/sim2sim_runs.sh OUT_DIR RUN_DIR...
#
# RUN_DIR is logs/rsl_rl/<experiment>/slurm-<jobid>[-<variant>] with an exported/ bundle. The
# training seed is read from the job log logs/slurm/piwbc-stage0-<jobid>.out, and each result
# is labelled <variant>-s<seed> (variant "base" when the folder has none). Each run is checked
# on Newton's own model and on MuJoCo Menagerie's G1, 10 seeds per command (evaluate.py).
# Set JOBS to change how many run at once (default 6). Run from the repository root.
set -euo pipefail

OUT=${1:?usage: $0 OUT_DIR RUN_DIR...}
shift
MENAGERIE=third_party/mujoco_menagerie/unitree_g1/scene_with_hands.xml
JOBS=${JOBS:-6}
mkdir -p "${OUT}" logs/sim2sim
[[ -f ${MENAGERIE} ]] || { echo "run scripts/fetch_menagerie.sh first" >&2; exit 1; }

evaluate() {
    local run=$1 name rest job variant seed label
    name=$(basename "${run}")  # e.g. slurm-4920328-airtime
    rest=${name#slurm-}
    job=${rest%%-*}
    variant=base
    [[ ${rest} == *-* ]] && variant=${rest#*-}
    seed=$(sed -n 's/^seed \([0-9][0-9]*\),.*/\1/p' "logs/slurm/piwbc-stage0-${job}.out" | head -1)
    [[ -n ${seed} ]] || { echo "no seed in the job log of ${run}" >&2; return 1; }
    label=${variant}-s${seed}
    uv run --frozen pi-wbc-sim2sim --label "${label}" \
        --policy "${run}/exported/policy.onnx" --policy-io "${run}/exported/policy_io.json" \
        --model "newton=${run}/exported/newton_model.xml" --model "menagerie=${MENAGERIE}" \
        --seeds 10 --out "${OUT}/${label}.json" > "logs/sim2sim/${label}.txt" 2>&1
    echo "${label}: done"
}

for run in "$@"; do
    while (( $(jobs -rp | wc -l) >= JOBS )); do wait -n; done
    evaluate "${run}" &
done
wait

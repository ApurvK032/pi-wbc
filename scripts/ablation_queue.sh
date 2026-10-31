#!/usr/bin/env bash
# Stage 1 ablation on the workstation: train a teacher and student with one randomization group
# (or all of them) switched off, one configuration after another, and test each student in the
# MuJoCo bench (design, section 11).
#
#   setsid nohup scripts/ablation_queue.sh SEED [CONFIG...] > logs/ablation_queue.log 2>&1 < /dev/null &
#
# CONFIG is "none" (every group off) or a group of g1_robust.ABLATION_OVERRIDES: friction, mass,
# actuators, latency, sensor_noise or pushes. The default is all seven, "none" first. Each runs
# scripts/teacher_student_local.sh at 3000 + 1000 iterations (as the full student) with that
# group switched off. Results: results/stage1/ablation/abl-<config>-s<seed>-<condition>.json.
# A configuration whose results are all there is skipped, so the queue can simply be started
# again after an interruption. Every configuration runs the commit the queue started at.
set -uo pipefail

SEED=${1:?usage: $0 SEED [CONFIG...]}
shift
CONFIGS=("$@")
[ ${#CONFIGS[@]} -eq 0 ] && CONFIGS=(none pushes latency friction mass actuators sensor_noise)
OUT=results/stage1/ablation
export CONDITIONS=${CONDITIONS:-nominal friction_0.4 payload_1kg delay_20ms noise push_1.0 push_2.0}
export PIWBC_COMMIT=${PIWBC_COMMIT:-$(git rev-parse HEAD)}
echo "[$(date +%H:%M)] ablation queue, seed ${SEED}, commit ${PIWBC_COMMIT:0:7}: ${CONFIGS[*]}"

for CONFIG in "${CONFIGS[@]}"; do
    LABEL=abl-${CONFIG//_/-}
    missing=0
    for COND in ${CONDITIONS}; do
        [ -f "${OUT}/${LABEL}-s${SEED}-${COND}.json" ] || missing=1
    done
    if [ "${missing}" -eq 0 ]; then
        echo "[$(date +%H:%M)] ${CONFIG}: results already there, skipped"
        continue
    fi
    echo "[$(date +%H:%M)] ${CONFIG}: start (${LABEL})"
    ABLATION=${CONFIG} scripts/teacher_student_local.sh 3000 1000 "${OUT}" "${LABEL}" "${SEED}"
    echo "[$(date +%H:%M)] ${CONFIG}: finished"
done
echo "[$(date +%H:%M)] ablation queue done"

#!/usr/bin/env bash
# Stage 1 ablation on the workstation: train a teacher and student with one randomization group
# (or all of them) switched off, one configuration after another, and test each student in the
# MuJoCo bench, on development seeds and in the background on the test seeds (design, section 9).
#
#   setsid nohup scripts/ablation_queue.sh SEED [CONFIG...] > logs/ablation_queue.log 2>&1 < /dev/null &
#
# CONFIG is "full" (every group on), "none" (every group off), a group of
# g1_robust.ABLATION_OVERRIDES (friction, mass, actuators, latency, sensor_noise, pushes),
# "curriculum" (no curriculum: full ranges from the start, g1_robust.TRAINING_ABLATIONS, and no
# warm-up for the student either, RAMP_STEPS=0) or "teacher" (no teacher: the student trained by
# PPO alone for 4000 iterations, the budget of teacher and student together). The default is all
# ten, "full" first. Each runs scripts/teacher_student_local.sh at 3000 + 1000 iterations with
# that change, the student with the warm-up (D-021); "full" reuses the teacher of the
# full-length run (pc-ts-full-teacher-s<seed>-3000it), and a teacher already trained is reused.
# Results:
# results/stage1/ablation/abl-<config>-s<seed>-<condition>.json (development seeds 0-9) and
# results/stage1/ablation_test/abl-<config>-s<seed>-<condition>.json.gz (test seeds 1000-1199).
# A configuration whose development results are all there is skipped, so the queue can simply
# be started again after an interruption. Every configuration runs the commit the queue
# started at.
set -uo pipefail

SEED=${1:?usage: $0 SEED [CONFIG...]}
shift
CONFIGS=("$@")
[ ${#CONFIGS[@]} -eq 0 ] && CONFIGS=(full pushes none latency friction mass actuators sensor_noise
    curriculum teacher)
OUT=results/stage1/ablation
export TEST_OUT=results/stage1/ablation_test
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
    TEACHER_LABEL=${LABEL}
    [ "${CONFIG}" = full ] && TEACHER_LABEL=ts-full
    echo "[$(date +%H:%M)] ${CONFIG}: start (${LABEL})"
    if [ "${CONFIG}" = teacher ]; then
        ABLATION=full PPO_STUDENT_ITERS=4000 \
            scripts/teacher_student_local.sh 3000 1000 "${OUT}" "${LABEL}" "${SEED}"
    elif [ "${CONFIG}" = curriculum ]; then
        ABLATION=${CONFIG} TEACHER_LABEL=${TEACHER_LABEL} RAMP_STEPS=0 \
            scripts/teacher_student_local.sh 3000 1000 "${OUT}" "${LABEL}" "${SEED}"
    else
        ABLATION=${CONFIG} TEACHER_LABEL=${TEACHER_LABEL} \
            scripts/teacher_student_local.sh 3000 1000 "${OUT}" "${LABEL}" "${SEED}"
    fi
    echo "[$(date +%H:%M)] ${CONFIG}: finished"
done
echo "[$(date +%H:%M)] ablation queue done"

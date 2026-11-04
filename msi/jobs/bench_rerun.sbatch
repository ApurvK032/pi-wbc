#!/bin/bash
# The MuJoCo bench on the cluster's CPUs: one array task per job of a job list, one CPU each.
#
# The job list is a JSON list of {"name", "out", "args"} (as logs/bench_rerun/jobs.json); each
# job is `python -m pi_wbc.sim2sim.evaluate <args>`, whose args end with `--out <out>` and name
# their inputs by paths relative to the repository root. The bench code comes from a git
# worktree of the bench commit (PIWBC_SNAPSHOT), so a result's provenance records that commit
# and its code hash, whatever this checkout holds. A job whose output already loads is skipped,
# so resubmitting the same array runs only what is missing.
#
# Submit from the repository root; account and partition stay on the command line so they
# never enter the repository:
#   git worktree add --detach logs/snapshots/bench-<commit> <commit>
#   mkdir -p logs/slurm
#   sbatch -A <account> -p <cpu partition> --array=0-<jobs - 1>%<max running> \
#       --export=ALL,PIWBC_SNAPSHOT=logs/snapshots/bench-<commit> \
#       msi/jobs/bench_rerun.sbatch <job list>
#
# Job logs: logs/slurm/piwbc-simbench-<array job>_<task>.out. PIWBC_PYTHON replaces the
# container's Python (a test outside the cluster: PIWBC_PYTHON=.venv/bin/python).
#SBATCH --job-name=piwbc-simbench
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=03:00:00
#SBATCH --output=logs/slurm/%x-%A_%a.out

set -uo pipefail
JOBS=${1:?usage: bench_rerun.sbatch <job list>}
SNAP=${PIWBC_SNAPSHOT:?set PIWBC_SNAPSHOT to a git worktree of the bench commit}
SIF=${PIWBC_SIF:-$HOME/containers/pi-wbc.sif}
cd "${SLURM_SUBMIT_DIR}" || exit 1
[ "${SNAP:0:1}" = / ] || SNAP="${PWD}/${SNAP}"
[ -d "${SNAP}/src/pi_wbc" ] || { echo "no bench code at ${SNAP}"; exit 1; }
export PYTHONPATH="${SNAP}/src"
export OMP_NUM_THREADS=1
echo "task ${SLURM_ARRAY_TASK_ID} of ${JOBS}; code $(git -C "${SNAP}" rev-parse --short HEAD);" \
    "node $(hostname -s); $(date '+%F %T')"
# the venv's own Python, not `uv run`: hundreds of tasks must not each try to sync the venv
read -r -a PY <<< "${PIWBC_PYTHON:-apptainer exec ${SIF} .venv/bin/python}"
"${PY[@]}" - "${JOBS}" "${SLURM_ARRAY_TASK_ID}" <<'EOF'
import json
import sys
from pathlib import Path

from pi_wbc.provenance import load_json
from pi_wbc.sim2sim import evaluate

job = json.loads(Path(sys.argv[1]).read_text())[int(sys.argv[2])]
try:
    load_json(job["out"])
    print(f"{job['name']}: output already there, skipped", flush=True)
    sys.exit(0)
except (OSError, ValueError, EOFError):
    pass
Path(job["out"]).parent.mkdir(parents=True, exist_ok=True)
print(f"{job['name']}: running", flush=True)
evaluate.main(job["args"])
EOF
status=$?
echo "exit ${status}; $(date '+%F %T')"
exit ${status}

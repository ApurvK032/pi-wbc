#!/usr/bin/env bash
# A second Python environment, .venv-isaacsim, with Isaac Sim 6.2 and Isaac Lab's OVRTX renderer
# on top of this project (D-023). The training environment (.venv, uv.lock) is not touched:
# Isaac Sim pins packages that conflict with it, so it is installed with Isaac Lab's tested
# overrides (tools/wheel_builder/uv-overrides.txt at Isaac Lab's v3.0.0 tag), as Isaac Lab's
# install guide does. About 9.5 GB to download and 27 GB on disk. From the repository root:
#
#   scripts/isaacsim_env.sh
#
# Isaac Sim and OVRTX ask for the NVIDIA Omniverse License Agreement on first start; set
# OMNI_KIT_ACCEPT_EULA=YES only if you accept it.
set -euo pipefail
export PATH=$HOME/.local/bin:$PATH
OVERRIDES=https://raw.githubusercontent.com/isaac-sim/IsaacLab/v3.0.0/tools/wheel_builder/uv-overrides.txt
tmp=$(mktemp -d)
trap 'rm -r "${tmp}"' EXIT
curl -sSfL "${OVERRIDES}" -o "${tmp}/uv-overrides.txt"
[ -d .venv-isaacsim ] || uv venv --python 3.12 .venv-isaacsim
UV_HTTP_TIMEOUT=300 uv pip install --python .venv-isaacsim/bin/python \
    "isaaclab[isaacsim,ov]==3.0.0" -e . \
    --overrides "${tmp}/uv-overrides.txt" \
    --extra-index-url https://pypi.nvidia.com --torch-backend cu130 \
    --index-strategy unsafe-best-match
.venv-isaacsim/bin/python -c "import importlib.metadata as m; print({p: m.version(p) for p in ['isaaclab', 'isaacsim', 'ovrtx', 'torch', 'newton', 'warp-lang', 'mujoco']})"

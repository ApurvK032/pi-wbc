#!/usr/bin/env bash
# Download NVIDIA's Decoupled WBC lower-body policies (the baseline controller, D-004) into
# third_party/. Code: Apache-2.0; weights: NVIDIA Open Model License (its text is fetched too).
# The weights are not redistributed by this repository. Pinned to one commit, checked by SHA-256.
set -euo pipefail

COMMIT=1e1ec4f5f27e6c3e03048f693bd7c60cce04afda # main on 2026-11-03
REPO=NVlabs/GR00T-WholeBodyControl
SRC=decoupled_wbc/sim2mujoco/resources/robots/g1
DEST="$(cd "$(dirname "$0")/.." && pwd)/third_party/gr00t_wbc"

# path in the repo, expected SHA-256 (git LFS files come from media.githubusercontent.com)
FILES=(
    "policy/GR00T-WholeBodyControl-Balance.onnx f645da599d4ca3d29ed273c8f4712620bb680d34977469ca3aeabe5bb9631c18 lfs"
    "policy/GR00T-WholeBodyControl-Walk.onnx 7c82255b6905ffcc4468fa7f8ddcf7b70db168cf1042107ccab887cb6a8e5407 lfs"
    "g1_gear_wbc.yaml - raw"
    "policy/NVIDIA Open Model License - raw"
)

mkdir -p "${DEST}/policy"
for entry in "${FILES[@]}"; do
    read -r -a parts <<< "${entry}"
    kind="${parts[-1]}"
    sha="${parts[-2]}"
    path="${entry% "${sha}" "${kind}"}"
    out="${DEST}/${path}"
    url_path="${SRC}/${path// /%20}"
    if [[ "${kind}" == lfs ]]; then
        url="https://media.githubusercontent.com/media/${REPO}/${COMMIT}/${url_path}"
    else
        url="https://raw.githubusercontent.com/${REPO}/${COMMIT}/${url_path}"
    fi
    if [[ ! -f "${out}" ]]; then
        curl -sSfL -o "${out}" "${url}"
    fi
    if [[ "${sha}" != - ]]; then
        echo "${sha}  ${out}" | sha256sum -c --quiet -
    fi
done
echo "${COMMIT}" > "${DEST}/COMMIT"
echo "NVIDIA Decoupled WBC policies at ${COMMIT:0:12} in ${DEST}"

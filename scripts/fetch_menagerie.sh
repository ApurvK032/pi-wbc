#!/usr/bin/env bash
# Download the Unitree G1 model from MuJoCo Menagerie (BSD-3-Clause) into third_party/.
# Only the unitree_g1 folder is fetched, pinned to one commit so every machine gets the same model.
set -euo pipefail

COMMIT=f3b6013629c7b94f8ee93fd977b161fa8350f9bc # main on 2026-10-23
DEST="$(cd "$(dirname "$0")/.." && pwd)/third_party/mujoco_menagerie"

if [[ -d "${DEST}/.git" ]] && [[ "$(git -C "${DEST}" rev-parse HEAD)" == "${COMMIT}" ]]; then
    echo "menagerie already at ${COMMIT:0:12} in ${DEST}"
    exit 0
fi

mkdir -p "${DEST}"
git -C "${DEST}" init -q
git -C "${DEST}" remote remove origin 2>/dev/null || true
git -C "${DEST}" remote add origin https://github.com/google-deepmind/mujoco_menagerie.git
git -C "${DEST}" sparse-checkout set --no-cone /unitree_g1/ /LICENSE
git -C "${DEST}" fetch -q --depth 1 --filter=blob:none origin "${COMMIT}"
git -C "${DEST}" checkout -q --detach FETCH_HEAD
echo "menagerie unitree_g1 at ${COMMIT:0:12} in ${DEST}"

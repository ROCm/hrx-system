#!/usr/bin/env bash
# Copyright 2026 The HRX Authors
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
trap 'kill -TERM 0' INT

HRX_OUTPUT_DIR="${HRX_OUTPUT_DIR:-/hrx/output}"
HRX_RELEASE_TYPE="${HRX_RELEASE_TYPE:-nightly}"
HRX_ARTIFACT_SET="${HRX_ARTIFACT_SET:-core}"

mkdir -p "${HRX_OUTPUT_DIR}/caches/ccache" "${HRX_OUTPUT_DIR}/caches/pip"
export CCACHE_DIR="${HRX_OUTPUT_DIR}/caches/ccache"
export PIP_CACHE_DIR="${HRX_OUTPUT_DIR}/caches/pip"

python3 -m pip install --upgrade boto3 zstandard

FETCH_ARGS=(
  --release-type "${HRX_RELEASE_TYPE}"
  --set "${HRX_ARTIFACT_SET}"
  --output-dir "${HRX_OUTPUT_DIR}/rocm-root"
  --download-cache-dir "${HRX_OUTPUT_DIR}/downloads"
)
if [[ -n "${HRX_RUN_ID:-}" ]]; then
  FETCH_ARGS+=(--run-id "${HRX_RUN_ID}")
else
  FETCH_ARGS+=(--latest)
fi

python3 /hrx/src/build_tools/fetch_rocm_artifacts.py "${FETCH_ARGS[@]}"
python3 /hrx/src/build_tools/build_core.py \
  --rocm-root "${HRX_OUTPUT_DIR}/rocm-root" \
  --build-dir "${HRX_OUTPUT_DIR}/build/hrx-core" \
  --install-prefix "${HRX_OUTPUT_DIR}/rocm-root" \
  --build-deps-prefix "${HRX_OUTPUT_DIR}/build-deps"

TEST_ARGS=(
  --rocm-root "${HRX_OUTPUT_DIR}/rocm-root"
  --build-dir "${HRX_OUTPUT_DIR}/build/hrx-core"
  --install-prefix "${HRX_OUTPUT_DIR}/rocm-root"
  --package-smoke-build-dir "${HRX_OUTPUT_DIR}/build/package-smoke"
)
if [[ "${HRX_TEST_GPU:-}" == "1" ]]; then
  TEST_ARGS+=(--gpu)
fi
python3 /hrx/src/build_tools/test_core.py "${TEST_ARGS[@]}"

python3 /hrx/src/build_tools/package_core.py \
  --rocm-root "${HRX_OUTPUT_DIR}/rocm-root" \
  --hrx-install "${HRX_OUTPUT_DIR}/rocm-root" \
  --output-dir "${HRX_OUTPUT_DIR}/dist"

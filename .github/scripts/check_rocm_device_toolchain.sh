#!/usr/bin/env bash
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -euo pipefail

: "${HRX_ROCM_ROOT:?}"

rocm_llvm_bin="${HRX_ROCM_ROOT}/lib/llvm/bin"
for tool in clang llvm-ar llvm-link lld llvm-objcopy; do
  tool_path="${rocm_llvm_bin}/${tool}"
  if [[ ! -x "${tool_path}" ]]; then
    echo "::error::ROCm device tool '${tool_path}' is not executable."
    exit 1
  fi
done

echo "ROCm device compiler: ${rocm_llvm_bin}/clang"
"${rocm_llvm_bin}/clang" --version
"${rocm_llvm_bin}/llvm-ar" --version

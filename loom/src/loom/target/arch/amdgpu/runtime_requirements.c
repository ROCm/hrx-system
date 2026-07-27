// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/runtime_requirements.h"

#include "loom/sanitizer/options.h"

loom_amdgpu_runtime_requirements_t
loom_amdgpu_runtime_requirements_from_target_pipeline_options(
    const loom_target_pipeline_options_t* options) {
  if (options == NULL ||
      !loom_sanitizer_options_is_enabled(&options->sanitizer)) {
    return LOOM_AMDGPU_RUNTIME_REQUIREMENT_NONE;
  }

  loom_amdgpu_runtime_requirements_t requirements =
      options->sanitizer.reporting_mode == LOOM_SANITIZER_REPORTING_MODE_TRAP
          ? LOOM_AMDGPU_RUNTIME_REQUIREMENT_NONE
          : LOOM_AMDGPU_RUNTIME_REQUIREMENT_FEEDBACK;
  if (iree_any_bit_set(options->sanitizer.checks,
                       LOOM_SANITIZER_CHECK_ACCESS)) {
    requirements |= LOOM_AMDGPU_RUNTIME_REQUIREMENT_ASAN_SHADOW;
  }
  if (iree_any_bit_set(options->sanitizer.checks, LOOM_SANITIZER_CHECK_RACE)) {
    requirements |= LOOM_AMDGPU_RUNTIME_REQUIREMENT_TSAN_SHADOW;
  }
  return requirements;
}

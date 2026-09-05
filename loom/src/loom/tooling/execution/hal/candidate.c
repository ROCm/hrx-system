// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/execution/hal/candidate.h"

static void loom_run_hal_candidate_initialize(
    loom_run_hal_candidate_t* out_candidate) {
  *out_candidate = (loom_run_hal_candidate_t){0};
}

static iree_status_t loom_run_hal_candidate_emit_selected_target(
    const loom_device_provider_t* provider, const loom_device_target_t* target,
    loom_run_module_t* run_module, const loom_compile_options_t* options,
    iree_allocator_t allocator, loom_run_hal_candidate_t* candidate) {
  const loom_artifact_target_t artifact_target =
      loom_device_target_artifact_target(target);
  iree_status_t status = loom_artifact_candidate_emit_target(
      provider->artifact_provider, &artifact_target, run_module->module,
      options, allocator, &candidate->artifact_candidate);
  if (iree_status_is_ok(status) && candidate->artifact_candidate.compiled) {
    candidate->device_artifact = (loom_device_artifact_t){
        .executable_target = target->executable_target,
        .artifact = &candidate->artifact_candidate.artifact,
    };
  }
  return status;
}

iree_status_t loom_run_hal_candidate_emit_target(
    const loom_device_provider_t* provider, const loom_device_target_t* target,
    loom_run_module_t* run_module, const loom_compile_options_t* options,
    iree_allocator_t allocator, loom_run_hal_candidate_t* out_candidate) {
  if (target == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HAL candidate target emission requires a "
                            "selected target");
  }
  loom_run_hal_candidate_initialize(out_candidate);
  iree_status_t status = loom_run_hal_candidate_emit_selected_target(
      provider, target, run_module, options, allocator, out_candidate);
  if (!iree_status_is_ok(status)) {
    loom_run_hal_candidate_deinitialize(out_candidate);
  }
  return status;
}

void loom_run_hal_candidate_deinitialize(loom_run_hal_candidate_t* candidate) {
  if (candidate == NULL) {
    return;
  }
  loom_artifact_candidate_deinitialize(&candidate->artifact_candidate);
  *candidate = (loom_run_hal_candidate_t){0};
}

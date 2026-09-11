// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/execution/hal/candidate.h"

static void loom_run_hal_candidate_initialize(
    const loom_device_provider_t* provider, iree_allocator_t allocator,
    loom_run_hal_candidate_t* out_candidate) {
  *out_candidate = (loom_run_hal_candidate_t){
      .host_allocator = allocator,
      .provider = provider,
  };
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
  loom_run_hal_candidate_initialize(provider, allocator, out_candidate);
  out_candidate->device_target = *target;
  iree_status_t status = loom_artifact_candidate_emit_target(
      provider->artifact_provider,
      &out_candidate->device_target.artifact_target, run_module->module,
      options, allocator, &out_candidate->artifact_candidate);
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

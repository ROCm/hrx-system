// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// HAL artifact candidates produced by Loom execution tooling.

#ifndef LOOM_TOOLING_EXECUTION_HAL_CANDIDATE_H_
#define LOOM_TOOLING_EXECUTION_HAL_CANDIDATE_H_

#include "iree/base/api.h"
#include "loom/tooling/compile/artifact.h"
#include "loom/tooling/compile/options.h"
#include "loom/tooling/execution/hal/device_provider.h"
#include "loom/tooling/execution/session.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_run_hal_candidate_t {
  // Host allocator used for owned candidate storage.
  iree_allocator_t host_allocator;
  // Device provider used for artifact emission.
  const loom_device_provider_t* provider;
  // Device target selected by the caller for artifact emission.
  loom_device_target_t device_target;
  // Offline compiler candidate emitted through |provider|.
  loom_artifact_candidate_t artifact_candidate;
} loom_run_hal_candidate_t;

// Emits |run_module| to a HAL artifact candidate using |target| as the
// selected target overlay. The caller retains ownership of |target| storage and
// must keep it live until |out_candidate| is deinitialized.
iree_status_t loom_run_hal_candidate_emit_target(
    const loom_device_provider_t* provider, const loom_device_target_t* target,
    loom_run_module_t* run_module, const loom_compile_options_t* options,
    iree_allocator_t allocator, loom_run_hal_candidate_t* out_candidate);

// Releases all artifact storage owned by |candidate|.
void loom_run_hal_candidate_deinitialize(loom_run_hal_candidate_t* candidate);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_EXECUTION_HAL_CANDIDATE_H_

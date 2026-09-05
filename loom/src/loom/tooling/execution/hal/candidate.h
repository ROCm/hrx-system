// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// HAL artifact candidates produced by Loom execution tooling.

#ifndef LOOM_TOOLING_EXECUTION_HAL_CANDIDATE_H_
#define LOOM_TOOLING_EXECUTION_HAL_CANDIDATE_H_

#include "iree/base/api.h"
#include "loom/tooling/compile/options.h"
#include "loom/tooling/execution/hal/artifact.h"
#include "loom/tooling/execution/hal/device_provider.h"
#include "loom/tooling/execution/hal/runtime.h"
#include "loom/tooling/execution/session.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_run_hal_candidate_t {
  // Offline compiler artifact candidate.
  loom_artifact_candidate_t artifact_candidate;
  // Device-loadable view of |artifact_candidate|.
  loom_device_artifact_t device_artifact;
} loom_run_hal_candidate_t;

// Selects a HAL device target satisfying |target_requirement| through
// |provider| and emits |run_module| to a HAL artifact candidate. NULL
// represents target-independent code. The module must already contain the
// prepared target-low entries intended for the artifact.
iree_status_t loom_run_hal_candidate_compile(
    const loom_device_provider_t* provider,
    const loom_run_hal_runtime_t* runtime, loom_run_module_t* run_module,
    const loom_target_facts_t* target_requirement,
    const loom_compile_options_t* options, iree_allocator_t allocator,
    loom_run_hal_candidate_t* out_candidate);

// Emits |run_module| to a HAL artifact candidate using |target| as the
// selected target overlay. The target is borrowed only for this call.
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

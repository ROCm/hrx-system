// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// HAL artifact candidates produced by Loom execution tooling.

#ifndef LOOM_TOOLING_EXECUTION_HAL_CANDIDATE_H_
#define LOOM_TOOLING_EXECUTION_HAL_CANDIDATE_H_

#include "iree/base/api.h"
#include "loom/tooling/execution/compile_options.h"
#include "loom/tooling/execution/hal/artifact.h"
#include "loom/tooling/execution/hal/runtime.h"
#include "loom/tooling/execution/session.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_run_hal_candidate_t {
  // Host allocator used for owned candidate storage.
  iree_allocator_t host_allocator;
  // Structured report for this candidate.
  loom_target_compile_report_t compile_report;
  // HAL artifact provider that produced |artifact|.
  const loom_run_hal_artifact_provider_t* provider;
  // HAL device target selected during candidate compilation.
  loom_run_hal_device_target_t device_target;
  // True when |device_target| storage is owned by this candidate.
  bool owns_device_target;
  // True when artifact bytes were produced.
  bool compiled;
  // HAL artifact bytes produced by |provider|.
  loom_run_hal_artifact_t artifact;
} loom_run_hal_candidate_t;

// Selects a HAL device target through |provider| and emits |run_module| to a
// HAL artifact candidate. The module must already contain the prepared
// target-low entries intended for the artifact.
iree_status_t loom_run_hal_candidate_compile(
    const loom_run_hal_artifact_provider_t* provider,
    const loom_run_hal_runtime_t* runtime, loom_run_module_t* run_module,
    const loom_run_candidate_compile_options_t* options,
    iree_allocator_t allocator, loom_run_hal_candidate_t* out_candidate);

// Emits |run_module| to a HAL artifact candidate using |target| as the
// selected target overlay. The caller retains ownership of |target| storage and
// must keep it live until |out_candidate| is deinitialized.
iree_status_t loom_run_hal_candidate_emit_target(
    const loom_run_hal_artifact_provider_t* provider,
    const loom_run_hal_device_target_t* target, loom_run_module_t* run_module,
    const loom_run_candidate_compile_options_t* options,
    iree_allocator_t allocator, loom_run_hal_candidate_t* out_candidate);

// Emits |run_module| using the module's target records instead of selecting an
// active HAL device target. This is the offline artifact path used by
// compilation tools; run/benchmark tools should use
// loom_run_hal_candidate_compile so they specialize to the device they will
// immediately execute on.
iree_status_t loom_run_hal_candidate_emit_module_target(
    const loom_run_hal_artifact_provider_t* provider,
    loom_run_module_t* run_module,
    const loom_run_candidate_compile_options_t* options,
    iree_allocator_t allocator, loom_run_hal_candidate_t* out_candidate);

// Releases all artifact storage owned by |candidate|.
void loom_run_hal_candidate_deinitialize(loom_run_hal_candidate_t* candidate);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_EXECUTION_HAL_CANDIDATE_H_

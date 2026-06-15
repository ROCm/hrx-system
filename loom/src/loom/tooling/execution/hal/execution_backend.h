// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Generic one-shot execution backend adapter for HAL artifact providers.

#ifndef LOOM_TOOLING_EXECUTION_HAL_EXECUTION_BACKEND_H_
#define LOOM_TOOLING_EXECUTION_HAL_EXECUTION_BACKEND_H_

#include "iree/base/api.h"
#include "loom/tooling/execution/execution_backend.h"
#include "loom/tooling/execution/hal/artifact.h"
#include "loom/tooling/execution/one_shot.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_run_hal_execution_backend_t {
  // Generic execution backend base registered with Loom execution tools.
  loom_run_execution_backend_t base;
  // HAL artifact provider adapted by the generic one-shot execution hooks.
  const loom_run_hal_artifact_provider_t* artifact_provider;
} loom_run_hal_execution_backend_t;

// Probes the HAL device target for a backend configured with
// loom_run_hal_execution_backend_t.
iree_status_t loom_run_hal_execution_backend_probe(
    const loom_run_execution_backend_t* backend,
    const loom_run_one_shot_probe_request_t* request);

// Compiles and invokes one HAL executable through a backend configured with
// loom_run_hal_execution_backend_t.
iree_status_t loom_run_hal_execution_backend_run_one_shot(
    const loom_run_execution_backend_t* backend,
    const loom_run_one_shot_request_t* request);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_EXECUTION_HAL_EXECUTION_BACKEND_H_

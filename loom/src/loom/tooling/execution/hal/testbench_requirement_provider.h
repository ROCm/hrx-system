// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// HAL testbench requirement provider composition.

#ifndef LOOM_TOOLING_EXECUTION_HAL_TESTBENCH_REQUIREMENT_PROVIDER_H_
#define LOOM_TOOLING_EXECUTION_HAL_TESTBENCH_REQUIREMENT_PROVIDER_H_

#include "iree/base/api.h"
#include "loom/tooling/testbench/requirements.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_run_hal_testbench_context_t
    loom_run_hal_testbench_context_t;

// Initializes one requirement provider against |context|.
typedef void (*loom_run_hal_testbench_requirement_provider_initializer_t)(
    loom_run_hal_testbench_context_t* context,
    loom_testbench_requirement_provider_t* out_provider);

// Static requirement provider initializers linked into a HAL testbench tool.
typedef struct loom_run_hal_testbench_requirement_initializer_set_t {
  // Initializers invoked in table order for each testbench evaluation.
  const loom_run_hal_testbench_requirement_provider_initializer_t* initializers;
  // Number of entries in |initializers|.
  iree_host_size_t initializer_count;
} loom_run_hal_testbench_requirement_initializer_set_t;

// Populates |providers| from |initializer_set| and binds each provider to
// |context|. Fails without invoking an initializer when |provider_capacity| is
// insufficient.
iree_status_t loom_run_hal_testbench_requirement_providers_populate(
    const loom_run_hal_testbench_requirement_initializer_set_t* initializer_set,
    loom_run_hal_testbench_context_t* context,
    iree_host_size_t provider_capacity,
    loom_testbench_requirement_provider_t* providers,
    iree_host_size_t* out_provider_count);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_EXECUTION_HAL_TESTBENCH_REQUIREMENT_PROVIDER_H_

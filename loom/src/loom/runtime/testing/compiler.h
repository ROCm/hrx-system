// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_RUNTIME_TESTING_COMPILER_H_
#define LOOM_RUNTIME_TESTING_COMPILER_H_

#include "loom/runtime/host/host.h"
#include "loomc/target/iree_hal.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns one process-lifetime HAL target provider descriptor.
typedef const loomc_iree_hal_target_provider_t*(
    LOOMC_API_PTR* loom_runtime_test_target_provider_fn_t)(void);

// Test-only compiler composition supplied by one physical target leaf. Source
// JIT tests consume this without linking unrelated configured targets.
typedef struct loom_runtime_test_compiler_t {
  // Factory for the exact LoomC compiler target package.
  loom_runtime_target_environment_create_fn_t target_environment_create;
  // Factory for the matching IREE HAL target-selection route.
  loom_runtime_test_target_provider_fn_t target_provider;
} loom_runtime_test_compiler_t;

// Implemented exactly once by the linked compiler target leaf.
const loom_runtime_test_compiler_t* loom_runtime_test_compiler(void);

// Creates a serving host with the linked test compiler's exact target route.
iree_status_t loom_runtime_test_host_create(iree_host_size_t worker_count,
                                            iree_allocator_t host_allocator,
                                            loom_runtime_host_t** out_host);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_RUNTIME_TESTING_COMPILER_H_

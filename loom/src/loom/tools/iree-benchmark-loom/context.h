// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Benchmark HAL context lifecycle.

#ifndef LOOM_TOOLS_IREE_BENCHMARK_LOOM_CONTEXT_H_
#define LOOM_TOOLS_IREE_BENCHMARK_LOOM_CONTEXT_H_

#include "iree/base/api.h"
#include "loom/tools/iree-benchmark-loom/configuration.h"
#include "loom/tools/iree-benchmark-loom/model.h"

#ifdef __cplusplus
extern "C" {
#endif

// Initializes shared HAL state used by benchmark compilation and dispatch.
void iree_benchmark_loom_hal_context_initialize(
    const iree_benchmark_loom_configuration_t* configuration,
    iree_allocator_t host_allocator,
    iree_benchmark_loom_hal_context_t* out_context);

// Releases HAL state owned by |context|.
void iree_benchmark_loom_hal_context_deinitialize(
    iree_benchmark_loom_hal_context_t* context);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLS_IREE_BENCHMARK_LOOM_CONTEXT_H_

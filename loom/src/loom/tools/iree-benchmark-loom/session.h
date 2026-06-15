// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Benchmark-configured Loom run session lifecycle.

#ifndef LOOM_TOOLS_IREE_BENCHMARK_LOOM_SESSION_H_
#define LOOM_TOOLS_IREE_BENCHMARK_LOOM_SESSION_H_

#include "iree/base/api.h"
#include "loom/tooling/execution/session.h"
#include "loom/tools/iree-benchmark-loom/configuration.h"

#ifdef __cplusplus
extern "C" {
#endif

// Initializes a run session with benchmark-linked dialects and descriptors.
iree_status_t iree_benchmark_loom_session_initialize(
    const iree_benchmark_loom_configuration_t* configuration,
    iree_allocator_t host_allocator, loom_run_session_t* out_session);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLS_IREE_BENCHMARK_LOOM_SESSION_H_

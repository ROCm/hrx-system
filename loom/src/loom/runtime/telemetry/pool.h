// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_RUNTIME_TELEMETRY_POOL_H_
#define LOOM_RUNTIME_TELEMETRY_POOL_H_

#include "iree/hal/pool.h"
#include "loom/runtime/telemetry/sample.h"

#ifdef __cplusplus
extern "C" {
#endif

// Emits one versioned numeric topic from an existing HAL pool observation.
// name follows the source-owned topic identifier contract. Statistics remain
// independent atomic observations, not a coherent total or device-wide usage;
// nested pools can account for the same backing. wait_count counts reuse
// prerequisites, not host waits or elapsed time.
// Performs no allocation, hardware query, or device wait. Descriptors and
// values are borrowed only during the synchronous emit call. The only possible
// failure is the sink's observation output/allocation failure.
iree_status_t loom_runtime_telemetry_sample_pool(
    const char* name, const iree_hal_pool_stats_t* statistics,
    loom_runtime_telemetry_sample_sink_t sink);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // LOOM_RUNTIME_TELEMETRY_POOL_H_

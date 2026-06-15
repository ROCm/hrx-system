// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Timing-stat helpers for benchmark result rows.

#ifndef LOOM_TOOLS_IREE_BENCHMARK_LOOM_TIMING_H_
#define LOOM_TOOLS_IREE_BENCHMARK_LOOM_TIMING_H_

#include "iree/base/api.h"
#include "loom/tools/iree-benchmark-loom/model.h"

#ifdef __cplusplus
extern "C" {
#endif

// Sorts |durations| in place and computes nearest-rank summary statistics.
void iree_benchmark_loom_compute_timing_stats(
    iree_duration_t* durations, iree_host_size_t count,
    iree_benchmark_loom_timing_stats_t* out_stats);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLS_IREE_BENCHMARK_LOOM_TIMING_H_

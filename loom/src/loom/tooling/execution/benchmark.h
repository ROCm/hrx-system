// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Reusable benchmark loops for Loom execution tooling.

#ifndef LOOM_TOOLING_EXECUTION_BENCHMARK_H_
#define LOOM_TOOLING_EXECUTION_BENCHMARK_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef uint32_t loom_run_benchmark_stop_reason_t;
enum loom_run_benchmark_stop_reason_e {
  // The benchmark has not produced a terminal stop reason.
  LOOM_RUN_BENCHMARK_STOP_REASON_NONE = 0u,
  // Minimum batch count and duration were reached with no stability criterion.
  LOOM_RUN_BENCHMARK_STOP_REASON_MINIMUM_REACHED = 1u,
  // Minimums were reached and the configured robust spread bound was satisfied.
  LOOM_RUN_BENCHMARK_STOP_REASON_STABLE = 2u,
  // Measurement stopped at the hard maximum batch count.
  LOOM_RUN_BENCHMARK_STOP_REASON_MAX_BATCH_COUNT = 3u,
};

typedef struct loom_run_benchmark_timing_stats_t {
  // Number of measured timing samples.
  iree_host_size_t count;
  // Sum of measured durations in nanoseconds.
  iree_duration_t total_ns;
  // Minimum measured duration in nanoseconds.
  iree_duration_t minimum_ns;
  // Maximum measured duration in nanoseconds.
  iree_duration_t maximum_ns;
  // Arithmetic mean of measured durations in nanoseconds.
  double mean_ns;
  // Nearest-rank median duration in nanoseconds.
  iree_duration_t p50_ns;
  // Nearest-rank p90 duration in nanoseconds.
  iree_duration_t p90_ns;
  // Relative p90-to-p50 spread in parts per million.
  uint64_t p90_to_p50_delta_ppm;
} loom_run_benchmark_timing_stats_t;

typedef struct loom_run_benchmark_options_t {
  // Number of logical operations performed by one measured batch.
  iree_host_size_t batch_size;
  // Minimum number of warmup batches run before measuring.
  iree_host_size_t warmup_batch_count;
  // Minimum warmup wall duration in nanoseconds before measuring.
  iree_duration_t warmup_min_duration_ns;
  // Minimum number of measured batches required before stopping.
  iree_host_size_t min_batch_count;
  // Minimum measured wall duration in nanoseconds required before stopping.
  iree_duration_t min_duration_ns;
  // Hard maximum number of measured batches.
  iree_host_size_t max_batch_count;
  // Maximum accepted p90-to-p50 relative spread in parts per million.
  uint64_t stable_p90_to_p50_delta_ppm;
} loom_run_benchmark_options_t;

typedef struct loom_run_benchmark_result_t {
  // Effective number of logical operations per batch.
  iree_host_size_t batch_size;
  // Number of warmup batches executed before measuring.
  iree_host_size_t warmup_batch_count;
  // Wall duration spent in warmup batches in nanoseconds.
  iree_duration_t warmup_duration_ns;
  // Number of measured batches executed.
  iree_host_size_t measured_batch_count;
  // Number of logical operations represented by measured batches.
  iree_host_size_t measured_operation_count;
  // Wall duration covered by measured batches in nanoseconds.
  iree_duration_t measured_duration_ns;
  // Terminal reason selected by the benchmark loop.
  loom_run_benchmark_stop_reason_t stop_reason;
  // Timing statistics over complete measured batches.
  loom_run_benchmark_timing_stats_t batch_timing;
  // Timing statistics over measured batches normalized by |batch_size|.
  loom_run_benchmark_timing_stats_t operation_timing;
} loom_run_benchmark_result_t;

typedef iree_status_t (*loom_run_benchmark_batch_fn_t)(void* user_data);

typedef struct loom_run_benchmark_batch_callback_t {
  // Callback executing one benchmark batch.
  loom_run_benchmark_batch_fn_t fn;
  // User data passed to |fn|.
  void* user_data;
} loom_run_benchmark_batch_callback_t;

// Initializes benchmark options with conservative timing defaults.
void loom_run_benchmark_options_initialize(
    loom_run_benchmark_options_t* out_options);

// Initializes an empty benchmark result.
void loom_run_benchmark_result_initialize(
    loom_run_benchmark_result_t* out_result);

// Returns a stable JSON-friendly name for |reason|.
iree_string_view_t loom_run_benchmark_stop_reason_name(
    loom_run_benchmark_stop_reason_t reason);

// Computes timing statistics from |durations|. The array is sorted in place.
iree_status_t loom_run_benchmark_compute_timing_stats(
    iree_duration_t* durations, iree_host_size_t count,
    loom_run_benchmark_timing_stats_t* out_stats);

// Runs a rigorous warmup and measured benchmark loop around |callback|.
iree_status_t loom_run_benchmark_run_batches(
    loom_run_benchmark_batch_callback_t callback,
    const loom_run_benchmark_options_t* options, iree_allocator_t allocator,
    loom_run_benchmark_result_t* out_result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_EXECUTION_BENCHMARK_H_

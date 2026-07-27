// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/iree-benchmark-loom/timing.h"

#include <stdlib.h>

static int iree_benchmark_loom_compare_duration(const void* lhs,
                                                const void* rhs) {
  const iree_duration_t lhs_duration = *(const iree_duration_t*)lhs;
  const iree_duration_t rhs_duration = *(const iree_duration_t*)rhs;
  return (lhs_duration > rhs_duration) - (lhs_duration < rhs_duration);
}

static iree_duration_t iree_benchmark_loom_nearest_rank_percentile(
    const iree_duration_t* sorted_durations, iree_host_size_t count,
    iree_host_size_t percentile) {
  IREE_ASSERT_ARGUMENT(sorted_durations);
  IREE_ASSERT(count > 0);
  const iree_host_size_t rank = (count * percentile + 99) / 100;
  iree_host_size_t index = rank == 0 ? 0 : rank - 1;
  if (index >= count) {
    index = count - 1;
  }
  return sorted_durations[index];
}

void iree_benchmark_loom_compute_timing_stats(
    iree_duration_t* durations, iree_host_size_t count,
    iree_benchmark_loom_timing_stats_t* out_stats) {
  IREE_ASSERT_ARGUMENT(durations);
  IREE_ASSERT_ARGUMENT(out_stats);
  IREE_ASSERT(count > 0);

  qsort(durations, count, sizeof(*durations),
        iree_benchmark_loom_compare_duration);
  int64_t total_ns = 0;
  for (iree_host_size_t i = 0; i < count; ++i) {
    total_ns += durations[i];
  }
  *out_stats = (iree_benchmark_loom_timing_stats_t){
      .count = count,
      .total_ns = total_ns,
      .minimum_ns = durations[0],
      .maximum_ns = durations[count - 1],
      .mean_ns = (double)total_ns / (double)count,
      .p50_ns =
          iree_benchmark_loom_nearest_rank_percentile(durations, count, 50),
      .p90_ns =
          iree_benchmark_loom_nearest_rank_percentile(durations, count, 90),
  };
}

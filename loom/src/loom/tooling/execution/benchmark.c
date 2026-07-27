// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/execution/benchmark.h"

#include <stdint.h>
#include <stdlib.h>

enum {
  LOOM_RUN_BENCHMARK_DEFAULT_MIN_DURATION_NS = 100 * 1000 * 1000,
  LOOM_RUN_BENCHMARK_DEFAULT_STABLE_P90_DELTA_PPM = 100 * 1000,
};

void loom_run_benchmark_options_initialize(
    loom_run_benchmark_options_t* out_options) {
  *out_options = (loom_run_benchmark_options_t){
      .batch_size = 1,
      .warmup_batch_count = 1,
      .warmup_min_duration_ns = 0,
      .min_batch_count = 10,
      .min_duration_ns = LOOM_RUN_BENCHMARK_DEFAULT_MIN_DURATION_NS,
      .max_batch_count = 1000,
      .stable_p90_to_p50_delta_ppm =
          LOOM_RUN_BENCHMARK_DEFAULT_STABLE_P90_DELTA_PPM,
  };
}

void loom_run_benchmark_result_initialize(
    loom_run_benchmark_result_t* out_result) {
  *out_result = (loom_run_benchmark_result_t){0};
}

iree_string_view_t loom_run_benchmark_stop_reason_name(
    loom_run_benchmark_stop_reason_t reason) {
  switch (reason) {
    case LOOM_RUN_BENCHMARK_STOP_REASON_MINIMUM_REACHED:
      return IREE_SV("minimum_reached");
    case LOOM_RUN_BENCHMARK_STOP_REASON_STABLE:
      return IREE_SV("stable");
    case LOOM_RUN_BENCHMARK_STOP_REASON_MAX_BATCH_COUNT:
      return IREE_SV("max_batch_count");
    case LOOM_RUN_BENCHMARK_STOP_REASON_NONE:
    default:
      return IREE_SV("none");
  }
}

static iree_status_t loom_run_benchmark_options_validate(
    loom_run_benchmark_batch_callback_t callback,
    const loom_run_benchmark_options_t* options) {
  if (!callback.fn) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "benchmark batch callback is required");
  }
  if (options->batch_size == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "benchmark batch_size must be positive");
  }
  if (options->min_batch_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "benchmark min_batch_count must be positive");
  }
  if (options->max_batch_count < options->min_batch_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "benchmark max_batch_count %" PRIhsz
                            " must be >= min_batch_count %" PRIhsz,
                            options->max_batch_count, options->min_batch_count);
  }
  return iree_ok_status();
}

static int loom_run_benchmark_compare_duration(const void* lhs,
                                               const void* rhs) {
  const iree_duration_t lhs_duration = *(const iree_duration_t*)lhs;
  const iree_duration_t rhs_duration = *(const iree_duration_t*)rhs;
  return (lhs_duration > rhs_duration) - (lhs_duration < rhs_duration);
}

static iree_duration_t loom_run_benchmark_nearest_rank_percentile(
    const iree_duration_t* sorted_durations, iree_host_size_t count,
    iree_host_size_t percentile) {
  const iree_host_size_t rank = (count * percentile + 99) / 100;
  iree_host_size_t index = rank == 0 ? 0 : rank - 1;
  if (index >= count) {
    index = count - 1;
  }
  return sorted_durations[index];
}

static uint64_t loom_run_benchmark_relative_delta_ppm(iree_duration_t baseline,
                                                      iree_duration_t value) {
  if (value <= baseline) {
    return 0;
  }
  if (baseline <= 0) {
    return UINT64_MAX;
  }
  const uint64_t delta = (uint64_t)(value - baseline);
  const uint64_t denominator = (uint64_t)baseline;
  if (delta > UINT64_MAX / 1000000ull) {
    return UINT64_MAX;
  }
  return (delta * 1000000ull) / denominator;
}

iree_status_t loom_run_benchmark_compute_timing_stats(
    iree_duration_t* durations, iree_host_size_t count,
    loom_run_benchmark_timing_stats_t* out_stats) {
  if (count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "benchmark timing stats require samples");
  }
  if (!durations) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "benchmark timing stats require durations");
  }

  iree_duration_t total_ns = 0;
  for (iree_host_size_t i = 0; i < count; ++i) {
    if (durations[i] < 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "benchmark timing sample %" PRIhsz " has a negative duration", i);
    }
    if (durations[i] > INT64_MAX - total_ns) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "benchmark timing total overflow");
    }
    total_ns += durations[i];
  }
  qsort(durations, count, sizeof(*durations),
        loom_run_benchmark_compare_duration);
  const iree_duration_t p50_ns =
      loom_run_benchmark_nearest_rank_percentile(durations, count, 50);
  const iree_duration_t p90_ns =
      loom_run_benchmark_nearest_rank_percentile(durations, count, 90);
  *out_stats = (loom_run_benchmark_timing_stats_t){
      .count = count,
      .total_ns = total_ns,
      .minimum_ns = durations[0],
      .maximum_ns = durations[count - 1],
      .mean_ns = (double)total_ns / (double)count,
      .p50_ns = p50_ns,
      .p90_ns = p90_ns,
      .p90_to_p50_delta_ppm =
          loom_run_benchmark_relative_delta_ppm(p50_ns, p90_ns),
  };
  return iree_ok_status();
}

static bool loom_run_benchmark_warmup_complete(
    const loom_run_benchmark_options_t* options,
    iree_host_size_t warmup_batch_count, iree_duration_t warmup_duration_ns) {
  return warmup_batch_count >= options->warmup_batch_count &&
         warmup_duration_ns >= options->warmup_min_duration_ns;
}

static iree_status_t loom_run_benchmark_warmup(
    loom_run_benchmark_batch_callback_t callback,
    const loom_run_benchmark_options_t* options,
    loom_run_benchmark_result_t* result) {
  const iree_time_t start_time_ns = iree_time_now();
  while (!loom_run_benchmark_warmup_complete(
      options, result->warmup_batch_count, result->warmup_duration_ns)) {
    IREE_RETURN_IF_ERROR(callback.fn(callback.user_data));
    ++result->warmup_batch_count;
    const iree_time_t now_ns = iree_time_now();
    result->warmup_duration_ns =
        now_ns >= start_time_ns ? now_ns - start_time_ns : 0;
  }
  return iree_ok_status();
}

static bool loom_run_benchmark_minimum_reached(
    const loom_run_benchmark_options_t* options,
    const loom_run_benchmark_result_t* result) {
  return result->measured_batch_count >= options->min_batch_count &&
         result->measured_duration_ns >= options->min_duration_ns;
}

static iree_status_t loom_run_benchmark_update_timing_stats(
    iree_duration_t* durations, const loom_run_benchmark_options_t* options,
    loom_run_benchmark_result_t* result) {
  IREE_RETURN_IF_ERROR(loom_run_benchmark_compute_timing_stats(
      durations, result->measured_batch_count, &result->batch_timing));
  for (iree_host_size_t i = 0; i < result->measured_batch_count; ++i) {
    durations[i] /= (iree_duration_t)options->batch_size;
  }
  return loom_run_benchmark_compute_timing_stats(
      durations, result->measured_batch_count, &result->operation_timing);
}

static iree_status_t loom_run_benchmark_record_measured_batch(
    loom_run_benchmark_batch_callback_t callback, iree_duration_t* durations,
    loom_run_benchmark_result_t* result) {
  const iree_time_t start_time_ns = iree_time_now();
  IREE_RETURN_IF_ERROR(callback.fn(callback.user_data));
  const iree_time_t end_time_ns = iree_time_now();
  const iree_duration_t duration_ns =
      end_time_ns >= start_time_ns ? end_time_ns - start_time_ns : 0;
  durations[result->measured_batch_count++] = duration_ns;
  result->measured_duration_ns += duration_ns;
  return iree_ok_status();
}

iree_status_t loom_run_benchmark_run_batches(
    loom_run_benchmark_batch_callback_t callback,
    const loom_run_benchmark_options_t* options, iree_allocator_t allocator,
    loom_run_benchmark_result_t* out_result) {
  loom_run_benchmark_result_initialize(out_result);
  IREE_RETURN_IF_ERROR(loom_run_benchmark_options_validate(callback, options));

  iree_duration_t* durations = NULL;
  iree_host_size_t duration_storage_size = 0;
  if (!iree_host_size_checked_mul(options->max_batch_count, sizeof(*durations),
                                  &duration_storage_size)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "benchmark duration storage size overflow");
  }
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(allocator, duration_storage_size,
                                             (void**)&durations));

  iree_status_t status = iree_ok_status();
  out_result->batch_size = options->batch_size;
  status = loom_run_benchmark_warmup(callback, options, out_result);
  while (iree_status_is_ok(status) &&
         out_result->measured_batch_count < options->max_batch_count) {
    status = loom_run_benchmark_record_measured_batch(callback, durations,
                                                      out_result);
    if (!iree_status_is_ok(status)) {
      break;
    }
    if (!loom_run_benchmark_minimum_reached(options, out_result)) {
      continue;
    }

    status = loom_run_benchmark_compute_timing_stats(
        durations, out_result->measured_batch_count, &out_result->batch_timing);
    if (!iree_status_is_ok(status)) {
      break;
    }
    if (options->stable_p90_to_p50_delta_ppm == 0) {
      out_result->stop_reason = LOOM_RUN_BENCHMARK_STOP_REASON_MINIMUM_REACHED;
      break;
    }
    if (out_result->batch_timing.p90_to_p50_delta_ppm <=
        options->stable_p90_to_p50_delta_ppm) {
      out_result->stop_reason = LOOM_RUN_BENCHMARK_STOP_REASON_STABLE;
      break;
    }
  }
  if (iree_status_is_ok(status) &&
      out_result->stop_reason == LOOM_RUN_BENCHMARK_STOP_REASON_NONE) {
    out_result->stop_reason = LOOM_RUN_BENCHMARK_STOP_REASON_MAX_BATCH_COUNT;
  }
  if (iree_status_is_ok(status)) {
    iree_host_size_t measured_operation_count = 0;
    if (!iree_host_size_checked_mul(out_result->measured_batch_count,
                                    options->batch_size,
                                    &measured_operation_count)) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "benchmark measured operation count overflow");
    } else {
      out_result->measured_operation_count = measured_operation_count;
    }
  }
  if (iree_status_is_ok(status)) {
    status =
        loom_run_benchmark_update_timing_stats(durations, options, out_result);
  }

  iree_allocator_free(allocator, durations);
  if (!iree_status_is_ok(status)) {
    loom_run_benchmark_result_initialize(out_result);
  }
  return status;
}

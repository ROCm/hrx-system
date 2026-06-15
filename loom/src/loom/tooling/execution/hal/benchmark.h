// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Batched HAL dispatch benchmarking for Loom execution sessions.

#ifndef LOOM_TOOLING_EXECUTION_HAL_BENCHMARK_H_
#define LOOM_TOOLING_EXECUTION_HAL_BENCHMARK_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/profile_sink.h"
#include "iree/hal/utils/statistics_sink.h"
#include "loom/tooling/execution/benchmark.h"
#include "loom/tooling/execution/hal/invocation.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef uint32_t loom_run_hal_benchmark_flags_t;
enum loom_run_hal_benchmark_flag_bits_e {
  // Performs only the normal timing benchmark.
  LOOM_RUN_HAL_BENCHMARK_FLAG_NONE = 0u,
  // Runs one final profiled batch after the measured timing window.
  LOOM_RUN_HAL_BENCHMARK_FLAG_PROFILE_FINAL_BATCH = 1u << 0,
};

#define LOOM_RUN_HAL_PROFILE_SUMMARY_MAX_ROWS 32
#define LOOM_RUN_HAL_PROFILE_FUNCTION_NAME_CAPACITY 128
#define LOOM_RUN_HAL_PROFILE_ARTIFACT_PATH_CAPACITY 512

typedef struct loom_run_hal_profile_row_summary_t {
  // Kind of aggregate statistic represented by this row.
  iree_hal_profile_statistics_row_type_t row_type;
  // Time domain used by raw duration fields.
  iree_hal_profile_statistics_time_domain_t time_domain;
  // Flags specifying which optional aggregate fields are populated.
  iree_hal_profile_statistics_row_flags_t flags;
  // Session-local physical device ordinal associated with this row.
  uint32_t physical_device_ordinal;
  // Session-local queue ordinal associated with this row, or UINT32_MAX.
  uint32_t queue_ordinal;
  // Queue or memory event type for operation/lifecycle rows, or zero.
  uint32_t event_type;
  // Session-local executable identifier, or zero.
  uint64_t executable_id;
  // Session-local command-buffer identifier, or zero.
  uint64_t command_buffer_id;
  // Executable function ordinal, or UINT32_MAX.
  uint32_t function_ordinal;
  // Command ordinal within a command buffer, or UINT32_MAX.
  uint32_t command_index;
  // Number of source samples accumulated into this row.
  uint64_t sample_count;
  // Number of source samples rejected from timing aggregates.
  uint64_t invalid_sample_count;
  // Sum of source operation counts when available.
  uint64_t operation_count;
  // Sum of source payload byte lengths when available.
  uint64_t payload_bytes;
  // Sum of source tile counts when available.
  uint64_t tile_count;
  // Sum of source per-tile durations in nanoseconds when available.
  uint64_t tile_duration_sum_ns;
  // Earliest valid source start time in |time_domain| units.
  uint64_t first_start_time;
  // Latest valid source end time in |time_domain| units.
  uint64_t last_end_time;
  // Sum of valid source durations in |time_domain| units.
  uint64_t total_duration;
  // Minimum valid source duration in |time_domain| units.
  uint64_t minimum_duration;
  // Maximum valid source duration in |time_domain| units.
  uint64_t maximum_duration;
  // True when raw durations were scaled into nanoseconds.
  bool has_scaled_duration_ns;
  // Sum of valid source durations in nanoseconds when scaling was available.
  uint64_t total_duration_ns;
  // Minimum valid source duration in nanoseconds when scaling was available.
  uint64_t minimum_duration_ns;
  // Maximum valid source duration in nanoseconds when scaling was available.
  uint64_t maximum_duration_ns;
  // Length of |function_name| when executable metadata named the function.
  iree_host_size_t function_name_length;
  // Function name copied from profile metadata and truncated when necessary.
  char function_name[LOOM_RUN_HAL_PROFILE_FUNCTION_NAME_CAPACITY];
} loom_run_hal_profile_row_summary_t;

typedef struct loom_run_hal_profile_summary_t {
  // True when final-batch profiling was requested.
  bool requested;
  // True when the final profiled batch completed and emitted through the sink.
  bool executed;
  // True when |artifact_path| names a raw HAL profile bundle.
  bool has_artifact_path;
  // True when profiling was requested but failed before producing evidence.
  bool has_error;
  // Terminal status code from the profiling failure.
  iree_status_code_t error_code;
  // Length of |artifact_path| when a raw profile bundle was requested.
  iree_host_size_t artifact_path_length;
  // Path to the raw HAL profile bundle, truncated when necessary.
  char artifact_path[LOOM_RUN_HAL_PROFILE_ARTIFACT_PATH_CAPACITY];
  // Length of the formatted profiling failure message.
  iree_host_size_t error_message_length;
  // Formatted profiling failure message, truncated when necessary.
  char error_message[512];
  // Producer profiling flags requested for the final batch.
  iree_hal_device_profiling_flags_t flags;
  // Structured profiling data families requested for the final batch.
  iree_hal_device_profiling_data_families_t data_families;
  // Aggregate profile rows received by the statistics sink.
  iree_host_size_t row_count;
  // Source records reported as dropped by the profile producer.
  uint64_t dropped_record_count;
  // Number of row summaries copied into |rows|.
  iree_host_size_t captured_row_count;
  // Number of aggregate rows omitted because |rows| was full.
  iree_host_size_t truncated_row_count;
  // Bounded copy of aggregate rows emitted by the HAL statistics sink.
  loom_run_hal_profile_row_summary_t
      rows[LOOM_RUN_HAL_PROFILE_SUMMARY_MAX_ROWS];
} loom_run_hal_profile_summary_t;

typedef struct loom_run_hal_benchmark_options_t {
  // Flags controlling optional HAL benchmark phases.
  loom_run_hal_benchmark_flags_t flags;
  // Generic warmup, measured timing, batch, and stability policy.
  loom_run_benchmark_options_t timing;
  // Command-buffer recording and queue execute options for the batch.
  loom_run_hal_dispatch_batch_options_t dispatch_batch;
  // Producer profiling flags used for a requested final profiled batch.
  iree_hal_device_profiling_flags_t profile_flags;
  // Structured profile families requested for a final profiled batch.
  iree_hal_device_profiling_data_families_t profile_data_families;
  // Optional capture filter used for expensive profile artifacts.
  iree_hal_profile_capture_filter_t profile_capture_filter;
  // Number of hardware/software counter selections in |profile_counter_sets|.
  iree_host_size_t profile_counter_set_count;
  // Borrowed counter selections used during profiling_begin.
  const iree_hal_profile_counter_set_selection_t* profile_counter_sets;
  // Optional borrowed path receiving the raw final-batch profile bundle.
  iree_string_view_t profile_artifact_path;
  // Optional borrowed sink receiving the same final-batch profile chunks.
  iree_hal_profile_sink_t* profile_artifact_sink;
} loom_run_hal_benchmark_options_t;

typedef struct loom_run_hal_benchmark_result_t {
  // Generic host timing benchmark result.
  loom_run_benchmark_result_t timing;
  // Number of physical binding lists rotated through command-buffer dispatches.
  iree_host_size_t binding_ring_count;
  // Number of pre-recorded command buffers rotated across benchmark batches.
  iree_host_size_t command_buffer_ring_count;
  // Final profiled-batch summary.
  loom_run_hal_profile_summary_t profile;
} loom_run_hal_benchmark_result_t;

// Initializes HAL benchmark options for rigorous batched dispatch timing.
void loom_run_hal_benchmark_options_initialize(
    loom_run_hal_benchmark_options_t* out_options);

// Initializes an empty HAL benchmark result.
void loom_run_hal_benchmark_result_initialize(
    loom_run_hal_benchmark_result_t* out_result);

// Prepares and times a reusable HAL command buffer containing a dispatch batch.
iree_status_t loom_run_hal_benchmark_dispatch_plan(
    const loom_run_hal_runtime_t* runtime,
    const loom_run_hal_prepared_candidate_t* candidate,
    const loom_run_hal_invocation_plan_t* plan,
    const loom_run_hal_benchmark_options_t* options, iree_allocator_t allocator,
    loom_run_hal_benchmark_result_t* out_result);

// Prepares and times reusable HAL command buffers whose dispatch slots cycle
// across |binding_lists|. Each measured batch still executes
// |options->dispatch_batch.dispatch_count| dispatches.
iree_status_t loom_run_hal_benchmark_dispatch_binding_ring(
    const loom_run_hal_runtime_t* runtime,
    const loom_run_hal_prepared_candidate_t* candidate,
    const loom_run_hal_invocation_plan_t* plan,
    iree_host_size_t binding_list_count, iree_vm_list_t* const* binding_lists,
    const loom_run_hal_benchmark_options_t* options, iree_allocator_t allocator,
    loom_run_hal_benchmark_result_t* out_result);

// Prepares and times reusable HAL command buffers whose logical operation is a
// fixed sequence of dispatches. |candidates| has |sequence_count| entries.
// |plans| is a flattened row-major array with
// |plan_ring_count * sequence_count| entries, indexed as ring slot first and
// sequence step second.
iree_status_t loom_run_hal_benchmark_dispatch_sequence_plan_ring(
    const loom_run_hal_runtime_t* runtime, iree_host_size_t sequence_count,
    const loom_run_hal_prepared_candidate_t* const* candidates,
    iree_host_size_t plan_ring_count,
    const loom_run_hal_invocation_plan_t* const* plans,
    const loom_run_hal_benchmark_options_t* options, iree_allocator_t allocator,
    loom_run_hal_benchmark_result_t* out_result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_EXECUTION_HAL_BENCHMARK_H_

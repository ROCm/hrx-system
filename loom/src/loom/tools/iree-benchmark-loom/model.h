// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared iree-benchmark-loom run, benchmark, and HAL candidate model.

#ifndef LOOM_TOOLS_IREE_BENCHMARK_LOOM_MODEL_H_
#define LOOM_TOOLS_IREE_BENCHMARK_LOOM_MODEL_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "loom/tooling/execution/compile_report_capture.h"
#include "loom/tooling/execution/hal/benchmark.h"
#include "loom/tooling/execution/hal/testbench_actual.h"
#include "loom/tooling/testbench/executor.h"
#include "loom/tools/iree-benchmark-loom/configuration.h"
#include "loom/tools/iree-benchmark-loom/output.h"
#include "loom/util/stream.h"

#ifdef __cplusplus
extern "C" {
#endif

// Invalid stable ordinal for optional benchmark model indexes.
#define IREE_BENCHMARK_LOOM_INDEX_INVALID IREE_HOST_SIZE_MAX

typedef enum iree_benchmark_loom_measure_e {
  // Full testbench case execution wall time.
  IREE_BENCHMARK_LOOM_MEASURE_CASE_END_TO_END = 0,
  // Prepared HAL dispatch completion timing with correctness outside timing.
  IREE_BENCHMARK_LOOM_MEASURE_DISPATCH_COMPLETE = 1,
} iree_benchmark_loom_measure_t;

typedef struct iree_benchmark_loom_benchmark_policy_t {
  // Parsed measurement mode.
  iree_benchmark_loom_measure_t measure_kind;
  // Declared measurement mode.
  iree_string_view_t measure;
  // Number of warmup iterations run before measured timing.
  iree_host_size_t warmup_iterations;
  // Number of measured iterations.
  iree_host_size_t iterations;
  // Counter-set selection storage referenced by |hal_options|.
  iree_hal_profile_counter_set_selection_t profile_counter_set;
  // HAL benchmark options for dispatch_complete measurement.
  loom_run_hal_benchmark_options_t hal_options;
} iree_benchmark_loom_benchmark_policy_t;

typedef struct iree_benchmark_loom_diagnostic_capture_t {
  // JSON array entries for diagnostics emitted by this candidate compile.
  iree_string_builder_t output;
  // Output stream backed by |output|.
  loom_output_stream_t stream;
  // True after |output| has been initialized.
  bool initialized;
  // True until the first diagnostic has been written.
  bool first_diagnostic;
  // Number of error diagnostics captured.
  iree_host_size_t error_count;
  // Number of warning diagnostics captured.
  iree_host_size_t warning_count;
  // Number of remark diagnostics captured.
  iree_host_size_t remark_count;
} iree_benchmark_loom_diagnostic_capture_t;

typedef struct iree_benchmark_loom_device_row_state_t {
  // True once a selected-device row has been appended.
  bool appended;
} iree_benchmark_loom_device_row_state_t;

typedef struct iree_benchmark_loom_candidate_identity_t {
  // Deterministic candidate identifier within the source/run selection.
  iree_string_view_t candidate_id;
  // Zero-based selected benchmark ordinal within this run.
  iree_host_size_t candidate_index;
} iree_benchmark_loom_candidate_identity_t;

typedef struct iree_benchmark_loom_selected_benchmark_t {
  // Stable candidate identity for rows produced by this selected benchmark.
  iree_benchmark_loom_candidate_identity_t identity;
  // Inline storage backing |identity.candidate_id|.
  char candidate_id_storage[32];
  // Borrowed benchmark plan selected from the module plan.
  const loom_testbench_benchmark_plan_t* benchmark_plan;
  // Borrowed case plan referenced by |benchmark_plan|.
  const loom_testbench_case_plan_t* case_plan;
  // Effective benchmark policy after command-line overrides.
  iree_benchmark_loom_benchmark_policy_t policy;
} iree_benchmark_loom_selected_benchmark_t;

typedef struct iree_benchmark_loom_timing_stats_t {
  // Number of measured timing samples.
  iree_host_size_t count;
  // Sum of measured durations in nanoseconds.
  int64_t total_ns;
  // Minimum measured duration in nanoseconds.
  int64_t minimum_ns;
  // Maximum measured duration in nanoseconds.
  int64_t maximum_ns;
  // Arithmetic mean of measured durations in nanoseconds.
  double mean_ns;
  // Nearest-rank median duration in nanoseconds.
  int64_t p50_ns;
  // Nearest-rank p90 duration in nanoseconds.
  int64_t p90_ns;
} iree_benchmark_loom_timing_stats_t;

typedef struct iree_benchmark_loom_data_cache_summary_t {
  // True when the dispatch benchmark has populated this summary.
  bool populated;
  // Number of HAL binding references in each dispatch binding set.
  iree_host_size_t binding_count;
  // Number of physical binding sets materialized from check ops.
  iree_host_size_t binding_ring_count;
  // Number of pre-recorded command buffers rotated across benchmark batches.
  iree_host_size_t command_buffer_ring_count;
  // Number of dispatch slots recorded in each command buffer.
  iree_host_size_t dispatches_per_batch;
  // Requested minimum byte size for the physical binding ring.
  uint64_t requested_min_ring_bytes;
  // Byte length of the first materialized binding set.
  uint64_t binding_set_bytes;
  // Sum of byte lengths across materialized binding sets.
  uint64_t binding_ring_bytes;
} iree_benchmark_loom_data_cache_summary_t;

typedef struct iree_benchmark_loom_benchmark_result_t {
  // Stable benchmark status string when it differs from the default.
  iree_string_view_t status;
  // True when failure fields below describe why the benchmark did not run.
  bool has_failure;
  // Product stage that failed: compile, prepare, benchmark, etc.
  iree_string_view_t failure_stage;
  // Failure kind within |failure_stage|.
  iree_string_view_t failure_kind;
  // Optional human-facing failure message.
  iree_string_view_t failure_message;
  // Number of error diagnostics associated with the failure.
  iree_host_size_t diagnostic_error_count;
  // Number of warning diagnostics associated with the failure.
  iree_host_size_t diagnostic_warning_count;
  // Number of remark diagnostics associated with the failure.
  iree_host_size_t diagnostic_remark_count;
  // JSON array entries for structured diagnostics associated with the failure.
  iree_string_view_t diagnostic_json;
  // Captured structured compile report for the benchmark candidate.
  const loom_run_compile_report_capture_t* compile_report_capture;
  // Sidecar compile report artifact path for debug/full bundles, if any.
  iree_string_view_t compile_report_artifact_path;
  // Sidecar artifact manifest path for debug/full bundles, if any.
  iree_string_view_t artifact_manifest_path;
  // Target-native executable artifact path for debug/full bundles, if any.
  iree_string_view_t target_artifact_path;
  // Target-native textual listing path for debug/full bundles, if any.
  iree_string_view_t target_listing_path;
  // HAL executable artifact path for debug/full bundles, if any.
  iree_string_view_t hal_executable_path;
  // True when benchmark setup and timing completed.
  bool executed;
  // True when no measured or warmup sample failed expectations.
  bool passed;
  // Sample compilation label for this benchmark result.
  iree_string_view_t sample_compilation;
  // True when |sample_ordinal| identifies the measured sample.
  bool has_sample_ordinal;
  // Concrete case sample ordinal measured by dispatch_complete.
  iree_host_size_t sample_ordinal;
  // Number of case samples run per benchmark iteration.
  iree_host_size_t samples_per_iteration;
  // Failed sample executions observed during warmup and measured iterations.
  iree_host_size_t failed_sample_count;
  // Timing summary for measured iterations.
  iree_benchmark_loom_timing_stats_t timing;
  // True when |hal_benchmark| contains dispatch_complete evidence.
  bool has_hal_benchmark;
  // HAL batch benchmark evidence.
  loom_run_hal_benchmark_result_t hal_benchmark;
  // Device-buffer reuse shape used by the HAL batch benchmark.
  iree_benchmark_loom_data_cache_summary_t data_cache;
} iree_benchmark_loom_benchmark_result_t;

typedef struct iree_benchmark_loom_hal_context_t {
  // Tool configuration with linked artifact-provider registries.
  const iree_benchmark_loom_configuration_t* configuration;
  // Optional artifact bundle receiving HAL profile artifact references.
  iree_benchmark_loom_artifact_bundle_t* artifact_bundle;
  // Shared HAL runtime and artifact-provider state.
  loom_run_hal_testbench_context_t execution;
} iree_benchmark_loom_hal_context_t;

typedef struct iree_benchmark_loom_hal_actual_provider_t {
  // Shared HAL runtime and artifact-provider state.
  iree_benchmark_loom_hal_context_t* context;
  // Shared HAL actual provider owning compilation and dispatch state.
  loom_run_hal_testbench_actual_provider_t execution;
  // Sample compilation label for rows emitted from this provider.
  iree_string_view_t sample_compilation;
  // Structured diagnostics emitted while compiling this candidate.
  iree_benchmark_loom_diagnostic_capture_t diagnostics;
  // Structured compile report populated while emitting this candidate.
  loom_run_compile_report_capture_t compile_report_capture;
  // Borrowed view into |compile_report_artifact_path_storage|.
  iree_string_view_t compile_report_artifact_path;
  // Owned debug/full bundle compile-report artifact path.
  char* compile_report_artifact_path_storage;
  // Borrowed view into |artifact_manifest_path_storage|.
  iree_string_view_t artifact_manifest_path;
  // Owned debug/full bundle artifact-manifest path.
  char* artifact_manifest_path_storage;
  // Borrowed view into |target_artifact_path_storage|.
  iree_string_view_t target_artifact_path;
  // Owned debug/full bundle target-native artifact path.
  char* target_artifact_path_storage;
  // Borrowed view into |target_listing_path_storage|.
  iree_string_view_t target_listing_path;
  // Owned debug/full bundle target-native listing path.
  char* target_listing_path_storage;
  // Borrowed view into |hal_executable_path_storage|.
  iree_string_view_t hal_executable_path;
  // Owned debug/full bundle HAL executable artifact path.
  char* hal_executable_path_storage;
  // True when |compile_report_capture| owns initialized capture state.
  bool compile_report_capture_initialized;
} iree_benchmark_loom_hal_actual_provider_t;

typedef struct iree_benchmark_loom_dispatch_comparison_candidate_t {
  // Selected benchmark/case/policy identity for this comparison member.
  const iree_benchmark_loom_selected_benchmark_t* selection;
  // Module owning the selected benchmark and case plans.
  const loom_module_t* module;
  // Deduplicated physical work item index used for setup.
  iree_host_size_t work_item_index;
  // Sample compilation label for this prepared candidate.
  iree_string_view_t sample_compilation;
  // First benchmark-local sample ordinal included in the comparison window.
  iree_host_size_t begin_sample;
  // One-past-end benchmark-local sample ordinal included in the comparison
  // window.
  iree_host_size_t end_sample;
  // Number of correctness samples executed before interleaved timing.
  iree_host_size_t correctness_sample_count;
  // Number of failed correctness samples observed before interleaved timing.
  iree_host_size_t correctness_failed_sample_count;
  // True when compile and correctness succeeded and timing windows can run.
  bool runnable;
  // Per-repetition p50 logical-operation timings collected for aggregate rows.
  iree_duration_t* p50_samples;
  // Per-repetition p90 logical-operation timings collected for aggregate rows.
  iree_duration_t* p90_samples;
  // Number of entries populated in |p50_samples| and |p90_samples|.
  iree_host_size_t sample_count;
  // Capacity of |p50_samples| and |p90_samples|.
  iree_host_size_t sample_capacity;
} iree_benchmark_loom_dispatch_comparison_candidate_t;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLS_IREE_BENCHMARK_LOOM_MODEL_H_

// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Typed benchmark lifecycle events and sink adapters.
//
// Event payloads are borrowed from the benchmark run and are only valid for the
// duration of the sink callback receiving them. Streaming sinks should render
// rows before returning, and aggregating sinks must copy any strings or derived
// JSON they need after the run advances.

#ifndef LOOM_TOOLS_IREE_BENCHMARK_LOOM_EVENT_H_
#define LOOM_TOOLS_IREE_BENCHMARK_LOOM_EVENT_H_

#include "iree/base/api.h"
#include "loom/ir/module.h"
#include "loom/tooling/testbench/testbench.h"
#include "loom/tools/iree-benchmark-loom/model.h"
#include "loom/tools/iree-benchmark-loom/options.h"
#include "loom/tools/iree-benchmark-loom/output.h"
#include "loom/tools/iree-benchmark-loom/work_plan.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum iree_benchmark_loom_event_kind_e {
  // Invalid or uninitialized event.
  IREE_BENCHMARK_LOOM_EVENT_NONE = 0,
  // Run metadata emitted once after output setup.
  IREE_BENCHMARK_LOOM_EVENT_RUN = 1,
  // One selected benchmark plan row.
  IREE_BENCHMARK_LOOM_EVENT_PLAN = 2,
  // Final aggregate run summary.
  IREE_BENCHMARK_LOOM_EVENT_SUMMARY = 3,
  // Selected HAL device metadata.
  IREE_BENCHMARK_LOOM_EVENT_DEVICE = 4,
  // Candidate compile metadata.
  IREE_BENCHMARK_LOOM_EVENT_COMPILE = 5,
  // One correctness sample result.
  IREE_BENCHMARK_LOOM_EVENT_SAMPLE = 6,
  // One benchmark measurement result.
  IREE_BENCHMARK_LOOM_EVENT_BENCHMARK_RESULT = 7,
  // Optional profile rows derived from a benchmark measurement.
  IREE_BENCHMARK_LOOM_EVENT_PROFILE = 8,
  // Parse, verify, planning, or infrastructure failure metadata.
  IREE_BENCHMARK_LOOM_EVENT_FAILURE = 9,
  // One interleaved comparison benchmark repetition.
  IREE_BENCHMARK_LOOM_EVENT_BENCHMARK_REPETITION = 10,
  // Aggregate interleaved comparison result.
  IREE_BENCHMARK_LOOM_EVENT_COMPARISON = 11,
  // Deduplicated planned work graph for dry-run/reporting sinks.
  IREE_BENCHMARK_LOOM_EVENT_WORK_PLAN = 12,
} iree_benchmark_loom_event_kind_t;

typedef struct iree_benchmark_loom_run_event_t {
  // Run identity shared by all emitted events.
  const iree_benchmark_loom_run_identity_t* run;
  // True when the run stops after planning selected benchmarks.
  bool dry_run;
  // Requested dispatch sample-compilation mode for this run.
  iree_benchmark_loom_sample_compilation_mode_t sample_compilation_mode;
} iree_benchmark_loom_run_event_t;

typedef struct iree_benchmark_loom_plan_event_t {
  // Run identity shared by all emitted events.
  const iree_benchmark_loom_run_identity_t* run;
  // Parsed module that owns case and benchmark plan references.
  const loom_module_t* module;
  // Selected benchmark record being reported.
  const iree_benchmark_loom_selected_benchmark_t* selection;
  // Effective benchmark runner options for plan-policy reporting.
  const iree_benchmark_loom_options_t* options;
  // Requested dispatch sample-compilation mode for this run.
  iree_benchmark_loom_sample_compilation_mode_t sample_compilation_mode;
} iree_benchmark_loom_plan_event_t;

typedef struct iree_benchmark_loom_summary_event_t {
  // Run identity shared by all emitted events.
  const iree_benchmark_loom_run_identity_t* run;
  // Optional artifact bundle observed during execution.
  const iree_benchmark_loom_artifact_bundle_t* artifact_bundle;
  // Number of planned check.case records.
  iree_host_size_t planned_case_count;
  // Number of planned check.benchmark records.
  iree_host_size_t planned_benchmark_count;
  // Number of selected benchmark candidates.
  iree_host_size_t selected_benchmark_count;
  // Number of logical benchmark samples selected for reporting.
  iree_host_size_t logical_sample_count;
  // Number of unique physical work items selected for execution.
  iree_host_size_t work_item_count;
  // Number of top-level failure rows emitted.
  iree_host_size_t failure_count;
  // Number of benchmark rows that failed or did not execute successfully.
  iree_host_size_t failed_benchmark_count;
  // Number of correctness samples executed.
  iree_host_size_t correctness_sample_count;
  // Number of correctness samples that failed expectations.
  iree_host_size_t correctness_failed_sample_count;
  // True when the run stopped after planning selected benchmarks.
  bool dry_run;
  // Requested dispatch sample-compilation mode for this run.
  iree_benchmark_loom_sample_compilation_mode_t sample_compilation_mode;
} iree_benchmark_loom_summary_event_t;

typedef struct iree_benchmark_loom_device_event_t {
  // Run identity shared by all emitted events.
  const iree_benchmark_loom_run_identity_t* run;
  // HAL context whose selected device metadata is reported.
  iree_benchmark_loom_hal_context_t* context;
} iree_benchmark_loom_device_event_t;

typedef struct iree_benchmark_loom_compile_event_t {
  // Run identity shared by all emitted events.
  const iree_benchmark_loom_run_identity_t* run;
  // Stable candidate identity for the compile row.
  const iree_benchmark_loom_candidate_identity_t* candidate;
  // Benchmark plan owning the candidate.
  const loom_testbench_benchmark_plan_t* benchmark_plan;
  // Case plan referenced by |benchmark_plan|.
  const loom_testbench_case_plan_t* case_plan;
  // HAL actual provider containing compile diagnostics and artifact paths.
  const iree_benchmark_loom_hal_actual_provider_t* provider;
} iree_benchmark_loom_compile_event_t;

typedef struct iree_benchmark_loom_sample_event_t {
  // Run identity shared by all emitted events.
  const iree_benchmark_loom_run_identity_t* run;
  // Stable candidate identity for the sample row.
  const iree_benchmark_loom_candidate_identity_t* candidate;
  // Stable physical work item ordinal, or IREE_BENCHMARK_LOOM_INDEX_INVALID.
  iree_host_size_t work_item_index;
  // Parsed module used to render sample parameter assignments.
  const loom_module_t* module;
  // Benchmark plan owning the sample.
  const loom_testbench_benchmark_plan_t* benchmark_plan;
  // Case plan referenced by |benchmark_plan|.
  const loom_testbench_case_plan_t* case_plan;
  // Optional sample-compilation label.
  iree_string_view_t sample_compilation;
  // Zero-based sample ordinal within the benchmark's selected sample space.
  iree_host_size_t benchmark_sample_ordinal;
  // Zero-based sample ordinal within the check.case cartesian sample space.
  iree_host_size_t case_sample_ordinal;
  // Borrowed correctness result for this sample.
  const loom_testbench_case_sample_result_t* sample_result;
} iree_benchmark_loom_sample_event_t;

typedef struct iree_benchmark_loom_benchmark_result_event_t {
  // Run identity shared by all emitted events.
  const iree_benchmark_loom_run_identity_t* run;
  // Stable candidate identity for the benchmark row.
  const iree_benchmark_loom_candidate_identity_t* candidate;
  // Stable physical work item ordinal, or IREE_BENCHMARK_LOOM_INDEX_INVALID.
  iree_host_size_t work_item_index;
  // Parsed module used to render sample parameter assignments.
  const loom_module_t* module;
  // Benchmark plan owning the result.
  const loom_testbench_benchmark_plan_t* benchmark_plan;
  // Case plan referenced by |benchmark_plan|.
  const loom_testbench_case_plan_t* case_plan;
  // Effective benchmark policy used by the measurement.
  const iree_benchmark_loom_benchmark_policy_t* policy;
  // Borrowed benchmark result payload.
  const iree_benchmark_loom_benchmark_result_t* benchmark_result;
  // Number of correctness samples executed before measurement.
  iree_host_size_t correctness_sample_count;
  // Number of correctness samples that failed expectations.
  iree_host_size_t correctness_failed_sample_count;
} iree_benchmark_loom_benchmark_result_event_t;

typedef struct iree_benchmark_loom_profile_event_t {
  // Run identity shared by all emitted events.
  const iree_benchmark_loom_run_identity_t* run;
  // Stable candidate identity for the profile rows.
  const iree_benchmark_loom_candidate_identity_t* candidate;
  // Stable physical work item ordinal, or IREE_BENCHMARK_LOOM_INDEX_INVALID.
  iree_host_size_t work_item_index;
  // Parsed module used to render sample parameter assignments.
  const loom_module_t* module;
  // Benchmark plan owning the profiled result.
  const loom_testbench_benchmark_plan_t* benchmark_plan;
  // Case plan referenced by |benchmark_plan|.
  const loom_testbench_case_plan_t* case_plan;
  // Effective benchmark policy used by the measurement.
  const iree_benchmark_loom_benchmark_policy_t* policy;
  // Borrowed benchmark result payload.
  const iree_benchmark_loom_benchmark_result_t* benchmark_result;
} iree_benchmark_loom_profile_event_t;

typedef struct iree_benchmark_loom_failure_event_t {
  // Run identity shared by all emitted events.
  const iree_benchmark_loom_run_identity_t* run;
  // Pipeline stage that rejected the input.
  iree_string_view_t stage;
  // Failure kind within |stage|.
  iree_string_view_t kind;
  // Human-readable failure summary.
  iree_string_view_t message;
  // Optional captured diagnostics for this failure.
  const iree_benchmark_loom_diagnostic_capture_t* diagnostics;
} iree_benchmark_loom_failure_event_t;

typedef struct iree_benchmark_loom_benchmark_repetition_event_t {
  // Run identity shared by all emitted events.
  const iree_benchmark_loom_run_identity_t* run;
  // Interleaved comparison candidate that produced this repetition.
  const iree_benchmark_loom_dispatch_comparison_candidate_t* candidate;
  // Baseline candidate identity for the comparison group.
  const iree_benchmark_loom_candidate_identity_t* baseline;
  // Comparison group label shared by all candidates in this comparison.
  iree_string_view_t comparison_group;
  // Interleaving method name.
  iree_string_view_t method;
  // Zero-based global execution order within the comparison schedule.
  iree_host_size_t order_index;
  // Zero-based repetition ordinal for |candidate|.
  iree_host_size_t repetition_index;
  // Compact schedule label used in ABABA/round-robin output.
  char schedule_token;
  // True when final-batch profiling was suppressed for interleaving stability.
  bool profile_suppressed;
  // Borrowed benchmark result payload for this repetition.
  const iree_benchmark_loom_benchmark_result_t* benchmark_result;
} iree_benchmark_loom_benchmark_repetition_event_t;

typedef struct iree_benchmark_loom_comparison_event_t {
  // Run identity shared by all emitted events.
  const iree_benchmark_loom_run_identity_t* run;
  // Baseline interleaved comparison candidate.
  const iree_benchmark_loom_dispatch_comparison_candidate_t* baseline;
  // Candidate compared against |baseline|.
  const iree_benchmark_loom_dispatch_comparison_candidate_t* candidate;
  // Comparison group label shared by all candidates in this comparison.
  iree_string_view_t comparison_group;
  // Interleaving method name.
  iree_string_view_t method;
} iree_benchmark_loom_comparison_event_t;

typedef struct iree_benchmark_loom_work_plan_event_t {
  // Run identity shared by all emitted events.
  const iree_benchmark_loom_run_identity_t* run;
  // Parsed module that owns case and benchmark plan references.
  const loom_module_t* module;
  // Selected benchmark and logical-to-physical work plan.
  const iree_benchmark_loom_work_plan_t* work_plan;
} iree_benchmark_loom_work_plan_event_t;

typedef struct iree_benchmark_loom_event_t {
  // Event kind and union discriminator.
  iree_benchmark_loom_event_kind_t kind;
  union {
    // Payload for IREE_BENCHMARK_LOOM_EVENT_RUN.
    iree_benchmark_loom_run_event_t run;
    // Payload for IREE_BENCHMARK_LOOM_EVENT_PLAN.
    iree_benchmark_loom_plan_event_t plan;
    // Payload for IREE_BENCHMARK_LOOM_EVENT_SUMMARY.
    iree_benchmark_loom_summary_event_t summary;
    // Payload for IREE_BENCHMARK_LOOM_EVENT_DEVICE.
    iree_benchmark_loom_device_event_t device;
    // Payload for IREE_BENCHMARK_LOOM_EVENT_COMPILE.
    iree_benchmark_loom_compile_event_t compile;
    // Payload for IREE_BENCHMARK_LOOM_EVENT_SAMPLE.
    iree_benchmark_loom_sample_event_t sample;
    // Payload for IREE_BENCHMARK_LOOM_EVENT_BENCHMARK_RESULT.
    iree_benchmark_loom_benchmark_result_event_t benchmark_result;
    // Payload for IREE_BENCHMARK_LOOM_EVENT_PROFILE.
    iree_benchmark_loom_profile_event_t profile;
    // Payload for IREE_BENCHMARK_LOOM_EVENT_FAILURE.
    iree_benchmark_loom_failure_event_t failure;
    // Payload for IREE_BENCHMARK_LOOM_EVENT_BENCHMARK_REPETITION.
    iree_benchmark_loom_benchmark_repetition_event_t benchmark_repetition;
    // Payload for IREE_BENCHMARK_LOOM_EVENT_COMPARISON.
    iree_benchmark_loom_comparison_event_t comparison;
    // Payload for IREE_BENCHMARK_LOOM_EVENT_WORK_PLAN.
    iree_benchmark_loom_work_plan_event_t work_plan;
  };
} iree_benchmark_loom_event_t;

typedef iree_status_t (*iree_benchmark_loom_event_sink_emit_fn_t)(
    void* user_data, const iree_benchmark_loom_event_t* event);

typedef struct iree_benchmark_loom_event_sink_t {
  // Callback invoked once per event with a borrowed event payload.
  iree_benchmark_loom_event_sink_emit_fn_t emit;
  // Opaque sink state passed to |emit|.
  void* user_data;
} iree_benchmark_loom_event_sink_t;

typedef struct iree_benchmark_loom_jsonl_event_sink_t {
  // Borrowed JSONL row sink receiving rendered events during emission.
  iree_benchmark_loom_jsonl_sink_t* jsonl_sink;
  // Renderer-local selected-device row suppression state.
  iree_benchmark_loom_device_row_state_t device_row_state;
} iree_benchmark_loom_jsonl_event_sink_t;

// Emits |event| to |sink|.
iree_status_t iree_benchmark_loom_event_sink_emit(
    const iree_benchmark_loom_event_sink_t* sink,
    const iree_benchmark_loom_event_t* event);

// Emits the run metadata event.
iree_status_t iree_benchmark_loom_event_sink_emit_run(
    const iree_benchmark_loom_event_sink_t* sink,
    const iree_benchmark_loom_run_identity_t* run, bool dry_run,
    iree_benchmark_loom_sample_compilation_mode_t sample_compilation_mode);

// Emits one selected benchmark plan event.
iree_status_t iree_benchmark_loom_event_sink_emit_plan(
    const iree_benchmark_loom_event_sink_t* sink,
    const iree_benchmark_loom_run_identity_t* run, const loom_module_t* module,
    const iree_benchmark_loom_selected_benchmark_t* selection,
    const iree_benchmark_loom_options_t* options,
    iree_benchmark_loom_sample_compilation_mode_t sample_compilation_mode);

// Emits the deduplicated planned work graph.
iree_status_t iree_benchmark_loom_event_sink_emit_work_plan(
    const iree_benchmark_loom_event_sink_t* sink,
    const iree_benchmark_loom_run_identity_t* run, const loom_module_t* module,
    const iree_benchmark_loom_work_plan_t* work_plan);

// Emits the terminal aggregate summary event.
iree_status_t iree_benchmark_loom_event_sink_emit_summary(
    const iree_benchmark_loom_event_sink_t* sink,
    const iree_benchmark_loom_run_identity_t* run,
    const iree_benchmark_loom_artifact_bundle_t* artifact_bundle,
    iree_host_size_t planned_case_count,
    iree_host_size_t planned_benchmark_count,
    iree_host_size_t selected_benchmark_count,
    iree_host_size_t logical_sample_count, iree_host_size_t work_item_count,
    iree_host_size_t failure_count, iree_host_size_t failed_benchmark_count,
    iree_host_size_t correctness_sample_count,
    iree_host_size_t correctness_failed_sample_count, bool dry_run,
    iree_benchmark_loom_sample_compilation_mode_t sample_compilation_mode);

// Emits selected HAL device metadata.
iree_status_t iree_benchmark_loom_event_sink_emit_device(
    const iree_benchmark_loom_event_sink_t* sink,
    const iree_benchmark_loom_run_identity_t* run,
    iree_benchmark_loom_hal_context_t* context);

// Emits candidate compile metadata.
iree_status_t iree_benchmark_loom_event_sink_emit_compile(
    const iree_benchmark_loom_event_sink_t* sink,
    const iree_benchmark_loom_run_identity_t* run,
    const iree_benchmark_loom_candidate_identity_t* candidate,
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    const loom_testbench_case_plan_t* case_plan,
    const iree_benchmark_loom_hal_actual_provider_t* provider);

// Emits one correctness sample result.
iree_status_t iree_benchmark_loom_event_sink_emit_sample(
    const iree_benchmark_loom_event_sink_t* sink,
    const iree_benchmark_loom_run_identity_t* run,
    const iree_benchmark_loom_candidate_identity_t* candidate,
    iree_host_size_t work_item_index, const loom_module_t* module,
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    const loom_testbench_case_plan_t* case_plan,
    iree_string_view_t sample_compilation,
    iree_host_size_t benchmark_sample_ordinal,
    iree_host_size_t case_sample_ordinal,
    const loom_testbench_case_sample_result_t* sample_result);

// Emits one benchmark measurement result.
iree_status_t iree_benchmark_loom_event_sink_emit_benchmark_result(
    const iree_benchmark_loom_event_sink_t* sink,
    const iree_benchmark_loom_run_identity_t* run,
    const iree_benchmark_loom_candidate_identity_t* candidate,
    iree_host_size_t work_item_index, const loom_module_t* module,
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    const loom_testbench_case_plan_t* case_plan,
    const iree_benchmark_loom_benchmark_policy_t* policy,
    const iree_benchmark_loom_benchmark_result_t* benchmark_result,
    iree_host_size_t correctness_sample_count,
    iree_host_size_t correctness_failed_sample_count);

// Emits optional profile rows for one benchmark measurement result.
iree_status_t iree_benchmark_loom_event_sink_emit_profile(
    const iree_benchmark_loom_event_sink_t* sink,
    const iree_benchmark_loom_run_identity_t* run,
    const iree_benchmark_loom_candidate_identity_t* candidate,
    iree_host_size_t work_item_index, const loom_module_t* module,
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    const loom_testbench_case_plan_t* case_plan,
    const iree_benchmark_loom_benchmark_policy_t* policy,
    const iree_benchmark_loom_benchmark_result_t* benchmark_result);

// Emits parse, verify, planning, or infrastructure failure metadata.
iree_status_t iree_benchmark_loom_event_sink_emit_failure(
    const iree_benchmark_loom_event_sink_t* sink,
    const iree_benchmark_loom_run_identity_t* run, iree_string_view_t stage,
    iree_string_view_t kind, iree_string_view_t message,
    const iree_benchmark_loom_diagnostic_capture_t* diagnostics);

// Emits one interleaved comparison benchmark repetition.
iree_status_t iree_benchmark_loom_event_sink_emit_benchmark_repetition(
    const iree_benchmark_loom_event_sink_t* sink,
    const iree_benchmark_loom_run_identity_t* run,
    const iree_benchmark_loom_dispatch_comparison_candidate_t* candidate,
    const iree_benchmark_loom_candidate_identity_t* baseline,
    iree_string_view_t comparison_group, iree_string_view_t method,
    iree_host_size_t order_index, iree_host_size_t repetition_index,
    char schedule_token, bool profile_suppressed,
    const iree_benchmark_loom_benchmark_result_t* benchmark_result);

// Emits one aggregate interleaved comparison result.
iree_status_t iree_benchmark_loom_event_sink_emit_comparison(
    const iree_benchmark_loom_event_sink_t* sink,
    const iree_benchmark_loom_run_identity_t* run,
    const iree_benchmark_loom_dispatch_comparison_candidate_t* baseline,
    const iree_benchmark_loom_dispatch_comparison_candidate_t* candidate,
    iree_string_view_t comparison_group, iree_string_view_t method);

// Initializes a JSONL event adapter over |jsonl_sink|.
void iree_benchmark_loom_jsonl_event_sink_initialize(
    iree_benchmark_loom_jsonl_sink_t* jsonl_sink,
    iree_benchmark_loom_jsonl_event_sink_t* out_adapter,
    iree_benchmark_loom_event_sink_t* out_sink);

// Releases borrowed references held by |adapter|.
void iree_benchmark_loom_jsonl_event_sink_deinitialize(
    iree_benchmark_loom_jsonl_event_sink_t* adapter);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLS_IREE_BENCHMARK_LOOM_EVENT_H_

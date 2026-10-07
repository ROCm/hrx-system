// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// HAL kernel-launch bridge for Loom check testbench actual-candidate execution.
//
// This layer is target-neutral: tools inject an execution session with its
// composed target environment and linked device providers, while this bridge
// owns HAL runtime selection, candidate compilation, dispatch input conversion,
// and the callback shape used by the testbench executor.

#ifndef LOOM_TOOLING_EXECUTION_HAL_TESTBENCH_ACTUAL_H_
#define LOOM_TOOLING_EXECUTION_HAL_TESTBENCH_ACTUAL_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "loom/analysis/kernel_launch_config.h"
#include "loom/pass/pipeline_snapshot.h"
#include "loom/sanitizer/options.h"
#include "loom/target/provider.h"
#include "loom/tooling/compile/options.h"
#include "loom/tooling/compile/pipeline.h"
#include "loom/tooling/execution/hal/candidate.h"
#include "loom/tooling/execution/hal/device_provider.h"
#include "loom/tooling/execution/hal/invocation.h"
#include "loom/tooling/execution/hal/runtime.h"
#include "loom/tooling/execution/session.h"
#include "loom/tooling/testbench/invocation.h"
#include "loom/tooling/testbench/requirements.h"
#include "loom/tooling/testbench/testbench.h"
#include "loom/tooling/testbench/value_materializer.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_tooling_config_set_t loom_tooling_config_set_t;

typedef struct loom_run_hal_testbench_context_t {
  // Linked device-provider registry selected by the tool binary.
  const loom_device_provider_registry_t* device_provider_registry;
  // Host allocator used for runtime and candidate storage.
  iree_allocator_t host_allocator;
  // Device event sink used when initializing |runtime|.
  iree_hal_device_event_sink_t device_event_sink;
  // HAL runtime services required by every module added to this context.
  iree_hal_device_runtime_feature_flags_t runtime_features;
  // Selected provider for the active device.
  const loom_device_provider_t* device_provider;
  // Shared HAL runtime used by kernel launches.
  loom_run_hal_runtime_t runtime;
  // True when |runtime| owns initialized HAL state.
  bool runtime_initialized;
} loom_run_hal_testbench_context_t;

// Initializes a HAL testbench context with a linked device-provider registry.
void loom_run_hal_testbench_context_initialize(
    const loom_device_provider_registry_t* device_provider_registry,
    iree_allocator_t host_allocator,
    loom_run_hal_testbench_context_t* out_context);

// Sets the device event sink used by future HAL runtime initialization.
void loom_run_hal_testbench_context_set_device_event_sink(
    loom_run_hal_testbench_context_t* context,
    iree_hal_device_event_sink_t device_event_sink);

// Adds runtime requirements for executable sanitizer operations in |module|
// and instrumentation requested by |sanitizer_options|.
//
// Requirements may be accumulated across modules before runtime creation. A
// module added afterward must require only services already provisioned by the
// active device.
iree_status_t loom_run_hal_testbench_context_add_module_runtime_requirements(
    loom_run_hal_testbench_context_t* context, const loom_module_t* module,
    const loom_sanitizer_options_t* sanitizer_options);

// Releases HAL runtime resources owned by |context|.
void loom_run_hal_testbench_context_deinitialize(
    loom_run_hal_testbench_context_t* context);

// Validates an explicit --device selection against the linked providers.
// Empty selection remains lazy and does not initialize a HAL runtime.
iree_status_t loom_run_hal_testbench_context_validate_explicit_device(
    loom_run_hal_testbench_context_t* context);

// Selects a linked device provider and initializes the HAL runtime on demand.
iree_status_t loom_run_hal_testbench_context_ensure_runtime(
    loom_run_hal_testbench_context_t* context);

// Returns host-visible fixture parameters for CPU generation and observations.
// HAL execution stages fixtures into device-local memory around kernel
// launches.
iree_hal_buffer_params_t loom_run_hal_testbench_host_visible_buffer_params(
    void);

// Finds the single kernel launch in |case_plan| accepted by the HAL bridge.
iree_status_t loom_run_hal_testbench_select_kernel_launch(
    const loom_testbench_case_plan_t* case_plan,
    const loom_testbench_invocation_plan_t** out_kernel_launch);

// Counts kernel launches in |case_plan| and validates their schedule shape.
iree_status_t loom_run_hal_testbench_count_kernel_launches(
    const loom_testbench_case_plan_t* case_plan,
    iree_host_size_t* out_kernel_launch_count);

typedef struct loom_run_hal_testbench_actual_provider_options_t {
  // Shared HAL context used to prepare and dispatch the candidate.
  loom_run_hal_testbench_context_t* context;
  // Execution session used to clone and compile the private module copy.
  loom_run_session_t* session;
  // Canonical parsed module that owns |kernel_launch|. Borrowed through
  // provider deinitialization.
  const loom_run_module_t* run_module;
  // User-selected pass pipeline.
  iree_string_view_t pipeline;
  // Optional explicit `family:selector` compiler target.
  iree_string_view_t target;
  // Sanitizer checks inserted by the target pipeline.
  loom_sanitizer_options_t sanitizer;
  // Config bindings materialized into the private compile copy.
  const loom_tooling_config_set_t* config_set;
  // Kernel launch selected from the owning check.case.
  const loom_testbench_invocation_plan_t* kernel_launch;
  // Diagnostic sink used while lowering and emitting the candidate.
  loom_diagnostic_sink_t diagnostic_sink;
  // Maximum diagnostics to emit before stopping. Zero uses the default.
  uint32_t max_errors;
  // Optional caller-owned structured compile report to populate.
  loom_target_compile_report_t* report;
  // Optional debug artifacts requested from the selected backend.
  loom_compile_artifact_flags_t artifact_flags;
  // Optional artifact manifest requested from the selected backend.
  loom_compile_artifact_manifest_options_t artifact_manifest;
} loom_run_hal_testbench_actual_provider_options_t;

typedef struct loom_run_hal_testbench_actual_provider_t {
  // Shared HAL context used to prepare and dispatch the candidate.
  loom_run_hal_testbench_context_t* context;
  // Execution session used to clone and compile the private module copy.
  loom_run_session_t* session;
  // Canonical parsed module that owns |kernel_launch|. Borrowed through
  // provider deinitialization.
  const loom_run_module_t* run_module;
  // User-selected pass pipeline.
  iree_string_view_t pipeline;
  // Selected named pipeline closure retained across deferred compilation.
  loom_pass_pipeline_snapshot_t pipeline_snapshot;
  // Optional explicit `family:selector` compiler target.
  iree_string_view_t target;
  // Sanitizer checks inserted by the target pipeline.
  loom_sanitizer_options_t sanitizer;
  // Config bindings materialized into the private compile copy.
  const loom_tooling_config_set_t* config_set;
  // Kernel launch selected from the owning check.case.
  const loom_testbench_invocation_plan_t* kernel_launch;
  // Diagnostic sink used while lowering and emitting the candidate.
  loom_diagnostic_sink_t diagnostic_sink;
  // Maximum diagnostics to emit before stopping. Zero uses the default.
  uint32_t max_errors;
  // Optional caller-owned structured compile report to populate.
  loom_target_compile_report_t* report;
  // Optional debug artifacts requested from the selected backend.
  loom_compile_artifact_flags_t artifact_flags;
  // Optional artifact manifest requested from the selected backend.
  loom_compile_artifact_manifest_options_t artifact_manifest;
  // Private compile module owned by this provider.
  loom_run_module_t compile_module;
  // Compiler-produced host launch program retained for repeated evaluation.
  loom_module_t* launch_config_module;
  // Exported function bound from |launch_config_module|.
  loom_kernel_launch_config_function_t launch_config_function;
  // Reusable fact storage for launch-function evaluation.
  loom_pass_value_fact_owner_t launch_config_fact_owner;
  // Reusable raw workload argument bits used during launch evaluation.
  uint64_t* workload_argument_bits;
  // Backend-produced HAL executable candidate.
  loom_run_hal_candidate_t candidate;
  // Target selected before the compile pipeline runs.
  loom_device_target_t compile_device_target;
  // Prepared executable retained for correctness and benchmark dispatches.
  loom_run_hal_prepared_candidate_t prepared_candidate;
  // Allocator-owned reflected logical parameter layout for the prepared
  // executable function. Parameter string views borrow executable storage.
  iree_hal_executable_function_parameter_t* function_parameters;
  // Number of entries in |function_parameters|. Zero means the backend did
  // not publish logical parameter reflection.
  iree_host_size_t function_parameter_count;
  // Dispatch options derived from the compiled source entry.
  loom_run_hal_invocation_options_t invocation_options;
  // Most recently evaluated compiled launch configuration.
  loom_kernel_launch_config_t resolved_launch_config;
  // Compiler products retained through artifact emission.
  loom_compile_pipeline_result_t pipeline_result;
  // Product stage that rejected the compile, when |compile_rejected| is true.
  iree_string_view_t compile_failure_stage;
  // Stable diagnostic category for |compile_rejected|.
  iree_string_view_t compile_failure_kind;
  // Optional human-facing explanation for |compile_rejected|.
  iree_string_view_t compile_failure_message;
  // Number of error diagnostics observed while compiling this candidate.
  iree_host_size_t diagnostic_error_count;
  // Number of warning diagnostics observed while compiling this candidate.
  iree_host_size_t diagnostic_warning_count;
  // Number of remark diagnostics observed while compiling this candidate.
  iree_host_size_t diagnostic_remark_count;
  // True when compile completed with product diagnostics instead of an
  // infrastructure failure.
  bool compile_rejected;
  // True when |compile_module| has been initialized.
  bool compile_module_initialized;
  // True when |launch_config_fact_owner| has been initialized.
  bool launch_config_evaluation_initialized;
  // True when |candidate| has been initialized.
  bool candidate_initialized;
  // True when |compile_device_target| owns provider-selected target storage.
  bool owns_compile_device_target;
  // True when |prepared_candidate| has been initialized.
  bool prepared_candidate_initialized;
  // True when HAL candidate emission populated the caller's compile report.
  bool compile_report_available;
} loom_run_hal_testbench_actual_provider_t;

typedef struct loom_run_hal_testbench_actual_sequence_execution_t
    loom_run_hal_testbench_actual_sequence_execution_t;

typedef struct loom_run_hal_testbench_actual_sequence_options_t {
  // Shared HAL context used to prepare and dispatch all actual candidates.
  loom_run_hal_testbench_context_t* context;
  // Execution session used to clone and compile each private module copy.
  loom_run_session_t* session;
  // Canonical parsed module that owns |case_plan|. Borrowed through sequence
  // deinitialization.
  const loom_run_module_t* run_module;
  // User-selected pass pipeline.
  iree_string_view_t pipeline;
  // Optional explicit `family:selector` compiler target.
  iree_string_view_t target;
  // Sanitizer checks inserted by the target pipeline.
  loom_sanitizer_options_t sanitizer;
  // Config bindings materialized into each private compile copy.
  const loom_tooling_config_set_t* config_set;
  // Case plan whose kernel launches are executed by the sequence.
  const loom_testbench_case_plan_t* case_plan;
  // Diagnostic sink used while lowering and emitting candidates.
  loom_diagnostic_sink_t diagnostic_sink;
  // Maximum diagnostics to emit before stopping. Zero uses the default.
  uint32_t max_errors;
  // Optional debug artifacts requested from the selected backend.
  loom_compile_artifact_flags_t artifact_flags;
  // Optional artifact manifest requested from the selected backend.
  loom_compile_artifact_manifest_options_t artifact_manifest;
} loom_run_hal_testbench_actual_sequence_options_t;

typedef struct loom_run_hal_testbench_actual_sequence_t {
  // Host allocator used for sequence-owned provider storage.
  iree_allocator_t host_allocator;
  // Sequence-owned providers in check.case source order.
  loom_run_hal_testbench_actual_provider_t* providers;
  // Number of entries in |providers|.
  iree_host_size_t provider_count;
  // Prepared ordered execution over |providers|.
  loom_run_hal_testbench_actual_sequence_execution_t* execution;
} loom_run_hal_testbench_actual_sequence_t;

// Initializes a compile-on-first-use HAL actual provider.
void loom_run_hal_testbench_actual_provider_initialize(
    const loom_run_hal_testbench_actual_provider_options_t* options,
    loom_run_hal_testbench_actual_provider_t* out_provider);

// Releases storage owned by |provider|.
void loom_run_hal_testbench_actual_provider_deinitialize(
    loom_run_hal_testbench_actual_provider_t* provider);

// Compiles and prepares the selected actual candidate if needed.
iree_status_t loom_run_hal_testbench_actual_provider_compile(
    loom_run_hal_testbench_actual_provider_t* provider);

// Materializes one invocation of an already-prepared provider.
//
// Workload values resolve launch geometry while ordinary inputs become direct
// constants or retained buffer bindings in HAL ABI order. Compilation is not
// performed here: |provider| must already own a prepared candidate so callers
// can keep runtime trial values outside product preparation.
iree_status_t loom_run_hal_testbench_actual_provider_materialize_invocation(
    loom_run_hal_testbench_actual_provider_t* provider,
    iree_host_size_t workload_count, const loom_testbench_value_t* workloads,
    iree_host_size_t input_count, const loom_testbench_value_t* inputs,
    loom_run_hal_invocation_options_t* out_options,
    loom_run_hal_binding_list_t* out_bindings);

// Creates ordered execution over actual providers in source order.
//
// Provider objects referenced by |providers| are borrowed until the returned
// execution is destroyed; the pointer array itself is needed only during this
// call. One execution may be reused across case executors but is submitted
// serially; concurrent invocation requires a distinct execution object.
iree_status_t loom_run_hal_testbench_actual_sequence_execution_create(
    const loom_testbench_case_plan_t* case_plan,
    iree_host_size_t provider_count,
    loom_run_hal_testbench_actual_provider_t* const* providers,
    iree_allocator_t host_allocator,
    loom_run_hal_testbench_actual_sequence_execution_t** out_execution);

// Releases storage and reusable command sequences owned by |execution|.
void loom_run_hal_testbench_actual_sequence_execution_destroy(
    loom_run_hal_testbench_actual_sequence_execution_t* execution);

// Returns a testbench provider backed by |execution|.
loom_testbench_invocation_provider_t
loom_run_hal_testbench_actual_sequence_execution_provider(
    loom_run_hal_testbench_actual_sequence_execution_t* execution);

// Testbench invocation callback for HAL kernel launches.
iree_status_t loom_run_hal_testbench_actual_invoke(
    void* user_data, const loom_testbench_invocation_plan_t* invocation,
    iree_host_size_t workload_count, const loom_testbench_value_t* workloads,
    iree_host_size_t input_count, const loom_testbench_value_t* inputs,
    iree_host_size_t result_count, loom_testbench_value_t* out_results);

// Initializes a compile-on-first-use provider sequence for every kernel launch
// in a check.case.
iree_status_t loom_run_hal_testbench_actual_sequence_initialize(
    const loom_run_hal_testbench_actual_sequence_options_t* options,
    loom_run_hal_testbench_actual_sequence_t* out_sequence);

// Releases storage owned by |sequence|.
void loom_run_hal_testbench_actual_sequence_deinitialize(
    loom_run_hal_testbench_actual_sequence_t* sequence);

// Returns a testbench provider backed by |sequence|.
loom_testbench_invocation_provider_t
loom_run_hal_testbench_actual_sequence_provider(
    loom_run_hal_testbench_actual_sequence_t* sequence);

// Appends borrowed testbench input values to HAL bindings/constants.
//
// When |input_parameters| is provided, scalar widths and HAL table offsets are
// taken from the loaded executable's reflected ABI. A NULL parameter list uses
// address carrier widths from |target_snapshot| for backends without parameter
// reflection. |target_snapshot| may be NULL only when no input is an address
// scalar.
iree_status_t loom_run_hal_testbench_invocation_inputs_from_values(
    const loom_testbench_value_t* inputs, const loom_type_t* input_types,
    const loom_target_snapshot_t* target_snapshot,
    const iree_hal_executable_function_parameter_t* input_parameters,
    iree_host_size_t input_count, loom_run_hal_invocation_options_t* options,
    iree_allocator_t allocator, loom_run_hal_binding_list_t* out_bindings);

// Materializes one kernel launch's geometry and HAL bindings from
// an already-materialized case sample value table.
iree_status_t loom_run_hal_testbench_materialize_invocation_from_table(
    const loom_testbench_value_table_t* table,
    loom_run_hal_testbench_actual_provider_t* provider,
    iree_allocator_t allocator, loom_run_hal_invocation_options_t* out_options,
    loom_run_hal_binding_list_t* out_bindings);

// Materializes one case sample's kernel launch as geometry and HAL bindings.
iree_status_t loom_run_hal_testbench_materialize_invocation_for_sample(
    const loom_module_t* module,
    const loom_testbench_value_materializer_options_t* materializer_options,
    const loom_testbench_case_plan_t* case_plan,
    loom_run_hal_testbench_actual_provider_t* provider,
    iree_host_size_t sample_ordinal, iree_allocator_t allocator,
    loom_run_hal_invocation_options_t* out_options,
    loom_run_hal_binding_list_t* out_bindings);

// Prepares a reusable HAL invocation plan for one testbench sample.
iree_status_t loom_run_hal_testbench_prepare_invocation_plan_for_sample(
    const loom_testbench_module_plan_t* module_plan,
    const loom_testbench_case_plan_t* case_plan,
    const loom_testbench_value_materializer_options_t* materializer_options,
    loom_run_hal_testbench_actual_provider_t* provider,
    iree_host_size_t sample_ordinal, iree_allocator_t allocator,
    loom_run_hal_invocation_plan_t* out_plan);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_EXECUTION_HAL_TESTBENCH_ACTUAL_H_

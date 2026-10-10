// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// HAL dispatch bridge for Loom check testbench actual-candidate execution.
//
// This layer is target-neutral: tools inject a public target environment and
// linked HAL target providers, while this bridge owns HAL runtime selection,
// candidate compilation, dispatch input conversion, and the callback shape
// used by the testbench executor.

#ifndef LOOM_TOOLING_EXECUTION_HAL_TESTBENCH_ACTUAL_H_
#define LOOM_TOOLING_EXECUTION_HAL_TESTBENCH_ACTUAL_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "loom/ir/module.h"
#include "loom/target/types.h"
#include "loom/tooling/execution/hal/invocation.h"
#include "loom/tooling/execution/hal/runtime.h"
#include "loom/tooling/testbench/compiled_provider.h"
#include "loom/tooling/testbench/invocation.h"
#include "loom/tooling/testbench/testbench.h"
#include "loom/tooling/testbench/value_materializer.h"
#include "loomc/compile.h"
#include "loomc/launch_config.h"
#include "loomc/sanitizer.h"
#include "loomc/target/iree_hal.h"

#ifdef __cplusplus
extern "C" {
#endif

// Connects one command-line HAL driver to its public compiler target adapter.
typedef struct loom_run_hal_target_route_t {
  // Canonical IREE HAL driver name accepted in a device URI.
  iree_string_view_t driver_name;
  // Public adapter that selects a compiler profile from the live device.
  const loomc_iree_hal_target_provider_t* provider;
} loom_run_hal_target_route_t;

typedef struct loom_run_hal_testbench_context_t {
  // Public target environment composed by the final tool binary.
  loomc_target_environment_t* target_environment;
  // Driver-to-target routes linked into the final tool binary.
  const loom_run_hal_target_route_t* target_routes;
  // Number of entries in |target_routes|.
  iree_host_size_t target_route_count;
  // Route selected by the explicit device URI, once known.
  const loom_run_hal_target_route_t* selected_target_route;
  // Host allocator used for runtime and candidate storage.
  iree_allocator_t host_allocator;
  // Device event sink used when initializing |runtime|.
  iree_hal_device_event_sink_t device_event_sink;
  // HAL runtime services required by every module added to this context.
  iree_hal_device_runtime_feature_flags_t runtime_features;
  // HAL driver parsed from the single active --device URI.
  iree_string_view_t driver_name;
  // Shared HAL runtime used by kernel launches.
  loom_run_hal_runtime_t runtime;
  // True when |runtime| owns initialized HAL state.
  bool runtime_initialized;
} loom_run_hal_testbench_context_t;

// Initializes a HAL testbench context with linked public target routes.
void loom_run_hal_testbench_context_initialize(
    loomc_target_environment_t* target_environment,
    const loom_run_hal_target_route_t* target_routes,
    iree_host_size_t target_route_count, iree_allocator_t host_allocator,
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
    loom_run_hal_testbench_context_t* context, const loomc_module_t* module,
    const loomc_sanitizer_options_t* sanitizer_options);

// Releases HAL runtime resources owned by |context|.
void loom_run_hal_testbench_context_deinitialize(
    loom_run_hal_testbench_context_t* context);

// Parses an explicit --device selection. Empty selection remains lazy and does
// not initialize a HAL runtime.
iree_status_t loom_run_hal_testbench_context_validate_explicit_device(
    loom_run_hal_testbench_context_t* context);

// Initializes the selected HAL runtime on demand.
iree_status_t loom_run_hal_testbench_context_ensure_runtime(
    loom_run_hal_testbench_context_t* context);

// Selects one compiler profile and exact loader target for the live device.
// The caller owns the returned result and target profile according to the
// public LoomC contracts.
iree_status_t loom_run_hal_testbench_context_select_target(
    loom_run_hal_testbench_context_t* context, loomc_string_view_t identifier,
    loomc_target_profile_t* target_profile,
    loomc_iree_hal_target_selection_t* out_selection,
    loomc_result_t** out_result);

// Converts a rejected public compiler result to a failed-precondition status.
// The first error diagnostic supplies the message when one is available.
iree_status_t loom_run_hal_testbench_require_successful_result(
    const loomc_result_t* result, iree_string_view_t fallback_message);

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
  // Public compiler services and configuration for private module clones.
  const loom_testbench_compilation_t* compilation;
  // Public source module compiled by this provider. Defaults to the module in
  // |compilation| and may name an exact-version tooling-generated derivative.
  const loomc_module_t* module;
  // Native read-only projection of |module| owning |invocation|.
  const loom_module_t* native_module;
  // Optional prepared pass program. NULL selects the target default.
  const loomc_pass_program_t* pass_program;
  // Optional caller-selected target profile validated against the live device.
  loomc_target_profile_t* requested_target_profile;
  // Optional sanitizer policy for the target default pass program.
  const loomc_sanitizer_options_t* sanitizer;
  // Kernel or finite pipeline invocation selected from its case or scenario.
  const loom_testbench_invocation_plan_t* invocation;
  // Observer receiving each public target-selection and compilation result.
  loom_testbench_compile_result_callback_t result_callback;
  // Optional compile report request.
  loomc_compile_report_options_t compile_report;
  // Optional artifact manifest request.
  loomc_artifact_manifest_options_t artifact_manifest;
  // Target-side artifact classes requested by the caller.
  loomc_emit_artifact_flags_t emit_artifact_flags;
} loom_run_hal_testbench_actual_provider_options_t;

typedef struct loom_run_hal_testbench_compile_artifacts_t {
  // Loadable target executable returned by compilation.
  const loomc_artifact_t* executable;
  // Compiled host launch-configuration program.
  const loomc_artifact_t* launch_config;
  // Optional requested JSON or text compile report.
  const loomc_artifact_t* compile_report;
  // Optional requested JSON artifact manifest.
  const loomc_artifact_t* artifact_manifest;
  // Optional target-owned textual listing.
  const loomc_artifact_t* target_listing;
} loom_run_hal_testbench_compile_artifacts_t;

typedef struct loom_run_hal_testbench_actual_provider_t {
  // Shared HAL context used to prepare and dispatch the candidate.
  loom_run_hal_testbench_context_t* context;
  // Public compiler services and configuration for private module clones.
  const loom_testbench_compilation_t* compilation;
  // Public source module compiled by this provider.
  const loomc_module_t* module;
  // Native read-only projection of |module| owning |invocation|.
  const loom_module_t* native_module;
  // Optional prepared pass program. NULL selects the target default.
  const loomc_pass_program_t* pass_program;
  // Optional caller-selected target profile validated against the live device.
  loomc_target_profile_t* requested_target_profile;
  // Optional sanitizer policy for the target default pass program.
  const loomc_sanitizer_options_t* sanitizer;
  // Kernel or finite pipeline invocation selected from its case or scenario.
  const loom_testbench_invocation_plan_t* invocation;
  // Observer receiving each public target-selection and compilation result.
  loom_testbench_compile_result_callback_t result_callback;
  // Optional compile report request.
  loomc_compile_report_options_t compile_report;
  // Optional artifact manifest request.
  loomc_artifact_manifest_options_t artifact_manifest;
  // Target-side artifact classes requested by the caller.
  loomc_emit_artifact_flags_t emit_artifact_flags;
  // Selected public target profile retained through provider teardown.
  loomc_target_profile_t* selected_target_profile;
  // Native target snapshot borrowed from |selected_target_profile|.
  const loom_target_snapshot_t* target_snapshot;
  // Exact live-device executable target paired with the emitted bytes.
  const iree_hal_executable_target_t* executable_target;
  // Public compiler result retained for artifact and diagnostic lifetimes.
  loomc_result_t* compiler_result;
  // Selected result-owned artifacts.
  loom_run_hal_testbench_compile_artifacts_t artifacts;
  // Loaded compiler-produced launch program retained for evaluation.
  loomc_launch_config_program_t* launch_config_program;
  // Exported function bound from |launch_config_program|.
  loomc_launch_config_function_t launch_config_function;
  // Reusable raw workload argument bits used during launch evaluation.
  uint64_t* workload_argument_bits;
  // Prepared executable retained for correctness and benchmark dispatches.
  loom_run_hal_prepared_candidate_t prepared_candidate;
  // Queue submission representation required by the prepared invocation.
  loom_run_hal_dispatch_sequence_representation_t sequence_representation;
  // Allocator-owned reflected logical parameter layout for the prepared
  // executable function. Parameter string views borrow executable storage.
  iree_hal_executable_function_parameter_t* function_parameters;
  // Number of entries in |function_parameters|. Zero means the backend did
  // not publish logical parameter reflection.
  iree_host_size_t function_parameter_count;
  // Dispatch options derived from the compiled source entry.
  loom_run_hal_invocation_options_t invocation_options;
  // Most recently evaluated compiled launch configuration.
  loomc_launch_config_t resolved_launch_config;
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
  // True when |prepared_candidate| has been initialized.
  bool prepared_candidate_initialized;
} loom_run_hal_testbench_actual_provider_t;

typedef struct loom_run_hal_testbench_actual_sequence_execution_t
    loom_run_hal_testbench_actual_sequence_execution_t;

typedef struct loom_run_hal_testbench_actual_sequence_options_t {
  // Shared HAL context used to prepare and dispatch all actual candidates.
  loom_run_hal_testbench_context_t* context;
  // Public compiler services and configuration for private module clones.
  const loom_testbench_compilation_t* compilation;
  // Native read-only projection of |compilation->module| owning |case_plan|.
  const loom_module_t* native_module;
  // Optional prepared pass program. NULL selects the target default.
  const loomc_pass_program_t* pass_program;
  // Optional caller-selected target profile validated against the live device.
  loomc_target_profile_t* requested_target_profile;
  // Optional sanitizer policy for the target default pass program.
  const loomc_sanitizer_options_t* sanitizer;
  // Case plan whose kernel launches are executed by the sequence.
  const loom_testbench_case_plan_t* case_plan;
  // Observer receiving each public target-selection and compilation result.
  loom_testbench_compile_result_callback_t result_callback;
  // Optional compile report request.
  loomc_compile_report_options_t compile_report;
  // Optional artifact manifest request.
  loomc_artifact_manifest_options_t artifact_manifest;
  // Target-side artifact classes requested by the caller.
  loomc_emit_artifact_flags_t emit_artifact_flags;
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

// Testbench invocation callback for HAL kernel and finite pipeline dispatches.
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

// Materializes one invocation's geometry and HAL bindings from an
// already-materialized case sample value table.
iree_status_t loom_run_hal_testbench_materialize_invocation_from_table(
    const loom_testbench_value_table_t* table,
    loom_run_hal_testbench_actual_provider_t* provider,
    iree_allocator_t allocator, loom_run_hal_invocation_options_t* out_options,
    loom_run_hal_binding_list_t* out_bindings);

// Materializes one case sample's invocation as geometry and HAL bindings.
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

// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// HAL actual-invocation bridge for Loom check testbench execution.
//
// This layer is target-neutral: tools inject a composed target environment and
// linked HAL artifact providers, while this bridge owns HAL runtime selection,
// candidate compilation, dispatch input conversion, and the callback shape used
// by the testbench executor.

#ifndef LOOM_TOOLING_EXECUTION_HAL_TESTBENCH_ACTUAL_H_
#define LOOM_TOOLING_EXECUTION_HAL_TESTBENCH_ACTUAL_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "loom/pass/types.h"
#include "loom/sanitizer/options.h"
#include "loom/target/provider.h"
#include "loom/tooling/execution/compile_options.h"
#include "loom/tooling/execution/hal/artifact.h"
#include "loom/tooling/execution/hal/candidate.h"
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
  // Linked artifact-provider registry selected by the tool binary.
  const loom_run_hal_artifact_provider_registry_t* artifact_provider_registry;
  // Host allocator used for runtime and candidate storage.
  iree_allocator_t host_allocator;
  // Device event sink used when initializing |runtime|.
  iree_hal_device_event_sink_t device_event_sink;
  // Sanitizer options used when deriving HAL runtime requirements.
  loom_sanitizer_options_t runtime_sanitizer_options;
  // Selected HAL artifact provider for the active device.
  const loom_run_hal_artifact_provider_t* artifact_provider;
  // Shared HAL runtime used by actual invocations.
  loom_run_hal_runtime_t runtime;
  // True when |runtime_sanitizer_options| has been set by the tool.
  bool has_runtime_sanitizer_options;
  // True when |runtime| owns initialized HAL state.
  bool runtime_initialized;
} loom_run_hal_testbench_context_t;

// Initializes a HAL testbench context with a linked artifact-provider registry.
void loom_run_hal_testbench_context_initialize(
    const loom_run_hal_artifact_provider_registry_t* artifact_provider_registry,
    iree_allocator_t host_allocator,
    loom_run_hal_testbench_context_t* out_context);

// Sets the device event sink used by future HAL runtime initialization.
void loom_run_hal_testbench_context_set_device_event_sink(
    loom_run_hal_testbench_context_t* context,
    iree_hal_device_event_sink_t device_event_sink);

// Sets the structured sanitizer policy used by future HAL runtime creation.
void loom_run_hal_testbench_context_set_runtime_sanitizer_options(
    loom_run_hal_testbench_context_t* context,
    const loom_sanitizer_options_t* sanitizer_options);

// Releases HAL runtime resources owned by |context|.
void loom_run_hal_testbench_context_deinitialize(
    loom_run_hal_testbench_context_t* context);

// Selects a linked artifact provider and initializes the HAL runtime on demand.
iree_status_t loom_run_hal_testbench_context_ensure_runtime(
    loom_run_hal_testbench_context_t* context);

// Returns the driver component of an IREE --device= URI.
iree_string_view_t loom_run_hal_testbench_device_uri_driver_name(
    iree_string_view_t device_uri);

// Returns host-visible buffer parameters suitable for correctness execution.
iree_hal_buffer_params_t loom_run_hal_testbench_host_visible_buffer_params(
    void);

// Finds the single semantic actual invocation in |case_plan| accepted by the
// HAL bridge. HAL kernels currently model outputs as in-place buffer arguments.
iree_status_t loom_run_hal_testbench_select_actual_invocation(
    const loom_testbench_case_plan_t* case_plan,
    const loom_testbench_invocation_plan_t** out_invocation);

// Counts actual invocations in |case_plan| and validates that each invocation
// uses in-place HAL buffer outputs.
iree_status_t loom_run_hal_testbench_count_actual_invocations(
    const loom_testbench_case_plan_t* case_plan,
    iree_host_size_t* out_actual_invocation_count);

typedef struct loom_run_hal_testbench_actual_provider_options_t {
  // Shared HAL context used to prepare and dispatch the candidate.
  loom_run_hal_testbench_context_t* context;
  // Execution session used to parse the private compile copy.
  loom_run_session_t* session;
  // Target environment used by the source-to-low pipeline.
  const loom_target_environment_t* target_environment;
  // Source filename used for diagnostics.
  iree_string_view_t filename;
  // Source text used to parse a private compile copy.
  iree_string_view_t source;
  // User-selected pass pipeline.
  iree_string_view_t pipeline;
  // Sanitizer checks inserted by the target pipeline.
  loom_sanitizer_options_t sanitizer;
  // Config bindings materialized into the private compile copy.
  const loom_tooling_config_set_t* config_set;
  // Module that owns |actual_invocation|.
  const loom_module_t* test_module;
  // Actual invocation selected from the owning check.case.
  const loom_testbench_invocation_plan_t* actual_invocation;
  // Optional case plan providing compile-time literals and sample parameters.
  const loom_testbench_case_plan_t* case_plan;
  // Case sample ordinal used when |has_sample_constant_ordinal| is true.
  iree_host_size_t sample_constant_ordinal;
  // True when sample parameter values become compile-time constants.
  bool has_sample_constant_ordinal;
  // Diagnostic sink used while parsing, lowering, and emitting the candidate.
  loom_diagnostic_sink_t diagnostic_sink;
  // Maximum diagnostics to emit before stopping. Zero uses the default.
  uint32_t max_errors;
  // Optional caller-owned structured compile report to populate.
  loom_target_compile_report_t* report;
  // Optional debug artifacts requested from the selected backend.
  loom_run_candidate_artifact_flags_t artifact_flags;
  // Optional artifact manifest requested from the selected backend.
  loom_run_candidate_artifact_manifest_options_t artifact_manifest;
} loom_run_hal_testbench_actual_provider_options_t;

typedef struct loom_run_hal_testbench_actual_provider_t {
  // Shared HAL context used to prepare and dispatch the candidate.
  loom_run_hal_testbench_context_t* context;
  // Execution session used to parse the private compile copy.
  loom_run_session_t* session;
  // Target environment used by the source-to-low pipeline.
  const loom_target_environment_t* target_environment;
  // Source filename used for diagnostics.
  iree_string_view_t filename;
  // Source text used to parse a private compile copy.
  iree_string_view_t source;
  // User-selected pass pipeline.
  iree_string_view_t pipeline;
  // Sanitizer checks inserted by the target pipeline.
  loom_sanitizer_options_t sanitizer;
  // Config bindings materialized into the private compile copy.
  const loom_tooling_config_set_t* config_set;
  // Module that owns |actual_invocation|.
  const loom_module_t* test_module;
  // Actual invocation selected from the owning check.case.
  const loom_testbench_invocation_plan_t* actual_invocation;
  // Optional case plan providing compile-time literals and sample parameters.
  const loom_testbench_case_plan_t* case_plan;
  // Case sample ordinal used when |has_sample_constant_ordinal| is true.
  iree_host_size_t sample_constant_ordinal;
  // True when sample parameter values become compile-time constants.
  bool has_sample_constant_ordinal;
  // Diagnostic sink used while parsing, lowering, and emitting the candidate.
  loom_diagnostic_sink_t diagnostic_sink;
  // Maximum diagnostics to emit before stopping. Zero uses the default.
  uint32_t max_errors;
  // Optional caller-owned structured compile report to populate.
  loom_target_compile_report_t* report;
  // Optional debug artifacts requested from the selected backend.
  loom_run_candidate_artifact_flags_t artifact_flags;
  // Optional artifact manifest requested from the selected backend.
  loom_run_candidate_artifact_manifest_options_t artifact_manifest;
  // Parsed compile module owned by this provider.
  loom_run_module_t compile_module;
  // Backend-produced HAL executable candidate.
  loom_run_hal_candidate_t candidate;
  // Target selected before the compile pipeline runs.
  loom_run_hal_device_target_t compile_device_target;
  // Prepared executable retained for correctness and benchmark dispatches.
  loom_run_hal_prepared_candidate_t prepared_candidate;
  // Dispatch options derived from the compiled source entry.
  loom_run_hal_invocation_options_t invocation_options;
  // Pass diagnostic counts from the Loom compile pipeline.
  loom_pass_run_result_t pass_result;
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
  // True when |candidate| has been initialized.
  bool candidate_initialized;
  // True when |compile_device_target| owns provider-selected target storage.
  bool compile_device_target_initialized;
  // True when |prepared_candidate| has been initialized.
  bool prepared_candidate_initialized;
  // True when HAL candidate emission populated the caller's compile report.
  bool compile_report_available;
  // Number of function-like region arguments replaced by selected sample
  // constants.
  iree_host_size_t sample_constant_argument_count;
} loom_run_hal_testbench_actual_provider_t;

typedef struct loom_run_hal_testbench_actual_sequence_options_t {
  // Shared HAL context used to prepare and dispatch all actual candidates.
  loom_run_hal_testbench_context_t* context;
  // Execution session used to parse each private compile copy.
  loom_run_session_t* session;
  // Target environment used by the source-to-low pipeline.
  const loom_target_environment_t* target_environment;
  // Source filename used for diagnostics.
  iree_string_view_t filename;
  // Source text used to parse each private compile copy.
  iree_string_view_t source;
  // User-selected pass pipeline.
  iree_string_view_t pipeline;
  // Sanitizer checks inserted by the target pipeline.
  loom_sanitizer_options_t sanitizer;
  // Config bindings materialized into each private compile copy.
  const loom_tooling_config_set_t* config_set;
  // Module that owns |case_plan|.
  const loom_module_t* test_module;
  // Case plan whose actual invocations are executed by the sequence.
  const loom_testbench_case_plan_t* case_plan;
  // Case sample ordinal used when |has_sample_constant_ordinal| is true.
  iree_host_size_t sample_constant_ordinal;
  // True when sample parameter values become compile-time constants.
  bool has_sample_constant_ordinal;
  // Diagnostic sink used while parsing, lowering, and emitting candidates.
  loom_diagnostic_sink_t diagnostic_sink;
  // Maximum diagnostics to emit before stopping. Zero uses the default.
  uint32_t max_errors;
  // Optional debug artifacts requested from the selected backend.
  loom_run_candidate_artifact_flags_t artifact_flags;
  // Optional artifact manifest requested from the selected backend.
  loom_run_candidate_artifact_manifest_options_t artifact_manifest;
} loom_run_hal_testbench_actual_sequence_options_t;

typedef struct loom_run_hal_testbench_actual_sequence_t {
  // Host allocator used for sequence-owned provider storage.
  iree_allocator_t host_allocator;
  // Sequence-owned providers in check.case source order.
  loom_run_hal_testbench_actual_provider_t* providers;
  // Number of entries in |providers|.
  iree_host_size_t provider_count;
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

// Testbench invocation callback for HAL actual invocations.
iree_status_t loom_run_hal_testbench_actual_invoke(
    void* user_data, const loom_testbench_invocation_plan_t* invocation,
    iree_host_size_t input_count, const loom_testbench_value_t* inputs,
    iree_host_size_t result_count, loom_testbench_value_t* out_results);

// Initializes a compile-on-first-use provider sequence for every actual
// invocation in a check.case.
iree_status_t loom_run_hal_testbench_actual_sequence_initialize(
    const loom_run_hal_testbench_actual_sequence_options_t* options,
    loom_run_hal_testbench_actual_sequence_t* out_sequence);

// Releases storage owned by |sequence|.
void loom_run_hal_testbench_actual_sequence_deinitialize(
    loom_run_hal_testbench_actual_sequence_t* sequence);

// Testbench invocation callback for HAL actual invocation sequences.
iree_status_t loom_run_hal_testbench_actual_sequence_invoke(
    void* user_data, const loom_testbench_invocation_plan_t* invocation,
    iree_host_size_t input_count, const loom_testbench_value_t* inputs,
    iree_host_size_t result_count, loom_testbench_value_t* out_results);

// Reports a compile rejection recorded by the provider for |invocation|, if
// any.
iree_status_t loom_run_hal_testbench_actual_sequence_query_issue(
    void* user_data, const loom_testbench_invocation_plan_t* invocation,
    loom_testbench_sample_issue_t* out_issue);

// Appends borrowed testbench input values to HAL bindings/constants.
//
// Scalar constants are packed according to the corresponding Loom source input
// type, not only the scalar carrier type. This matters for index/offset values,
// which have target ABI widths independent of the materialized scalar storage.
iree_status_t loom_run_hal_testbench_invocation_inputs_from_values(
    const loom_testbench_value_t* inputs, const loom_type_t* input_types,
    iree_host_size_t input_count, loom_run_hal_invocation_options_t* options,
    iree_allocator_t allocator, loom_run_hal_binding_list_t* out_bindings);

// Extracts the selected actual invocation inputs from an already-materialized
// case sample value table as HAL bindings/constants.
iree_status_t loom_run_hal_testbench_create_invocation_inputs_from_table(
    const loom_testbench_value_table_t* table,
    const loom_testbench_invocation_plan_t* invocation,
    const loom_run_hal_invocation_options_t* base_options,
    iree_allocator_t allocator, loom_run_hal_invocation_options_t* out_options,
    loom_run_hal_binding_list_t* out_bindings);

// Materializes one case sample and extracts the selected actual invocation
// inputs as HAL bindings/constants.
iree_status_t loom_run_hal_testbench_create_invocation_inputs_for_sample(
    const loom_module_t* module,
    const loom_testbench_value_materializer_options_t* materializer_options,
    const loom_testbench_case_plan_t* case_plan,
    const loom_testbench_invocation_plan_t* invocation,
    iree_host_size_t sample_ordinal,
    const loom_run_hal_invocation_options_t* base_options,
    iree_allocator_t allocator, loom_run_hal_invocation_options_t* out_options,
    loom_run_hal_binding_list_t* out_bindings);

// Prepares a reusable HAL invocation plan for one testbench sample.
iree_status_t loom_run_hal_testbench_prepare_invocation_plan_for_sample(
    const loom_testbench_module_plan_t* module_plan,
    const loom_testbench_case_plan_t* case_plan,
    const loom_testbench_value_materializer_options_t* materializer_options,
    const loom_testbench_invocation_plan_t* invocation,
    const loom_run_hal_invocation_options_t* base_options,
    iree_host_size_t sample_ordinal, iree_allocator_t allocator,
    loom_run_hal_invocation_plan_t* out_plan);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_EXECUTION_HAL_TESTBENCH_ACTUAL_H_

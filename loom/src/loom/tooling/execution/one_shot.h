// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// One-shot execution request/result adapters for CLI and lit-style tools.

#ifndef LOOM_TOOLING_EXECUTION_ONE_SHOT_H_
#define LOOM_TOOLING_EXECUTION_ONE_SHOT_H_

#include "iree/base/api.h"
#include "loom/ir/module.h"
#include "loom/tooling/execution/compile_options.h"
#include "loom/tooling/execution/session.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_run_compile_report_capture_t
    loom_run_compile_report_capture_t;

enum {
  // Maximum number of HAL dispatch constants accepted by the one-shot front
  // door.
  LOOM_RUN_ONE_SHOT_HAL_MAX_CONSTANT_COUNT = 64,
  // Maximum number of HAL dispatch bindings accepted by the one-shot front
  // door.
  LOOM_RUN_ONE_SHOT_HAL_MAX_BINDING_COUNT = 64,
};

typedef struct loom_run_one_shot_value_specs_t {
  // Textual value specs in IREE function I/O syntax.
  const iree_string_view_t* values;
  // Number of entries in |values|.
  iree_host_size_t count;
} loom_run_one_shot_value_specs_t;

typedef struct loom_run_one_shot_binding_specs_t {
  // Textual binding specs in HAL binding ordinal order.
  const iree_string_view_t* values;
  // Number of entries in |values|.
  iree_host_size_t count;
} loom_run_one_shot_binding_specs_t;

// Invocation options for the current one-shot run front door.
//
// This intentionally preserves the existing textual VM/HAL input specs used by
// CLI and loom-check adapters without depending on the concrete VM or HAL
// invocation headers. Benchmark/tune hot loops should use typed invocation
// plans instead of this structure.
typedef struct loom_run_one_shot_options_t {
  // VM function name to invoke. Empty selects the single export.
  iree_string_view_t vm_function_name;
  // VM function input specs.
  loom_run_one_shot_value_specs_t vm_inputs;
  // VM function output materialization specs.
  loom_run_one_shot_value_specs_t vm_outputs;
  // VM expected output specs. When present, these take precedence over outputs.
  loom_run_one_shot_value_specs_t vm_expected_outputs;
  // Maximum number of VM output elements to format.
  iree_host_size_t vm_max_output_element_count;
  // HAL executable function symbol to dispatch. Empty selects the only named
  // function in the prepared executable.
  iree_string_view_t hal_function_name;
  // HAL dispatch workgroup count in x/y/z order.
  uint32_t hal_workgroup_count[3];
  // HAL dispatch constants in ABI order.
  uint32_t hal_constants[LOOM_RUN_ONE_SHOT_HAL_MAX_CONSTANT_COUNT];
  // Number of entries in |hal_constants|.
  iree_host_size_t hal_constant_count;
  // HAL dispatch binding specs.
  loom_run_one_shot_binding_specs_t hal_bindings;
  // Optional HAL binding specs compared after dispatch.
  loom_run_one_shot_binding_specs_t hal_expected_bindings;
  // Optional path receiving the target-native HAL artifact, such as HSACO.
  iree_string_view_t hal_target_artifact_output_path;
  // Optional path receiving the executable artifact passed to the HAL loader.
  iree_string_view_t hal_executable_output_path;
  // Stops after HAL executable emission without dispatching.
  bool hal_emit_only;
  // Maximum number of HAL output elements to format.
  iree_host_size_t hal_max_output_element_count;
} loom_run_one_shot_options_t;

typedef struct loom_run_one_shot_result_t {
  // Human-readable output or comparison diagnostics.
  iree_string_builder_t output;
  // Process-style exit code: zero for success, non-zero for mismatches.
  int exit_code;
} loom_run_one_shot_result_t;

struct loom_run_one_shot_probe_request_t {
  // Host allocator for transient probe allocations.
  iree_allocator_t host_allocator;
  // Result receiving human-readable probe output.
  loom_run_one_shot_result_t* result;
};

struct loom_run_one_shot_request_t {
  // Parsed module to compile and invoke.
  loom_run_module_t* run_module;
  // Candidate compile options shared by backends.
  const loom_run_candidate_compile_options_t* compile_options;
  // Invocation options selected by the front door.
  const loom_run_one_shot_options_t* options;
  // Optional compile report capture emitted before backend candidate teardown.
  loom_run_compile_report_capture_t* compile_report_capture;
  // Host allocator for transient execution allocations.
  iree_allocator_t host_allocator;
  // Result receiving human-readable output and exit status.
  loom_run_one_shot_result_t* result;
};

// Initializes one-shot options to match the existing iree-run-loom defaults.
void loom_run_one_shot_options_initialize(
    loom_run_one_shot_options_t* out_options);

// Applies the static dispatch workgroup count from a source kernel launch
// config when |function_name| names a kernel, or when the module has exactly
// one kernel and |function_name| is empty. Returns false when no unique source
// kernel with a fully static workgroup count is available.
bool loom_run_one_shot_options_apply_static_hal_workgroup_count(
    const loom_module_t* module, iree_string_view_t function_name,
    loom_run_one_shot_options_t* options);

// Initializes a one-shot result. Must be paired with
// loom_run_one_shot_result_deinitialize().
void loom_run_one_shot_result_initialize(
    iree_allocator_t allocator, loom_run_one_shot_result_t* out_result);

// Releases storage owned by |result|.
void loom_run_one_shot_result_deinitialize(loom_run_one_shot_result_t* result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_EXECUTION_ONE_SHOT_H_

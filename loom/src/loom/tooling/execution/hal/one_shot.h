// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// One-shot HAL compilation and execution for CLI and lit-style tools.

#ifndef LOOM_TOOLING_EXECUTION_HAL_ONE_SHOT_H_
#define LOOM_TOOLING_EXECUTION_HAL_ONE_SHOT_H_

#include "iree/base/api.h"
#include "loom/ir/module.h"
#include "loom/tooling/compile/options.h"
#include "loom/tooling/compile/request.h"
#include "loom/tooling/execution/hal/device_provider.h"
#include "loom/tooling/execution/session.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_compile_report_capture_t loom_compile_report_capture_t;
typedef struct loom_target_environment_t loom_target_environment_t;

enum {
  // Maximum number of dispatch constants accepted by the one-shot front door.
  LOOM_RUN_HAL_ONE_SHOT_MAX_CONSTANT_COUNT = 64,
  // Maximum number of dispatch bindings accepted by the one-shot front door.
  LOOM_RUN_HAL_ONE_SHOT_MAX_BINDING_COUNT = 64,
};

typedef struct loom_run_hal_one_shot_binding_specs_t {
  // Textual binding specs in HAL binding ordinal order.
  const iree_string_view_t* values;
  // Number of entries in |values|.
  iree_host_size_t count;
} loom_run_hal_one_shot_binding_specs_t;

// Invocation options for the one-shot HAL run front door.
//
// Textual input specs remain at the CLI boundary. Benchmark and tuning hot
// loops use typed invocation plans instead.
typedef struct loom_run_hal_one_shot_options_t {
  // Executable function symbol to dispatch. Empty selects the only function.
  iree_string_view_t function_name;
  // Dispatch workgroup count in x/y/z order.
  uint32_t workgroup_count[3];
  // Dispatch constants in ABI order.
  uint32_t constants[LOOM_RUN_HAL_ONE_SHOT_MAX_CONSTANT_COUNT];
  // Number of entries in |constants|.
  iree_host_size_t constant_count;
  // Dispatch binding specs in binding ordinal order.
  loom_run_hal_one_shot_binding_specs_t bindings;
  // Optional binding specs compared after dispatch.
  loom_run_hal_one_shot_binding_specs_t expected_bindings;
  // Optional path receiving the target-native artifact.
  iree_string_view_t target_artifact_output_path;
  // Optional path receiving the executable artifact passed to the HAL loader.
  iree_string_view_t executable_output_path;
  // Stops after executable emission without dispatching.
  bool emit_only;
  // Maximum number of output elements to format.
  iree_host_size_t max_output_element_count;
} loom_run_hal_one_shot_options_t;

typedef struct loom_run_hal_one_shot_result_t {
  // Human-readable output or comparison diagnostics.
  iree_string_builder_t output;
  // Process-style exit code: zero for success, non-zero for mismatches.
  int exit_code;
} loom_run_hal_one_shot_result_t;

typedef struct loom_run_hal_one_shot_probe_request_t {
  // Configured target environment used to resolve an explicit target.
  const loom_target_environment_t* target_environment;
  // Optional family-qualified target profile to probe.
  iree_string_view_t target;
  // Host allocator for transient probe allocations.
  iree_allocator_t host_allocator;
  // Result receiving human-readable probe output.
  loom_run_hal_one_shot_result_t* result;
} loom_run_hal_one_shot_probe_request_t;

typedef struct loom_run_hal_one_shot_request_t {
  // Execution session owning compiler arenas and descriptor registries.
  loom_run_session_t* session;
  // Parsed module to compile and invoke.
  loom_run_module_t* run_module;
  // Configured target environment used for specialization and lowering.
  const loom_target_environment_t* target_environment;
  // Shared product, format, and target constraints for this kernel compile.
  loom_compile_request_options_t compile_request_options;
  // Pass pipeline selected by the command line.
  iree_string_view_t pipeline;
  // Candidate compile options.
  const loom_compile_options_t* compile_options;
  // Invocation options selected by the front door.
  const loom_run_hal_one_shot_options_t* options;
  // Optional compile report capture emitted before candidate teardown.
  loom_compile_report_capture_t* compile_report_capture;
  // Host allocator for transient execution allocations.
  iree_allocator_t host_allocator;
  // Result receiving human-readable output and exit status.
  loom_run_hal_one_shot_result_t* result;
} loom_run_hal_one_shot_request_t;

// Initializes one-shot options to the iree-run-loom defaults.
void loom_run_hal_one_shot_options_initialize(
    loom_run_hal_one_shot_options_t* out_options);

// Applies a static dispatch workgroup count from a source kernel. Returns
// false when no unique source kernel with a fully static count is available.
bool loom_run_hal_one_shot_options_apply_static_workgroup_count(
    const loom_module_t* module, iree_string_view_t function_name,
    loom_run_hal_one_shot_options_t* options);

// Initializes a one-shot result. Must be paired with deinitialize.
void loom_run_hal_one_shot_result_initialize(
    iree_allocator_t allocator, loom_run_hal_one_shot_result_t* out_result);

// Releases storage owned by |result|.
void loom_run_hal_one_shot_result_deinitialize(
    loom_run_hal_one_shot_result_t* result);

// Probes |device_provider|'s requested or preferred live target.
iree_status_t loom_run_hal_one_shot_probe(
    const loom_device_provider_t* device_provider,
    const loom_run_hal_one_shot_probe_request_t* request);

// Compiles and invokes one executable through |device_provider|.
iree_status_t loom_run_hal_one_shot_run(
    const loom_device_provider_t* device_provider,
    const loom_run_hal_one_shot_request_t* request);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_EXECUTION_HAL_ONE_SHOT_H_

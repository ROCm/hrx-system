// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Binding of compiled semantic-function providers to testbench plans.

#ifndef LOOM_TOOLING_TESTBENCH_COMPILED_PROVIDER_H_
#define LOOM_TOOLING_TESTBENCH_COMPILED_PROVIDER_H_

#include "iree/base/api.h"
#include "loom/tooling/testbench/invocation.h"
#include "loomc/compile.h"
#include "loomc/config.h"
#include "loomc/module.h"
#include "loomc/result.h"
#include "loomc/workspace.h"

#ifdef __cplusplus
extern "C" {
#endif

// Immutable source and reusable public compiler or emission services for one
// testbench invocation. Source execution providers clone |module| before
// compilation. Calls sharing |workspace| are serialized by the owning runner.
typedef struct loom_testbench_compilation_t {
  // Context owning the compiler, module, and dialect registry.
  loomc_context_t* context;
  // Prepared source compiler, or NULL when only emitting a prepared module.
  loomc_compiler_t* compiler;
  // Invocation workspace reused after each private clone is released.
  loomc_workspace_t* workspace;
  // Canonical immutable source module retained by the runner.
  const loomc_module_t* module;
  // Borrowed compile-time configuration applied to private clones.
  const loomc_config_options_t* config;
} loom_testbench_compilation_t;

// Observes one complete public compiler result before it is released.
typedef iree_status_t (*loom_testbench_compile_result_fn_t)(
    void* user_data, const loomc_result_t* result);

typedef struct loom_testbench_compile_result_callback_t {
  // Result observer, or NULL when diagnostics need no external presentation.
  loom_testbench_compile_result_fn_t fn;
  // Caller-owned state passed to |fn|.
  void* user_data;
} loom_testbench_compile_result_callback_t;

// Delivers |result| to |callback| when an observer is present.
static inline iree_status_t loom_testbench_observe_compile_result(
    loom_testbench_compile_result_callback_t callback,
    const loomc_result_t* result) {
  return callback.fn ? callback.fn(callback.user_data, result)
                     : iree_ok_status();
}

// Binds one borrowed function-call provider to the runner's selected calls.
// The caller owns callback state. Compilation inputs and invocation plans
// remain live through the final call. Providers compile private module clones
// and report each complete public compiler result through |result_callback|.
typedef loom_testbench_invocation_provider_t(
    IREE_API_PTR* loom_testbench_function_call_provider_fn_t)(
    void* user_data, const loom_testbench_compilation_t* compilation,
    loom_testbench_invocation_plan_list_t invocations,
    loom_testbench_compile_result_callback_t result_callback);

typedef struct loom_testbench_function_call_provider_callback_t {
  // Binding callback, or NULL when no function executor is linked.
  loom_testbench_function_call_provider_fn_t fn;
  // Caller-owned executor state passed to |fn|.
  void* user_data;
} loom_testbench_function_call_provider_callback_t;

typedef struct loom_testbench_execution_profile_t
    loom_testbench_execution_profile_t;
typedef struct loom_source_table_resolver_t loom_source_table_resolver_t;
typedef struct loom_diagnostic_sink_t loom_diagnostic_sink_t;

// Binds one externally selected scenario execution profile to compiler inputs.
// The caller owns profile state through the last prepared product's teardown.
typedef loom_testbench_execution_profile_t (
    *loom_testbench_bind_scenario_profile_fn_t)(
    void* user_data, const loom_testbench_compilation_t* compilation,
    const loom_source_table_resolver_t* sources,
    loom_diagnostic_sink_t diagnostic_sink,
    loom_testbench_compile_result_callback_t result_callback);

typedef struct loom_testbench_bind_scenario_profile_callback_t {
  // Profile binding callback, or NULL when the profile is unavailable.
  loom_testbench_bind_scenario_profile_fn_t fn;
  // Caller-owned profile state passed to |fn|.
  void* user_data;
} loom_testbench_bind_scenario_profile_callback_t;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_TESTBENCH_COMPILED_PROVIDER_H_

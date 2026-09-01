// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared iree-test-loom command-line implementation.

#ifndef LOOM_TOOLS_IREE_TEST_LOOM_MAIN_H_
#define LOOM_TOOLS_IREE_TEST_LOOM_MAIN_H_

#include "iree/base/api.h"
#include "loom/target/provider.h"
#include "loom/tooling/execution/hal/device_provider.h"
#include "loom/tooling/execution/session.h"
#include "loom/tooling/testbench/invocation.h"
#include "loom/tooling/testbench/requirements.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_run_hal_testbench_context_t
    loom_run_hal_testbench_context_t;
typedef struct loom_tooling_config_set_t loom_tooling_config_set_t;

// Prepares one module-lifetime function-call provider shared by all cases.
//
// The callback may compile or otherwise materialize the module once. On
// success, every prepared case borrows |out_provider| until the matching
// deinitialize callback. A module without semantic function calls does not
// invoke this callback.
typedef iree_status_t (*iree_test_loom_prepare_function_call_provider_fn_t)(
    void* user_data, loom_run_session_t* session,
    const loom_target_environment_t* target_environment,
    const loom_run_module_t* run_module,
    const loom_testbench_module_plan_t* module_plan,
    iree_string_view_t pipeline, const loom_tooling_config_set_t* config_set,
    iree_allocator_t host_allocator,
    loom_testbench_invocation_provider_t* out_provider);

// Releases state produced by successful function-call provider preparation.
typedef void (*iree_test_loom_deinitialize_function_call_provider_fn_t)(
    void* user_data);

typedef struct iree_test_loom_function_call_provider_callback_t {
  // Prepares the provider after parsing and planning the complete module.
  iree_test_loom_prepare_function_call_provider_fn_t prepare;
  // Releases provider state before the parsed module and session are released.
  iree_test_loom_deinitialize_function_call_provider_fn_t deinitialize;
  // Caller-owned state passed to both callbacks.
  void* user_data;
} iree_test_loom_function_call_provider_callback_t;

// Appends target-linked requirement providers to |providers|.
typedef iree_status_t (*iree_test_loom_populate_requirement_providers_fn_t)(
    void* user_data, loom_run_hal_testbench_context_t* hal_context,
    iree_host_size_t provider_capacity,
    loom_testbench_requirement_provider_t* providers,
    iree_host_size_t* inout_provider_count);

typedef struct iree_test_loom_populate_requirement_providers_callback_t {
  // Callback implementation, or NULL when no extra providers are linked.
  iree_test_loom_populate_requirement_providers_fn_t fn;
  // Opaque callback state passed to |fn|.
  void* user_data;
} iree_test_loom_populate_requirement_providers_callback_t;

typedef struct iree_test_loom_configuration_t {
  // Null-terminated executable name used in help and diagnostics.
  const char* tool_name;
  // Registers target-specific dialects and attrs linked into this runner.
  loom_run_register_context_callback_t register_context;
  // Target environment composed from linked execution providers.
  const loom_target_environment_t* target_environment;
  // Linked device providers available to kernel launches.
  const loom_device_provider_registry_t* device_provider_registry;
  // Appends target-specific requirement providers linked into this runner.
  iree_test_loom_populate_requirement_providers_callback_t
      populate_requirement_providers;
  // Prepares the linked provider used for semantic function calls.
  iree_test_loom_function_call_provider_callback_t function_call_provider;
  // Target-low descriptor registry package linked into this runner.
  loom_run_initialize_low_descriptor_registry_callback_t
      initialize_low_descriptor_registry;
} iree_test_loom_configuration_t;

// Runs the configured iree-test-loom command-line tool.
int iree_test_loom_main(int argc, char** argv,
                        const iree_test_loom_configuration_t* configuration);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLS_IREE_TEST_LOOM_MAIN_H_

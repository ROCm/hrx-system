// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Test-only loom-check providers that inspect exact-version native IR.

#ifndef LOOM_TOOLS_LOOM_CHECK_TEST_NATIVE_PROVIDER_H_
#define LOOM_TOOLS_LOOM_CHECK_TEST_NATIVE_PROVIDER_H_

#include "loom/error/emitter.h"
#include "loom/error/source.h"
#include "loom/ir/module.h"
#include "loom/target/low_descriptor_registry.h"
#include "loom/tools/loom-check/execute.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_target_environment_t loom_target_environment_t;
typedef struct loom_check_test_native_diagnostic_context_t
    loom_check_test_native_diagnostic_context_t;

// Mutable exact-version module projection supplied only to checker tests.
typedef struct loom_check_test_native_module_t {
  // Mutable module owned by the emit request's public module.
  loom_module_t* module;
  // Resolver for source-backed operation locations in |module|.
  loom_source_resolver_t source_resolver;
  // Native target environment borrowed from the public compiler session.
  const loom_target_environment_t* target_environment;
  // Target-low descriptors composed by the runner's public environment.
  loom_target_low_descriptor_registry_t low_registry;
  // Native diagnostic projection active for this provider invocation.
  loom_check_test_native_diagnostic_context_t* diagnostic_context;
} loom_check_test_native_module_t;

// Materializes native diagnostic emissions into the generic checker result.
typedef struct loom_check_test_native_diagnostic_emitter_capture_t {
  // Active native diagnostic context owned by the provider adapter.
  loom_check_test_native_diagnostic_context_t* context;
  // Module containing any operation referenced by emitted diagnostics.
  const loom_module_t* module;
  // Resolver for source-backed operation locations in |module|.
  loom_source_resolver_t source_resolver;
  // Subsystem identity to store in materialized diagnostics.
  loom_emitter_t emitter;
  // Number of diagnostics materialized through this capture.
  iree_host_size_t emission_count;
} loom_check_test_native_diagnostic_emitter_capture_t;

// Diagnostic emitter callback for a
// loom_check_test_native_diagnostic_emitter_capture_t user_data pointer.
iree_status_t loom_check_test_native_diagnostic_emitter_capture_emit(
    void* user_data, const loom_diagnostic_emission_t* emission);

// Native implementation invoked with one verified mutable module projection.
typedef iree_status_t (*loom_check_test_native_emit_provider_execute_fn_t)(
    const loom_check_emit_provider_t* provider,
    const loom_check_emit_provider_request_t* request,
    const loom_check_test_native_module_t* native_module);

// Test-only provider pairing ordinary emit dispatch with native inspection.
// |base| must remain the first member so registries can borrow it directly.
typedef struct loom_check_test_native_emit_provider_t {
  // Ordinary provider registered with the checker test runner.
  loom_check_emit_provider_t base;
  // Exact-version native implementation invoked by |base.execute|.
  loom_check_test_native_emit_provider_execute_fn_t execute_native;
} loom_check_test_native_emit_provider_t;

// Projects, invokes, and re-verifies the admitted public module for a
// loom_check_test_native_emit_provider_t registered through its |base| member.
iree_status_t loom_check_test_native_emit_provider_execute(
    const loom_check_emit_provider_t* provider,
    const loom_check_emit_provider_request_t* request);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLS_LOOM_CHECK_TEST_NATIVE_PROVIDER_H_

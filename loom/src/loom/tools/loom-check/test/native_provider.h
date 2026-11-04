// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Test-only loom-check providers that inspect exact-version native IR.

#ifndef LOOM_TOOLS_LOOM_CHECK_TEST_NATIVE_PROVIDER_H_
#define LOOM_TOOLS_LOOM_CHECK_TEST_NATIVE_PROVIDER_H_

#include "loom/tools/loom-check/execute.h"

#ifdef __cplusplus
extern "C" {
#endif

// Native implementation invoked with one verified mutable module projection.
typedef iree_status_t (*loom_check_test_native_emit_provider_execute_fn_t)(
    const loom_check_emit_provider_t* provider,
    const loom_check_emit_provider_request_t* request,
    const loom_check_emit_native_module_t* native_module);

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

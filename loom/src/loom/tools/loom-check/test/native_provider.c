// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/loom-check/test/native_provider.h"

#include "loom/tools/loom-check/compile.h"

typedef struct loom_check_test_native_emit_context_t {
  // Test-only native provider selected by ordinary checker dispatch.
  const loom_check_test_native_emit_provider_t* provider;
  // Common request owning the admitted public module.
  const loom_check_emit_provider_request_t* request;
} loom_check_test_native_emit_context_t;

static iree_status_t loom_check_test_native_emit_consume(
    void* user_data, const loom_check_emit_native_module_t* native_module) {
  const loom_check_test_native_emit_context_t* context =
      (const loom_check_test_native_emit_context_t*)user_data;
  return context->provider->execute_native(&context->provider->base,
                                           context->request, native_module);
}

iree_status_t loom_check_test_native_emit_provider_execute(
    const loom_check_emit_provider_t* provider,
    const loom_check_emit_provider_request_t* request) {
  const loom_check_test_native_emit_provider_t* native_provider =
      (const loom_check_test_native_emit_provider_t*)provider;
  const loom_check_test_native_emit_context_t context = {
      .provider = native_provider,
      .request = request,
  };
  return loom_check_compile_with_native_module(
      request, loom_check_test_native_emit_consume, (void*)&context);
}

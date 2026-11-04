// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/loom-check/test/native_provider.h"

#include "loom/tools/loom-check/compile_diagnostics.h"
#include "loom/tools/loom-check/diagnostics.h"
#include "loomc/interop.h"
#include "loomc/iree.h"

static iree_status_t loom_check_test_native_project_module(
    const loom_check_emit_provider_request_t* request, bool* out_projected) {
  loomc_module_interop_view_t verified_view = {0};
  loomc_result_t* result = NULL;
  iree_status_t status = iree_status_from_loomc(loomc_module_get_interop_view(
      request->public_module,
      loomc_allocator_from_iree(request->host_allocator), &verified_view,
      &result));
  if (iree_status_is_ok(status)) {
    status = loom_check_compile_append_result(request->diagnostic_collector,
                                              result, out_projected);
  }
  loomc_result_release(result);
  return status;
}

iree_status_t loom_check_test_native_emit_provider_execute(
    const loom_check_emit_provider_t* provider,
    const loom_check_emit_provider_request_t* request) {
  const loom_check_test_native_emit_provider_t* native_provider =
      (const loom_check_test_native_emit_provider_t*)provider;
  bool projected = false;
  iree_status_t status =
      loom_check_test_native_project_module(request, &projected);
  if (!iree_status_is_ok(status) || !projected) {
    return status;
  }

  const loomc_module_mutable_interop_view_t mutable_view =
      loomc_module_get_mutable_interop_view(request->public_module);
  const loom_check_test_native_module_t native_module = {
      .module = mutable_view.module,
      .source_resolver =
          {
              .fn = loom_source_table_resolve,
              .user_data = (void*)mutable_view.source_table,
          },
  };
  request->diagnostic_collector->module = native_module.module;
  status = native_provider->execute_native(provider, request, &native_module);
  request->diagnostic_collector->module = NULL;

  if (iree_status_is_ok(status) && !loom_check_diagnostic_collector_has_error(
                                       request->diagnostic_collector)) {
    projected = false;
    status = loom_check_test_native_project_module(request, &projected);
  }
  return status;
}

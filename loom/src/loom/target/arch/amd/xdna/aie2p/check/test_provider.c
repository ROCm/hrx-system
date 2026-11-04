// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/check/test_provider.h"

#include "loom/error/error_catalog.h"
#include "loom/error/parameter.h"
#include "loom/target/arch/amd/xdna/aie2p/check/array_plan.h"
#include "loom/tools/loom-check/diagnostics.h"
#include "loom/tools/loom-check/execute.h"

static bool loom_aie2p_check_test_matches(
    const loom_check_emit_provider_t* provider,
    iree_string_view_t target_name) {
  (void)provider;
  return iree_string_view_equal(target_name,
                                IREE_SV("aie2p-array-plan-with-remark"));
}

static iree_status_t loom_aie2p_check_test_execute(
    const loom_check_emit_provider_t* provider,
    const loom_check_emit_provider_request_t* request) {
  (void)provider;
  const loom_diagnostic_param_t params[] = {
      loom_param_string(IREE_SV("aie2p.array-plan.test")),
      loom_param_string(IREE_SV("non-error diagnostic regression")),
  };
  loom_diagnostic_t diagnostic = {
      .severity = LOOM_DIAGNOSTIC_REMARK,
      .error = LOOM_ERR_FOLD_001,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
      .emitter = LOOM_EMITTER_PASS,
  };
  diagnostic.origin.filename = request->filename;
  diagnostic.origin.start_line = 1;
  IREE_RETURN_IF_ERROR(loom_check_diagnostic_collector_sink(
      request->diagnostic_collector, &diagnostic));
  return loom_aie2p_array_plan_check_emit_provider.base.execute(
      &loom_aie2p_array_plan_check_emit_provider.base, request);
}

static iree_status_t loom_aie2p_check_test_append_names(
    const loom_check_emit_provider_t* provider,
    iree_string_builder_t* builder) {
  (void)provider;
  return iree_string_builder_append_cstring(builder,
                                            "aie2p-array-plan-with-remark");
}

static const loom_check_emit_provider_t kAie2pCheckTestEmitProvider = {
    .name = IREE_SVL("AIE2P test"),
    .match = loom_aie2p_check_test_matches,
    .execute = loom_aie2p_check_test_execute,
    .append_names = loom_aie2p_check_test_append_names,
};

static const loom_check_emit_provider_t* const kAie2pCheckTestEmitProviders[] =
    {
        &kAie2pCheckTestEmitProvider,
};

const loom_check_provider_t loom_aie2p_check_test_provider = {
    .name = IREE_SVL("amd-xdna-aie2p-test"),
    .emit_providers = kAie2pCheckTestEmitProviders,
    .emit_provider_count = IREE_ARRAYSIZE(kAie2pCheckTestEmitProviders),
};

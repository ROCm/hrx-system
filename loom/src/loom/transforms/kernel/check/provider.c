// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/kernel/check/provider.h"

#include "loom/format/text/printer.h"
#include "loom/tooling/compile/pipeline.h"
#include "loom/tools/loom-check/diagnostics.h"
#include "loom/tools/loom-check/execute.h"
#include "loom/transforms/kernel/launch_config_module.h"
#include "loom/verify/verify.h"

static bool loom_kernel_transform_check_emit_provider_matches(
    const loom_check_emit_provider_t* provider,
    iree_string_view_t target_name) {
  (void)provider;
  return iree_string_view_equal(target_name, IREE_SV("launch-config-module"));
}

static iree_status_t loom_kernel_transform_check_emit_provider_execute(
    const loom_check_emit_provider_t* provider,
    const loom_check_emit_provider_request_t* request) {
  (void)provider;
  if (!iree_string_view_is_empty(
          iree_string_view_trim(request->target_options))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "launch-config-module emit does not accept target options");
  }

  loom_compile_pipeline_options_t pipeline_options = {0};
  loom_compile_pipeline_options_initialize(&pipeline_options);
  pipeline_options.default_pipeline =
      LOOM_COMPILE_DEFAULT_PIPELINE_EXPANDED_SOURCE;
  pipeline_options.target_environment =
      request->environment->target_environment;
  pipeline_options.low_descriptor_registry = request->low_registry;
  pipeline_options.diagnostic_sink =
      (loom_diagnostic_sink_t){.fn = loom_check_diagnostic_collector_sink,
                               .user_data = request->diagnostic_collector};
  pipeline_options.source_resolver = request->source_resolver;
  pipeline_options.max_errors = 20;

  loom_compile_pipeline_result_t pipeline_result = {0};
  iree_status_t status =
      loom_compile_run_pipeline(request->module, &pipeline_options,
                                request->block_pool, &pipeline_result);
  loom_module_t* launch_module = NULL;
  if (iree_status_is_ok(status) && pipeline_result.pass.error_count == 0) {
    status = loom_kernel_launch_config_module_materialize(
        request->module, request->block_pool, request->host_allocator,
        &launch_module);
  }
  loom_verify_result_t verify_result = {0};
  if (iree_status_is_ok(status) && pipeline_result.pass.error_count == 0) {
    const loom_verify_options_t verify_options = {
        .sink = {.fn = loom_check_diagnostic_collector_sink,
                 .user_data = request->diagnostic_collector},
        .max_errors = 20,
        .source_resolver = request->source_resolver,
    };
    status = loom_verify_module(launch_module, &verify_options, &verify_result);
  }
  if (iree_status_is_ok(status) && pipeline_result.pass.error_count == 0) {
    if (verify_result.error_count == 0) {
      status = loom_text_print_module_to_builder(
          launch_module, &request->result->actual_output,
          LOOM_TEXT_PRINT_DEFAULT);
    }
  }
  if (launch_module != NULL) loom_module_free(launch_module);
  loom_compile_pipeline_result_deinitialize(&pipeline_result);
  return status;
}

static iree_status_t loom_kernel_transform_check_emit_provider_append_names(
    const loom_check_emit_provider_t* provider,
    iree_string_builder_t* builder) {
  (void)provider;
  return iree_string_builder_append_cstring(builder, "launch-config-module");
}

static const loom_check_emit_provider_t kLoomKernelTransformCheckEmitProvider =
    {
        .name = IREE_SVL("kernel launch configuration module"),
        .match = loom_kernel_transform_check_emit_provider_matches,
        .execute = loom_kernel_transform_check_emit_provider_execute,
        .append_names = loom_kernel_transform_check_emit_provider_append_names,
};

static const loom_check_emit_provider_t* const
    kLoomKernelTransformCheckEmitProviders[] = {
        &kLoomKernelTransformCheckEmitProvider,
};

const loom_check_provider_t loom_kernel_transform_check_provider = {
    .name = IREE_SVL("kernel transforms"),
    .emit_providers = kLoomKernelTransformCheckEmitProviders,
    .emit_provider_count =
        IREE_ARRAYSIZE(kLoomKernelTransformCheckEmitProviders),
};

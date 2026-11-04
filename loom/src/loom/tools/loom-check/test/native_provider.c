// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/loom-check/test/native_provider.h"

#include "loom/codegen/low/text_asm.h"
#include "loom/target/provider.h"
#include "loom/tools/loom-check/compile.h"
#include "loom/tools/loom-check/compile_diagnostics.h"
#include "loom/tools/loom-check/diagnostics.h"
#include "loomc/interop.h"
#include "loomc/iree.h"

struct loom_check_test_native_diagnostic_context_t {
  // Collector receiving each native diagnostic emission.
  loom_check_diagnostic_collector_t* collector;
  // Module whose types are rendered for the current emission.
  const loom_module_t* module;
  // Target-aware Low assembly printer configuration.
  loom_low_descriptor_text_print_context_t low_text;
};

static iree_status_t loom_check_test_native_format_type(
    loom_type_t type, void* user_data, loom_output_stream_t* stream) {
  const loom_check_test_native_diagnostic_context_t* context =
      (const loom_check_test_native_diagnostic_context_t*)user_data;
  return loom_text_print_type_with_options(type, context->module, stream,
                                           &context->low_text.options);
}

iree_status_t loom_check_test_native_diagnostic_emitter_capture_emit(
    void* user_data, const loom_diagnostic_emission_t* emission) {
  loom_check_test_native_diagnostic_emitter_capture_t* capture = user_data;
  loom_diagnostic_t diagnostic;
  loom_diagnostic_related_location_t
      related_locations[LOOM_DIAGNOSTIC_MAX_RELATED_LOCATIONS];
  loom_source_resolve_diagnostic(capture->module, capture->source_resolver,
                                 capture->emitter, emission, related_locations,
                                 &diagnostic);

  const loom_module_t* previous_module = capture->context->module;
  capture->context->module =
      emission->module ? emission->module : capture->module;
  iree_status_t status = loom_check_diagnostic_collector_sink(
      capture->context->collector, &diagnostic);
  capture->context->module = previous_module;
  IREE_RETURN_IF_ERROR(status);
  ++capture->emission_count;
  return iree_ok_status();
}

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
  const loom_target_environment_t* target_environment =
      loomc_target_environment_get_interop_view(
          request->environment->compile_session->target_environment);
  loom_check_test_native_diagnostic_context_t diagnostic_context = {
      .collector = request->diagnostic_collector,
      .module = mutable_view.module,
  };
  loom_check_test_native_module_t native_module = {
      .module = mutable_view.module,
      .source_resolver =
          {
              .fn = loom_source_table_resolve,
              .user_data = (void*)mutable_view.source_table,
          },
      .target_environment = target_environment,
      .low_registry =
          loom_target_environment_low_descriptor_registry(target_environment),
      .diagnostic_context = &diagnostic_context,
  };
  loom_low_descriptor_text_print_context_initialize(
      &native_module.low_registry.registry, &diagnostic_context.low_text);
  const loom_type_formatter_t previous_type_formatter =
      request->diagnostic_collector->type_formatter;
  request->diagnostic_collector->type_formatter = (loom_type_formatter_t){
      .fn = loom_check_test_native_format_type,
      .user_data = &diagnostic_context,
  };
  status = native_provider->execute_native(provider, request, &native_module);
  request->diagnostic_collector->type_formatter = previous_type_formatter;

  if (iree_status_is_ok(status) && !loom_check_diagnostic_collector_has_error(
                                       request->diagnostic_collector)) {
    projected = false;
    status = loom_check_test_native_project_module(request, &projected);
  }
  return status;
}

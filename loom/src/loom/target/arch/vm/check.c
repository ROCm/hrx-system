// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/vm/check.h"

#include "iree/vm/bytecode/disassembler.h"
#include "loom/target/arch/vm/module.h"
#include "loom/target/arch/vm/provider.h"
#include "loom/tools/loom-check/diagnostics.h"
#include "loom/tools/loom-check/execute.h"
#include "loom/tools/loom-check/source_low.h"

static bool loom_vm_check_match(const loom_check_emit_provider_t* provider,
                                iree_string_view_t target_name) {
  return iree_string_view_equal(target_name, IREE_SV("vm-dis"));
}

static iree_status_t loom_vm_check_write(void* user_data,
                                         iree_string_view_t fragment) {
  return iree_string_builder_append_string(user_data, fragment);
}

static iree_status_t loom_vm_check_emit(
    const loom_check_emit_provider_t* provider,
    const loom_check_emit_provider_request_t* request) {
  loom_check_source_low_request_t source_request;
  IREE_RETURN_IF_ERROR(
      loom_check_source_low_parse(request->target_options, &source_request));
  if (source_request.options & ~LOOM_CHECK_SOURCE_LOW_OPTION_TARGET) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "vm-dis accepts only @function and target options");
  }
  loom_check_prepare_source_low_options_t prepare_options;
  loom_check_prepare_source_low_options_initialize(&prepare_options);
  loom_target_specialization_request_t specialization = {0};
  if (iree_any_bit_set(source_request.options,
                       LOOM_CHECK_SOURCE_LOW_OPTION_TARGET)) {
    IREE_RETURN_IF_ERROR(loom_check_resolve_source_target(
        request->module, request->environment->target_environment,
        source_request.function_name, &source_request.target, &specialization));
    prepare_options.target_specializations =
        (loom_target_specialization_request_list_t){&specialization, 1};
  }
  loom_compile_pipeline_result_t pipeline_result = {0};
  iree_status_t status = loom_check_prepare_source_low_module(
      request->module, &prepare_options, request->low_registry,
      request->environment, request->source_resolver,
      request->diagnostic_collector, request->block_pool, &pipeline_result);

  if (!iree_status_is_ok(status) || request->diagnostic_collector->count) {
    loom_compile_pipeline_result_deinitialize(&pipeline_result);
    return status;
  }

  loom_check_diagnostic_emitter_capture_t capture = {
      .diagnostic_collector = request->diagnostic_collector,
      .module = request->module,
      .source_resolver = request->source_resolver,
      .emitter = LOOM_EMITTER_PASS,
  };
  const loom_target_emit_request_t emit_request = {
      .low_descriptor_registry = &request->low_registry->registry,
      .module = request->module,
      .function_versions = &pipeline_result.function_versions.list,
      .diagnostic_emitter = {.fn = loom_check_diagnostic_emitter_capture_emit,
                             .user_data = &capture},
      .scratch_arena = request->case_arena,
      .allocator = request->host_allocator,
  };
  loom_target_emit_artifact_t artifact = {0};
  bool emitted = false;
  status = loom_vm_module_emit(&emit_request, &emitted, &artifact);
  iree_byte_span_t contents = iree_byte_span_empty();
  if (iree_status_is_ok(status) && emitted) {
    status = iree_byte_sequence_clone(artifact.contents,
                                      request->host_allocator, &contents);
  }
  loom_target_emit_artifact_release(&artifact);
  if (iree_status_is_ok(status) && emitted) {
    status = iree_vm_bytecode_disassemble_module(
        iree_make_const_byte_span(contents.data, contents.data_length),
        (iree_vm_bytecode_disassembler_write_callback_t){
            .fn = loom_vm_check_write,
            .user_data = &request->result->actual_output},
        request->host_allocator);
  }
  iree_allocator_free(request->host_allocator, contents.data);
  loom_compile_pipeline_result_deinitialize(&pipeline_result);
  return status;
}

static iree_status_t loom_vm_check_append_names(
    const loom_check_emit_provider_t* provider,
    iree_string_builder_t* builder) {
  return iree_string_builder_append_cstring(builder, "vm-dis");
}

static const loom_check_emit_provider_t loom_vm_check_emit_provider = {
    .name = IREE_SVL("vm"),
    .match = loom_vm_check_match,
    .execute = loom_vm_check_emit,
    .append_names = loom_vm_check_append_names,
};

static const loom_check_emit_provider_t* const loom_vm_check_emit_providers[] =
    {
        &loom_vm_check_emit_provider,
};

const loom_check_provider_t loom_vm_check_provider = {
    .name = IREE_SVL("vm"),
    .target_provider = &loom_vm_target_provider,
    .emit_providers = loom_vm_check_emit_providers,
    .emit_provider_count = IREE_ARRAYSIZE(loom_vm_check_emit_providers),
};

// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/vm/check/loom_check.h"

#include "iree/base/byte_sequence.h"
#include "iree/vm/bytecode/module.h"
#include "iree/vm/bytecode/tooling/dump.h"
#include "loom/target/emit/vm/module_emitter.h"
#include "loom/tools/loom-check/diagnostics.h"

static bool loom_vm_loom_check_emit_provider_matches(
    const loom_check_emit_provider_t* provider,
    iree_string_view_t target_name) {
  return iree_string_view_equal(target_name, IREE_SV("vm-dis"));
}

typedef enum loom_vm_loom_check_input_e {
  LOOM_VM_LOOM_CHECK_INPUT_SOURCE_LOW = 0,
  LOOM_VM_LOOM_CHECK_INPUT_LOW = 1,
} loom_vm_loom_check_input_t;

static iree_status_t loom_vm_loom_check_parse_input(
    iree_string_view_t target_options, loom_vm_loom_check_input_t* out_input) {
  target_options = iree_string_view_trim(target_options);
  if (iree_string_view_is_empty(target_options) ||
      iree_string_view_equal(target_options, IREE_SV("input=source-low"))) {
    *out_input = LOOM_VM_LOOM_CHECK_INPUT_SOURCE_LOW;
    return iree_ok_status();
  }
  if (iree_string_view_equal(target_options, IREE_SV("input=low"))) {
    *out_input = LOOM_VM_LOOM_CHECK_INPUT_LOW;
    return iree_ok_status();
  }
  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT,
      "vm-dis expected no options, 'input=source-low', or 'input=low'; got "
      "'%.*s'",
      (int)target_options.size, target_options.data);
}

static iree_status_t loom_vm_loom_check_prepare_module(
    const loom_check_emit_provider_request_t* request,
    loom_vm_loom_check_input_t input) {
  loom_check_prepare_source_low_options_t options = {0};
  loom_check_prepare_source_low_options_initialize(&options);
  options.pipeline = input == LOOM_VM_LOOM_CHECK_INPUT_LOW ? IREE_SV("none")
                                                           : IREE_SV("default");
  options.default_pipeline = LOOM_COMPILE_DEFAULT_PIPELINE_PREPARED_LOW;
  return loom_check_prepare_source_low_module(
      request->module, &options, request->low_registry, request->environment,
      request->source_resolver, request->diagnostic_collector,
      request->block_pool);
}

static iree_status_t loom_vm_loom_check_append_dump(void* user_data,
                                                    iree_string_view_t text) {
  return iree_string_builder_append_string((iree_string_builder_t*)user_data,
                                           text);
}

static iree_status_t loom_vm_loom_check_emit_provider_execute(
    const loom_check_emit_provider_t* provider,
    const loom_check_emit_provider_request_t* request) {
  loom_vm_loom_check_input_t input = LOOM_VM_LOOM_CHECK_INPUT_SOURCE_LOW;
  IREE_RETURN_IF_ERROR(
      loom_vm_loom_check_parse_input(request->target_options, &input));

  IREE_RETURN_IF_ERROR(loom_vm_loom_check_prepare_module(request, input));
  if (request->diagnostic_collector->count != 0) return iree_ok_status();

  loom_check_diagnostic_emitter_capture_t capture = {
      .diagnostic_collector = request->diagnostic_collector,
      .module = request->module,
      .source_resolver = request->source_resolver,
      .emitter = LOOM_EMITTER_PASS,
  };
  const loom_vm_module_emitter_options_t options = {
      .descriptor_registry = &request->low_registry->registry,
      .diagnostic_emitter =
          {
              .fn = loom_check_diagnostic_emitter_capture_emit,
              .user_data = &capture,
          },
  };

  iree_byte_sequence_t* contents = NULL;
  iree_byte_span_t contiguous_contents = iree_byte_span_empty();
  iree_status_t status =
      loom_vm_emit_module(request->module, &options, request->case_arena,
                          request->host_allocator, &contents);
  if (iree_status_is_ok(status) && contents != NULL) {
    status = iree_byte_sequence_clone(contents, request->host_allocator,
                                      &contiguous_contents);
  }
  const iree_const_byte_span_t module_contents = iree_make_const_byte_span(
      contiguous_contents.data, contiguous_contents.data_length);
  if (iree_status_is_ok(status) && contents != NULL) {
    status = iree_vm_bytecode_module_verify(module_contents,
                                            request->host_allocator);
  }
  if (iree_status_is_ok(status) && contents != NULL) {
    status = iree_vm_bytecode_module_dump(
        IREE_SV("module"), module_contents,
        (iree_vm_bytecode_dump_write_callback_t){
            .fn = loom_vm_loom_check_append_dump,
            .user_data = &request->result->actual_output,
        },
        request->host_allocator);
  }
  iree_allocator_free(request->host_allocator, contiguous_contents.data);
  iree_byte_sequence_release(contents);
  return status;
}

static iree_status_t loom_vm_loom_check_emit_provider_append_names(
    const loom_check_emit_provider_t* provider,
    iree_string_builder_t* builder) {
  return iree_string_builder_append_cstring(builder, "vm-dis");
}

const loom_check_emit_provider_t loom_vm_loom_check_emit_provider = {
    .name = IREE_SVL("vm"),
    .match = loom_vm_loom_check_emit_provider_matches,
    .execute = loom_vm_loom_check_emit_provider_execute,
    .append_names = loom_vm_loom_check_emit_provider_append_names,
};

// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/target/vm/check/loom_check.h"

#include "iree/vm/bytecode/disassembler.h"
#include "loom/tools/loom-check/compile.h"
#include "loom/tools/loom-check/execute.h"

typedef struct loom_vm_check_request_t {
  // Optional root function name without a leading '@'.
  iree_string_view_t function_name;
  // Optional complete target profile specification.
  iree_string_view_t target;
  // True when |target| was explicitly supplied.
  bool has_target;
} loom_vm_check_request_t;

static iree_status_t loom_vm_check_parse_request(
    iree_string_view_t text, loom_vm_check_request_t* out_request) {
  *out_request = (loom_vm_check_request_t){0};
  text = iree_string_view_trim(text);
  if (iree_string_view_starts_with_char(text, '@')) {
    iree_string_view_t symbol = iree_string_view_empty();
    iree_string_view_split(text, ' ', &symbol, &text);
    out_request->function_name =
        iree_string_view_substr(symbol, 1, IREE_HOST_SIZE_MAX);
    if (iree_string_view_is_empty(out_request->function_name)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "vm-dis requires a nonempty function symbol");
    }
  }

  text = iree_string_view_trim(text);
  while (!iree_string_view_is_empty(text)) {
    iree_string_view_t token = iree_string_view_empty();
    iree_string_view_split(text, ' ', &token, &text);
    iree_string_view_t name = iree_string_view_empty();
    iree_string_view_t value = iree_string_view_empty();
    iree_string_view_split(token, '=', &name, &value);
    if (!iree_string_view_equal(name, IREE_SV("target"))) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "vm-dis accepts only @function and target options");
    }
    if (out_request->has_target) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "duplicate vm-dis option 'target'");
    }
    out_request->target = value;
    out_request->has_target = true;
    text = iree_string_view_trim(text);
  }
  if (!iree_string_view_is_empty(out_request->function_name) &&
      !out_request->has_target) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "vm-dis @function requires "
                            "target=family:selector");
  }
  return iree_ok_status();
}

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
  loom_vm_check_request_t vm_request;
  IREE_RETURN_IF_ERROR(
      loom_vm_check_parse_request(request->target_options, &vm_request));
  const loom_check_compile_artifact_options_t compile_options = {
      .artifact_format = IREE_SV("vm"),
      .root = vm_request.function_name,
      .target = vm_request.target,
      .lower_source_to_low = true,
      .control_flow_lowering = LOOMC_TARGET_CONTROL_FLOW_LOWERING_CFG,
  };
  loomc_source_t* artifact_source = NULL;
  iree_status_t status =
      loom_check_compile_artifact(request, &compile_options, &artifact_source);
  const loomc_byte_span_t source_contents =
      loomc_source_contents(artifact_source);
  const iree_const_byte_span_t contents = iree_make_const_byte_span(
      source_contents.data, source_contents.data_length);
  if (iree_status_is_ok(status) && artifact_source != NULL) {
    status = iree_vm_bytecode_disassemble_module(
        contents,
        (iree_vm_bytecode_disassembler_write_callback_t){
            .fn = loom_vm_check_write,
            .user_data = &request->result->actual_output},
        request->host_allocator);
  }
  loomc_source_release(artifact_source);
  return status;
}

static iree_status_t loom_vm_check_append_names(
    const loom_check_emit_provider_t* provider,
    iree_string_builder_t* builder) {
  return iree_string_builder_append_cstring(builder, "vm-dis");
}

const loom_check_emit_provider_t loom_vm_loom_check_emit_provider = {
    .name = IREE_SVL("vm"),
    .match = loom_vm_check_match,
    .execute = loom_vm_check_emit,
    .append_names = loom_vm_check_append_names,
};

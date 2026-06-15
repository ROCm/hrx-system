// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/llvmir/check/loom_check.h"

#include "iree/io/vec_stream.h"
#include "loom/codegen/low/function.h"
#include "loom/ir/module.h"
#include "loom/ops/llvmir/ops.h"
#include "loom/target/emit/llvmir/bitcode_writer.h"
#include "loom/target/emit/llvmir/module_emitter.h"
#include "loom/target/emit/llvmir/target_env.h"
#include "loom/target/emit/llvmir/text_writer.h"
#include "loom/target/emit/llvmir/verify.h"
#include "loom/target/tool/llvm.h"
#include "loom/tools/loom-check/diagnostics.h"
#include "loom/util/stream.h"

typedef enum loom_llvmir_loom_check_emit_format_e {
  LOOM_LLVMIR_LOOM_CHECK_EMIT_TEXT = 0,
  LOOM_LLVMIR_LOOM_CHECK_EMIT_BODY_TEXT = 1,
  LOOM_LLVMIR_LOOM_CHECK_EMIT_BITCODE_DISASSEMBLY = 2,
  LOOM_LLVMIR_LOOM_CHECK_EMIT_OBJECT = 3,
} loom_llvmir_loom_check_emit_format_t;

typedef enum loom_llvmir_loom_check_disassembly_filter_flag_bits_e {
  LOOM_LLVMIR_LOOM_CHECK_DISASSEMBLY_FILTER_NONE = 0u,
  LOOM_LLVMIR_LOOM_CHECK_DISASSEMBLY_FILTER_SOURCE_FILENAME = 1u << 0,
} loom_llvmir_loom_check_disassembly_filter_flag_bits_t;
typedef uint8_t loom_llvmir_loom_check_disassembly_filter_flags_t;

typedef struct loom_llvmir_loom_check_byte_buffer_t {
  // Allocated byte storage owned by this buffer.
  uint8_t* data;
  // Number of valid bytes in |data|.
  iree_host_size_t length;
} loom_llvmir_loom_check_byte_buffer_t;

static bool loom_llvmir_loom_check_case_has_requirement(
    const loom_check_case_t* test_case, iree_string_view_t requirement) {
  for (iree_host_size_t i = 0; i < test_case->requirement_count; ++i) {
    if (iree_string_view_equal(test_case->requirements[i], requirement)) {
      return true;
    }
  }
  return false;
}

static iree_status_t loom_llvmir_loom_check_fail_missing_requirement(
    iree_string_view_t emit_target, iree_string_view_t requirement,
    loom_check_result_t* result) {
  result->raw_outcome = LOOM_CHECK_FAIL;
  result->final_outcome = LOOM_CHECK_FAIL;
  return iree_string_builder_append_format(
      &result->detail,
      "RUN: emit %.*s requires '// REQUIRES: %.*s'; external tool "
      "dependencies must be declared even when they are available\n",
      (int)emit_target.size, emit_target.data, (int)requirement.size,
      requirement.data);
}

static iree_status_t loom_llvmir_loom_check_require_declared_requirement(
    const loom_check_case_t* test_case, iree_string_view_t requirement,
    loom_check_result_t* result, bool* out_continue_execution) {
  if (loom_llvmir_loom_check_case_has_requirement(test_case, requirement)) {
    return iree_ok_status();
  }
  *out_continue_execution = false;
  return loom_llvmir_loom_check_fail_missing_requirement(test_case->emit_target,
                                                         requirement, result);
}

iree_status_t loom_llvmir_loom_check_emit_provider_check_requirements(
    const loom_check_emit_provider_t* provider,
    const loom_check_case_t* test_case, loom_check_result_t* result,
    bool* out_continue_execution) {
  iree_string_view_t emit_target =
      iree_string_view_trim(test_case->emit_target);
  iree_string_view_t target_name = iree_string_view_empty();
  iree_string_view_t target_options = iree_string_view_empty();
  iree_string_view_split(emit_target, ' ', &target_name, &target_options);
  (void)target_options;
  target_name = iree_string_view_trim(target_name);

  if (iree_string_view_equal(target_name, IREE_SV("llvmir-bitcode"))) {
    return loom_llvmir_loom_check_require_declared_requirement(
        test_case, IREE_SV("llvm-dis"), result, out_continue_execution);
  }
  if (iree_string_view_equal(target_name, IREE_SV("llvmir-object"))) {
    IREE_RETURN_IF_ERROR(loom_llvmir_loom_check_require_declared_requirement(
        test_case, IREE_SV("llc"), result, out_continue_execution));
  }
  return iree_ok_status();
}

bool loom_llvmir_loom_check_emit_provider_matches(
    const loom_check_emit_provider_t* provider,
    iree_string_view_t target_name) {
  return iree_string_view_equal(target_name, IREE_SV("llvmir")) ||
         iree_string_view_equal(target_name, IREE_SV("llvmir-text")) ||
         iree_string_view_equal(target_name, IREE_SV("llvmir-body")) ||
         iree_string_view_equal(target_name, IREE_SV("llvmir-text-body")) ||
         iree_string_view_equal(target_name, IREE_SV("llvmir-bitcode")) ||
         iree_string_view_equal(target_name, IREE_SV("llvmir-object"));
}

static iree_status_t loom_llvmir_loom_check_parse_emit_format(
    iree_string_view_t target_name,
    loom_llvmir_loom_check_emit_format_t* out_format) {
  if (iree_string_view_equal(target_name, IREE_SV("llvmir")) ||
      iree_string_view_equal(target_name, IREE_SV("llvmir-text"))) {
    *out_format = LOOM_LLVMIR_LOOM_CHECK_EMIT_TEXT;
    return iree_ok_status();
  }
  if (iree_string_view_equal(target_name, IREE_SV("llvmir-body")) ||
      iree_string_view_equal(target_name, IREE_SV("llvmir-text-body"))) {
    *out_format = LOOM_LLVMIR_LOOM_CHECK_EMIT_BODY_TEXT;
    return iree_ok_status();
  }
  if (iree_string_view_equal(target_name, IREE_SV("llvmir-bitcode"))) {
    *out_format = LOOM_LLVMIR_LOOM_CHECK_EMIT_BITCODE_DISASSEMBLY;
    return iree_ok_status();
  }
  if (iree_string_view_equal(target_name, IREE_SV("llvmir-object"))) {
    *out_format = LOOM_LLVMIR_LOOM_CHECK_EMIT_OBJECT;
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unknown LLVMIR emit target '%.*s'",
                          (int)target_name.size, target_name.data);
}

static iree_string_view_t loom_llvmir_loom_check_consume_line(
    iree_string_view_t* remaining) {
  iree_host_size_t newline_position =
      iree_string_view_find(*remaining, IREE_SV("\n"), 0);
  if (newline_position == IREE_STRING_VIEW_NPOS) {
    iree_string_view_t line = *remaining;
    *remaining = iree_string_view_empty();
    return line;
  }
  iree_string_view_t line =
      iree_string_view_substr(*remaining, 0, newline_position);
  *remaining = iree_string_view_substr(*remaining, newline_position + 1,
                                       IREE_HOST_SIZE_MAX);
  return line;
}

static iree_status_t loom_llvmir_loom_check_filter_disassembly(
    iree_string_view_t input,
    loom_llvmir_loom_check_disassembly_filter_flags_t flags,
    iree_string_builder_t* output) {
  iree_string_view_t remaining = input;
  while (!iree_string_view_is_empty(remaining)) {
    iree_string_view_t line = loom_llvmir_loom_check_consume_line(&remaining);
    iree_string_view_t trimmed = iree_string_view_trim(line);
    if (iree_string_view_starts_with(trimmed, IREE_SV(";"))) {
      continue;
    }
    if (iree_any_bit_set(
            flags, LOOM_LLVMIR_LOOM_CHECK_DISASSEMBLY_FILTER_SOURCE_FILENAME) &&
        iree_string_view_starts_with(trimmed, IREE_SV("source_filename ="))) {
      continue;
    }
    IREE_RETURN_IF_ERROR(iree_string_builder_append_string(output, line));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(output, "\n"));
  }
  return iree_ok_status();
}

static bool loom_llvmir_loom_check_is_module_header_line(
    iree_string_view_t line) {
  line = iree_string_view_trim(line);
  return iree_string_view_starts_with(line, IREE_SV("source_filename =")) ||
         iree_string_view_starts_with(line, IREE_SV("target datalayout =")) ||
         iree_string_view_starts_with(line, IREE_SV("target triple ="));
}

static iree_status_t loom_llvmir_loom_check_strip_module_header(
    iree_string_view_t input, iree_string_builder_t* output) {
  bool stripped_header = false;
  bool emitted_body_line = false;
  iree_string_view_t remaining = input;
  while (!iree_string_view_is_empty(remaining)) {
    iree_string_view_t line = loom_llvmir_loom_check_consume_line(&remaining);
    if (loom_llvmir_loom_check_is_module_header_line(line)) {
      stripped_header = true;
      continue;
    }
    if (stripped_header && !emitted_body_line &&
        iree_string_view_is_empty(iree_string_view_trim(line))) {
      continue;
    }
    IREE_RETURN_IF_ERROR(iree_string_builder_append_string(output, line));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(output, "\n"));
    emitted_body_line = true;
  }
  return iree_ok_status();
}

static iree_status_t loom_llvmir_loom_check_write_text(
    const loom_llvmir_module_t* lowered_module, loom_check_result_t* result) {
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&result->actual_output, &stream);
  return loom_llvmir_text_write_module(lowered_module, &stream);
}

static iree_status_t loom_llvmir_loom_check_write_body_text(
    const loom_llvmir_module_t* lowered_module, iree_allocator_t allocator,
    loom_check_result_t* result) {
  iree_string_builder_t module_text;
  iree_string_builder_initialize(allocator, &module_text);

  loom_output_stream_t stream;
  loom_output_stream_for_builder(&module_text, &stream);
  iree_status_t status = loom_llvmir_text_write_module(lowered_module, &stream);
  if (iree_status_is_ok(status)) {
    status = loom_llvmir_loom_check_strip_module_header(
        iree_string_builder_view(&module_text), &result->actual_output);
  }

  iree_string_builder_deinitialize(&module_text);
  return status;
}

static void loom_llvmir_loom_check_byte_buffer_deinitialize(
    loom_llvmir_loom_check_byte_buffer_t* buffer, iree_allocator_t allocator) {
  iree_allocator_free(allocator, buffer->data);
  *buffer = (loom_llvmir_loom_check_byte_buffer_t){0};
}

static iree_status_t loom_llvmir_loom_check_write_bitcode_bytes(
    const loom_llvmir_module_t* lowered_module, iree_allocator_t allocator,
    loom_llvmir_loom_check_byte_buffer_t* out_bitcode) {
  *out_bitcode = (loom_llvmir_loom_check_byte_buffer_t){0};
  iree_io_stream_t* bitcode_stream = NULL;
  iree_status_t status = iree_io_vec_stream_create(
      IREE_IO_STREAM_MODE_READABLE | IREE_IO_STREAM_MODE_WRITABLE |
          IREE_IO_STREAM_MODE_SEEKABLE,
      4096, allocator, &bitcode_stream);

  uint8_t* bitcode_data = NULL;
  iree_host_size_t bitcode_length = 0;
  if (iree_status_is_ok(status)) {
    status = loom_llvmir_bitcode_write_module(lowered_module, bitcode_stream);
  }
  if (iree_status_is_ok(status)) {
    iree_io_stream_pos_t stream_length = iree_io_stream_length(bitcode_stream);
    if (stream_length <= 0 ||
        (uint64_t)stream_length > (uint64_t)IREE_HOST_SIZE_MAX) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "LLVM bitcode output length is invalid");
    } else {
      bitcode_length = (iree_host_size_t)stream_length;
    }
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_allocator_malloc(allocator, bitcode_length, (void**)&bitcode_data);
  }
  if (iree_status_is_ok(status)) {
    status = iree_io_stream_seek(bitcode_stream, IREE_IO_STREAM_SEEK_SET, 0);
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_io_stream_read(bitcode_stream, bitcode_length, bitcode_data, NULL);
  }
  if (iree_status_is_ok(status)) {
    *out_bitcode = (loom_llvmir_loom_check_byte_buffer_t){
        .data = bitcode_data,
        .length = bitcode_length,
    };
    bitcode_data = NULL;
  }

  iree_allocator_free(allocator, bitcode_data);
  iree_io_stream_release(bitcode_stream);
  return status;
}

static iree_status_t loom_llvmir_loom_check_write_bitcode_disassembly(
    const loom_llvmir_module_t* lowered_module, iree_allocator_t allocator,
    loom_check_result_t* result) {
  loom_llvmir_loom_check_byte_buffer_t bitcode = {0};
  iree_status_t status = loom_llvmir_loom_check_write_bitcode_bytes(
      lowered_module, allocator, &bitcode);

  loom_llvm_tool_output_t disassembly = {0};
  if (iree_status_is_ok(status)) {
    loom_llvm_toolchain_t toolchain;
    loom_llvm_toolchain_initialize_from_environment(&toolchain);
    status = loom_llvm_tool_disassemble_bitcode(
        &toolchain, iree_make_const_byte_span(bitcode.data, bitcode.length),
        allocator, &disassembly);
  }
  if (iree_status_is_ok(status)) {
    loom_llvmir_loom_check_disassembly_filter_flags_t filter_flags =
        LOOM_LLVMIR_LOOM_CHECK_DISASSEMBLY_FILTER_SOURCE_FILENAME;
    status = loom_llvmir_loom_check_filter_disassembly(
        iree_make_string_view(disassembly.data, disassembly.length),
        filter_flags, &result->actual_output);
  }

  loom_llvm_tool_output_deinitialize(&disassembly, allocator);
  loom_llvmir_loom_check_byte_buffer_deinitialize(&bitcode, allocator);
  return status;
}

static iree_status_t loom_llvmir_loom_check_write_object(
    const loom_llvmir_module_t* lowered_module,
    const loom_llvmir_target_profile_t* profile, iree_allocator_t allocator,
    loom_check_result_t* result) {
  loom_llvmir_loom_check_byte_buffer_t bitcode = {0};
  iree_status_t status = loom_llvmir_loom_check_write_bitcode_bytes(
      lowered_module, allocator, &bitcode);

  loom_llvm_tool_output_t object = {0};
  if (iree_status_is_ok(status)) {
    loom_llvm_toolchain_t toolchain;
    loom_llvm_toolchain_initialize_from_environment(&toolchain);
    loom_llvmir_target_profile_llc_arguments_t llc_arguments = {0};
    status = loom_llvmir_target_profile_llc_arguments(profile, &llc_arguments);
    if (iree_status_is_ok(status)) {
      status = loom_llvm_tool_compile_object(
          &toolchain, iree_make_const_byte_span(bitcode.data, bitcode.length),
          llc_arguments.values, llc_arguments.count, allocator, &object);
    }
  }
  if (iree_status_is_ok(status) && object.length == 0) {
    status =
        iree_make_status(IREE_STATUS_DATA_LOSS, "LLVM object output is empty");
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_format(
        &result->actual_output, "object emitted: %.*s\n",
        (int)profile->name.size, profile->name.data);
  }

  loom_llvm_tool_output_deinitialize(&object, allocator);
  loom_llvmir_loom_check_byte_buffer_deinitialize(&bitcode, allocator);
  return status;
}

static iree_string_view_t loom_llvmir_loom_check_string_attr(
    const loom_module_t* module, const loom_op_t* op, uint8_t attr_index) {
  const loom_attribute_t attr = loom_op_attrs(op)[attr_index];
  if (loom_attr_is_absent(attr)) {
    return iree_string_view_empty();
  }
  const loom_string_id_t string_id = loom_attr_as_string_id(attr);
  if (string_id == LOOM_STRING_ID_INVALID ||
      string_id >= module->strings.count) {
    return iree_string_view_empty();
  }
  return module->strings.entries[string_id];
}

static bool loom_llvmir_loom_check_lookup_symbol_id(
    const loom_module_t* module, iree_string_view_t symbol_name,
    loom_symbol_id_t* out_symbol_id) {
  *out_symbol_id = LOOM_SYMBOL_ID_INVALID;
  const loom_string_id_t symbol_name_id =
      loom_module_lookup_string(module, symbol_name);
  if (symbol_name_id == LOOM_STRING_ID_INVALID) {
    return false;
  }
  const loom_symbol_id_t symbol_id =
      loom_module_find_symbol(module, symbol_name_id);
  if (symbol_id == LOOM_SYMBOL_ID_INVALID) {
    return false;
  }
  *out_symbol_id = symbol_id;
  return true;
}

static const loom_op_t* loom_llvmir_loom_check_resolve_llvmir_target_op(
    const loom_check_emit_provider_request_t* request,
    iree_string_view_t symbol_name) {
  loom_symbol_id_t symbol_id = LOOM_SYMBOL_ID_INVALID;
  if (!loom_llvmir_loom_check_lookup_symbol_id(request->module, symbol_name,
                                               &symbol_id)) {
    return NULL;
  }
  const loom_op_t* op = request->module->symbols.entries[symbol_id].defining_op;
  return op != NULL && loom_llvmir_target_isa(op) ? op : NULL;
}

static void loom_llvmir_loom_check_initialize_projection_profile(
    const loom_check_emit_provider_request_t* request,
    iree_string_view_t symbol_name, const loom_op_t* target_op,
    loom_llvmir_target_profile_storage_t* out_storage) {
  out_storage->target_env = (loom_llvmir_target_env_t){
      .name = symbol_name,
      .target_triple = loom_llvmir_loom_check_string_attr(
          request->module, target_op, loom_llvmir_target_triple_ATTR_INDEX),
      .data_layout = loom_llvmir_loom_check_string_attr(
          request->module, target_op,
          loom_llvmir_target_data_layout_ATTR_INDEX),
  };
  out_storage->profile = (loom_llvmir_target_profile_t){
      .name = symbol_name,
      .target_env = &out_storage->target_env,
      .target_cpu = loom_llvmir_loom_check_string_attr(
          request->module, target_op, loom_llvmir_target_cpu_ATTR_INDEX),
      .target_features = loom_llvmir_loom_check_string_attr(
          request->module, target_op, loom_llvmir_target_features_ATTR_INDEX),
  };
}

static iree_status_t loom_llvmir_loom_check_resolve_object_profile(
    const loom_check_emit_provider_request_t* request,
    loom_llvmir_target_profile_storage_t* out_profile_storage,
    const loom_llvmir_target_profile_t** out_profile) {
  *out_profile_storage = (loom_llvmir_target_profile_storage_t){0};
  *out_profile = NULL;
  if (iree_string_view_starts_with(request->target_options, IREE_SV("@"))) {
    const iree_string_view_t symbol_name =
        iree_string_view_substr(request->target_options, 1, IREE_HOST_SIZE_MAX);
    if (iree_string_view_is_empty(symbol_name)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "target symbol name is required");
    }
    const loom_op_t* target_op =
        loom_llvmir_loom_check_resolve_llvmir_target_op(request, symbol_name);
    if (target_op == NULL) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "target symbol @%.*s is not an llvmir.target",
                              (int)symbol_name.size, symbol_name.data);
    }
    loom_llvmir_loom_check_initialize_projection_profile(
        request, symbol_name, target_op, out_profile_storage);
    *out_profile = &out_profile_storage->profile;
    return iree_ok_status();
  }

  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "llvmir-object requires target symbol such as "
                          "@target, got '%.*s'",
                          (int)request->target_options.size,
                          request->target_options.data);
}

static bool loom_llvmir_loom_check_module_has_low_functions(
    const loom_module_t* module) {
  const loom_block_t* block = loom_region_const_entry_block(module->body);
  const loom_op_t* op = NULL;
  loom_block_for_each_op(block, op) {
    if (loom_low_function_def_isa(op)) {
      return true;
    }
  }
  return false;
}

static iree_status_t loom_llvmir_loom_check_prepare_low_module(
    const loom_check_emit_provider_request_t* request) {
  if (loom_llvmir_loom_check_module_has_low_functions(request->module)) {
    return iree_ok_status();
  }
  loom_check_prepare_source_low_options_t prepare_options = {0};
  loom_check_prepare_source_low_options_initialize(&prepare_options);
  return loom_check_prepare_source_low_module(
      request->module, &prepare_options, request->low_registry,
      request->environment, request->source_resolver,
      request->diagnostic_collector, request->block_pool);
}

iree_status_t loom_llvmir_loom_check_emit_provider_execute(
    const loom_check_emit_provider_t* provider,
    const loom_check_emit_provider_request_t* request) {
  const loom_llvmir_loom_check_emit_provider_t* llvmir_provider =
      (const loom_llvmir_loom_check_emit_provider_t*)provider;
  loom_llvmir_loom_check_emit_format_t format;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_loom_check_parse_emit_format(request->target_name, &format));

  loom_llvmir_target_profile_storage_t profile_storage = {0};
  const loom_llvmir_target_profile_t* profile = NULL;
  if (format == LOOM_LLVMIR_LOOM_CHECK_EMIT_OBJECT) {
    IREE_RETURN_IF_ERROR(loom_llvmir_loom_check_resolve_object_profile(
        request, &profile_storage, &profile));
  }

  IREE_RETURN_IF_ERROR(loom_llvmir_loom_check_prepare_low_module(request));
  if (request->diagnostic_collector->count != 0) {
    return iree_ok_status();
  }

  loom_check_diagnostic_emitter_capture_t diagnostic_capture = {
      .diagnostic_collector = request->diagnostic_collector,
      .module = request->module,
      .source_resolver = request->source_resolver,
      .emitter = LOOM_EMITTER_PASS,
  };
  loom_llvmir_module_t* lowered_module = NULL;
  const loom_llvmir_target_profile_provider_t** target_profile_providers = NULL;
  loom_llvmir_target_profile_registry_t target_profile_registry = {0};
  iree_status_t status = iree_ok_status();
  if (llvmir_provider->target_profile_provider_function_count != 0) {
    const iree_host_size_t byte_length =
        llvmir_provider->target_profile_provider_function_count *
        sizeof(*target_profile_providers);
    status = iree_allocator_malloc(request->host_allocator, byte_length,
                                   (void**)&target_profile_providers);
  }
  if (iree_status_is_ok(status) && target_profile_providers != NULL) {
    for (iree_host_size_t i = 0;
         i < llvmir_provider->target_profile_provider_function_count; ++i) {
      target_profile_providers[i] =
          llvmir_provider->target_profile_provider_functions[i]();
      if (target_profile_providers[i] == NULL) {
        status = iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "LLVMIR loom-check target profile provider returned null");
        break;
      }
    }
  }
  if (iree_status_is_ok(status) && target_profile_providers != NULL) {
    target_profile_registry = (loom_llvmir_target_profile_registry_t){
        .providers = target_profile_providers,
        .provider_count =
            llvmir_provider->target_profile_provider_function_count,
    };
  }
  loom_llvmir_emit_low_module_options_t options = {0};
  loom_llvmir_emit_low_module_options_initialize(&options);
  options.target_profile_registry =
      target_profile_providers != NULL ? &target_profile_registry : NULL;
  if (iree_status_is_ok(status)) {
    status = loom_llvmir_emit_low_module(
        request->module, &request->low_registry->registry,
        loom_target_selection_empty(),
        (iree_diagnostic_emitter_t){
            .fn = loom_check_diagnostic_emitter_capture_emit,
            .user_data = &diagnostic_capture,
        },
        request->case_arena, &options, &lowered_module,
        request->host_allocator);
  }
  const bool has_lowered_module =
      request->diagnostic_collector->count == 0 && lowered_module != NULL;
  if (iree_status_is_ok(status) && has_lowered_module) {
    status = loom_llvmir_verify_module(lowered_module);
  }
  if (iree_status_is_ok(status) && has_lowered_module) {
    switch (format) {
      case LOOM_LLVMIR_LOOM_CHECK_EMIT_TEXT:
        status =
            loom_llvmir_loom_check_write_text(lowered_module, request->result);
        break;
      case LOOM_LLVMIR_LOOM_CHECK_EMIT_BODY_TEXT:
        status = loom_llvmir_loom_check_write_body_text(
            lowered_module, request->host_allocator, request->result);
        break;
      case LOOM_LLVMIR_LOOM_CHECK_EMIT_BITCODE_DISASSEMBLY:
        status = loom_llvmir_loom_check_write_bitcode_disassembly(
            lowered_module, request->host_allocator, request->result);
        break;
      case LOOM_LLVMIR_LOOM_CHECK_EMIT_OBJECT:
        status = loom_llvmir_loom_check_write_object(
            lowered_module, profile, request->host_allocator, request->result);
        break;
    }
  }
  loom_llvmir_module_free(lowered_module);
  iree_allocator_free(request->host_allocator, target_profile_providers);
  return status;
}

iree_status_t loom_llvmir_loom_check_emit_provider_append_names(
    const loom_check_emit_provider_t* provider,
    iree_string_builder_t* builder) {
  return iree_string_builder_append_cstring(
      builder,
      "llvmir, llvmir-text, llvmir-body, llvmir-text-body, llvmir-bitcode, "
      "llvmir-object");
}

static bool loom_llvmir_loom_check_requirement_provider_matches(
    const loom_check_requirement_provider_t* provider,
    iree_string_view_t requirement) {
  return iree_string_view_equal(requirement, IREE_SV("llvm-as")) ||
         iree_string_view_equal(requirement, IREE_SV("llvm-dis")) ||
         iree_string_view_equal(requirement, IREE_SV("opt")) ||
         iree_string_view_equal(requirement, IREE_SV("llc")) ||
         iree_string_view_equal(requirement, IREE_SV("llvm-mc")) ||
         iree_string_view_equal(requirement, IREE_SV("llvm-objdump"));
}

static iree_status_t loom_llvmir_loom_check_query_llvm_tool(
    loom_llvm_tool_kind_t tool_kind, iree_allocator_t allocator) {
  loom_llvm_toolchain_t toolchain;
  loom_llvm_toolchain_initialize_from_environment(&toolchain);
  loom_llvm_tool_output_t version_text = {0};
  iree_status_t status = loom_llvm_tool_query_version(&toolchain, tool_kind,
                                                      allocator, &version_text);
  loom_llvm_tool_output_deinitialize(&version_text, allocator);
  return status;
}

static iree_status_t loom_llvmir_loom_check_requirement_provider_query(
    const loom_check_requirement_provider_t* provider,
    const loom_check_environment_t* environment, iree_string_view_t requirement,
    iree_allocator_t allocator) {
  if (iree_string_view_equal(requirement, IREE_SV("llvm-as"))) {
    return loom_llvmir_loom_check_query_llvm_tool(LOOM_LLVM_TOOL_LLVM_AS,
                                                  allocator);
  }
  if (iree_string_view_equal(requirement, IREE_SV("llvm-dis"))) {
    return loom_llvmir_loom_check_query_llvm_tool(LOOM_LLVM_TOOL_LLVM_DIS,
                                                  allocator);
  }
  if (iree_string_view_equal(requirement, IREE_SV("opt"))) {
    return loom_llvmir_loom_check_query_llvm_tool(LOOM_LLVM_TOOL_OPT,
                                                  allocator);
  }
  if (iree_string_view_equal(requirement, IREE_SV("llc"))) {
    return loom_llvmir_loom_check_query_llvm_tool(LOOM_LLVM_TOOL_LLC,
                                                  allocator);
  }
  if (iree_string_view_equal(requirement, IREE_SV("llvm-mc"))) {
    return loom_llvmir_loom_check_query_llvm_tool(LOOM_LLVM_TOOL_LLVM_MC,
                                                  allocator);
  }
  if (iree_string_view_equal(requirement, IREE_SV("llvm-objdump"))) {
    return loom_llvmir_loom_check_query_llvm_tool(LOOM_LLVM_TOOL_LLVM_OBJDUMP,
                                                  allocator);
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unknown LLVMIR loom-check requirement '%.*s'",
                          (int)requirement.size, requirement.data);
}

static iree_status_t loom_llvmir_loom_check_requirement_provider_append_names(
    const loom_check_requirement_provider_t* provider,
    iree_string_builder_t* builder) {
  return iree_string_builder_append_cstring(
      builder, "llvm-as, llvm-dis, opt, llc, llvm-mc, llvm-objdump");
}

const loom_check_requirement_provider_t
    loom_llvmir_loom_check_requirement_provider = {
        .name = IREE_SVL("llvmir"),
        .match = loom_llvmir_loom_check_requirement_provider_matches,
        .query = loom_llvmir_loom_check_requirement_provider_query,
        .append_names =
            loom_llvmir_loom_check_requirement_provider_append_names,
};

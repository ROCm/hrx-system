// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/x86/loom_check.h"

#include "iree/io/vec_stream.h"
#include "loom/analysis/symbol_facts.h"
#include "loom/codegen/low/target_binding.h"
#include "loom/ops/low/ops.h"
#include "loom/target/emit/native/x86/abi.h"
#include "loom/target/emit/native/x86/function.h"
#include "loom/tools/loom-check/diagnostics.h"
#include "loom/tools/loom-check/test/low/native_emit.h"

static bool loom_x86_loom_check_emit_provider_matches(
    const loom_check_emit_provider_t* provider,
    iree_string_view_t target_name) {
  (void)provider;
  return iree_string_view_equal(target_name, IREE_SV("x86-native-bytes"));
}

static iree_status_t loom_x86_loom_check_append_native_bytes(
    void* user_data, iree_const_byte_span_t block) {
  iree_string_builder_t* builder = (iree_string_builder_t*)user_data;
  for (iree_host_size_t i = 0; i < block.data_length; ++i) {
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_format(builder, " %02x", block.data[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_x86_loom_check_emit_native_bytes(
    const loom_x86_function_t* function,
    const loom_x86_module_abi_t* module_abi, iree_string_builder_t* builder,
    iree_arena_allocator_t* arena) {
  uint32_t* symbol_indices = NULL;
  loom_native_object_fixup_t* fixups = NULL;
  iree_io_stream_t* stream = NULL;
  iree_status_t status = iree_arena_allocate_array(
      arena, module_abi->symbol_count, sizeof(*symbol_indices),
      (void**)&symbol_indices);
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < module_abi->symbol_count; ++i) {
      symbol_indices[i] = (uint32_t)i;
    }
    status = iree_arena_allocate_array(arena, function->symbol_fixup_count,
                                       sizeof(*fixups), (void**)&fixups);
  }
  if (iree_status_is_ok(status)) {
    status = iree_io_vec_stream_create(IREE_IO_STREAM_MODE_WRITABLE |
                                           IREE_IO_STREAM_MODE_SEEKABLE |
                                           IREE_IO_STREAM_MODE_RESIZABLE,
                                       256, iree_allocator_system(), &stream);
  }
  if (iree_status_is_ok(status)) {
    status = loom_x86_function_write(function, symbol_indices, 0, fixups,
                                     stream, arena, NULL);
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(builder, "code:");
  }
  if (iree_status_is_ok(status)) {
    status = iree_io_vec_stream_enumerate_blocks(
        stream, loom_x86_loom_check_append_native_bytes, builder);
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(builder, "\n");
  }
  iree_io_stream_release(stream);
  return status;
}

static iree_status_t loom_x86_loom_check_prepare_abis(
    const loom_check_emit_provider_request_t* request, loom_module_t* module,
    const loom_target_low_descriptor_registry_t* low_registry,
    loom_x86_module_abi_t* out_module_abi) {
  iree_host_size_t function_count = 0;
  for (iree_host_size_t i = 0; i < module->symbols.count; ++i) {
    const loom_op_t* op = module->symbols.entries[i].defining_op;
    function_count +=
        op != NULL && (loom_low_func_def_isa(op) || loom_low_func_decl_isa(op));
  }
  IREE_RETURN_IF_ERROR(
      loom_x86_module_abi_initialize(module->symbols.count, function_count,
                                     request->case_arena, out_module_abi));
  loom_symbol_fact_table_t symbol_facts = {0};
  loom_symbol_fact_table_initialize(&symbol_facts, request->case_arena);
  iree_host_size_t function_index = 0;
  for (loom_symbol_id_t symbol_id = 0; symbol_id < module->symbols.count;
       ++symbol_id) {
    loom_op_t* op = module->symbols.entries[symbol_id].defining_op;
    if (op == NULL ||
        (!loom_low_func_def_isa(op) && !loom_low_func_decl_isa(op))) {
      continue;
    }
    loom_low_resolved_target_t target = {0};
    IREE_RETURN_IF_ERROR(loom_low_resolve_function_target(
        module, &symbol_facts, op, NULL, &low_registry->registry,
        (iree_diagnostic_emitter_t){0}, &target));
    if (target.descriptor_set == NULL) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "x86 native function has no resolved target");
    }
    bool supported = false;
    iree_string_view_t constraint = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_x86_function_abi_prepare(
        module, loom_func_like_cast(module, op), &target, request->case_arena,
        &supported, &constraint, &out_module_abi->functions[function_index]));
    if (!supported) {
      return iree_status_allocate_copy(IREE_STATUS_UNIMPLEMENTED,
                                       iree_make_cstring_view(__FILE__),
                                       __LINE__, constraint);
    }
    loom_x86_module_abi_bind(out_module_abi, function_index++, symbol_id);
  }
  return iree_ok_status();
}

static iree_status_t loom_x86_loom_check_emit_provider_execute(
    const loom_check_emit_provider_t* provider,
    const loom_check_emit_provider_request_t* request,
    const loom_check_test_native_module_t* native_module) {
  (void)provider;
  iree_string_view_t function_symbol_name =
      iree_string_view_trim(request->target_options);
  if (!iree_string_view_consume_prefix(&function_symbol_name, IREE_SV("@")) ||
      iree_string_view_is_empty(function_symbol_name) ||
      iree_string_view_find_first_of(function_symbol_name, IREE_SV(" \t\r\n"),
                                     0) != IREE_STRING_VIEW_NPOS) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "x86-native-bytes requires a single low function symbol");
  }

  loom_x86_module_abi_t module_abi = {0};
  IREE_RETURN_IF_ERROR(loom_x86_loom_check_prepare_abis(
      request, native_module->module, &native_module->low_registry,
      &module_abi));
  const loom_string_id_t function_name_id =
      loom_module_lookup_string(native_module->module, function_symbol_name);
  const loom_symbol_id_t function_symbol_id =
      function_name_id == LOOM_STRING_ID_INVALID
          ? LOOM_SYMBOL_ID_INVALID
          : loom_module_find_symbol(native_module->module, function_name_id);
  const loom_x86_function_abi_t* function_abi =
      function_symbol_id == LOOM_SYMBOL_ID_INVALID
          ? NULL
          : loom_x86_module_abi_lookup(
                &module_abi,
                (loom_symbol_ref_t){.symbol_id = function_symbol_id});

  loom_low_emission_frame_t frame = {0};
  bool frame_accepted = false;
  const loom_low_emission_frame_spill_free_options_t spill_free_options = {0};
  // Reserve RSP just as the object provider does before allocation; the native
  // envelope uses it. Incoming and outgoing locations follow the normal ABI.
  loom_low_allocation_reserved_range_t reserved_ranges[2] = {
      {{0}},
  };
  const iree_host_size_t reserved_range_count =
      function_abi != NULL
          ? loom_x86_function_reserved_ranges(function_abi, reserved_ranges)
          : 0;
  const loom_low_emission_frame_options_t frame_options = {
      .resolved_target = function_abi != NULL ? &function_abi->target : NULL,
      .call_contracts =
          {
              .validate = loom_x86_function_call_contract_validate,
              .query = loom_x86_function_call_contract,
              .query_common_clobbers = loom_x86_function_common_call_clobbers,
              .user_data = &module_abi,
          },
      .schedule_strategy = LOOM_LOW_SCHEDULE_STRATEGY_SOURCE_PRIORITY,
      .synchronous_storage_spaces = LOOM_LOW_STORAGE_SPACE_SET_STACK |
                                    LOOM_LOW_STORAGE_SPACE_SET_PRIVATE |
                                    LOOM_LOW_STORAGE_SPACE_SET_SCRATCH,
      .allocation_entry_locations =
          function_abi != NULL ? function_abi->call_contract.arguments : NULL,
      .allocation_entry_location_count =
          function_abi != NULL ? function_abi->call_contract.argument_count : 0,
      .allocation_exit_locations =
          function_abi != NULL ? function_abi->call_contract.results : NULL,
      .allocation_exit_location_count =
          function_abi != NULL ? function_abi->call_contract.result_count : 0,
      .allocation_reserved_ranges = reserved_ranges,
      .allocation_reserved_range_count = reserved_range_count,
  };
  IREE_RETURN_IF_ERROR(loom_check_low_emit_packetize_function(
      request, native_module, function_symbol_name, &frame_options,
      &spill_free_options, &frame, &frame_accepted));
  if (request->diagnostic_collector != NULL &&
      loom_check_diagnostic_collector_has_error(
          request->diagnostic_collector)) {
    return iree_ok_status();
  }
  if (!frame_accepted) {
    return iree_ok_status();
  }
  loom_check_test_native_diagnostic_emitter_capture_t capture = {
      .context = native_module->diagnostic_context,
      .module = native_module->module,
      .source_resolver = native_module->source_resolver,
  };
  IREE_ASSERT_NE(function_abi, NULL);
  loom_x86_function_t function;
  bool accepted = false;
  IREE_RETURN_IF_ERROR(loom_x86_function_prepare(
      &frame, function_abi, &module_abi,
      (iree_diagnostic_emitter_t){
          .fn = loom_check_test_native_diagnostic_emitter_capture_emit,
          .user_data = &capture,
      },
      request->case_arena, &accepted, &function));
  if (!accepted) {
    return iree_ok_status();
  }
  return loom_x86_loom_check_emit_native_bytes(&function, &module_abi,
                                               &request->result->actual_output,
                                               request->case_arena);
}

static iree_status_t loom_x86_loom_check_emit_provider_append_names(
    const loom_check_emit_provider_t* provider,
    iree_string_builder_t* builder) {
  (void)provider;
  return iree_string_builder_append_cstring(builder, "x86-native-bytes");
}

const loom_check_test_native_emit_provider_t
    loom_x86_native_loom_check_emit_provider = {
        .base =
            {
                .name = IREE_SVL("x86-native"),
                .match = loom_x86_loom_check_emit_provider_matches,
                .execute = loom_check_test_native_emit_provider_execute,
                .append_names = loom_x86_loom_check_emit_provider_append_names,
            },
        .execute_native = loom_x86_loom_check_emit_provider_execute,
};

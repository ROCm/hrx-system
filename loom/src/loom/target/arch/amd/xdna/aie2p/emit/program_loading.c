// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/emit/program_loading.h"

#include <string.h>

#include "loom/codegen/low/builder.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/amd/xdna/aie2p/descriptors/core_descriptors.h"

typedef struct loom_aie2p_program_loading_builder_t {
  // Ordinary mutable Low insertion point.
  loom_builder_t builder;
  // Static descriptor vocabulary for native scalar/control instructions.
  const loom_low_descriptor_set_t* descriptors;
  // Scalar carrier for queue words, selectors, and completion status.
  loom_type_t scalar;
  // Module-interned integer immediate attribute.
  loom_string_id_t integer_name;
} loom_aie2p_program_loading_builder_t;

static iree_status_t loom_aie2p_program_loading_initialize(
    loom_module_t* module, loom_aie2p_program_loading_builder_t* out_builder) {
  *out_builder = (loom_aie2p_program_loading_builder_t){
      .descriptors = loom_aie2p_core_descriptor_set(),
  };
  loom_builder_initialize(module, &module->arena, loom_module_block(module),
                          &out_builder->builder);
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      out_builder->descriptors, AIE2P_CORE_REG_CLASS_ID_AIE2P_ER, 1,
      &out_builder->scalar));
  return loom_module_intern_string(module, IREE_SV("i"),
                                   &out_builder->integer_name);
}

static iree_status_t loom_aie2p_program_loading_constant(
    loom_aie2p_program_loading_builder_t* code, uint32_t value,
    loom_value_id_t* out_value) {
  const uint32_t descriptor = (int32_t)value >= -1024 && (int32_t)value <= 1023
                                  ? AIE2P_CORE_DESCRIPTOR_REF_CONSTANT_I32_SHORT
                                  : AIE2P_CORE_DESCRIPTOR_REF_CONSTANT_I32;
  const loom_named_attr_t attribute = {.name_id = code->integer_name,
                                       .value = loom_attr_i64((int32_t)value)};
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_const(
      &code->builder, code->descriptors,
      &code->descriptors->descriptors[descriptor],
      loom_make_named_attr_slice(&attribute, 1), code->scalar,
      LOOM_LOCATION_UNKNOWN, &op));
  *out_value = loom_low_const_result(op);
  return iree_ok_status();
}

static iree_status_t loom_aie2p_program_loading_op(
    loom_aie2p_program_loading_builder_t* code, uint32_t descriptor,
    const loom_value_id_t* operands, iree_host_size_t count,
    loom_value_id_t* out_value) {
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_op(
      &code->builder, code->descriptors,
      &code->descriptors->descriptors[descriptor], 0, operands, count,
      loom_named_attr_slice_empty(), out_value ? &code->scalar : NULL,
      out_value ? 1 : 0, NULL, 0, LOOM_LOCATION_UNKNOWN, &op));
  if (out_value) {
    *out_value = loom_op_results(op)[0];
  }
  return iree_ok_status();
}

iree_status_t loom_aie2p_program_loading_build(loom_module_t* module,
                                               uint32_t queue_header,
                                               uint32_t queue_command,
                                               loom_symbol_ref_t* out_helper) {
  loom_aie2p_program_loading_builder_t code;
  IREE_RETURN_IF_ERROR(loom_aie2p_program_loading_initialize(module, &code));
  char name[48];
  uint32_t ordinal = 0;
  do {
    iree_snprintf(name, sizeof(name), "__aie2p_code_load_%u", ordinal++);
  } while (loom_module_lookup_string(module, iree_make_cstring_view(name)) !=
           LOOM_STRING_ID_INVALID);
  loom_string_id_t name_id, contract;
  IREE_RETURN_IF_ERROR(loom_module_intern_string(
      module, iree_make_cstring_view(name), &name_id));
  IREE_RETURN_IF_ERROR(loom_module_intern_string(
      module, IREE_SV("amd.xdna.aie2p.core"), &contract));
  *out_helper = (loom_symbol_ref_t){.module_id = 0};
  IREE_RETURN_IF_ERROR(
      loom_module_add_symbol(module, name_id, &out_helper->symbol_id));
  loom_op_t* function = NULL;
  IREE_RETURN_IF_ERROR(loom_low_func_def_build(
      &code.builder, 0, 0, 0, 0, 0, 0, 0, 0, 0, contract,
      loom_symbol_ref_null(), 0, loom_named_attr_slice_empty(),
      loom_named_attr_slice_empty(), LOOM_STRING_ID_INVALID,
      loom_named_attr_slice_empty(), *out_helper, &code.scalar, 1, NULL, 0,
      NULL, 0, NULL, 0, LOOM_LOCATION_UNKNOWN, &function));
  loom_region_t* region = loom_low_func_def_body(function);
  loom_builder_enter_region(&code.builder, function, region);
  const loom_value_id_t descriptor =
      loom_region_entry_block(region)->arg_ids[0];
  const uint32_t headers[] = {queue_header, queue_command};
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(headers); ++i) {
    loom_value_id_t word;
    IREE_RETURN_IF_ERROR(
        loom_aie2p_program_loading_constant(&code, headers[i], &word));
    IREE_RETURN_IF_ERROR(loom_aie2p_program_loading_op(
        &code, AIE2P_CORE_DESCRIPTOR_REF_STREAM_WRITE_I32, &word, 1, NULL));
  }
  IREE_RETURN_IF_ERROR(loom_aie2p_program_loading_op(
      &code, AIE2P_CORE_DESCRIPTOR_REF_STREAM_WRITE_LAST_I32, &descriptor, 1,
      NULL));
  loom_value_id_t header, status, header_mask, status_mask, header_error,
      status_error;
  IREE_RETURN_IF_ERROR(loom_aie2p_program_loading_op(
      &code, AIE2P_CORE_DESCRIPTOR_REF_STREAM_READ_I32, NULL, 0, &header));
  IREE_RETURN_IF_ERROR(loom_aie2p_program_loading_op(
      &code, AIE2P_CORE_DESCRIPTOR_REF_STREAM_READ_I32, NULL, 0, &status));
  IREE_RETURN_IF_ERROR(
      loom_aie2p_program_loading_constant(&code, 0x7000u, &header_mask));
  IREE_RETURN_IF_ERROR(
      loom_aie2p_program_loading_constant(&code, 15u, &status_mask));
  const loom_value_id_t header_inputs[] = {header, header_mask};
  const loom_value_id_t status_inputs[] = {status, status_mask};
  IREE_RETURN_IF_ERROR(
      loom_aie2p_program_loading_op(&code, AIE2P_CORE_DESCRIPTOR_REF_AND_I32,
                                    header_inputs, 2, &header_error));
  IREE_RETURN_IF_ERROR(
      loom_aie2p_program_loading_op(&code, AIE2P_CORE_DESCRIPTOR_REF_AND_I32,
                                    status_inputs, 2, &status_error));
  const loom_value_id_t errors[] = {header_error, status_error};
  loom_value_id_t error;
  IREE_RETURN_IF_ERROR(loom_aie2p_program_loading_op(
      &code, AIE2P_CORE_DESCRIPTOR_REF_OR_I32, errors, 2, &error));
  loom_block_t* failed = NULL;
  loom_block_t* complete = NULL;
  IREE_RETURN_IF_ERROR(loom_region_append_block(module, region, &failed));
  IREE_RETURN_IF_ERROR(loom_region_append_block(module, region, &complete));
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_cond_br_build(
      &code.builder, error, failed, complete, LOOM_LOCATION_UNKNOWN, &op));
  loom_builder_set_block(&code.builder, failed);
  // A failed control transfer cannot authorize instruction fetch or normal
  // completion. Remain in resident code until context error recovery retires
  // this invocation, as for a failed payload transfer.
  loom_value_id_t discarded;
  IREE_RETURN_IF_ERROR(loom_aie2p_program_loading_op(
      &code, AIE2P_CORE_DESCRIPTOR_REF_STREAM_READ_I32, NULL, 0, &discarded));
  IREE_RETURN_IF_ERROR(loom_low_br_build(&code.builder, failed, NULL, 0,
                                         LOOM_LOCATION_UNKNOWN, &op));
  loom_builder_set_block(&code.builder, complete);
  return loom_low_return_build(&code.builder, NULL, 0, LOOM_LOCATION_UNKNOWN,
                               &op);
}

iree_status_t loom_aie2p_program_loading_insert(
    loom_module_t* module, loom_symbol_ref_t helper,
    const uint32_t* descriptors,
    loom_aie2p_program_clone_function_t* function) {
  const loom_low_call_boundary_index_t original = function->call_boundaries;
  iree_host_size_t added_count = 0;
  for (iree_host_size_t i = 0; i < original.count; ++i) {
    added_count += descriptors[i] != UINT32_MAX;
  }
  if (added_count == 0) {
    return iree_ok_status();
  }
  const loom_op_t** calls = NULL;
  loom_symbol_ref_t* callees = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(&module->arena, original.count + added_count,
                                sizeof(*calls), (void**)&calls));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      &module->arena, function->direct_callees.count + 1, sizeof(*callees),
      (void**)&callees));
  if (function->direct_callees.count != 0) {
    memcpy(callees, function->direct_callees.values,
           function->direct_callees.count * sizeof(*callees));
  }
  callees[function->direct_callees.count] = helper;
  loom_aie2p_program_loading_builder_t code;
  IREE_RETURN_IF_ERROR(loom_aie2p_program_loading_initialize(module, &code));
  iree_host_size_t cursor = 0;
  for (iree_host_size_t i = 0; i < original.count; ++i) {
    if (descriptors[i] != UINT32_MAX) {
      loom_builder_set_before(&code.builder, (loom_op_t*)original.calls[i]);
      loom_value_id_t descriptor;
      IREE_RETURN_IF_ERROR(loom_aie2p_program_loading_constant(
          &code, descriptors[i], &descriptor));
      loom_op_t* call = NULL;
      IREE_RETURN_IF_ERROR(loom_low_func_call_build(
          &code.builder, 0, 0, 0, 0, helper, &descriptor, 1, NULL, 0, NULL, 0,
          original.calls[i]->location, &call));
      calls[cursor++] = call;
    }
    calls[cursor++] = original.calls[i];
  }
  function->direct_callees =
      loom_make_symbol_ref_array(callees, function->direct_callees.count + 1);
  function->call_boundaries =
      (loom_low_call_boundary_index_t){.calls = calls, .count = cursor};
  return iree_ok_status();
}

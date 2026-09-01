// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/spirv/function_emitter.h"

#include <inttypes.h>

#include "loom/codegen/low/function.h"
#include "loom/ir/context.h"
#include "loom/ir/local_value_domain.h"
#include "loom/ir/module.h"
#include "loom/ir/types.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/spirv/descriptors/descriptors.h"
#include "loom/target/arch/spirv/packet_rows.h"
#include "loom/target/emit/spirv/binary_format.h"
#include "loom/target/emit/spirv/function_control.h"
#include "loom/target/emit/spirv/module_abi.h"
#include "loom/target/emit/spirv/module_instructions.h"
#include "loom/target/emit/spirv/module_storage.h"
#include "loom/target/emit/spirv/module_types.h"
#include "loom/target/emit/spirv/module_values.h"
#include "loom/target/types.h"

iree_status_t loom_spirv_emit_label_id(loom_spirv_emit_state_t* state,
                                       uint32_t label_id) {
  const uint32_t operands[] = {label_id};
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      LOOM_SPIRV_OP_LABEL, operands, IREE_ARRAYSIZE(operands)));
  state->current_label_id = label_id;
  return iree_ok_status();
}

static iree_string_view_t loom_spirv_emit_string_or_empty(
    const loom_module_t* module, loom_string_id_t string_id) {
  if (string_id == LOOM_STRING_ID_INVALID ||
      string_id >= module->strings.count) {
    return iree_string_view_empty();
  }
  return module->strings.entries[string_id];
}

static iree_string_view_t loom_spirv_emit_symbol_name(
    const loom_module_t* module, loom_symbol_ref_t symbol_ref) {
  if (!loom_symbol_ref_is_valid(symbol_ref) || symbol_ref.module_id != 0 ||
      symbol_ref.symbol_id >= module->symbols.count) {
    return IREE_SV("<unnamed>");
  }
  const loom_symbol_t* symbol = &module->symbols.entries[symbol_ref.symbol_id];
  return loom_spirv_emit_string_or_empty(module, symbol->name_id);
}

static iree_string_view_t loom_spirv_emit_function_name(
    const loom_spirv_emit_state_t* state) {
  return loom_spirv_emit_symbol_name(
      state->module, loom_low_function_callee(state->function_op));
}

static iree_string_view_t loom_spirv_emit_export_name(
    const loom_spirv_emit_state_t* state) {
  const iree_string_view_t export_symbol =
      loom_low_resolved_target_bundle(state->target)
          ->export_plan->export_symbol;
  if (!iree_string_view_is_empty(export_symbol)) {
    return export_symbol;
  }
  return loom_spirv_emit_function_name(state);
}

static iree_status_t loom_spirv_emit_op_name(loom_spirv_emit_state_t* state,
                                             uint32_t id,
                                             iree_string_view_t name) {
  if (id == 0 || iree_string_view_is_empty(name)) {
    return iree_ok_status();
  }
  const uint32_t prefix_operands[] = {id};
  return loom_spirv_binary_write_string_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_DEBUG),
      LOOM_SPIRV_OP_NAME, prefix_operands, IREE_ARRAYSIZE(prefix_operands),
      name, NULL, 0);
}

static iree_status_t loom_spirv_emit_value_name(loom_spirv_emit_state_t* state,
                                                loom_value_id_t value_id,
                                                uint32_t id) {
  if (value_id >= state->module->values.count) {
    return iree_ok_status();
  }
  const loom_value_t* value = loom_module_value(state->module, value_id);
  return loom_spirv_emit_op_name(
      state, id,
      loom_spirv_emit_string_or_empty(state->module, value->name_id));
}

static iree_status_t loom_spirv_emit_prepare_immediate_name_ids(
    loom_spirv_emit_state_t* state) {
  const iree_host_size_t immediate_count =
      state->target->descriptor_set->immediate_count;
  if (immediate_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->scratch_arena, immediate_count, sizeof(*state->immediate_name_ids),
      (void**)&state->immediate_name_ids));
  state->immediate_name_id_count = immediate_count;
  for (iree_host_size_t i = 0; i < immediate_count; ++i) {
    const loom_low_immediate_t* immediate =
        &state->target->descriptor_set->immediates[i];
    const iree_string_view_t name = loom_low_descriptor_set_string(
        state->target->descriptor_set, immediate->field_name_string_offset);
    state->immediate_name_ids[i] =
        loom_module_lookup_string(state->module, name);
  }
  return iree_ok_status();
}

iree_status_t loom_spirv_emit_define_value(
    loom_spirv_emit_state_t* state, loom_value_id_t value_id,
    loom_spirv_module_value_ref_t value_ref, bool emit_name) {
  loom_spirv_module_value_table_define(&state->value_table, value_id,
                                       value_ref);
  if (emit_name) {
    IREE_RETURN_IF_ERROR(
        loom_spirv_emit_value_name(state, value_id, value_ref.id));
  }
  return iree_ok_status();
}

iree_status_t loom_spirv_emit_reserve_value_ref(
    loom_spirv_emit_state_t* state, loom_value_id_t value_id, uint32_t type_id,
    loom_spirv_value_type_t value_type, uint32_t* out_result_id) {
  *out_result_id = loom_spirv_module_value_table_reserve(
      &state->value_table, state->builder, value_id, type_id, value_type);
  return iree_ok_status();
}

static iree_status_t loom_spirv_emit_prepare_packet_result(
    loom_spirv_emit_state_t* state, const loom_low_descriptor_packet_t* packet,
    uint32_t type_id, loom_spirv_value_type_t value_type,
    uint32_t* out_result_id) {
  return loom_spirv_emit_reserve_value_ref(state,
                                           loom_op_const_results(packet->op)[0],
                                           type_id, value_type, out_result_id);
}

static iree_status_t loom_spirv_emit_define_packet_result(
    loom_spirv_emit_state_t* state, const loom_low_descriptor_packet_t* packet,
    uint32_t id, uint32_t type_id, loom_spirv_value_type_t value_type) {
  const loom_spirv_module_value_ref_t value_ref = {
      .id = id, .type_id = type_id, .value_type = value_type};
  return loom_spirv_emit_define_value(
      state, loom_op_const_results(packet->op)[0], value_ref, true);
}

iree_status_t loom_spirv_emit_lookup_value(
    loom_spirv_emit_state_t* state, loom_value_id_t value_id,
    loom_spirv_module_value_ref_t* out_value_ref) {
  *out_value_ref =
      loom_spirv_module_value_table_lookup(&state->value_table, value_id);
  return iree_ok_status();
}

static loom_spirv_module_abi_context_t loom_spirv_emit_abi_context(
    loom_spirv_emit_state_t* state) {
  return (loom_spirv_module_abi_context_t){
      .module = state->module,
      .function_op = state->function_op,
      .target = state->target,
      .scratch_arena = state->scratch_arena,
      .builder = state->builder,
      .type_context = state->type_context,
      .raw_bda_layout = state->context->raw_bda_layout,
      .value_table = &state->value_table,
  };
}

static bool loom_spirv_emit_builtin_variable_info(
    uint32_t builtin, uint8_t* out_slot, iree_string_view_t* out_name) {
  switch (builtin) {
    case LOOM_SPIRV_BUILT_IN_WORKGROUP_ID:
      *out_slot = 0;
      *out_name = IREE_SV("workgroup_id");
      return true;
    case LOOM_SPIRV_BUILT_IN_LOCAL_INVOCATION_ID:
      *out_slot = 1;
      *out_name = IREE_SV("local_invocation_id");
      return true;
    case LOOM_SPIRV_BUILT_IN_GLOBAL_INVOCATION_ID:
      *out_slot = 2;
      *out_name = IREE_SV("global_invocation_id");
      return true;
  }
  return false;
}

static iree_status_t loom_spirv_emit_builtin_variable(
    loom_spirv_emit_state_t* state, uint32_t builtin,
    uint32_t* out_variable_id) {
  uint8_t slot = 0;
  iree_string_view_t name = iree_string_view_empty();
  if (!loom_spirv_emit_builtin_variable_info(builtin, &slot, &name)) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "unknown SPIR-V builtin packet row %" PRIu32,
                            builtin);
  }
  uint32_t variable_id = state->context->builtin_variable_ids[slot];
  if (variable_id != 0) {
    state->builtin_interface_variable_ids[slot] = variable_id;
    *out_variable_id = variable_id;
    return iree_ok_status();
  }

  uint32_t u32_type_id = 0;
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_type_u32(state->type_context, &u32_type_id));
  uint32_t vector_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_vector(
      state->type_context, u32_type_id, 3, &vector_type_id));
  uint32_t pointer_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_pointer(
      state->type_context, LOOM_SPIRV_STORAGE_CLASS_INPUT, vector_type_id,
      /*pointer_array_stride=*/0, &pointer_type_id));

  variable_id = loom_spirv_emit_allocate_id(state);
  const uint32_t decoration_operands[] = {
      variable_id,
      LOOM_SPIRV_DECORATION_BUILT_IN,
      builtin,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_ANNOTATION),
      LOOM_SPIRV_OP_DECORATE, decoration_operands,
      IREE_ARRAYSIZE(decoration_operands)));
  const uint32_t variable_operands[] = {
      pointer_type_id,
      variable_id,
      LOOM_SPIRV_STORAGE_CLASS_INPUT,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_DECLARATION),
      LOOM_SPIRV_OP_VARIABLE, variable_operands,
      IREE_ARRAYSIZE(variable_operands)));
  IREE_RETURN_IF_ERROR(loom_spirv_emit_op_name(state, variable_id, name));
  state->context->builtin_variable_ids[slot] = variable_id;
  state->builtin_interface_variable_ids[slot] = variable_id;
  *out_variable_id = variable_id;
  return iree_ok_status();
}

static loom_named_attr_slice_t loom_spirv_emit_packet_attrs(
    const loom_low_descriptor_packet_t* packet) {
  switch (packet->kind) {
    case LOOM_LOW_DESCRIPTOR_PACKET_CONST:
      return loom_low_const_attrs(packet->op);
    case LOOM_LOW_DESCRIPTOR_PACKET_OP:
      return loom_low_op_attrs(packet->op);
    case LOOM_LOW_DESCRIPTOR_PACKET_NONE:
      break;
  }
  return loom_make_named_attr_slice(NULL, 0);
}

static const loom_low_immediate_t* loom_spirv_emit_descriptor_immediate(
    loom_spirv_emit_state_t* state, const loom_low_descriptor_packet_t* packet,
    uint8_t descriptor_immediate_index) {
  if (descriptor_immediate_index >= packet->descriptor->immediate_count) {
    return NULL;
  }
  const uint32_t immediate_row =
      packet->descriptor->immediate_start + descriptor_immediate_index;
  if (immediate_row >= state->target->descriptor_set->immediate_count) {
    return NULL;
  }
  return &state->target->descriptor_set->immediates[immediate_row];
}

static iree_string_view_t loom_spirv_emit_immediate_name(
    loom_spirv_emit_state_t* state, const loom_low_immediate_t* immediate) {
  if (immediate == NULL) {
    return IREE_SV("<unknown>");
  }
  return loom_low_descriptor_set_string(state->target->descriptor_set,
                                        immediate->field_name_string_offset);
}

static iree_status_t loom_spirv_emit_lookup_i64_immediate(
    loom_spirv_emit_state_t* state, const loom_low_descriptor_packet_t* packet,
    const loom_spirv_packet_row_t* row, int64_t* out_value) {
  const loom_low_immediate_t* immediate =
      loom_spirv_emit_descriptor_immediate(state, packet, row->immediate_index);
  IREE_ASSERT(immediate != NULL);
  const uint32_t immediate_row =
      packet->descriptor->immediate_start + row->immediate_index;
  IREE_ASSERT_LT(immediate_row, state->immediate_name_id_count);
  const loom_string_id_t expected_name_id =
      state->immediate_name_ids[immediate_row];
  const loom_named_attr_slice_t attrs = loom_spirv_emit_packet_attrs(packet);
  for (iree_host_size_t i = 0; i < attrs.count; ++i) {
    const loom_named_attr_t* attr = &attrs.entries[i];
    if (attr->name_id != expected_name_id) {
      continue;
    }
    IREE_ASSERT_EQ(attr->value.kind, LOOM_ATTR_I64);
    *out_value = loom_attr_as_i64(attr->value);
    return iree_ok_status();
  }
  if (iree_any_bit_set(immediate->flags,
                       LOOM_LOW_IMMEDIATE_FLAG_DEFAULT_VALUE)) {
    *out_value = immediate->default_value;
    return iree_ok_status();
  }
  IREE_CHECK_UNREACHABLE("verified low descriptor immediate attribute");
  return iree_ok_status();
}

static void loom_spirv_emit_validate_packet_shape(
    const loom_op_t* op, const loom_low_descriptor_packet_t* packet,
    const loom_spirv_packet_row_t* row) {
  (void)packet;
  IREE_ASSERT_EQ(op->result_count, row->result_count);
  IREE_ASSERT_EQ(op->operand_count, row->operand_count);
}

static iree_status_t loom_spirv_emit_load_packet_operands(
    loom_spirv_emit_state_t* state, const loom_low_descriptor_packet_t* packet,
    const loom_spirv_packet_row_t* row,
    loom_spirv_module_value_ref_t* out_operands) {
  const loom_value_id_t* operand_values = loom_op_const_operands(packet->op);
  for (uint8_t i = 0; i < row->operand_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_spirv_emit_lookup_value(state, operand_values[i],
                                                      &out_operands[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_spirv_emit_scalar_constant_packet(
    loom_spirv_emit_state_t* state, const loom_low_descriptor_packet_t* packet,
    const loom_spirv_packet_row_t* row) {
  int64_t value = 0;
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_lookup_i64_immediate(state, packet, row, &value));
  uint32_t type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_id_for_value_type(
      state->type_context, loom_spirv_packet_row_result_type(row), &type_id));
  uint32_t result_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_prepare_packet_result(
      state, packet, type_id, loom_spirv_packet_row_result_type(row),
      &result_id));
  const uint64_t literal = (uint64_t)value;
  uint32_t operands[] = {
      type_id,
      result_id,
      (uint32_t)literal,
      (uint32_t)(literal >> 32),
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_DECLARATION),
      row->opcode, operands,
      2 + row->payload.scalar_constant.literal_word_count));
  return loom_spirv_emit_define_packet_result(
      state, packet, result_id, type_id,
      loom_spirv_packet_row_result_type(row));
}

static iree_status_t loom_spirv_emit_boolean_constant_packet(
    loom_spirv_emit_state_t* state, const loom_low_descriptor_packet_t* packet,
    const loom_spirv_packet_row_t* row) {
  uint32_t type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_id_for_value_type(
      state->type_context, loom_spirv_packet_row_result_type(row), &type_id));
  uint32_t result_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_prepare_packet_result(
      state, packet, type_id, loom_spirv_packet_row_result_type(row),
      &result_id));
  const uint32_t operands[] = {
      type_id,
      result_id,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_DECLARATION),
      row->opcode, operands, IREE_ARRAYSIZE(operands)));
  return loom_spirv_emit_define_packet_result(
      state, packet, result_id, type_id,
      loom_spirv_packet_row_result_type(row));
}

static iree_status_t loom_spirv_emit_binary_same_type_packet(
    loom_spirv_emit_state_t* state, const loom_low_descriptor_packet_t* packet,
    const loom_spirv_packet_row_t* row) {
  loom_spirv_module_value_ref_t operands[2] = {0};
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_load_packet_operands(state, packet, row, operands));
  IREE_ASSERT_EQ(operands[0].type_id, operands[1].type_id);
  uint32_t result_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_prepare_packet_result(
      state, packet, operands[0].type_id,
      loom_spirv_packet_row_result_type(row), &result_id));
  const uint32_t instruction_operands[] = {
      operands[0].type_id,
      result_id,
      operands[0].id,
      operands[1].id,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      row->opcode, instruction_operands, IREE_ARRAYSIZE(instruction_operands)));
  return loom_spirv_emit_define_packet_result(
      state, packet, result_id, operands[0].type_id,
      loom_spirv_packet_row_result_type(row));
}

static iree_status_t loom_spirv_emit_unary_typed_packet(
    loom_spirv_emit_state_t* state, const loom_low_descriptor_packet_t* packet,
    const loom_spirv_packet_row_t* row) {
  loom_spirv_module_value_ref_t operands[1] = {0};
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_load_packet_operands(state, packet, row, operands));
  uint32_t result_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_id_for_value_type(
      state->type_context, loom_spirv_packet_row_result_type(row),
      &result_type_id));
  uint32_t result_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_prepare_packet_result(
      state, packet, result_type_id, loom_spirv_packet_row_result_type(row),
      &result_id));
  const uint32_t instruction_operands[] = {
      result_type_id,
      result_id,
      operands[0].id,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      row->opcode, instruction_operands, IREE_ARRAYSIZE(instruction_operands)));
  return loom_spirv_emit_define_packet_result(
      state, packet, result_id, result_type_id,
      loom_spirv_packet_row_result_type(row));
}

static iree_status_t loom_spirv_emit_composite_construct_packet(
    loom_spirv_emit_state_t* state, const loom_low_descriptor_packet_t* packet,
    const loom_spirv_packet_row_t* row) {
  loom_spirv_module_value_ref_t operands[LOOM_SPIRV_PACKET_MAX_OPERAND_COUNT] =
      {0};
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_load_packet_operands(state, packet, row, operands));
  uint32_t result_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_id_for_value_type(
      state->type_context, loom_spirv_packet_row_result_type(row),
      &result_type_id));
  uint32_t result_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_prepare_packet_result(
      state, packet, result_type_id, loom_spirv_packet_row_result_type(row),
      &result_id));
  uint32_t instruction_operands[2 + LOOM_SPIRV_PACKET_MAX_OPERAND_COUNT] = {
      result_type_id,
      result_id,
  };
  for (uint8_t i = 0; i < row->operand_count; ++i) {
    instruction_operands[2 + i] = operands[i].id;
  }
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      row->opcode, instruction_operands, 2 + row->operand_count));
  return loom_spirv_emit_define_packet_result(
      state, packet, result_id, result_type_id,
      loom_spirv_packet_row_result_type(row));
}

static iree_status_t loom_spirv_emit_composite_extract_packet(
    loom_spirv_emit_state_t* state, const loom_low_descriptor_packet_t* packet,
    const loom_spirv_packet_row_t* row) {
  loom_spirv_module_value_ref_t operands[1] = {0};
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_load_packet_operands(state, packet, row, operands));
  int64_t component_index = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_lookup_i64_immediate(state, packet, row,
                                                            &component_index));
  uint32_t result_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_id_for_value_type(
      state->type_context, loom_spirv_packet_row_result_type(row),
      &result_type_id));
  uint32_t result_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_prepare_packet_result(
      state, packet, result_type_id, loom_spirv_packet_row_result_type(row),
      &result_id));
  const uint32_t instruction_operands[] = {
      result_type_id,
      result_id,
      operands[0].id,
      (uint32_t)component_index,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      row->opcode, instruction_operands, IREE_ARRAYSIZE(instruction_operands)));
  return loom_spirv_emit_define_packet_result(
      state, packet, result_id, result_type_id,
      loom_spirv_packet_row_result_type(row));
}

static iree_status_t loom_spirv_emit_composite_insert_packet(
    loom_spirv_emit_state_t* state, const loom_low_descriptor_packet_t* packet,
    const loom_spirv_packet_row_t* row) {
  loom_spirv_module_value_ref_t operands[2] = {0};
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_load_packet_operands(state, packet, row, operands));
  int64_t component_index = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_lookup_i64_immediate(state, packet, row,
                                                            &component_index));
  uint32_t result_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_id_for_value_type(
      state->type_context, loom_spirv_packet_row_result_type(row),
      &result_type_id));
  uint32_t result_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_prepare_packet_result(
      state, packet, result_type_id, loom_spirv_packet_row_result_type(row),
      &result_id));
  const uint32_t instruction_operands[] = {
      result_type_id,
      result_id,
      operands[0].id,
      operands[1].id,
      (uint32_t)component_index,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      row->opcode, instruction_operands, IREE_ARRAYSIZE(instruction_operands)));
  return loom_spirv_emit_define_packet_result(
      state, packet, result_id, result_type_id,
      loom_spirv_packet_row_result_type(row));
}

static iree_status_t loom_spirv_emit_load_builtin_packet(
    loom_spirv_emit_state_t* state, const loom_low_descriptor_packet_t* packet,
    const loom_spirv_packet_row_t* row) {
  uint32_t variable_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_builtin_variable(
      state, row->payload.builtin_load.builtin, &variable_id));
  uint32_t u32_type_id = 0;
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_type_u32(state->type_context, &u32_type_id));
  uint32_t vector_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_vector(
      state->type_context, u32_type_id, 3, &vector_type_id));

  uint32_t vector_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_module_emit_unary_result(
      state->builder, LOOM_SPIRV_OP_LOAD, vector_type_id, variable_id,
      &vector_id));

  uint32_t result_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_id_for_value_type(
      state->type_context, loom_spirv_packet_row_result_type(row),
      &result_type_id));
  uint32_t result_id = 0;
  uint32_t component_id = 0;
  if (result_type_id == u32_type_id) {
    IREE_RETURN_IF_ERROR(loom_spirv_emit_prepare_packet_result(
        state, packet, result_type_id, loom_spirv_packet_row_result_type(row),
        &result_id));
    component_id = result_id;
  } else {
    component_id = loom_spirv_emit_allocate_id(state);
  }
  const uint32_t extract_operands[] = {
      u32_type_id,
      component_id,
      vector_id,
      row->payload.builtin_load.component_index,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      LOOM_SPIRV_OP_COMPOSITE_EXTRACT, extract_operands,
      IREE_ARRAYSIZE(extract_operands)));

  if (result_type_id != u32_type_id) {
    IREE_RETURN_IF_ERROR(loom_spirv_emit_prepare_packet_result(
        state, packet, result_type_id, loom_spirv_packet_row_result_type(row),
        &result_id));
    const uint32_t bitcast_operands[] = {
        result_type_id,
        result_id,
        component_id,
    };
    IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
        loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
        LOOM_SPIRV_OP_BITCAST, bitcast_operands,
        IREE_ARRAYSIZE(bitcast_operands)));
  }
  return loom_spirv_emit_define_packet_result(
      state, packet, result_id, result_type_id,
      loom_spirv_packet_row_result_type(row));
}

static iree_status_t loom_spirv_emit_integer_mul_add_packet(
    loom_spirv_emit_state_t* state, const loom_low_descriptor_packet_t* packet,
    const loom_spirv_packet_row_t* row) {
  loom_spirv_module_value_ref_t operands[3] = {0};
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_load_packet_operands(state, packet, row, operands));
  IREE_ASSERT_EQ(operands[0].type_id, operands[1].type_id);
  IREE_ASSERT_EQ(operands[0].type_id, operands[2].type_id);
  uint32_t product_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_module_emit_binary_result(
      state->builder, LOOM_SPIRV_OP_I_MUL, operands[0].type_id, operands[0].id,
      operands[1].id, &product_id));
  uint32_t result_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_prepare_packet_result(
      state, packet, operands[0].type_id,
      loom_spirv_packet_row_result_type(row), &result_id));
  const uint32_t instruction_operands[] = {
      operands[0].type_id,
      result_id,
      product_id,
      operands[2].id,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      LOOM_SPIRV_OP_I_ADD, instruction_operands,
      IREE_ARRAYSIZE(instruction_operands)));
  return loom_spirv_emit_define_packet_result(
      state, packet, result_id, operands[0].type_id,
      loom_spirv_packet_row_result_type(row));
}

static iree_status_t loom_spirv_emit_compare_same_type_packet(
    loom_spirv_emit_state_t* state, const loom_low_descriptor_packet_t* packet,
    const loom_spirv_packet_row_t* row) {
  loom_spirv_module_value_ref_t operands[2] = {0};
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_load_packet_operands(state, packet, row, operands));
  IREE_ASSERT_EQ(operands[0].type_id, operands[1].type_id);
  uint32_t result_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_id_for_value_type(
      state->type_context, loom_spirv_packet_row_result_type(row),
      &result_type_id));
  uint32_t result_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_prepare_packet_result(
      state, packet, result_type_id, loom_spirv_packet_row_result_type(row),
      &result_id));
  const uint32_t instruction_operands[] = {
      result_type_id,
      result_id,
      operands[0].id,
      operands[1].id,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      row->opcode, instruction_operands, IREE_ARRAYSIZE(instruction_operands)));
  return loom_spirv_emit_define_packet_result(
      state, packet, result_id, result_type_id,
      loom_spirv_packet_row_result_type(row));
}

static iree_status_t loom_spirv_emit_select_packet(
    loom_spirv_emit_state_t* state, const loom_low_descriptor_packet_t* packet,
    const loom_spirv_packet_row_t* row) {
  loom_spirv_module_value_ref_t operands[3] = {0};
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_load_packet_operands(state, packet, row, operands));
  IREE_ASSERT_EQ(operands[1].type_id, operands[2].type_id);
  uint32_t result_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_id_for_value_type(
      state->type_context, loom_spirv_packet_row_result_type(row),
      &result_type_id));
  IREE_ASSERT_EQ(result_type_id, operands[1].type_id);
  uint32_t result_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_prepare_packet_result(
      state, packet, result_type_id, loom_spirv_packet_row_result_type(row),
      &result_id));
  const uint32_t instruction_operands[] = {
      result_type_id, result_id, operands[0].id, operands[1].id, operands[2].id,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      row->opcode, instruction_operands, IREE_ARRAYSIZE(instruction_operands)));
  return loom_spirv_emit_define_packet_result(
      state, packet, result_id, result_type_id,
      loom_spirv_packet_row_result_type(row));
}

static iree_status_t loom_spirv_emit_typed_physical_storage_buffer_pointer(
    loom_spirv_emit_state_t* state, loom_spirv_module_value_ref_t address,
    loom_spirv_value_type_t pointer_type, uint32_t pointer_type_id,
    uint32_t* out_pointer_id) {
  if (address.value_type.value_class ==
      LOOM_SPIRV_VALUE_CLASS_PTR_PHYSICAL_STORAGE_BUFFER) {
    IREE_ASSERT(loom_spirv_value_type_equal(address.value_type, pointer_type));
    *out_pointer_id = address.id;
    return iree_ok_status();
  }
  IREE_ASSERT_EQ(address.value_type.value_class,
                 LOOM_SPIRV_VALUE_CLASS_STORAGE_BUFFER_ADDRESS);
  const uint32_t result_id = loom_spirv_emit_allocate_id(state);
  const uint32_t operands[] = {
      pointer_type_id,
      result_id,
      address.id,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      LOOM_SPIRV_OP_CONVERT_U_TO_PTR, operands, IREE_ARRAYSIZE(operands)));
  *out_pointer_id = result_id;
  return iree_ok_status();
}

static iree_status_t loom_spirv_emit_ptr_access_chain_packet(
    loom_spirv_emit_state_t* state, const loom_low_descriptor_packet_t* packet,
    const loom_spirv_packet_row_t* row) {
  loom_spirv_module_value_ref_t operands[2] = {0};
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_load_packet_operands(state, packet, row, operands));
  uint32_t result_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_id_for_value_type(
      state->type_context, loom_spirv_packet_row_result_type(row),
      &result_type_id));
  uint32_t base_pointer_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_typed_physical_storage_buffer_pointer(
      state, operands[0], loom_spirv_packet_row_result_type(row),
      result_type_id, &base_pointer_id));
  uint32_t result_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_prepare_packet_result(
      state, packet, result_type_id, loom_spirv_packet_row_result_type(row),
      &result_id));
  const uint32_t instruction_operands[] = {
      result_type_id,
      result_id,
      base_pointer_id,
      operands[1].id,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      row->opcode, instruction_operands, IREE_ARRAYSIZE(instruction_operands)));
  return loom_spirv_emit_define_packet_result(
      state, packet, result_id, result_type_id,
      loom_spirv_packet_row_result_type(row));
}

static iree_status_t loom_spirv_emit_access_chain_packet(
    loom_spirv_emit_state_t* state, const loom_low_descriptor_packet_t* packet,
    const loom_spirv_packet_row_t* row) {
  loom_spirv_module_value_ref_t operands[2] = {0};
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_load_packet_operands(state, packet, row, operands));
  uint32_t result_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_id_for_value_type(
      state->type_context, loom_spirv_packet_row_result_type(row),
      &result_type_id));
  uint32_t result_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_prepare_packet_result(
      state, packet, result_type_id, loom_spirv_packet_row_result_type(row),
      &result_id));
  const uint32_t instruction_operands[] = {
      result_type_id,
      result_id,
      operands[0].id,
      operands[1].id,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      row->opcode, instruction_operands, IREE_ARRAYSIZE(instruction_operands)));
  return loom_spirv_emit_define_packet_result(
      state, packet, result_id, result_type_id,
      loom_spirv_packet_row_result_type(row));
}

static iree_status_t loom_spirv_emit_load_aligned_packet(
    loom_spirv_emit_state_t* state, const loom_low_descriptor_packet_t* packet,
    const loom_spirv_packet_row_t* row) {
  loom_spirv_module_value_ref_t operands[1] = {0};
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_load_packet_operands(state, packet, row, operands));
  uint32_t result_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_id_for_value_type(
      state->type_context, loom_spirv_packet_row_result_type(row),
      &result_type_id));
  uint32_t result_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_prepare_packet_result(
      state, packet, result_type_id, loom_spirv_packet_row_result_type(row),
      &result_id));
  const uint32_t instruction_operands[] = {
      result_type_id,        result_id,
      operands[0].id,        LOOM_SPIRV_MEMORY_ACCESS_ALIGNED_MASK,
      row->memory_alignment,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      row->opcode, instruction_operands, IREE_ARRAYSIZE(instruction_operands)));
  return loom_spirv_emit_define_packet_result(
      state, packet, result_id, result_type_id,
      loom_spirv_packet_row_result_type(row));
}

static iree_status_t loom_spirv_emit_store_aligned_packet(
    loom_spirv_emit_state_t* state, const loom_low_descriptor_packet_t* packet,
    const loom_spirv_packet_row_t* row) {
  loom_spirv_module_value_ref_t operands[2] = {0};
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_load_packet_operands(state, packet, row, operands));
  const uint32_t instruction_operands[] = {
      operands[0].id,
      operands[1].id,
      LOOM_SPIRV_MEMORY_ACCESS_ALIGNED_MASK,
      row->memory_alignment,
  };
  return loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      row->opcode, instruction_operands, IREE_ARRAYSIZE(instruction_operands));
}

static iree_status_t loom_spirv_emit_cooperative_matrix_layout_operands(
    loom_spirv_emit_state_t* state, const loom_spirv_packet_row_t* row,
    uint32_t* out_layout_id, uint32_t* out_stride_id) {
  IREE_RETURN_IF_ERROR(loom_spirv_emit_u32_constant(
      state->type_context, row->payload.cooperative_matrix.layout,
      out_layout_id));
  return loom_spirv_emit_u32_constant(state->type_context,
                                      row->payload.cooperative_matrix.stride,
                                      out_stride_id);
}

static iree_status_t loom_spirv_emit_cooperative_matrix_load_packet(
    loom_spirv_emit_state_t* state, const loom_low_descriptor_packet_t* packet,
    const loom_spirv_packet_row_t* row) {
  loom_spirv_module_value_ref_t operands[1] = {0};
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_load_packet_operands(state, packet, row, operands));
  uint32_t result_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_id_for_value_type(
      state->type_context, loom_spirv_packet_row_result_type(row),
      &result_type_id));
  uint32_t layout_id = 0;
  uint32_t stride_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_cooperative_matrix_layout_operands(
      state, row, &layout_id, &stride_id));
  uint32_t result_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_prepare_packet_result(
      state, packet, result_type_id, loom_spirv_packet_row_result_type(row),
      &result_id));
  const uint32_t instruction_operands[] = {
      result_type_id,
      result_id,
      operands[0].id,
      layout_id,
      stride_id,
      LOOM_SPIRV_MEMORY_ACCESS_ALIGNED_MASK,
      row->memory_alignment,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      row->opcode, instruction_operands, IREE_ARRAYSIZE(instruction_operands)));
  return loom_spirv_emit_define_packet_result(
      state, packet, result_id, result_type_id,
      loom_spirv_packet_row_result_type(row));
}

static iree_status_t loom_spirv_emit_cooperative_matrix_store_packet(
    loom_spirv_emit_state_t* state, const loom_low_descriptor_packet_t* packet,
    const loom_spirv_packet_row_t* row) {
  loom_spirv_module_value_ref_t operands[2] = {0};
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_load_packet_operands(state, packet, row, operands));
  uint32_t layout_id = 0;
  uint32_t stride_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_cooperative_matrix_layout_operands(
      state, row, &layout_id, &stride_id));
  const uint32_t instruction_operands[] = {
      operands[0].id,
      operands[1].id,
      layout_id,
      stride_id,
      LOOM_SPIRV_MEMORY_ACCESS_ALIGNED_MASK,
      row->memory_alignment,
  };
  return loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      row->opcode, instruction_operands, IREE_ARRAYSIZE(instruction_operands));
}

static iree_status_t loom_spirv_emit_cooperative_matrix_mul_add_packet(
    loom_spirv_emit_state_t* state, const loom_low_descriptor_packet_t* packet,
    const loom_spirv_packet_row_t* row) {
  loom_spirv_module_value_ref_t operands[3] = {0};
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_load_packet_operands(state, packet, row, operands));
  uint32_t result_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_id_for_value_type(
      state->type_context, loom_spirv_packet_row_result_type(row),
      &result_type_id));
  uint32_t result_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_prepare_packet_result(
      state, packet, result_type_id, loom_spirv_packet_row_result_type(row),
      &result_id));
  uint32_t instruction_operands[6] = {
      result_type_id, result_id, operands[0].id, operands[1].id, operands[2].id,
  };
  uint8_t operand_count = 5;
  if (row->payload.cooperative_matrix.operands != 0) {
    instruction_operands[operand_count++] =
        row->payload.cooperative_matrix.operands;
  }
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      row->opcode, instruction_operands, operand_count));
  return loom_spirv_emit_define_packet_result(
      state, packet, result_id, result_type_id,
      loom_spirv_packet_row_result_type(row));
}

static iree_status_t loom_spirv_emit_control_barrier_packet(
    loom_spirv_emit_state_t* state, const loom_spirv_packet_row_t* row) {
  uint32_t execution_scope_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_u32_constant(
      state->type_context, row->payload.barrier.execution_scope,
      &execution_scope_id));
  uint32_t memory_scope_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_u32_constant(
      state->type_context, row->payload.barrier.memory_scope,
      &memory_scope_id));
  uint32_t memory_semantics_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_u32_constant(
      state->type_context, row->payload.barrier.memory_semantics,
      &memory_semantics_id));
  const uint32_t instruction_operands[] = {
      execution_scope_id,
      memory_scope_id,
      memory_semantics_id,
  };
  return loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      row->opcode, instruction_operands, IREE_ARRAYSIZE(instruction_operands));
}

static iree_status_t loom_spirv_emit_transfer(loom_spirv_emit_state_t* state,
                                              const loom_op_t* op) {
  loom_spirv_module_value_ref_t source = {0};
  IREE_RETURN_IF_ERROR(loom_spirv_emit_lookup_value(
      state, loom_op_const_operands(op)[0], &source));
  return loom_spirv_emit_define_value(state, loom_op_const_results(op)[0],
                                      source, false);
}

static iree_status_t loom_spirv_emit_descriptor_packet(
    loom_spirv_emit_state_t* state, const loom_op_t* op,
    const loom_low_descriptor_packet_t* packet) {
  for (uint16_t i = 0; i < packet->descriptor->feature_mask_word_count; ++i) {
    const uint32_t feature_mask_row =
        packet->descriptor->feature_mask_word_start + i;
    const uint64_t feature_bits =
        state->target->descriptor_set->feature_mask_words[feature_mask_row];
    IREE_ASSERT(i == 0 || feature_bits == 0);
    loom_spirv_module_builder_require_feature_bits(
        state->builder, (loom_spirv_feature_bits_t)feature_bits);
  }

  const loom_spirv_packet_row_t* row =
      loom_spirv_packet_row_for_descriptor_ordinal(packet->descriptor_ordinal);
  IREE_ASSERT(row != NULL);
  loom_spirv_emit_validate_packet_shape(op, packet, row);
  switch (row->form) {
    case LOOM_SPIRV_PACKET_FORM_SCALAR_CONSTANT:
      return loom_spirv_emit_scalar_constant_packet(state, packet, row);
    case LOOM_SPIRV_PACKET_FORM_BOOLEAN_CONSTANT:
      return loom_spirv_emit_boolean_constant_packet(state, packet, row);
    case LOOM_SPIRV_PACKET_FORM_BINARY_SAME_TYPE:
      return loom_spirv_emit_binary_same_type_packet(state, packet, row);
    case LOOM_SPIRV_PACKET_FORM_UNARY_TYPED:
      return loom_spirv_emit_unary_typed_packet(state, packet, row);
    case LOOM_SPIRV_PACKET_FORM_COMPOSITE_CONSTRUCT:
      return loom_spirv_emit_composite_construct_packet(state, packet, row);
    case LOOM_SPIRV_PACKET_FORM_COMPOSITE_EXTRACT:
      return loom_spirv_emit_composite_extract_packet(state, packet, row);
    case LOOM_SPIRV_PACKET_FORM_COMPOSITE_INSERT:
      return loom_spirv_emit_composite_insert_packet(state, packet, row);
    case LOOM_SPIRV_PACKET_FORM_LOAD_BUILTIN:
      return loom_spirv_emit_load_builtin_packet(state, packet, row);
    case LOOM_SPIRV_PACKET_FORM_INTEGER_MUL_ADD:
      return loom_spirv_emit_integer_mul_add_packet(state, packet, row);
    case LOOM_SPIRV_PACKET_FORM_COMPARE_SAME_TYPE:
      return loom_spirv_emit_compare_same_type_packet(state, packet, row);
    case LOOM_SPIRV_PACKET_FORM_SELECT:
      return loom_spirv_emit_select_packet(state, packet, row);
    case LOOM_SPIRV_PACKET_FORM_PTR_ACCESS_CHAIN:
      return loom_spirv_emit_ptr_access_chain_packet(state, packet, row);
    case LOOM_SPIRV_PACKET_FORM_ACCESS_CHAIN:
      return loom_spirv_emit_access_chain_packet(state, packet, row);
    case LOOM_SPIRV_PACKET_FORM_LOAD_ALIGNED:
      return loom_spirv_emit_load_aligned_packet(state, packet, row);
    case LOOM_SPIRV_PACKET_FORM_STORE_ALIGNED:
      return loom_spirv_emit_store_aligned_packet(state, packet, row);
    case LOOM_SPIRV_PACKET_FORM_COOPERATIVE_MATRIX_LOAD:
      return loom_spirv_emit_cooperative_matrix_load_packet(state, packet, row);
    case LOOM_SPIRV_PACKET_FORM_COOPERATIVE_MATRIX_STORE:
      return loom_spirv_emit_cooperative_matrix_store_packet(state, packet,
                                                             row);
    case LOOM_SPIRV_PACKET_FORM_COOPERATIVE_MATRIX_MUL_ADD:
      return loom_spirv_emit_cooperative_matrix_mul_add_packet(state, packet,
                                                               row);
    case LOOM_SPIRV_PACKET_FORM_CONTROL_BARRIER:
      return loom_spirv_emit_control_barrier_packet(state, row);
    case LOOM_SPIRV_PACKET_FORM_UNSUPPORTED:
      break;
  }
  IREE_CHECK_UNREACHABLE("verified SPIR-V binary packet row");
  return iree_ok_status();
}

iree_status_t loom_spirv_emit_low_op(loom_spirv_emit_state_t* state,
                                     const loom_op_t* op) {
  if (loom_traits_are_compile_time_only(op->traits)) {
    if (op->result_count == 0) return iree_ok_status();
    IREE_ASSERT(loom_traits_are_fact_identity(op->traits));
    IREE_ASSERT_EQ(op->operand_count, op->result_count);
    const loom_value_id_t* operands = loom_op_const_operands(op);
    const loom_value_id_t* results = loom_op_const_results(op);
    for (uint16_t i = 0; i < op->result_count; ++i) {
      loom_spirv_module_value_ref_t value_ref = {0};
      IREE_RETURN_IF_ERROR(
          loom_spirv_emit_lookup_value(state, operands[i], &value_ref));
      IREE_RETURN_IF_ERROR(loom_spirv_emit_define_value(
          state, results[i], value_ref, /*emit_name=*/false));
    }
    return iree_ok_status();
  }
  if (loom_low_scf_if_isa(op)) {
    return loom_spirv_emit_scf_if(state, op);
  }
  if (loom_low_scf_for_isa(op)) {
    return loom_spirv_emit_scf_for(state, op);
  }
  if (loom_low_scf_while_isa(op)) {
    return loom_spirv_emit_scf_while(state, op);
  }
  if (loom_low_scf_yield_isa(op) || loom_low_scf_condition_isa(op)) {
    IREE_CHECK_UNREACHABLE("verified SPIR-V low.scf terminator parent");
    return iree_ok_status();
  }
  if (loom_low_resource_isa(op)) {
    IREE_ASSERT_EQ(state->abi_plan.kind,
                   LOOM_SPIRV_MODULE_ABI_PLAN_HAL_KERNEL_RAW_BDA);
    loom_spirv_module_abi_context_t context =
        loom_spirv_emit_abi_context(state);
    return loom_spirv_module_abi_materialize_resource(&context,
                                                      &state->abi_plan, op);
  }
  if (loom_low_storage_reserve_isa(op)) {
    return loom_spirv_module_workgroup_storage_emit_reserve(
        &state->value_domain, &state->workgroup_storage, op);
  }
  if (loom_low_storage_address_isa(op)) {
    loom_spirv_module_value_ref_t value_ref = {0};
    IREE_RETURN_IF_ERROR(loom_spirv_module_workgroup_storage_emit_address(
        &state->value_domain, &state->workgroup_storage, op,
        state->type_context, state->builder, &value_ref));
    return loom_spirv_emit_define_value(
        state, loom_low_storage_address_result(op), value_ref, true);
  }
  if (loom_low_copy_isa(op) || loom_low_move_isa(op)) {
    return loom_spirv_emit_transfer(state, op);
  }

  loom_low_descriptor_packet_t packet = {0};
  loom_low_descriptor_packet_initialize(state->target->descriptor_set, op,
                                        &packet);
  if (packet.kind == LOOM_LOW_DESCRIPTOR_PACKET_NONE) {
    IREE_CHECK_UNREACHABLE("verified SPIR-V structural op");
    return iree_ok_status();
  }
  return loom_spirv_emit_descriptor_packet(state, op, &packet);
}

static iree_status_t loom_spirv_emit_return(loom_spirv_emit_state_t* state,
                                            const loom_op_t* op) {
  loom_spirv_module_abi_context_t context = loom_spirv_emit_abi_context(state);
  IREE_RETURN_IF_ERROR(loom_spirv_module_abi_store_return_values(
      &context, &state->abi_plan, op));
  return loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      LOOM_SPIRV_OP_RETURN, NULL, 0);
}

static iree_status_t loom_spirv_emit_entry_point(
    loom_spirv_emit_state_t* state) {
  const uint32_t prefix_operands[] = {
      LOOM_SPIRV_EXECUTION_MODEL_GL_COMPUTE,
      state->function_id,
  };
  uint32_t interface_operands[LOOM_SPIRV_BUILTIN_VARIABLE_COUNT] = {0};
  iree_host_size_t interface_operand_count = 0;
  for (iree_host_size_t i = 0; i < LOOM_SPIRV_BUILTIN_VARIABLE_COUNT; ++i) {
    if (state->builtin_interface_variable_ids[i] != 0) {
      interface_operands[interface_operand_count++] =
          state->builtin_interface_variable_ids[i];
    }
  }
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_string_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_ENTRY_POINT),
      LOOM_SPIRV_OP_ENTRY_POINT, prefix_operands,
      IREE_ARRAYSIZE(prefix_operands), loom_spirv_emit_export_name(state),
      interface_operands, interface_operand_count));
  uint32_t workgroup_size_x = 1;
  uint32_t workgroup_size_y = 1;
  uint32_t workgroup_size_z = 1;
  if (loom_low_kernel_def_isa(state->function_op)) {
    const int64_t x = loom_low_kernel_def_workgroup_size_x(state->function_op);
    const int64_t y = loom_low_kernel_def_workgroup_size_y(state->function_op);
    const int64_t z = loom_low_kernel_def_workgroup_size_z(state->function_op);
    workgroup_size_x = x > 0 ? (uint32_t)x : 1u;
    workgroup_size_y = y > 0 ? (uint32_t)y : 1u;
    workgroup_size_z = z > 0 ? (uint32_t)z : 1u;
  }
  const uint32_t execution_mode_operands[] = {
      state->function_id, LOOM_SPIRV_EXECUTION_MODE_LOCAL_SIZE,
      workgroup_size_x,   workgroup_size_y,
      workgroup_size_z,
  };
  return loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_EXECUTION_MODE),
      LOOM_SPIRV_OP_EXECUTION_MODE, execution_mode_operands,
      IREE_ARRAYSIZE(execution_mode_operands));
}

static iree_status_t loom_spirv_emit_function_signature(
    loom_spirv_emit_state_t* state, uint32_t* out_result_type_id,
    uint32_t* out_function_type_id) {
  uint32_t result_type_id = 0;
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_type_void(state->type_context, &result_type_id));
  uint32_t function_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_function(
      state->type_context, result_type_id, /*parameter_type_ids=*/NULL,
      /*parameter_count=*/0, &function_type_id));
  *out_result_type_id = result_type_id;
  *out_function_type_id = function_type_id;
  return iree_ok_status();
}

static iree_status_t loom_spirv_emit_cfg_op_error(
    loom_spirv_emit_state_t* state, const loom_op_t* op) {
  const loom_op_vtable_t* vtable = loom_op_vtable(state->module, op);
  const iree_string_view_t op_name =
      vtable != NULL ? loom_op_vtable_name(vtable) : IREE_SV("<unknown>");
  return iree_make_status(
      IREE_STATUS_FAILED_PRECONDITION,
      "verified SPIR-V low function still contains CFG op '%.*s'",
      (int)op_name.size, op_name.data);
}

static iree_status_t loom_spirv_emit_function_entry_block(
    loom_spirv_emit_state_t* state, const loom_block_t* block) {
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_label_id(state, loom_spirv_emit_allocate_id(state)));
  loom_spirv_module_abi_context_t context = loom_spirv_emit_abi_context(state);
  IREE_RETURN_IF_ERROR(
      loom_spirv_module_abi_materialize_entry_args(&context, &state->abi_plan));
  for (const loom_op_t* op = block->first_op; op != NULL; op = op->next_op) {
    if (loom_low_return_isa(op)) {
      IREE_RETURN_IF_ERROR(loom_spirv_emit_return(state, op));
    } else if (loom_low_br_isa(op) || loom_low_cond_br_isa(op)) {
      return loom_spirv_emit_cfg_op_error(state, op);
    } else {
      IREE_RETURN_IF_ERROR(loom_spirv_emit_low_op(state, op));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_spirv_emit_function_body(
    loom_spirv_emit_state_t* state) {
  if (state->body->block_count != 1) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "verified SPIR-V low function has %u top-level blocks; expected one",
        (unsigned)state->body->block_count);
  }
  const loom_block_t* block = loom_region_const_entry_block(state->body);
  IREE_RETURN_IF_ERROR(loom_spirv_emit_function_entry_block(state, block));
  return loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      LOOM_SPIRV_OP_FUNCTION_END, NULL, 0);
}

static iree_status_t loom_spirv_emit_function_contents(
    loom_spirv_emit_state_t* state) {
  const loom_region_t* body = loom_low_function_const_body(state->function_op);
  IREE_ASSERT(body != NULL);
  IREE_ASSERT_NE(body->block_count, 0u);
  state->body = body;
  const loom_block_t* entry_block = loom_region_const_entry_block(body);
  state->function_id = loom_spirv_emit_allocate_id(state);
  IREE_RETURN_IF_ERROR(loom_spirv_emit_op_name(
      state, state->function_id, loom_spirv_emit_function_name(state)));
  loom_spirv_module_abi_context_t context = loom_spirv_emit_abi_context(state);
  IREE_RETURN_IF_ERROR(loom_spirv_module_abi_build_plan(&context, entry_block,
                                                        &state->abi_plan));

  uint32_t result_type_id = 0;
  uint32_t function_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_function_signature(
      state, &result_type_id, &function_type_id));

  const uint32_t function_operands[] = {
      result_type_id,
      state->function_id,
      LOOM_SPIRV_FUNCTION_CONTROL_NONE,
      function_type_id,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      LOOM_SPIRV_OP_FUNCTION, function_operands,
      IREE_ARRAYSIZE(function_operands)));
  IREE_RETURN_IF_ERROR(loom_spirv_emit_function_body(state));
  return loom_spirv_emit_entry_point(state);
}

static void loom_spirv_emit_function_state_deinitialize(
    loom_spirv_emit_state_t* state) {
  if (loom_local_value_domain_is_acquired(&state->value_domain)) {
    loom_local_value_domain_release(&state->value_domain);
  }
}

static iree_status_t loom_spirv_emit_function_state_initialize(
    loom_spirv_function_emission_context_t* context, loom_op_t* low_function_op,
    const loom_low_resolved_target_t* target,
    loom_spirv_emit_state_t* out_state) {
  *out_state = (loom_spirv_emit_state_t){
      .context = context,
      .module = context->module,
      .function_op = low_function_op,
      .target = target,
      .scratch_arena = context->scratch_arena,
      .builder = context->builder,
      .type_context = context->type_context,
  };

  const loom_region_t* body = loom_low_function_const_body(low_function_op);
  iree_status_t status = iree_ok_status();
  if (body == NULL || body->block_count == 0) {
    status =
        iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                         "verified SPIR-V low function has no function body");
  }
  if (iree_status_is_ok(status)) {
    status = loom_local_value_domain_acquire_for_region_tree(
        context->module, body, context->scratch_arena,
        &out_state->value_domain);
  }
  if (iree_status_is_ok(status)) {
    status = loom_spirv_module_workgroup_storage_initialize(
        &out_state->value_domain, &out_state->workgroup_storage,
        context->scratch_arena);
  }
  if (iree_status_is_ok(status)) {
    status = loom_spirv_module_value_table_initialize(&out_state->value_domain,
                                                      &out_state->value_table,
                                                      context->scratch_arena);
  }
  if (iree_status_is_ok(status)) {
    status = loom_spirv_emit_prepare_immediate_name_ids(out_state);
  }
  if (!iree_status_is_ok(status)) {
    loom_spirv_emit_function_state_deinitialize(out_state);
  }
  return status;
}

iree_status_t loom_spirv_emit_low_function(
    loom_spirv_function_emission_context_t* context, loom_op_t* low_function_op,
    const loom_low_resolved_target_t* target) {
  if (!loom_low_function_def_isa(low_function_op)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "SPIR-V emission requires a low function "
                            "definition");
  }

  loom_spirv_emit_state_t function_state = {0};
  iree_status_t status = loom_spirv_emit_function_state_initialize(
      context, low_function_op, target, &function_state);
  if (iree_status_is_ok(status)) {
    status = loom_spirv_emit_function_contents(&function_state);
  }
  loom_spirv_emit_function_state_deinitialize(&function_state);
  return status;
}

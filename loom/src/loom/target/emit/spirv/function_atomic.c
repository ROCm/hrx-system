// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/spirv/function_atomic.h"

#include "loom/target/emit/spirv/binary_format.h"
#include "loom/target/emit/spirv/module_instructions.h"
#include "loom/target/emit/spirv/module_types.h"

// Atomic ordering ordinals match loom_atomic_ordering_t and the descriptor
// enum domains. Sequentially consistent ordering is deliberately absent from
// every SPIR-V atomic descriptor because the Vulkan environment forbids it.
enum loom_spirv_atomic_ordering_e {
  LOOM_SPIRV_ATOMIC_ORDERING_RELAXED = 0,
  LOOM_SPIRV_ATOMIC_ORDERING_ACQUIRE = 1,
  LOOM_SPIRV_ATOMIC_ORDERING_RELEASE = 2,
  LOOM_SPIRV_ATOMIC_ORDERING_ACQUIRE_RELEASE = 3,
};
typedef uint8_t loom_spirv_atomic_ordering_t;

// Floating-point CAS operation ordinals match the generated packet rows.
enum loom_spirv_atomic_float_operation_e {
  LOOM_SPIRV_ATOMIC_FLOAT_OPERATION_EXCHANGE = 0,
  LOOM_SPIRV_ATOMIC_FLOAT_OPERATION_ADD = 1,
  LOOM_SPIRV_ATOMIC_FLOAT_OPERATION_MINIMUM = 2,
  LOOM_SPIRV_ATOMIC_FLOAT_OPERATION_MAXIMUM = 3,
  LOOM_SPIRV_ATOMIC_FLOAT_OPERATION_MINNUM = 4,
  LOOM_SPIRV_ATOMIC_FLOAT_OPERATION_MAXNUM = 5,
};

static uint32_t loom_spirv_atomic_memory_semantics(
    loom_spirv_atomic_ordering_t ordering, uint32_t storage_semantics) {
  switch (ordering) {
    case LOOM_SPIRV_ATOMIC_ORDERING_RELAXED:
      // Vulkan requires relaxed atomics to use None, without a storage-class
      // semantics bit.
      return LOOM_SPIRV_MEMORY_SEMANTICS_NONE;
    case LOOM_SPIRV_ATOMIC_ORDERING_ACQUIRE:
      return LOOM_SPIRV_MEMORY_SEMANTICS_ACQUIRE_MASK | storage_semantics;
    case LOOM_SPIRV_ATOMIC_ORDERING_RELEASE:
      return LOOM_SPIRV_MEMORY_SEMANTICS_RELEASE_MASK | storage_semantics;
    case LOOM_SPIRV_ATOMIC_ORDERING_ACQUIRE_RELEASE:
      return LOOM_SPIRV_MEMORY_SEMANTICS_ACQUIRE_RELEASE_MASK |
             storage_semantics;
  }
  IREE_CHECK_UNREACHABLE("verified SPIR-V atomic ordering");
  return LOOM_SPIRV_MEMORY_SEMANTICS_NONE;
}

static iree_status_t loom_spirv_emit_atomic_constants(
    loom_spirv_emit_state_t* state, const loom_spirv_packet_row_t* row,
    loom_spirv_atomic_ordering_t ordering, uint32_t* out_scope_id,
    uint32_t* out_semantics_id) {
  IREE_RETURN_IF_ERROR(loom_spirv_emit_u32_constant(
      state->type_context, row->payload.atomic.memory_scope, out_scope_id));
  return loom_spirv_emit_u32_constant(
      state->type_context,
      loom_spirv_atomic_memory_semantics(ordering,
                                         row->payload.atomic.storage_semantics),
      out_semantics_id);
}

static iree_status_t loom_spirv_emit_atomic_result(
    loom_spirv_emit_state_t* state, const loom_low_descriptor_packet_t* packet,
    const loom_spirv_packet_row_t* row, loom_spirv_module_value_ref_t value_ref,
    uint32_t* out_result_id) {
  if (row->result_count == 0) {
    *out_result_id = loom_spirv_emit_allocate_id(state);
    return iree_ok_status();
  }
  return loom_spirv_emit_reserve_value_ref(
      state, loom_op_const_results(packet->op)[0], value_ref.type_id,
      loom_spirv_packet_row_result_type(row), out_result_id);
}

static iree_status_t loom_spirv_emit_define_atomic_result(
    loom_spirv_emit_state_t* state, const loom_low_descriptor_packet_t* packet,
    const loom_spirv_packet_row_t* row, uint32_t result_id,
    loom_spirv_module_value_ref_t value_ref) {
  if (row->result_count == 0) return iree_ok_status();
  value_ref.id = result_id;
  value_ref.value_type = loom_spirv_packet_row_result_type(row);
  return loom_spirv_emit_define_value(
      state, loom_op_const_results(packet->op)[0], value_ref,
      /*emit_name=*/true);
}

iree_status_t loom_spirv_emit_atomic_packet(
    loom_spirv_emit_state_t* state, const loom_low_descriptor_packet_t* packet,
    const loom_spirv_packet_row_t* row) {
  const loom_value_id_t* operand_values = loom_op_const_operands(packet->op);
  loom_spirv_module_value_ref_t pointer = {0};
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_lookup_value(state, operand_values[0], &pointer));
  loom_spirv_module_value_ref_t value = {0};
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_lookup_value(state, operand_values[1], &value));

  int64_t ordering_value = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_lookup_packet_i64_immediate(
      state, packet, row->immediate_index, &ordering_value));
  IREE_ASSERT(ordering_value >= LOOM_SPIRV_ATOMIC_ORDERING_RELAXED &&
              ordering_value <= LOOM_SPIRV_ATOMIC_ORDERING_ACQUIRE_RELEASE);
  const loom_spirv_atomic_ordering_t ordering =
      (loom_spirv_atomic_ordering_t)ordering_value;
  uint32_t scope_id = 0;
  uint32_t semantics_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_atomic_constants(
      state, row, ordering, &scope_id, &semantics_id));

  uint32_t result_id = 0;
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_atomic_result(state, packet, row, value, &result_id));
  const uint32_t instruction_operands[] = {
      value.type_id, result_id, pointer.id, scope_id, semantics_id, value.id,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      row->opcode, instruction_operands, IREE_ARRAYSIZE(instruction_operands)));
  return loom_spirv_emit_define_atomic_result(state, packet, row, result_id,
                                              value);
}

iree_status_t loom_spirv_emit_atomic_compare_exchange_packet(
    loom_spirv_emit_state_t* state, const loom_low_descriptor_packet_t* packet,
    const loom_spirv_packet_row_t* row) {
  const loom_value_id_t* operand_values = loom_op_const_operands(packet->op);
  loom_spirv_module_value_ref_t pointer = {0};
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_lookup_value(state, operand_values[0], &pointer));
  loom_spirv_module_value_ref_t replacement = {0};
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_lookup_value(state, operand_values[1], &replacement));
  loom_spirv_module_value_ref_t expected = {0};
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_lookup_value(state, operand_values[2], &expected));
  IREE_ASSERT_EQ(replacement.type_id, expected.type_id);

  int64_t failure_ordering_value = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_lookup_packet_i64_immediate(
      state, packet, row->immediate_index, &failure_ordering_value));
  IREE_ASSERT(failure_ordering_value >= LOOM_SPIRV_ATOMIC_ORDERING_RELAXED &&
              failure_ordering_value <= LOOM_SPIRV_ATOMIC_ORDERING_ACQUIRE);
  const loom_spirv_atomic_ordering_t success_ordering =
      (loom_spirv_atomic_ordering_t)row->payload.atomic.success_ordering;
  const loom_spirv_atomic_ordering_t failure_ordering =
      (loom_spirv_atomic_ordering_t)failure_ordering_value;
  uint32_t scope_id = 0;
  uint32_t success_semantics_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_atomic_constants(
      state, row, success_ordering, &scope_id, &success_semantics_id));
  uint32_t failure_semantics_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_u32_constant(
      state->type_context,
      loom_spirv_atomic_memory_semantics(failure_ordering,
                                         row->payload.atomic.storage_semantics),
      &failure_semantics_id));

  uint32_t result_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_atomic_result(state, packet, row,
                                                     replacement, &result_id));
  const uint32_t instruction_operands[] = {
      replacement.type_id,  result_id,
      pointer.id,           scope_id,
      success_semantics_id, failure_semantics_id,
      replacement.id,       expected.id,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      row->opcode, instruction_operands, IREE_ARRAYSIZE(instruction_operands)));
  return loom_spirv_emit_define_atomic_result(state, packet, row, result_id,
                                              replacement);
}

static iree_status_t loom_spirv_emit_atomic_integer_type(
    loom_spirv_emit_state_t* state, const loom_spirv_packet_row_t* row,
    const loom_spirv_module_value_ref_t* pointer,
    uint32_t* out_integer_type_id) {
  const loom_spirv_scalar_type_t integer_scalar_type =
      (loom_spirv_scalar_type_t)row->payload.atomic.integer_scalar;
  IREE_ASSERT_EQ(pointer->value_type.scalar_type, integer_scalar_type);
  return loom_spirv_emit_type_scalar(state->type_context, integer_scalar_type,
                                     out_integer_type_id);
}

static iree_status_t loom_spirv_emit_select(loom_spirv_emit_state_t* state,
                                            uint32_t result_type_id,
                                            uint32_t condition_id,
                                            uint32_t true_value_id,
                                            uint32_t false_value_id,
                                            uint32_t* out_result_id) {
  const uint32_t result_id = loom_spirv_emit_allocate_id(state);
  const uint32_t operands[] = {
      result_type_id, result_id, condition_id, true_value_id, false_value_id,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      LOOM_SPIRV_OP_SELECT, operands, IREE_ARRAYSIZE(operands)));
  *out_result_id = result_id;
  return iree_ok_status();
}

static iree_status_t loom_spirv_emit_branch(loom_spirv_emit_state_t* state,
                                            uint32_t label_id) {
  const uint32_t operands[] = {label_id};
  return loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      LOOM_SPIRV_OP_BRANCH, operands, IREE_ARRAYSIZE(operands));
}

static iree_status_t loom_spirv_emit_float_atomic_result(
    loom_spirv_emit_state_t* state, const loom_low_descriptor_packet_t* packet,
    const loom_spirv_packet_row_t* row,
    const loom_spirv_module_value_ref_t* float_value, uint32_t integer_id) {
  if (row->result_count == 0) return iree_ok_status();
  uint32_t result_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_atomic_result(state, packet, row,
                                                     *float_value, &result_id));
  const uint32_t operands[] = {
      float_value->type_id,
      result_id,
      integer_id,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      LOOM_SPIRV_OP_BITCAST, operands, IREE_ARRAYSIZE(operands)));
  return loom_spirv_emit_define_atomic_result(state, packet, row, result_id,
                                              *float_value);
}

iree_status_t loom_spirv_emit_atomic_float_bitcast_packet(
    loom_spirv_emit_state_t* state, const loom_low_descriptor_packet_t* packet,
    const loom_spirv_packet_row_t* row) {
  const loom_value_id_t* operand_values = loom_op_const_operands(packet->op);
  loom_spirv_module_value_ref_t pointer = {0};
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_lookup_value(state, operand_values[0], &pointer));
  loom_spirv_module_value_ref_t value = {0};
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_lookup_value(state, operand_values[1], &value));

  int64_t ordering_value = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_lookup_packet_i64_immediate(
      state, packet, row->immediate_index, &ordering_value));
  IREE_ASSERT(ordering_value >= LOOM_SPIRV_ATOMIC_ORDERING_RELAXED &&
              ordering_value <= LOOM_SPIRV_ATOMIC_ORDERING_ACQUIRE_RELEASE);
  uint32_t scope_id = 0;
  uint32_t semantics_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_atomic_constants(
      state, row, (loom_spirv_atomic_ordering_t)ordering_value, &scope_id,
      &semantics_id));

  uint32_t integer_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_atomic_integer_type(state, row, &pointer,
                                                           &integer_type_id));
  uint32_t integer_value_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_module_emit_unary_result(
      state->builder, LOOM_SPIRV_OP_BITCAST, integer_type_id, value.id,
      &integer_value_id));

  const uint32_t integer_result_id = loom_spirv_emit_allocate_id(state);
  const uint32_t operands[] = {
      integer_type_id, integer_result_id, pointer.id,
      scope_id,        semantics_id,      integer_value_id,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      row->opcode, operands, IREE_ARRAYSIZE(operands)));
  return loom_spirv_emit_float_atomic_result(state, packet, row, &value,
                                             integer_result_id);
}

iree_status_t loom_spirv_emit_atomic_float_compare_exchange_packet(
    loom_spirv_emit_state_t* state, const loom_low_descriptor_packet_t* packet,
    const loom_spirv_packet_row_t* row) {
  const loom_value_id_t* operand_values = loom_op_const_operands(packet->op);
  loom_spirv_module_value_ref_t pointer = {0};
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_lookup_value(state, operand_values[0], &pointer));
  loom_spirv_module_value_ref_t replacement = {0};
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_lookup_value(state, operand_values[1], &replacement));
  loom_spirv_module_value_ref_t expected = {0};
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_lookup_value(state, operand_values[2], &expected));
  IREE_ASSERT_EQ(replacement.type_id, expected.type_id);

  int64_t failure_ordering_value = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_lookup_packet_i64_immediate(
      state, packet, row->immediate_index, &failure_ordering_value));
  IREE_ASSERT(failure_ordering_value >= LOOM_SPIRV_ATOMIC_ORDERING_RELAXED &&
              failure_ordering_value <= LOOM_SPIRV_ATOMIC_ORDERING_ACQUIRE);
  uint32_t scope_id = 0;
  uint32_t success_semantics_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_atomic_constants(
      state, row,
      (loom_spirv_atomic_ordering_t)row->payload.atomic.success_ordering,
      &scope_id, &success_semantics_id));
  uint32_t failure_semantics_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_u32_constant(
      state->type_context,
      loom_spirv_atomic_memory_semantics(
          (loom_spirv_atomic_ordering_t)failure_ordering_value,
          row->payload.atomic.storage_semantics),
      &failure_semantics_id));

  uint32_t integer_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_atomic_integer_type(state, row, &pointer,
                                                           &integer_type_id));
  uint32_t replacement_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_module_emit_unary_result(
      state->builder, LOOM_SPIRV_OP_BITCAST, integer_type_id, replacement.id,
      &replacement_id));
  uint32_t expected_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_module_emit_unary_result(
      state->builder, LOOM_SPIRV_OP_BITCAST, integer_type_id, expected.id,
      &expected_id));

  const uint32_t integer_result_id = loom_spirv_emit_allocate_id(state);
  const uint32_t operands[] = {
      integer_type_id,      integer_result_id,    pointer.id,     scope_id,
      success_semantics_id, failure_semantics_id, replacement_id, expected_id,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      row->opcode, operands, IREE_ARRAYSIZE(operands)));
  return loom_spirv_emit_float_atomic_result(state, packet, row, &replacement,
                                             integer_result_id);
}

static iree_status_t loom_spirv_emit_float_atomic_minmax(
    loom_spirv_emit_state_t* state, uint8_t operation, uint32_t float_type_id,
    uint32_t integer_type_id, uint32_t current_float_id,
    uint32_t current_integer_id, uint32_t value_float_id,
    uint32_t value_integer_id, uint32_t* out_result_id) {
  const bool is_minimum =
      operation == LOOM_SPIRV_ATOMIC_FLOAT_OPERATION_MINIMUM ||
      operation == LOOM_SPIRV_ATOMIC_FLOAT_OPERATION_MINNUM;
  const bool propagates_nan =
      operation == LOOM_SPIRV_ATOMIC_FLOAT_OPERATION_MINIMUM ||
      operation == LOOM_SPIRV_ATOMIC_FLOAT_OPERATION_MAXIMUM;
  IREE_ASSERT(operation >= LOOM_SPIRV_ATOMIC_FLOAT_OPERATION_MINIMUM &&
              operation <= LOOM_SPIRV_ATOMIC_FLOAT_OPERATION_MAXNUM);

  uint32_t bool_type_id = 0;
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_type_bool(state->type_context, &bool_type_id));
  uint32_t ordered_comparison_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_module_emit_binary_result(
      state->builder,
      is_minimum ? LOOM_SPIRV_OP_F_ORD_LESS_THAN
                 : LOOM_SPIRV_OP_F_ORD_GREATER_THAN,
      bool_type_id, current_float_id, value_float_id, &ordered_comparison_id));
  uint32_t numeric_result_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_select(
      state, float_type_id, ordered_comparison_id, current_float_id,
      value_float_id, &numeric_result_id));

  uint32_t equal_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_module_emit_binary_result(
      state->builder, LOOM_SPIRV_OP_F_ORD_EQUAL, bool_type_id, current_float_id,
      value_float_id, &equal_id));
  uint32_t tied_integer_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_module_emit_binary_result(
      state->builder,
      is_minimum ? LOOM_SPIRV_OP_BITWISE_OR : LOOM_SPIRV_OP_BITWISE_AND,
      integer_type_id, current_integer_id, value_integer_id, &tied_integer_id));
  uint32_t tied_float_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_module_emit_unary_result(
      state->builder, LOOM_SPIRV_OP_BITCAST, float_type_id, tied_integer_id,
      &tied_float_id));
  uint32_t ordered_result_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_select(state, float_type_id, equal_id,
                                              tied_float_id, numeric_result_id,
                                              &ordered_result_id));

  uint32_t current_nan_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_module_emit_unary_result(
      state->builder, LOOM_SPIRV_OP_IS_NAN, bool_type_id, current_float_id,
      &current_nan_id));
  uint32_t value_nan_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_module_emit_unary_result(
      state->builder, LOOM_SPIRV_OP_IS_NAN, bool_type_id, value_float_id,
      &value_nan_id));
  if (propagates_nan) {
    uint32_t any_nan_id = 0;
    IREE_RETURN_IF_ERROR(loom_spirv_module_emit_binary_result(
        state->builder, LOOM_SPIRV_OP_LOGICAL_OR, bool_type_id, current_nan_id,
        value_nan_id, &any_nan_id));
    uint32_t nan_result_id = 0;
    IREE_RETURN_IF_ERROR(loom_spirv_emit_select(
        state, float_type_id, current_nan_id, current_float_id, value_float_id,
        &nan_result_id));
    return loom_spirv_emit_select(state, float_type_id, any_nan_id,
                                  nan_result_id, ordered_result_id,
                                  out_result_id);
  }

  uint32_t current_filtered_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_select(
      state, float_type_id, current_nan_id, value_float_id, ordered_result_id,
      &current_filtered_id));
  return loom_spirv_emit_select(state, float_type_id, value_nan_id,
                                current_float_id, current_filtered_id,
                                out_result_id);
}

iree_status_t loom_spirv_emit_atomic_float_cas_packet(
    loom_spirv_emit_state_t* state, const loom_low_descriptor_packet_t* packet,
    const loom_spirv_packet_row_t* row) {
  const loom_value_id_t* operand_values = loom_op_const_operands(packet->op);
  loom_spirv_module_value_ref_t pointer = {0};
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_lookup_value(state, operand_values[0], &pointer));
  loom_spirv_module_value_ref_t value = {0};
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_lookup_value(state, operand_values[1], &value));

  int64_t ordering_value = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_lookup_packet_i64_immediate(
      state, packet, row->immediate_index, &ordering_value));
  IREE_ASSERT(ordering_value >= LOOM_SPIRV_ATOMIC_ORDERING_RELAXED &&
              ordering_value <= LOOM_SPIRV_ATOMIC_ORDERING_ACQUIRE_RELEASE);
  uint32_t scope_id = 0;
  uint32_t success_semantics_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_atomic_constants(
      state, row, (loom_spirv_atomic_ordering_t)ordering_value, &scope_id,
      &success_semantics_id));
  uint32_t relaxed_semantics_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_u32_constant(
      state->type_context, LOOM_SPIRV_MEMORY_SEMANTICS_NONE,
      &relaxed_semantics_id));

  uint32_t integer_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_atomic_integer_type(state, row, &pointer,
                                                           &integer_type_id));
  uint32_t value_integer_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_module_emit_unary_result(
      state->builder, LOOM_SPIRV_OP_BITCAST, integer_type_id, value.id,
      &value_integer_id));

  const uint32_t initial_id = loom_spirv_emit_allocate_id(state);
  const uint32_t load_operands[] = {
      integer_type_id, initial_id, pointer.id, scope_id, relaxed_semantics_id,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      LOOM_SPIRV_OP_ATOMIC_LOAD, load_operands, IREE_ARRAYSIZE(load_operands)));

  const uint32_t preheader_label_id = state->current_label_id;
  const uint32_t header_label_id = loom_spirv_emit_allocate_id(state);
  const uint32_t continue_label_id = loom_spirv_emit_allocate_id(state);
  const uint32_t merge_label_id = loom_spirv_emit_allocate_id(state);
  const uint32_t expected_id = loom_spirv_emit_allocate_id(state);
  const uint32_t observed_id = loom_spirv_emit_allocate_id(state);
  IREE_RETURN_IF_ERROR(loom_spirv_emit_branch(state, header_label_id));

  IREE_RETURN_IF_ERROR(loom_spirv_emit_label_id(state, header_label_id));
  const uint32_t phi_operands[] = {
      integer_type_id,    expected_id, initial_id,
      preheader_label_id, observed_id, continue_label_id,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      LOOM_SPIRV_OP_PHI, phi_operands, IREE_ARRAYSIZE(phi_operands)));
  uint32_t expected_float_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_module_emit_unary_result(
      state->builder, LOOM_SPIRV_OP_BITCAST, value.type_id, expected_id,
      &expected_float_id));

  uint32_t replacement_float_id = 0;
  if (row->payload.atomic.float_operation ==
      LOOM_SPIRV_ATOMIC_FLOAT_OPERATION_ADD) {
    IREE_RETURN_IF_ERROR(loom_spirv_module_emit_binary_result(
        state->builder, LOOM_SPIRV_OP_F_ADD, value.type_id, expected_float_id,
        value.id, &replacement_float_id));
  } else {
    IREE_RETURN_IF_ERROR(loom_spirv_emit_float_atomic_minmax(
        state, row->payload.atomic.float_operation, value.type_id,
        integer_type_id, expected_float_id, expected_id, value.id,
        value_integer_id, &replacement_float_id));
  }
  uint32_t replacement_integer_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_module_emit_unary_result(
      state->builder, LOOM_SPIRV_OP_BITCAST, integer_type_id,
      replacement_float_id, &replacement_integer_id));

  const uint32_t compare_exchange_operands[] = {
      integer_type_id,
      observed_id,
      pointer.id,
      scope_id,
      success_semantics_id,
      relaxed_semantics_id,
      replacement_integer_id,
      expected_id,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      row->opcode, compare_exchange_operands,
      IREE_ARRAYSIZE(compare_exchange_operands)));
  uint32_t bool_type_id = 0;
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_type_bool(state->type_context, &bool_type_id));
  uint32_t success_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_module_emit_binary_result(
      state->builder, LOOM_SPIRV_OP_I_EQUAL, bool_type_id, observed_id,
      expected_id, &success_id));
  const uint32_t loop_merge_operands[] = {
      merge_label_id,
      continue_label_id,
      LOOM_SPIRV_LOOP_CONTROL_NONE,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      LOOM_SPIRV_OP_LOOP_MERGE, loop_merge_operands,
      IREE_ARRAYSIZE(loop_merge_operands)));
  const uint32_t branch_operands[] = {
      success_id,
      merge_label_id,
      continue_label_id,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      LOOM_SPIRV_OP_BRANCH_CONDITIONAL, branch_operands,
      IREE_ARRAYSIZE(branch_operands)));

  IREE_RETURN_IF_ERROR(loom_spirv_emit_label_id(state, continue_label_id));
  IREE_RETURN_IF_ERROR(loom_spirv_emit_branch(state, header_label_id));
  IREE_RETURN_IF_ERROR(loom_spirv_emit_label_id(state, merge_label_id));
  return loom_spirv_emit_float_atomic_result(state, packet, row, &value,
                                             observed_id);
}

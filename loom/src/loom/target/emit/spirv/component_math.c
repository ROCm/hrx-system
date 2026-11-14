// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/spirv/component_math.h"

#include "loom/target/emit/spirv/binary_format.h"

static iree_status_t loom_spirv_emit_component_math_body(
    loom_spirv_function_emission_context_t* context,
    const loom_spirv_packet_row_t* row, uint32_t matrix_type_id,
    uint32_t instruction_set_id, uint32_t function_id) {
  loom_spirv_module_builder_t* builder = context->builder;
  loom_spirv_type_context_t* types = context->type_context;
  loom_spirv_binary_writer_t* writer = loom_spirv_module_builder_section(
      builder, LOOM_SPIRV_MODULE_SECTION_COMPONENT_FUNCTION);
  const loom_spirv_value_type_t matrix = loom_spirv_packet_row_result_type(row);
  uint32_t scalar_type_id = 0;
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_type_scalar(types, matrix.scalar_type, &scalar_type_id));
  uint32_t matrix_pointer_type_id = 0;
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_type_pointer(types, LOOM_SPIRV_STORAGE_CLASS_FUNCTION,
                                   matrix_type_id, 0, &matrix_pointer_type_id));
  uint32_t scalar_pointer_type_id = 0;
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_type_pointer(types, LOOM_SPIRV_STORAGE_CLASS_FUNCTION,
                                   scalar_type_id, 0, &scalar_pointer_type_id));
  const uint32_t parameter_types[] = {matrix_type_id, matrix_type_id};
  uint32_t function_type_id = 0;
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_type_function(types, matrix_type_id, parameter_types,
                                    row->operand_count, &function_type_id));
  uint32_t index_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_u32(types, &index_type_id));
  uint32_t bool_type_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_type_bool(types, &bool_type_id));
  uint32_t zero = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_u32_constant(types, 0, &zero));
  uint32_t one = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_u32_constant(types, 1, &one));

  const uint32_t function[] = {matrix_type_id, function_id,
                               LOOM_SPIRV_FUNCTION_CONTROL_INLINE_MASK,
                               function_type_id};
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      writer, LOOM_SPIRV_OP_FUNCTION, function, IREE_ARRAYSIZE(function)));
  uint32_t parameters[2];
  uint32_t variables[2];
  iree_status_t status = iree_ok_status();
  for (uint8_t i = 0; i < row->operand_count && iree_status_is_ok(status);
       ++i) {
    parameters[i] = loom_spirv_module_builder_allocate_id(builder);
    const uint32_t parameter[] = {matrix_type_id, parameters[i]};
    status = loom_spirv_binary_write_instruction(
        writer, LOOM_SPIRV_OP_FUNCTION_PARAMETER, parameter,
        IREE_ARRAYSIZE(parameter));
  }
  IREE_RETURN_IF_ERROR(status);
  const uint32_t entry = loom_spirv_module_builder_allocate_id(builder);
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      writer, LOOM_SPIRV_OP_LABEL, &entry, 1));
  for (uint8_t i = 0; i < row->operand_count && iree_status_is_ok(status);
       ++i) {
    variables[i] = loom_spirv_module_builder_allocate_id(builder);
    const uint32_t variable[] = {matrix_pointer_type_id, variables[i],
                                 LOOM_SPIRV_STORAGE_CLASS_FUNCTION};
    status = loom_spirv_binary_write_instruction(
        writer, LOOM_SPIRV_OP_VARIABLE, variable, IREE_ARRAYSIZE(variable));
  }
  IREE_RETURN_IF_ERROR(status);
  for (uint8_t i = 0; i < row->operand_count && iree_status_is_ok(status);
       ++i) {
    const uint32_t store[] = {variables[i], parameters[i]};
    status = loom_spirv_binary_write_instruction(writer, LOOM_SPIRV_OP_STORE,
                                                 store, IREE_ARRAYSIZE(store));
  }
  IREE_RETURN_IF_ERROR(status);

  // No host-side lane count describes this storage. Even the same logical
  // matrix shape can have different component counts on different devices.
  const uint32_t length_id = loom_spirv_module_builder_allocate_id(builder);
  const uint32_t length[] = {index_type_id, length_id, matrix_type_id};
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      writer, LOOM_SPIRV_OP_COOPERATIVE_MATRIX_LENGTH_KHR, length,
      IREE_ARRAYSIZE(length)));
  const uint32_t header = loom_spirv_module_builder_allocate_id(builder);
  const uint32_t body = loom_spirv_module_builder_allocate_id(builder);
  const uint32_t continuation = loom_spirv_module_builder_allocate_id(builder);
  const uint32_t merge = loom_spirv_module_builder_allocate_id(builder);
  const uint32_t index = loom_spirv_module_builder_allocate_id(builder);
  const uint32_t next = loom_spirv_module_builder_allocate_id(builder);
  const uint32_t condition = loom_spirv_module_builder_allocate_id(builder);
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      writer, LOOM_SPIRV_OP_BRANCH, &header, 1));
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      writer, LOOM_SPIRV_OP_LABEL, &header, 1));
  const uint32_t phi[] = {index_type_id, index, zero,
                          entry,         next,  continuation};
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      writer, LOOM_SPIRV_OP_PHI, phi, IREE_ARRAYSIZE(phi)));
  const uint32_t compare[] = {bool_type_id, condition, index, length_id};
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      writer, LOOM_SPIRV_OP_U_LESS_THAN, compare, IREE_ARRAYSIZE(compare)));
  const uint32_t loop[] = {merge, continuation,
                           LOOM_SPIRV_LOOP_CONTROL_UNROLL_MASK};
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      writer, LOOM_SPIRV_OP_LOOP_MERGE, loop, IREE_ARRAYSIZE(loop)));
  const uint32_t branch[] = {condition, body, merge};
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      writer, LOOM_SPIRV_OP_BRANCH_CONDITIONAL, branch,
      IREE_ARRAYSIZE(branch)));
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      writer, LOOM_SPIRV_OP_LABEL, &body, 1));

  // Overwrite the first local matrix only after loading every input component.
  // Independent operands of this identical matrix type share the mapping;
  // distinct component types or matrix uses require a conversion contract.
  uint32_t pointers[2];
  const uint32_t result = loom_spirv_module_builder_allocate_id(builder);
  const loom_spirv_packet_payload_t* payload =
      loom_spirv_packet_row_payload(row);
  uint32_t instruction[6] = {scalar_type_id, result, instruction_set_id,
                             payload->extended_instruction.instruction};
  for (uint8_t i = 0; i < row->operand_count && iree_status_is_ok(status);
       ++i) {
    pointers[i] = loom_spirv_module_builder_allocate_id(builder);
    instruction[4 + i] = loom_spirv_module_builder_allocate_id(builder);
    const uint32_t access[] = {scalar_pointer_type_id, pointers[i],
                               variables[i], index};
    status = loom_spirv_binary_write_instruction(
        writer, LOOM_SPIRV_OP_ACCESS_CHAIN, access, IREE_ARRAYSIZE(access));
    if (iree_status_is_ok(status)) {
      const uint32_t load[] = {scalar_type_id, instruction[4 + i], pointers[i]};
      status = loom_spirv_binary_write_instruction(writer, LOOM_SPIRV_OP_LOAD,
                                                   load, IREE_ARRAYSIZE(load));
    }
  }
  IREE_RETURN_IF_ERROR(status);
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      writer, row->opcode, instruction, 4 + row->operand_count));
  const uint32_t store[] = {pointers[0], result};
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      writer, LOOM_SPIRV_OP_STORE, store, IREE_ARRAYSIZE(store)));
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      writer, LOOM_SPIRV_OP_BRANCH, &continuation, 1));
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      writer, LOOM_SPIRV_OP_LABEL, &continuation, 1));
  const uint32_t increment[] = {index_type_id, next, index, one};
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      writer, LOOM_SPIRV_OP_I_ADD, increment, IREE_ARRAYSIZE(increment)));
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      writer, LOOM_SPIRV_OP_BRANCH, &header, 1));
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      writer, LOOM_SPIRV_OP_LABEL, &merge, 1));
  const uint32_t result_matrix = loom_spirv_module_builder_allocate_id(builder);
  const uint32_t load[] = {matrix_type_id, result_matrix, variables[0]};
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      writer, LOOM_SPIRV_OP_LOAD, load, IREE_ARRAYSIZE(load)));
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      writer, LOOM_SPIRV_OP_RETURN_VALUE, &result_matrix, 1));
  return loom_spirv_binary_write_instruction(writer, LOOM_SPIRV_OP_FUNCTION_END,
                                             NULL, 0);
}

iree_status_t loom_spirv_emit_component_math_function(
    loom_spirv_function_emission_context_t* context,
    const loom_spirv_packet_row_t* row, uint32_t matrix_type_id,
    uint32_t instruction_set_id, uint32_t* out_function_id) {
  if (*context->component_math_function_ids == NULL) {
    const iree_host_size_t size =
        loom_spirv_component_math_function_count * sizeof(uint32_t);
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate(context->scratch_arena, size,
                            (void**)context->component_math_function_ids));
    memset(*context->component_math_function_ids, 0, size);
  }
  const loom_spirv_packet_payload_t* payload =
      loom_spirv_packet_row_payload(row);
  uint32_t* function_id =
      &(*context->component_math_function_ids)
          [payload->extended_instruction.component_function - 1];
  if (*function_id == 0) {
    *function_id = loom_spirv_module_builder_allocate_id(context->builder);
    IREE_RETURN_IF_ERROR(loom_spirv_emit_component_math_body(
        context, row, matrix_type_id, instruction_set_id, *function_id));
  }
  *out_function_id = *function_id;
  return iree_ok_status();
}

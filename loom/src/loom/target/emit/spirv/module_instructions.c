// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/spirv/module_instructions.h"

#include "loom/target/arch/spirv/isa.h"
#include "loom/target/emit/spirv/binary_format.h"

enum {
  LOOM_SPIRV_ACCESS_CHAIN_MAX_INDEX_COUNT = 8,
};

iree_status_t loom_spirv_module_emit_access_chain(
    loom_spirv_module_builder_t* builder, uint32_t result_type_id,
    uint32_t base_pointer_id, const uint32_t* index_ids, uint8_t index_count,
    uint32_t* out_result_id) {
  IREE_ASSERT_ARGUMENT(builder);
  IREE_ASSERT_ARGUMENT(out_result_id);

  if (index_count > LOOM_SPIRV_ACCESS_CHAIN_MAX_INDEX_COUNT) {
    IREE_CHECK_UNREACHABLE("SPIR-V access-chain helper capacity");
  }
  uint32_t operands[3 + LOOM_SPIRV_ACCESS_CHAIN_MAX_INDEX_COUNT] = {0};
  operands[0] = result_type_id;
  operands[1] = loom_spirv_module_builder_allocate_id(builder);
  operands[2] = base_pointer_id;
  for (uint8_t i = 0; i < index_count; ++i) {
    operands[3 + i] = index_ids[i];
  }
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_module_builder_section(builder,
                                        LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      LOOM_SPIRV_OP_ACCESS_CHAIN, operands, 3 + index_count));
  *out_result_id = operands[1];
  return iree_ok_status();
}

iree_status_t loom_spirv_module_emit_load_aligned(
    loom_spirv_module_builder_t* builder, uint32_t result_type_id,
    uint32_t pointer_id, uint32_t alignment, uint32_t* out_result_id) {
  IREE_ASSERT_ARGUMENT(builder);
  IREE_ASSERT_ARGUMENT(out_result_id);

  const uint32_t result_id = loom_spirv_module_builder_allocate_id(builder);
  const uint32_t operands[] = {
      result_type_id, result_id,
      pointer_id,     LOOM_SPIRV_MEMORY_ACCESS_ALIGNED_MASK,
      alignment,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_module_builder_section(builder,
                                        LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      LOOM_SPIRV_OP_LOAD, operands, IREE_ARRAYSIZE(operands)));
  *out_result_id = result_id;
  return iree_ok_status();
}

iree_status_t loom_spirv_module_emit_unary_result(
    loom_spirv_module_builder_t* builder, uint32_t opcode,
    uint32_t result_type_id, uint32_t operand_id, uint32_t* out_result_id) {
  IREE_ASSERT_ARGUMENT(builder);
  IREE_ASSERT_ARGUMENT(out_result_id);

  const uint32_t result_id = loom_spirv_module_builder_allocate_id(builder);
  const uint32_t operands[] = {
      result_type_id,
      result_id,
      operand_id,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_module_builder_section(builder,
                                        LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      opcode, operands, IREE_ARRAYSIZE(operands)));
  *out_result_id = result_id;
  return iree_ok_status();
}

iree_status_t loom_spirv_module_emit_binary_result(
    loom_spirv_module_builder_t* builder, uint32_t opcode,
    uint32_t result_type_id, uint32_t lhs_id, uint32_t rhs_id,
    uint32_t* out_result_id) {
  IREE_ASSERT_ARGUMENT(builder);
  IREE_ASSERT_ARGUMENT(out_result_id);

  const uint32_t result_id = loom_spirv_module_builder_allocate_id(builder);
  const uint32_t operands[] = {
      result_type_id,
      result_id,
      lhs_id,
      rhs_id,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_module_builder_section(builder,
                                        LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      opcode, operands, IREE_ARRAYSIZE(operands)));
  *out_result_id = result_id;
  return iree_ok_status();
}

iree_status_t loom_spirv_module_emit_bitcast_if_needed(
    loom_spirv_module_builder_t* builder, uint32_t result_type_id,
    uint32_t operand_type_id, uint32_t operand_id, uint32_t* out_result_id) {
  IREE_ASSERT_ARGUMENT(builder);
  IREE_ASSERT_ARGUMENT(out_result_id);

  if (result_type_id == operand_type_id) {
    *out_result_id = operand_id;
    return iree_ok_status();
  }
  return loom_spirv_module_emit_unary_result(builder, LOOM_SPIRV_OP_BITCAST,
                                             result_type_id, operand_id,
                                             out_result_id);
}

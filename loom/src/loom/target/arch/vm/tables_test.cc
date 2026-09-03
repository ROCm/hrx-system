// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/vm/tables.h"

#include "iree/testing/gtest.h"
#include "loom/ops/scalar/ops.h"

namespace {

TEST(TablesTest, ProjectsInstructionLayout) {
  const uint32_t descriptor =
      loom_vm_instruction_descriptors[IREE_VM_BYTECODE_OPCODE_CONTROL_SWITCH];
  EXPECT_EQ(loom_vm_instruction_descriptor_byte_length(descriptor), 8u);
  EXPECT_EQ(loom_vm_instruction_descriptor_field_count(descriptor), 3u);

  const uint16_t field_base =
      loom_vm_instruction_descriptor_field_base(descriptor);
  ASSERT_LT(field_base + 2u, loom_vm_instruction_field_count);
  EXPECT_EQ(loom_vm_instruction_fields[field_base + 0].byte_offset, 1u);
  EXPECT_EQ(loom_vm_instruction_fields[field_base + 0].role,
            LOOM_VM_INSTRUCTION_FIELD_ROLE_OPERAND);
  EXPECT_EQ(loom_vm_instruction_fields[field_base + 1].byte_offset, 2u);
  EXPECT_EQ(loom_vm_instruction_fields[field_base + 1].role,
            LOOM_VM_INSTRUCTION_FIELD_ROLE_CONSTRAINT_MEMBER);
  EXPECT_EQ(loom_vm_instruction_fields[field_base + 2].byte_offset, 4u);
  EXPECT_EQ(loom_vm_instruction_fields[field_base + 2].role,
            LOOM_VM_INSTRUCTION_FIELD_ROLE_CONSTRAINT_MEMBER);
}

TEST(TablesTest, ProjectsSourceLoweringsInLookupOrder) {
  const loom_vm_source_lowering_t* add_i32 = nullptr;
  for (uint16_t i = 0; i < loom_vm_source_lowering_count; ++i) {
    const loom_vm_source_lowering_t* row = &loom_vm_source_lowerings[i];
    if (row->source_op_kind == LOOM_OP_SCALAR_ADDI &&
        row->scalar_type == LOOM_SCALAR_TYPE_I32) {
      add_i32 = row;
    }
    if (i == 0) continue;
    const loom_vm_source_lowering_t* previous =
        &loom_vm_source_lowerings[i - 1];
    EXPECT_TRUE(previous->source_op_kind < row->source_op_kind ||
                (previous->source_op_kind == row->source_op_kind &&
                 previous->scalar_type < row->scalar_type));
  }
  ASSERT_NE(add_i32, nullptr);
  EXPECT_EQ(add_i32->opcode, IREE_VM_BYTECODE_OPCODE_INTEGER_ADD_I32);
}

}  // namespace

// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/bytecode/disassembler.h"

#include "iree/testing/gtest.h"
#include "iree/vm/bytecode/wire/core.h"
#include "iree/vm/bytecode/wire/module.h"

namespace {

TEST(DisassemblerTest, LooksUpGeneratedInstructionNames) {
  EXPECT_TRUE(
      iree_string_view_equal(iree_vm_bytecode_disassembler_instruction_name(
                                 IREE_VM_BYTECODE_OPCODE_INTEGER_ADD_I32),
                             IREE_SV("integer.add.i32")));
  EXPECT_TRUE(iree_string_view_is_empty(
      iree_vm_bytecode_disassembler_instruction_name(0xFF)));
}

TEST(DisassemblerTest, LooksUpGeneratedModuleRecordNames) {
  EXPECT_TRUE(
      iree_string_view_equal(iree_vm_bytecode_disassembler_module_record_name(
                                 IREE_VM_BYTECODE_MODULE_RECORD_SIGNATURE_ROW),
                             IREE_SV("signature_row")));
  EXPECT_TRUE(iree_string_view_is_empty(
      iree_vm_bytecode_disassembler_module_record_name(UINT8_MAX)));
}

}  // namespace

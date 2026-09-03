// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/bytecode/tooling.h"

#include "iree/vm/bytecode/tooling/data.inl"
#include "iree/vm/bytecode/wire/module.h"

static iree_string_view_t iree_vm_bytecode_tooling_string_at(uint16_t offset) {
  const uint8_t* bstring = iree_vm_bytecode_tooling_strings + offset;
  return iree_make_string_view((const char*)bstring + 1, bstring[0]);
}

IREE_API_EXPORT iree_string_view_t
iree_vm_bytecode_instruction_name(uint8_t opcode) {
  return iree_vm_bytecode_tooling_string_at(
      iree_vm_bytecode_instruction_name_offsets[opcode]);
}

IREE_API_EXPORT iree_string_view_t
iree_vm_bytecode_module_record_name(uint8_t record_ordinal) {
  if (record_ordinal >= IREE_VM_BYTECODE_MODULE_RECORD_COUNT) {
    return iree_string_view_empty();
  }
  return iree_vm_bytecode_tooling_string_at(
      iree_vm_bytecode_module_record_name_offsets[record_ordinal]);
}

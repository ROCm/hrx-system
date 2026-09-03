// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Compact VM instruction and source-lowering table contracts.

#ifndef LOOM_TARGET_ARCH_VM_TABLES_H_
#define LOOM_TARGET_ARCH_VM_TABLES_H_

#include "iree/base/api.h"
#include "iree/vm/bytecode/wire/core.h"
#include "loom/ir/ir.h"
#include "loom/ir/scalar_type.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Semantic use of one encoded VM instruction field.
enum loom_vm_instruction_field_role_e {
  // No field role. Reserved for zero-initialized rows.
  LOOM_VM_INSTRUCTION_FIELD_ROLE_NONE = 0,
  // Destination value populated by the instruction.
  LOOM_VM_INSTRUCTION_FIELD_ROLE_RESULT = 1,
  // Source value consumed by the instruction.
  LOOM_VM_INSTRUCTION_FIELD_ROLE_OPERAND = 2,
  // Immediate value encoded directly in the instruction.
  LOOM_VM_INSTRUCTION_FIELD_ROLE_IMMEDIATE = 3,
  // Member of a constraint evaluated across several fields.
  LOOM_VM_INSTRUCTION_FIELD_ROLE_CONSTRAINT_MEMBER = 4,
  // Canonical wire padding with no semantic value.
  LOOM_VM_INSTRUCTION_FIELD_ROLE_PADDING = 5,
};
typedef uint8_t loom_vm_instruction_field_role_t;

// Physical layout and semantic role of one VM instruction field.
typedef struct loom_vm_instruction_field_t {
  // Byte offset from the beginning of the instruction record.
  uint8_t byte_offset;
  // Encoded byte length of the field.
  uint8_t byte_length;
  // Semantic role from loom_vm_instruction_field_role_e.
  loom_vm_instruction_field_role_t role;
  // Reserved for future table revisions and always zero.
  uint8_t reserved;
} loom_vm_instruction_field_t;

// Direct source-op and scalar-type mapping to one VM opcode.
typedef struct loom_vm_source_lowering_t {
  // Loom source operation kind.
  loom_op_kind_t source_op_kind;
  // Required source and result scalar type.
  loom_scalar_type_t scalar_type;
  // VM instruction opcode selected for this combination.
  iree_vm_bytecode_opcode_t opcode;
} loom_vm_source_lowering_t;

// Packed instruction descriptors indexed directly by VM opcode. Zero denotes
// an unknown opcode. Nonzero words encode, from most- to least-significant
// bits, field_base:u16, field_count:u8, and byte_length:u8.
extern const uint32_t
    loom_vm_instruction_descriptors[IREE_VM_BYTECODE_OPCODE_CAPACITY];

// Instruction fields referenced by the packed descriptor table.
extern const loom_vm_instruction_field_t loom_vm_instruction_fields[];

// Number of rows in |loom_vm_instruction_fields|.
extern const uint16_t loom_vm_instruction_field_count;

// Source lowering rows sorted by (source_op_kind, scalar_type).
extern const loom_vm_source_lowering_t loom_vm_source_lowerings[];

// Number of rows in |loom_vm_source_lowerings|.
extern const uint16_t loom_vm_source_lowering_count;

// Returns the encoded record byte length from a packed descriptor.
static inline uint8_t loom_vm_instruction_descriptor_byte_length(
    uint32_t descriptor) {
  return (uint8_t)descriptor;
}

// Returns the field count from a packed descriptor.
static inline uint8_t loom_vm_instruction_descriptor_field_count(
    uint32_t descriptor) {
  return (uint8_t)(descriptor >> 8);
}

// Returns the first field ordinal from a packed descriptor.
static inline uint16_t loom_vm_instruction_descriptor_field_base(
    uint32_t descriptor) {
  return (uint16_t)(descriptor >> 16);
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // LOOM_TARGET_ARCH_VM_TABLES_H_

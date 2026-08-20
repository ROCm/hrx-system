// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// GENERATED FILE: DO NOT EDIT.
// Regenerate with:
//   iree-bazel-run //runtime/src/iree/vm/bytecode/spec:generate
// Natural-layout instruction records for core.family.function.
// Multi-byte fields are little-endian and naturally aligned.
// clang-format off
#ifndef IREE_VM_BYTECODE_GENERATED_WIRE_CORE_FUNCTION_H_
#define IREE_VM_BYTECODE_GENERATED_WIRE_CORE_FUNCTION_H_

#include <stdint.h>

// Page 0x00, opcode 0x20: Writes a canonical null function value.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Destination function-register ordinal.
  uint8_t dst_f8;
  // Canonical zero padding.
  uint16_t zero_padding_u16;
} iree_vm_isa_func_null_record_t;

// Page 0x00, opcode 0x21: Tests a complete function carrier for canonical null.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Destination Boolean value-register ordinal.
  uint8_t dst_v8;
  // Source function-register ordinal.
  uint8_t src_f8;
  // Canonical zero padding.
  uint8_t zero_padding_u8;
} iree_vm_isa_func_compare_null_record_t;

// Page 0x00, opcode 0x22: Copies one complete function carrier.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Destination function-register ordinal.
  uint8_t dst_f8;
  // Source function-register ordinal.
  uint8_t src_f8;
  // Canonical zero padding.
  uint8_t zero_padding_u8;
} iree_vm_isa_func_copy_record_t;

// Page 0x00, opcode 0x23: Materializes a verified local or imported function
// target.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Destination function-register ordinal.
  uint8_t dst_f8;
  // Local, required-import, or optional-import target kind.
  uint8_t target_kind_u8;
  // Canonical zero padding.
  uint8_t zero_padding_u8;
  // Direct ordinal in the domain selected by target_kind_u8.
  uint16_t target_ordinal_u16;
  // Exact structural callable type expected for the target.
  uint16_t callable_type_ordinal_u16;
} iree_vm_isa_func_address_record_t;

// Page 0x00, opcode 0x24: Tests whether an optional import resolved.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Destination Boolean value-register ordinal.
  uint8_t dst_v8;
  // Direct optional-import ordinal.
  uint16_t import_ordinal_u16;
} iree_vm_isa_func_import_resolved_record_t;

// Page 0x00, opcode 0x25: Loads one complete function-local cell.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Destination function-register ordinal.
  uint8_t dst_f8;
  // Direct function-local cell ordinal.
  uint16_t local_ordinal_u16;
} iree_vm_isa_func_stack_load_record_t;

// Page 0x00, opcode 0x26: Stores one complete function carrier into a local
// cell.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Source function-register ordinal.
  uint8_t src_f8;
  // Direct function-local cell ordinal.
  uint16_t local_ordinal_u16;
} iree_vm_isa_func_stack_store_record_t;

#endif  // IREE_VM_BYTECODE_GENERATED_WIRE_CORE_FUNCTION_H_
// clang-format on

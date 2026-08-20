// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// GENERATED FILE: DO NOT EDIT.
// Regenerate with:
//   iree-bazel-run //runtime/src/iree/vm/bytecode/spec:generate
// Natural-layout instruction records for core.family.global.
// Multi-byte fields are little-endian and naturally aligned.
// clang-format off
#ifndef IREE_VM_BYTECODE_GENERATED_WIRE_CORE_GLOBAL_H_
#define IREE_VM_BYTECODE_GENERATED_WIRE_CORE_GLOBAL_H_

#include <stdint.h>

// Page 0x00, opcode 0x30: Loads a set-once value global.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Value-register side of the global transfer.
  uint8_t dst_v8;
  // Direct module-local value.immutable global ordinal.
  uint16_t global_u16;
} iree_vm_isa_global_value_immutable_load_record_t;

// Page 0x00, opcode 0x31: Initializes one set-once value global.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Value-register side of the global transfer.
  uint8_t src_v8;
  // Direct module-local value.immutable global ordinal.
  uint16_t global_u16;
} iree_vm_isa_global_value_immutable_store_record_t;

// Page 0x00, opcode 0x32: Loads a mutable value global.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Value-register side of the global transfer.
  uint8_t dst_v8;
  // Direct module-local value.mutable global ordinal.
  uint16_t global_u16;
} iree_vm_isa_global_value_mutable_load_record_t;

// Page 0x00, opcode 0x33: Stores a mutable value global.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Value-register side of the global transfer.
  uint8_t src_v8;
  // Direct module-local value.mutable global ordinal.
  uint16_t global_u16;
} iree_vm_isa_global_value_mutable_store_record_t;

// Page 0x00, opcode 0x34: Borrows a set-once ref global.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Ref-register side of the global transfer.
  uint8_t dst_r8;
  // Direct module-local ref.immutable global ordinal.
  uint16_t global_u16;
} iree_vm_isa_global_ref_immutable_load_borrow_record_t;

// Page 0x00, opcode 0x35: Publishes a ref into a set-once global.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Ref-register side of the global transfer.
  uint8_t src_r8;
  // Direct module-local ref.immutable global ordinal.
  uint16_t global_u16;
} iree_vm_isa_global_ref_immutable_store_move_record_t;

// Page 0x00, opcode 0x36: Retains a snapshot of a mutable ref global.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Ref-register side of the global transfer.
  uint8_t dst_r8;
  // Direct module-local ref.mutable global ordinal.
  uint16_t global_u16;
} iree_vm_isa_global_ref_mutable_load_retain_record_t;

// Page 0x00, opcode 0x37: Moves a ref into a mutable global.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Ref-register side of the global transfer.
  uint8_t src_r8;
  // Direct module-local ref.mutable global ordinal.
  uint16_t global_u16;
} iree_vm_isa_global_ref_mutable_store_move_record_t;

// Page 0x00, opcode 0x38: Loads a set-once function global.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Function-register side of the global transfer.
  uint8_t dst_f8;
  // Direct module-local func.immutable global ordinal.
  uint16_t global_u16;
} iree_vm_isa_global_func_immutable_load_record_t;

// Page 0x00, opcode 0x39: Initializes one set-once function global.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Function-register side of the global transfer.
  uint8_t src_f8;
  // Direct module-local func.immutable global ordinal.
  uint16_t global_u16;
} iree_vm_isa_global_func_immutable_store_record_t;

// Page 0x00, opcode 0x3A: Loads a mutable function global.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Function-register side of the global transfer.
  uint8_t dst_f8;
  // Direct module-local func.mutable global ordinal.
  uint16_t global_u16;
} iree_vm_isa_global_func_mutable_load_record_t;

// Page 0x00, opcode 0x3B: Stores a mutable function global.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Function-register side of the global transfer.
  uint8_t src_f8;
  // Direct module-local func.mutable global ordinal.
  uint16_t global_u16;
} iree_vm_isa_global_func_mutable_store_record_t;

#endif  // IREE_VM_BYTECODE_GENERATED_WIRE_CORE_GLOBAL_H_
// clang-format on

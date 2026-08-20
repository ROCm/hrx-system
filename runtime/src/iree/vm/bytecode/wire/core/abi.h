// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// GENERATED FILE: DO NOT EDIT.
// Regenerate with:
//   iree-bazel-run //runtime/src/iree/vm/bytecode/spec:generate
// Natural-layout instruction records for core.family.abi.
// Multi-byte fields are little-endian and naturally aligned.
// clang-format off
#ifndef IREE_VM_BYTECODE_GENERATED_WIRE_CORE_ABI_H_
#define IREE_VM_BYTECODE_GENERATED_WIRE_CORE_ABI_H_

#include <stdint.h>

// Page 0x00, opcode 0xC0: Loads one exact value argument overflow cell.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Destination value-register ordinal.
  uint8_t dst_v8;
  // Zero-based argument.value overflow-slot ordinal.
  uint16_t slot_u16;
} iree_vm_isa_value_abi_argument_load_record_t;

// Page 0x00, opcode 0xC1: Stores one exact value result overflow cell.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Source value-register ordinal.
  uint8_t src_v8;
  // Zero-based result.value overflow-slot ordinal.
  uint16_t slot_u16;
} iree_vm_isa_value_abi_result_store_record_t;

// Page 0x00, opcode 0xC2: Borrows one ref argument overflow slot into a
// register.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Ref-register side of the overflow transfer.
  uint8_t dst_r8;
  // Zero-based argument.ref overflow-slot ordinal.
  uint16_t slot_u16;
} iree_vm_isa_ref_abi_argument_load_borrow_record_t;

// Page 0x00, opcode 0xC3: Consumes one ref argument overflow slot into a
// register.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Ref-register side of the overflow transfer.
  uint8_t dst_r8;
  // Zero-based argument.ref overflow-slot ordinal.
  uint16_t slot_u16;
} iree_vm_isa_ref_abi_argument_load_move_record_t;

// Page 0x00, opcode 0xC4: Publishes one ref register into a result overflow
// slot.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Ref-register side of the overflow transfer.
  uint8_t src_r8;
  // Zero-based result.ref overflow-slot ordinal.
  uint16_t slot_u16;
} iree_vm_isa_ref_abi_result_store_move_record_t;

// Page 0x00, opcode 0xC5: Loads one complete function argument overflow
// carrier.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Function-register side of the overflow transfer.
  uint8_t dst_f8;
  // Zero-based argument.function overflow-slot ordinal.
  uint16_t slot_u16;
} iree_vm_isa_func_abi_argument_load_record_t;

// Page 0x00, opcode 0xC6: Stores one complete function result overflow carrier.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Function-register side of the overflow transfer.
  uint8_t src_f8;
  // Zero-based result.function overflow-slot ordinal.
  uint16_t slot_u16;
} iree_vm_isa_func_abi_result_store_record_t;

#endif  // IREE_VM_BYTECODE_GENERATED_WIRE_CORE_ABI_H_
// clang-format on

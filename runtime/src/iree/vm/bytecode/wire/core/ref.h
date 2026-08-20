// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// GENERATED FILE: DO NOT EDIT.
// Regenerate with:
//   iree-bazel-run //runtime/src/iree/vm/bytecode/spec:generate
// Natural-layout instruction records for core.family.ref.
// Multi-byte fields are little-endian and naturally aligned.
// clang-format off
#ifndef IREE_VM_BYTECODE_GENERATED_WIRE_CORE_REF_H_
#define IREE_VM_BYTECODE_GENERATED_WIRE_CORE_REF_H_

#include <stdint.h>

// Page 0x00, opcode 0xC8: Clears a ref register to canonical null.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Destination ref-register ordinal to clear.
  uint8_t dst_r8;
  // Canonical zero padding.
  uint16_t zero_padding_u16;
} iree_vm_isa_ref_null_record_t;

// Page 0x00, opcode 0xC9: Tests a ref register for canonical null.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Destination Boolean value-register ordinal.
  uint8_t dst_v8;
  // Source ref-register ordinal inspected without dereference.
  uint8_t src_r8;
  // Canonical zero padding.
  uint8_t zero_padding_u8;
} iree_vm_isa_ref_compare_null_record_t;

// Page 0x00, opcode 0xCA: Tests exact typed object identity of two refs.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Destination Boolean value-register ordinal.
  uint8_t dst_v8;
  // Left ref-register ordinal.
  uint8_t lhs_r8;
  // Right ref-register ordinal.
  uint8_t rhs_r8;
} iree_vm_isa_ref_compare_eq_record_t;

// Page 0x00, opcode 0xCB: Creates an owned copy of a ref in another register.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Destination ref-register ordinal replaced by the new owner.
  uint8_t dst_r8;
  // Source ref-register ordinal retained without mutation.
  uint8_t src_r8;
  // Canonical zero padding.
  uint8_t zero_padding_u8;
} iree_vm_isa_ref_retain_record_t;

// Page 0x00, opcode 0xCC: Destructively transfers one complete ref-register
// state.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Destination ref-register ordinal replaced by the moved state.
  uint8_t dst_r8;
  // Source ref-register ordinal cleared by the transfer.
  uint8_t src_r8;
  // Canonical zero padding.
  uint8_t zero_padding_u8;
} iree_vm_isa_ref_move_record_t;

// Page 0x00, opcode 0xCD: Releases or clears one ref register.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Source ref-register ordinal to discard.
  uint8_t src_r8;
  // Canonical zero padding.
  uint16_t zero_padding_u16;
} iree_vm_isa_ref_discard_record_t;

// Page 0x00, opcode 0xCE: Retains a local ref slot into a register.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Destination ref-register ordinal.
  uint8_t dst_r8;
  // Direct function-local ref-slot ordinal.
  uint16_t slot_u16;
} iree_vm_isa_ref_stack_load_retain_record_t;

// Page 0x00, opcode 0xCF: Moves a local ref slot into a register.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Destination ref-register ordinal.
  uint8_t dst_r8;
  // Direct function-local ref-slot ordinal.
  uint16_t slot_u16;
} iree_vm_isa_ref_stack_load_move_record_t;

// Page 0x00, opcode 0xD0: Retains a ref register into a local slot.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Source ref-register ordinal retained without mutation.
  uint8_t src_r8;
  // Direct function-local ref-slot ordinal.
  uint16_t slot_u16;
} iree_vm_isa_ref_stack_store_retain_record_t;

// Page 0x00, opcode 0xD1: Moves a ref register into a local slot.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Source ref-register ordinal cleared by the transfer.
  uint8_t src_r8;
  // Direct function-local ref-slot ordinal.
  uint16_t slot_u16;
} iree_vm_isa_ref_stack_store_move_record_t;

// Page 0x00, opcode 0xD2: Releases or clears one local ref slot.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Canonical zero padding.
  uint8_t zero_padding_u8;
  // Direct function-local ref-slot ordinal.
  uint16_t slot_u16;
} iree_vm_isa_ref_stack_discard_record_t;

#endif  // IREE_VM_BYTECODE_GENERATED_WIRE_CORE_REF_H_
// clang-format on

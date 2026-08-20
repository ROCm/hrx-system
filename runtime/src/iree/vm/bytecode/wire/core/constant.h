// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// GENERATED FILE: DO NOT EDIT.
// Regenerate with:
//   iree-bazel-run //runtime/src/iree/vm/bytecode/spec:generate
// Natural-layout instruction records for core.family.constant.
// Multi-byte fields are little-endian and naturally aligned.
// clang-format off
#ifndef IREE_VM_BYTECODE_GENERATED_WIRE_CORE_CONSTANT_H_
#define IREE_VM_BYTECODE_GENERATED_WIRE_CORE_CONSTANT_H_

#include <stdint.h>

// Page 0x00, opcode 0x18: Clears a complete value-register cell.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Destination value-register ordinal.
  uint8_t dst_v8;
  // Canonical zero padding.
  uint16_t zero_padding_u16;
} iree_vm_isa_constant_zero_record_t;

// Page 0x00, opcode 0x19: Sign-extends a 16-bit immediate into a value cell.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Destination value-register ordinal.
  uint8_t dst_v8;
  // Signed little-endian two's-complement immediate.
  int16_t immediate_i16;
} iree_vm_isa_constant_s16_record_t;

// Page 0x00, opcode 0x1A: Loads an inline 32-bit pattern into a value cell.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Destination value-register ordinal.
  uint8_t dst_v8;
  // Canonical zero padding.
  uint16_t zero_padding_u16;
  // Arbitrary little-endian low 32 bits.
  uint32_t bits_u32le;
} iree_vm_isa_constant_i32_record_t;

// Page 0x00, opcode 0x1B: Loads an inline 64-bit pattern into a value cell.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Destination value-register ordinal.
  uint8_t dst_v8;
  // Canonical zero padding.
  uint16_t zero_padding_u16;
  // Arbitrary little-endian low 32 bits.
  uint32_t bits_low_u32le;
  // Arbitrary little-endian high 32 bits.
  uint32_t bits_high_u32le;
} iree_vm_isa_constant_i64_record_t;

// Page 0x00, opcode 0x1C: Loads the low 32 bits of a module constant-pool cell.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Destination value-register ordinal.
  uint8_t dst_v8;
  // Direct module constant-pool ordinal.
  uint16_t pool_u16;
} iree_vm_isa_constant_pool_load_i32_record_t;

// Page 0x00, opcode 0x1D: Loads a complete module constant-pool cell.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Destination value-register ordinal.
  uint8_t dst_v8;
  // Direct module constant-pool ordinal.
  uint16_t pool_u16;
} iree_vm_isa_constant_pool_load_i64_record_t;

#endif  // IREE_VM_BYTECODE_GENERATED_WIRE_CORE_CONSTANT_H_
// clang-format on

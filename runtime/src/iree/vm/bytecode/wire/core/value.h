// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// GENERATED FILE: DO NOT EDIT.
// Regenerate with:
//   iree-bazel-run //runtime/src/iree/vm/bytecode/spec:generate
// Natural-layout instruction records for core.family.value.
// Multi-byte fields are little-endian and naturally aligned.
// clang-format off
#ifndef IREE_VM_BYTECODE_GENERATED_WIRE_CORE_VALUE_H_
#define IREE_VM_BYTECODE_GENERATED_WIRE_CORE_VALUE_H_

#include <stdint.h>

// Page 0x00, opcode 0x10: Copies one complete value-register cell.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Destination value-register ordinal.
  uint8_t dst_v8;
  // Source value-register ordinal.
  uint8_t src_v8;
  // Canonical zero padding.
  uint8_t zero_padding_u8;
} iree_vm_isa_value_copy_record_t;

// Page 0x00, opcode 0x11: Selects one complete value-register cell.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Destination value-register ordinal.
  uint8_t dst_v8;
  // Complete 64-bit truth-condition value register.
  uint8_t condition_v8;
  // Source selected when condition_v8 is nonzero.
  uint8_t true_v8;
  // Source selected when condition_v8 is zero.
  uint8_t false_v8;
  // Canonical zero padding.
  uint8_t zero_padding_u8[3];
} iree_vm_isa_value_select_record_t;

#endif  // IREE_VM_BYTECODE_GENERATED_WIRE_CORE_VALUE_H_
// clang-format on

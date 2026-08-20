// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// GENERATED FILE: DO NOT EDIT.
// Regenerate with:
//   iree-bazel-run //runtime/src/iree/vm/bytecode/spec:generate
// Natural-layout instruction records for hal.family.buffer.
// Multi-byte fields are little-endian and naturally aligned.
// clang-format off
#ifndef IREE_VM_BYTECODE_GENERATED_WIRE_HAL_BUFFER_H_
#define IREE_VM_BYTECODE_GENERATED_WIRE_HAL_BUFFER_H_

#include <stdint.h>

// Page 0xF0, opcode 0x03: Allocates one long-lived HAL buffer on an explicit
// device.
typedef struct {
  // Architectural instruction page selector.
  uint8_t page_u8;
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // result_nonnull exact hal.buffer ref.
  uint8_t dst_r8;
  // required exact hal.device ref.
  uint8_t device_r8;
  // Unsigned HAL buffer-usage flag bits.
  uint8_t usage_v8;
  // Unsigned HAL memory-access flag bits.
  uint8_t access_v8;
  // Unsigned HAL memory-type flag bits.
  uint8_t memory_type_v8;
  // Complete u64 queue-affinity bitset.
  uint8_t affinity_v8;
  // Unsigned minimum allocation alignment in bytes.
  uint8_t min_alignment_v8;
  // Unsigned requested allocation size in device bytes.
  uint8_t allocation_size_v8;
  // Canonical zero padding.
  uint16_t zero_padding_u16;
} iree_vm_isa_hal_buffer_allocate_record_t;

// Page 0xF0, opcode 0x04: Creates one deterministic scoped CPU mapping.
typedef struct {
  // Architectural instruction page selector.
  uint8_t page_u8;
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // result_nonnull exact vm.buffer ref.
  uint8_t dst_r8;
  // required exact hal.buffer ref.
  uint8_t source_buffer_r8;
  // Unsigned HAL-buffer source offset.
  uint8_t source_offset_v8;
  // Unsigned requested mapping length.
  uint8_t source_length_v8;
  // Static mapping access: READ=1, WRITE=2, or READ_WRITE=3.
  uint8_t access_u8;
  // Canonical zero padding.
  uint8_t zero_padding_u8;
} iree_vm_isa_hal_buffer_map_record_t;

// Page 0xF0, opcode 0x05: Consumes and synchronously closes one mapped
// vm.buffer root.
typedef struct {
  // Architectural instruction page selector.
  uint8_t page_u8;
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Required mapped-root vm.buffer consumed after successful preflight.
  uint8_t mapped_buffer_r8;
  // Canonical zero padding.
  uint8_t zero_padding_u8;
} iree_vm_isa_hal_buffer_unmap_record_t;

#endif  // IREE_VM_BYTECODE_GENERATED_WIRE_HAL_BUFFER_H_
// clang-format on

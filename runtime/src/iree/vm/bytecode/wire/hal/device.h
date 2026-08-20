// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// GENERATED FILE: DO NOT EDIT.
// Regenerate with:
//   iree-bazel-run //runtime/src/iree/vm/bytecode/spec:generate
// Natural-layout instruction records for hal.family.device.
// Multi-byte fields are little-endian and naturally aligned.
// clang-format off
#ifndef IREE_VM_BYTECODE_GENERATED_WIRE_HAL_DEVICE_H_
#define IREE_VM_BYTECODE_GENERATED_WIRE_HAL_DEVICE_H_

#include <stdint.h>

// Page 0xF0, opcode 0x01: Returns the exact immutable device-group count.
typedef struct {
  // Architectural instruction page selector.
  uint8_t page_u8;
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Unsigned 64-bit device-count result.
  uint8_t dst_v8;
  // required exact hal.device_group ref.
  uint8_t group_r8;
} iree_vm_isa_hal_device_group_count_record_t;

// Page 0xF0, opcode 0x02: Retains and returns one device selected by index.
typedef struct {
  // Architectural instruction page selector.
  uint8_t page_u8;
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // result_nonnull exact hal.device ref.
  uint8_t dst_r8;
  // required exact hal.device_group ref.
  uint8_t group_r8;
  // Unsigned 64-bit device index.
  uint8_t index_v8;
  // Canonical zero padding.
  uint8_t zero_padding_u8[3];
} iree_vm_isa_hal_device_group_get_record_t;

// Page 0xF0, opcode 0x0A: Loads an ordered selected executable-function table.
typedef struct {
  // Architectural instruction page selector.
  uint8_t page_u8;
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // result_nonnull exact hal.executable_function_table ref.
  uint8_t dst_r8;
  // required exact hal.device ref.
  uint8_t device_r8;
  // Complete u64 queue-affinity bitset.
  uint8_t affinity_v8;
  // Canonical zero padding.
  uint8_t zero_padding0_u8;
  // Nonempty host-registered executable resolver name.
  uint16_t resolver_string_u16;
  // nullable exact vm.buffer ref.
  uint8_t payload_vm_buffer_r8_nullable;
  // Canonical zero padding.
  uint8_t zero_padding1_u8;
  // Canonical executable-name-table rodata ordinal.
  uint16_t name_table_u16;
  // Four-byte-aligned local base of selected u32 name indices.
  uint16_t selected_ordinal_base_u16;
  // Number of selected u32 name indices.
  uint16_t selected_ordinal_count_u16;
} iree_vm_isa_hal_executable_load_record_t;

#endif  // IREE_VM_BYTECODE_GENERATED_WIRE_HAL_DEVICE_H_
// clang-format on

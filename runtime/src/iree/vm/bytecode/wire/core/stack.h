// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// GENERATED FILE: DO NOT EDIT.
// Regenerate with:
//   iree-bazel-run //runtime/src/iree/vm/bytecode/spec:generate
// Natural-layout instruction records for core.family.stack.
// Multi-byte fields are little-endian and naturally aligned.
// clang-format off
#ifndef IREE_VM_BYTECODE_GENERATED_WIRE_CORE_STACK_H_
#define IREE_VM_BYTECODE_GENERATED_WIRE_CORE_STACK_H_

#include <stdint.h>

// Page 0x00, opcode 0xA8: Loads one fixed lane group from local bytes.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // First destination value-register ordinal.
  uint8_t dst_v8;
  // Static base of an alignment-independent local-byte lane range.
  uint16_t base_u16;
  // Closed lane element-width and lane-count selector.
  uint8_t format_u8;
  // Canonical zero padding.
  uint8_t zero_padding_u8[3];
} iree_vm_isa_stack_load_record_t;

// Page 0x00, opcode 0xA9: Stores one fixed lane group into local bytes.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Canonical zero padding.
  uint8_t zero_padding_u8;
  // Static base of an alignment-independent local-byte lane range.
  uint16_t base_u16;
  // First source value-register ordinal.
  uint8_t src_v8;
  // Closed lane element-width and lane-count selector.
  uint8_t format_u8;
  // Canonical zero padding.
  uint16_t zero_padding_u16;
} iree_vm_isa_stack_store_record_t;

// Page 0x00, opcode 0xAA: Dynamically indexes a checked local-byte lane load.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // First lane value-register ordinal.
  uint8_t dst_v8;
  // Static base of an alignment-independent local-byte lane range.
  uint16_t base_u16;
  // Unsigned 64-bit dynamic index value.
  uint8_t index_v8;
  // Nonzero byte scale applied to index_v8.
  uint8_t scale_u8;
  // Closed lane element-width and lane-count selector.
  uint8_t format_u8;
  // Canonical zero padding.
  uint8_t zero_padding_u8;
} iree_vm_isa_stack_load_indexed_record_t;

// Page 0x00, opcode 0xAB: Dynamically indexes a checked local-byte lane store.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Canonical zero padding.
  uint8_t zero_padding_u8;
  // Static base of an alignment-independent local-byte lane range.
  uint16_t base_u16;
  // Unsigned 64-bit dynamic index value.
  uint8_t index_v8;
  // Nonzero byte scale applied to index_v8.
  uint8_t scale_u8;
  // First lane value-register ordinal.
  uint8_t src_v8;
  // Closed lane element-width and lane-count selector.
  uint8_t format_u8;
} iree_vm_isa_stack_store_indexed_record_t;

// Page 0x00, opcode 0xAC: Repeats a low-byte pattern across a local-byte range.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Canonical zero padding.
  uint8_t zero_padding_u8;
  // Static local-byte target base.
  uint16_t target_base_u16;
  // Exact byte length of the statically checked local range.
  uint16_t length_u16;
  // Low-byte pattern source.
  uint8_t pattern_v8;
  // Pattern width in bytes.
  uint8_t pattern_width_u8;
} iree_vm_isa_stack_fill_record_t;

// Page 0x00, opcode 0xAD: Moves one possibly overlapping local-byte range.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Canonical zero padding.
  uint8_t zero_padding_u8;
  // Static local-byte target base.
  uint16_t target_u16;
  // Static local-byte source base.
  uint16_t source_u16;
  // Exact byte length of the statically checked local range.
  uint16_t length_u16;
} iree_vm_isa_stack_copy_record_t;

// Page 0x00, opcode 0xAE: Lexicographically compares two local-byte ranges.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Canonical signed i32 comparison result.
  uint8_t dst_v8;
  // Static left range base.
  uint16_t lhs_u16;
  // Static right range base.
  uint16_t rhs_u16;
  // Exact byte length of the statically checked local range.
  uint16_t length_u16;
} iree_vm_isa_stack_compare_record_t;

// Page 0x00, opcode 0xAF: Copies immutable module bytes into local bytes.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Canonical zero padding.
  uint8_t zero_padding_u8;
  // Static local-byte target base.
  uint16_t target_u16;
  // Direct module rodata-block ordinal.
  uint16_t rodata_u16;
  // Exact byte length of the statically checked local range.
  uint16_t length_u16;
  // Static source offset in the selected rodata block.
  uint32_t source_offset_u32;
} iree_vm_isa_stack_copy_rodata_record_t;

// Page 0x00, opcode 0xB0: Copies a checked vm.buffer range into local bytes.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Canonical zero padding.
  uint8_t zero_padding_u8;
  // Static local-byte target base.
  uint16_t target_u16;
  // Required exact vm.buffer ref borrowed for this instruction.
  uint8_t buffer_r8;
  // Unsigned dynamic source offset in buffer_r8.
  uint8_t source_offset_v8;
  // Exact byte length of the statically checked local range.
  uint16_t length_u16;
} iree_vm_isa_stack_copy_from_buffer_record_t;

// Page 0x00, opcode 0xB1: Copies local bytes into a checked vm.buffer range.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Required exact vm.buffer ref borrowed for this instruction.
  uint8_t buffer_r8;
  // Unsigned dynamic target offset in buffer_r8.
  uint8_t target_offset_v8;
  // Canonical zero padding.
  uint8_t zero_padding_u8;
  // Static local-byte source base.
  uint16_t source_u16;
  // Exact byte length of the statically checked local range.
  uint16_t length_u16;
} iree_vm_isa_stack_copy_to_buffer_record_t;

// Page 0x00, opcode 0xB2: Fills aligned i32 cells from one signed immediate.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Canonical zero padding.
  uint8_t zero_padding_u8;
  // 4-byte-aligned local target base.
  uint16_t target_u16;
  // Number of i32 cells to write.
  uint16_t count_u16;
  // Signed little-endian two's-complement fill immediate.
  int16_t immediate_i16;
} iree_vm_isa_stack_const_s16_i32_record_t;

// Page 0x00, opcode 0xB3: Fills aligned i64 cells from one signed immediate.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Canonical zero padding.
  uint8_t zero_padding_u8;
  // 8-byte-aligned local target base.
  uint16_t target_u16;
  // Number of i64 cells to write.
  uint16_t count_u16;
  // Signed little-endian two's-complement fill immediate.
  int16_t immediate_i16;
} iree_vm_isa_stack_const_s16_i64_record_t;

// Page 0x00, opcode 0xB4: Writes 2 zero-extended u16 immediates as i32 cells.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Canonical zero padding.
  uint8_t zero_padding_u8;
  // 4-byte-aligned local target base.
  uint16_t target_u16;
  // Exact 2-element little-endian immediate payload.
  uint16_t immediates_le[2];
} iree_vm_isa_stack_pack_i32_u16_x2_record_t;

// Page 0x00, opcode 0xB5: Writes 4 zero-extended u16 immediates as i32 cells.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Canonical zero padding.
  uint8_t zero_padding_u8;
  // 4-byte-aligned local target base.
  uint16_t target_u16;
  // Exact 4-element little-endian immediate payload.
  uint16_t immediates_le[4];
} iree_vm_isa_stack_pack_i32_u16_x4_record_t;

// Page 0x00, opcode 0xB6: Writes 8 zero-extended u16 immediates as i32 cells.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Canonical zero padding.
  uint8_t zero_padding_u8;
  // 4-byte-aligned local target base.
  uint16_t target_u16;
  // Exact 8-element little-endian immediate payload.
  uint16_t immediates_le[8];
} iree_vm_isa_stack_pack_i32_u16_x8_record_t;

// Page 0x00, opcode 0xB7: Writes 2 zero-extended u32 immediates as i64 cells.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Canonical zero padding.
  uint8_t zero_padding_u8;
  // 8-byte-aligned local target base.
  uint16_t target_u16;
  // Exact 2-element little-endian immediate payload.
  uint32_t immediates_le[2];
} iree_vm_isa_stack_pack_i64_u32_x2_record_t;

// Page 0x00, opcode 0xB8: Writes 4 zero-extended u32 immediates as i64 cells.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Canonical zero padding.
  uint8_t zero_padding_u8;
  // 8-byte-aligned local target base.
  uint16_t target_u16;
  // Exact 4-element little-endian immediate payload.
  uint32_t immediates_le[4];
} iree_vm_isa_stack_pack_i64_u32_x4_record_t;

// Page 0x00, opcode 0xB9: Writes 8 zero-extended u32 immediates as i64 cells.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Canonical zero padding.
  uint8_t zero_padding_u8;
  // 8-byte-aligned local target base.
  uint16_t target_u16;
  // Exact 8-element little-endian immediate payload.
  uint32_t immediates_le[8];
} iree_vm_isa_stack_pack_i64_u32_x8_record_t;

#endif  // IREE_VM_BYTECODE_GENERATED_WIRE_CORE_STACK_H_
// clang-format on

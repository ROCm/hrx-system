// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// GENERATED FILE: DO NOT EDIT.
// Regenerate with:
//   iree-bazel-run //runtime/src/iree/vm/bytecode/spec:generate
// Natural-layout instruction records for core.family.buffer.
// Multi-byte fields are little-endian and naturally aligned.
// clang-format off
#ifndef IREE_VM_BYTECODE_GENERATED_WIRE_CORE_BUFFER_H_
#define IREE_VM_BYTECODE_GENERATED_WIRE_CORE_BUFFER_H_

#include <stdint.h>

// Selects the exact replacement function for an atomic carrier. Integer
// arithmetic wraps at carrier width; floating operations inherit the
// selected-width floating profile and float.minmax rules.
typedef uint8_t iree_vm_isa_buffer_atomic_kind_t;
enum {
  // Replaces the carrier bits with operand bits.
  IREE_VM_ISA_BUFFER_ATOMIC_KIND_EXCHANGE_INTEGER = 0x00,
  // Replaces the f32/f64 carrier bits with operand bits.
  IREE_VM_ISA_BUFFER_ATOMIC_KIND_EXCHANGE_FLOAT = 0x01,
  // Replaces with unsigned modular old+operand.
  IREE_VM_ISA_BUFFER_ATOMIC_KIND_ADD_INTEGER = 0x02,
  // Replaces with selected-width IEEE old+operand.
  IREE_VM_ISA_BUFFER_ATOMIC_KIND_ADD_FLOAT = 0x03,
  // Replaces with unsigned modular old-operand.
  IREE_VM_ISA_BUFFER_ATOMIC_KIND_SUBTRACT_INTEGER = 0x04,
  // Replaces with the bitwise old AND operand.
  IREE_VM_ISA_BUFFER_ATOMIC_KIND_AND_INTEGER = 0x05,
  // Replaces with the bitwise old OR operand.
  IREE_VM_ISA_BUFFER_ATOMIC_KIND_OR_INTEGER = 0x06,
  // Replaces with the bitwise old XOR operand.
  IREE_VM_ISA_BUFFER_ATOMIC_KIND_XOR_INTEGER = 0x07,
  // Replaces with the two's-complement signed minimum.
  IREE_VM_ISA_BUFFER_ATOMIC_KIND_MINIMUM_SIGNED = 0x08,
  // Replaces with the two's-complement signed maximum.
  IREE_VM_ISA_BUFFER_ATOMIC_KIND_MAXIMUM_SIGNED = 0x09,
  // Replaces with the unsigned minimum.
  IREE_VM_ISA_BUFFER_ATOMIC_KIND_MINIMUM_UNSIGNED = 0x0A,
  // Replaces with the unsigned maximum.
  IREE_VM_ISA_BUFFER_ATOMIC_KIND_MAXIMUM_UNSIGNED = 0x0B,
  // Replaces with float.minmax minimum.
  IREE_VM_ISA_BUFFER_ATOMIC_KIND_MINIMUM_FLOAT = 0x0C,
  // Replaces with float.minmax maximum.
  IREE_VM_ISA_BUFFER_ATOMIC_KIND_MAXIMUM_FLOAT = 0x0D,
  // Replaces with float.minmax minnum.
  IREE_VM_ISA_BUFFER_ATOMIC_KIND_MINNUM_FLOAT = 0x0E,
  // Replaces with float.minmax maxnum.
  IREE_VM_ISA_BUFFER_ATOMIC_KIND_MAXNUM_FLOAT = 0x0F,
};

enum {
  IREE_VM_ISA_BUFFER_ATOMIC_KIND_EXCHANGE_INTEGER_SINCE_MINOR = 0,
  IREE_VM_ISA_BUFFER_ATOMIC_KIND_EXCHANGE_FLOAT_SINCE_MINOR = 0,
  IREE_VM_ISA_BUFFER_ATOMIC_KIND_ADD_INTEGER_SINCE_MINOR = 0,
  IREE_VM_ISA_BUFFER_ATOMIC_KIND_ADD_FLOAT_SINCE_MINOR = 0,
  IREE_VM_ISA_BUFFER_ATOMIC_KIND_SUBTRACT_INTEGER_SINCE_MINOR = 0,
  IREE_VM_ISA_BUFFER_ATOMIC_KIND_AND_INTEGER_SINCE_MINOR = 0,
  IREE_VM_ISA_BUFFER_ATOMIC_KIND_OR_INTEGER_SINCE_MINOR = 0,
  IREE_VM_ISA_BUFFER_ATOMIC_KIND_XOR_INTEGER_SINCE_MINOR = 0,
  IREE_VM_ISA_BUFFER_ATOMIC_KIND_MINIMUM_SIGNED_SINCE_MINOR = 0,
  IREE_VM_ISA_BUFFER_ATOMIC_KIND_MAXIMUM_SIGNED_SINCE_MINOR = 0,
  IREE_VM_ISA_BUFFER_ATOMIC_KIND_MINIMUM_UNSIGNED_SINCE_MINOR = 0,
  IREE_VM_ISA_BUFFER_ATOMIC_KIND_MAXIMUM_UNSIGNED_SINCE_MINOR = 0,
  IREE_VM_ISA_BUFFER_ATOMIC_KIND_MINIMUM_FLOAT_SINCE_MINOR = 0,
  IREE_VM_ISA_BUFFER_ATOMIC_KIND_MAXIMUM_FLOAT_SINCE_MINOR = 0,
  IREE_VM_ISA_BUFFER_ATOMIC_KIND_MINNUM_FLOAT_SINCE_MINOR = 0,
  IREE_VM_ISA_BUFFER_ATOMIC_KIND_MAXNUM_FLOAT_SINCE_MINOR = 0,
};

// Selects the naturally aligned raw-storage carrier width.
typedef uint8_t iree_vm_isa_buffer_atomic_carrier_t;
enum {
  // Uses a four-byte carrier and low 32 value-cell bits.
  IREE_VM_ISA_BUFFER_ATOMIC_CARRIER_I32 = 0x00,
  // Uses an eight-byte carrier and the complete value cell.
  IREE_VM_ISA_BUFFER_ATOMIC_CARRIER_I64 = 0x01,
};

enum {
  IREE_VM_ISA_BUFFER_ATOMIC_CARRIER_I32_SINCE_MINOR = 0,
  IREE_VM_ISA_BUFFER_ATOMIC_CARRIER_I64_SINCE_MINOR = 0,
};

// Selects the minimum C11-style synchronization ordering. A target may
// strengthen but never weaken it.
typedef uint8_t iree_vm_isa_buffer_atomic_ordering_t;
enum {
  // Guarantees atomicity without inter-operation ordering.
  IREE_VM_ISA_BUFFER_ATOMIC_ORDERING_RELAXED = 0x00,
  // Applies acquire ordering to the operation's read.
  IREE_VM_ISA_BUFFER_ATOMIC_ORDERING_ACQUIRE = 0x01,
  // Applies release ordering to the operation's write.
  IREE_VM_ISA_BUFFER_ATOMIC_ORDERING_RELEASE = 0x02,
  // Applies acquire ordering to the read and release ordering to the write.
  IREE_VM_ISA_BUFFER_ATOMIC_ORDERING_ACQ_REL = 0x03,
  // Participates in one sequentially consistent total order.
  IREE_VM_ISA_BUFFER_ATOMIC_ORDERING_SEQ_CST = 0x04,
};

enum {
  IREE_VM_ISA_BUFFER_ATOMIC_ORDERING_RELAXED_SINCE_MINOR = 0,
  IREE_VM_ISA_BUFFER_ATOMIC_ORDERING_ACQUIRE_SINCE_MINOR = 0,
  IREE_VM_ISA_BUFFER_ATOMIC_ORDERING_RELEASE_SINCE_MINOR = 0,
  IREE_VM_ISA_BUFFER_ATOMIC_ORDERING_ACQ_REL_SINCE_MINOR = 0,
  IREE_VM_ISA_BUFFER_ATOMIC_ORDERING_SEQ_CST_SINCE_MINOR = 0,
};

// Selects the minimum synchronization domain. CPU interpretation may strengthen
// a narrower domain to process/system scope.
typedef uint8_t iree_vm_isa_buffer_atomic_scope_t;
enum {
  // Requires ordering only within the current thread.
  IREE_VM_ISA_BUFFER_ATOMIC_SCOPE_THREAD = 0x00,
  // Requires ordering among invocations in one execution subgroup.
  IREE_VM_ISA_BUFFER_ATOMIC_SCOPE_SUBGROUP = 0x01,
  // Requires ordering among invocations in one workgroup.
  IREE_VM_ISA_BUFFER_ATOMIC_SCOPE_WORKGROUP = 0x02,
  // Requires ordering among agents on one logical device.
  IREE_VM_ISA_BUFFER_ATOMIC_SCOPE_DEVICE = 0x03,
  // Requires ordering across every participating system agent.
  IREE_VM_ISA_BUFFER_ATOMIC_SCOPE_SYSTEM = 0x04,
};

enum {
  IREE_VM_ISA_BUFFER_ATOMIC_SCOPE_THREAD_SINCE_MINOR = 0,
  IREE_VM_ISA_BUFFER_ATOMIC_SCOPE_SUBGROUP_SINCE_MINOR = 0,
  IREE_VM_ISA_BUFFER_ATOMIC_SCOPE_WORKGROUP_SINCE_MINOR = 0,
  IREE_VM_ISA_BUFFER_ATOMIC_SCOPE_DEVICE_SINCE_MINOR = 0,
  IREE_VM_ISA_BUFFER_ATOMIC_SCOPE_SYSTEM_SINCE_MINOR = 0,
};

// Page 0x00, opcode 0xD8: Allocates one zeroed READ|WRITE vm.buffer.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Non-null exact vm.buffer result replacing the destination ref.
  uint8_t dst_r8;
  // Unsigned 64-bit requested byte length.
  uint8_t length_v8;
  // Canonical zero padding.
  uint8_t zero_padding_u8;
} iree_vm_isa_buffer_allocate_record_t;

// Page 0x00, opcode 0xD9: Returns a buffer's immutable byte length.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Unsigned 64-bit byte-length result.
  uint8_t dst_v8;
  // Required exact vm.buffer ref borrowed for this instruction.
  uint8_t buffer_r8;
  // Canonical zero padding.
  uint8_t zero_padding_u8;
} iree_vm_isa_buffer_length_record_t;

// Page 0x00, opcode 0xDA: Materializes an owned buffer view over one exact
// range.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Non-null exact vm.buffer result replacing the destination ref.
  uint8_t dst_r8;
  // Required exact vm.buffer ref borrowed for this instruction.
  uint8_t buffer_r8;
  // Unsigned 64-bit source byte offset.
  uint8_t offset_v8;
  // Unsigned 64-bit view byte length.
  uint8_t length_v8;
  // Canonical zero padding.
  uint8_t zero_padding_u8[3];
} iree_vm_isa_buffer_subspan_record_t;

// Page 0x00, opcode 0xDB: Loads from a checked buffer lane group.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // First destination value-register ordinal.
  uint8_t dst_v8;
  // Required exact vm.buffer ref borrowed for this instruction.
  uint8_t buffer_r8;
  // Unsigned 64-bit address base.
  uint8_t base_v8;
  // Unsigned 64-bit scaled index.
  uint8_t index_v8;
  // Unsigned scale including the useful zero case.
  uint8_t scale_u8;
  // Closed lane element-width and lane-count selector.
  uint8_t format_u8;
  // Canonical zero padding.
  uint8_t zero_padding_u8;
} iree_vm_isa_buffer_load_record_t;

// Page 0x00, opcode 0xDC: Stores to a checked buffer lane group.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Required exact vm.buffer ref borrowed for this instruction.
  uint8_t buffer_r8;
  // Unsigned 64-bit address base.
  uint8_t base_v8;
  // Unsigned 64-bit scaled index.
  uint8_t index_v8;
  // Unsigned scale including the useful zero case.
  uint8_t scale_u8;
  // First source value-register ordinal.
  uint8_t src_v8;
  // Closed lane element-width and lane-count selector.
  uint8_t format_u8;
  // Canonical zero padding.
  uint8_t zero_padding_u8;
} iree_vm_isa_buffer_store_record_t;

// Page 0x00, opcode 0xDD: Atomically reduces one raw buffer carrier.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Required exact vm.buffer ref borrowed for this instruction.
  uint8_t buffer_r8;
  // Unsigned carrier byte offset.
  uint8_t offset_v8;
  // Low carrier-width update operand.
  uint8_t operand_v8;
  // Packed closed-selector components and canonical zero bits.
  uint8_t selector0_u8;
  // Packed closed-selector components and canonical zero bits.
  uint8_t selector1_u8;
  // Canonical zero padding.
  uint16_t zero_padding_u16;
} iree_vm_isa_buffer_atomic_reduce_record_t;

// Page 0x00, opcode 0xDE: Atomically updates and returns one raw buffer
// carrier.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Carrier value immediately preceding the committed update.
  uint8_t old_v8;
  // Required exact vm.buffer ref borrowed for this instruction.
  uint8_t buffer_r8;
  // Unsigned carrier byte offset.
  uint8_t offset_v8;
  // Low carrier-width update operand.
  uint8_t operand_v8;
  // Packed closed-selector components and canonical zero bits.
  uint8_t selector0_u8;
  // Packed closed-selector components and canonical zero bits.
  uint8_t selector1_u8;
  // Canonical zero padding.
  uint8_t zero_padding_u8;
} iree_vm_isa_buffer_atomic_rmw_record_t;

// Page 0x00, opcode 0xDF: Performs strong exact-bit atomic compare-exchange.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Actual carrier value observed by the comparison attempt.
  uint8_t old_v8;
  // Required exact vm.buffer ref borrowed for this instruction.
  uint8_t buffer_r8;
  // Unsigned carrier byte offset.
  uint8_t offset_v8;
  // Low carrier-width expected bit pattern.
  uint8_t expected_v8;
  // Low carrier-width replacement bit pattern.
  uint8_t replacement_v8;
  // Packed closed-selector components and canonical zero bits.
  uint8_t selector0_u8;
  // Packed closed-selector components and canonical zero bits.
  uint8_t selector1_u8;
} iree_vm_isa_buffer_atomic_cmpxchg_record_t;

// Page 0x00, opcode 0xE0: Repeats a byte pattern across a checked writable
// range.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Required exact vm.buffer ref borrowed for this instruction.
  uint8_t buffer_r8;
  // Unsigned byte offset.
  uint8_t offset_v8;
  // Unsigned byte length.
  uint8_t length_v8;
  // Low-byte repeating pattern.
  uint8_t pattern_v8;
  // Pattern width in bytes.
  uint8_t pattern_width_u8;
  // Canonical zero padding.
  uint16_t zero_padding_u16;
} iree_vm_isa_buffer_fill_record_t;

// Page 0x00, opcode 0xE1: Moves one checked buffer range to another.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Required exact vm.buffer ref borrowed for this instruction.
  uint8_t target_r8;
  // Unsigned target byte offset.
  uint8_t target_offset_v8;
  // Required exact vm.buffer ref borrowed for this instruction.
  uint8_t source_r8;
  // Unsigned source byte offset.
  uint8_t source_offset_v8;
  // Unsigned byte length.
  uint8_t length_v8;
  // Canonical zero padding.
  uint16_t zero_padding_u16;
} iree_vm_isa_buffer_copy_record_t;

// Page 0x00, opcode 0xE2: Lexicographically compares two checked buffer ranges.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Canonical signed i32 comparison result.
  uint8_t dst_v8;
  // Required exact vm.buffer ref borrowed for this instruction.
  uint8_t lhs_r8;
  // Unsigned left byte offset.
  uint8_t lhs_offset_v8;
  // Required exact vm.buffer ref borrowed for this instruction.
  uint8_t rhs_r8;
  // Unsigned right byte offset.
  uint8_t rhs_offset_v8;
  // Unsigned byte length.
  uint8_t length_v8;
  // Canonical zero padding.
  uint8_t zero_padding_u8;
} iree_vm_isa_buffer_compare_record_t;

// Page 0x00, opcode 0xE3: Copies module rodata into a checked writable buffer
// range.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Required exact vm.buffer ref borrowed for this instruction.
  uint8_t target_r8;
  // Unsigned target byte offset.
  uint8_t target_offset_v8;
  // Canonical zero padding.
  uint8_t zero_padding0_u8;
  // Direct module rodata-block ordinal.
  uint16_t rodata_u16;
  // Unsigned byte length.
  uint8_t length_v8;
  // Canonical zero padding.
  uint8_t zero_padding1_u8;
  // Statically in-range or one-past-end rodata offset.
  uint32_t source_offset_u32;
} iree_vm_isa_buffer_copy_rodata_record_t;

// Page 0x00, opcode 0xE4: Borrows a module-owned read-only rodata buffer view.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Non-null exact vm.buffer result replacing the destination ref.
  uint8_t dst_r8;
  // Direct module rodata-block ordinal.
  uint16_t rodata_u16;
} iree_vm_isa_buffer_rodata_load_record_t;

#endif  // IREE_VM_BYTECODE_GENERATED_WIRE_CORE_BUFFER_H_
// clang-format on

// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// GENERATED FILE: DO NOT EDIT.
// Regenerate with:
//   iree-bazel-run //runtime/src/iree/vm/bytecode/spec:generate
// Natural-layout instruction records for core.family.integer.
// Multi-byte fields are little-endian and naturally aligned.
// clang-format off
#ifndef IREE_VM_BYTECODE_GENERATED_WIRE_CORE_INTEGER_H_
#define IREE_VM_BYTECODE_GENERATED_WIRE_CORE_INTEGER_H_

#include <stdint.h>

// Selects an equality, signed-order, or unsigned-order predicate over the
// instruction-selected low integer width.
typedef uint8_t iree_vm_isa_integer_compare_t;
enum {
  // True when both low-width bit patterns are equal.
  IREE_VM_ISA_INTEGER_COMPARE_EQ = 0x00,
  // True when both low-width bit patterns differ.
  IREE_VM_ISA_INTEGER_COMPARE_NE = 0x01,
  // True when lhs is signed less than rhs.
  IREE_VM_ISA_INTEGER_COMPARE_SLT = 0x02,
  // True when lhs is signed less than or equal to rhs.
  IREE_VM_ISA_INTEGER_COMPARE_SLE = 0x03,
  // True when lhs is signed greater than rhs.
  IREE_VM_ISA_INTEGER_COMPARE_SGT = 0x04,
  // True when lhs is signed greater than or equal to rhs.
  IREE_VM_ISA_INTEGER_COMPARE_SGE = 0x05,
  // True when lhs is unsigned less than rhs.
  IREE_VM_ISA_INTEGER_COMPARE_ULT = 0x06,
  // True when lhs is unsigned less than or equal to rhs.
  IREE_VM_ISA_INTEGER_COMPARE_ULE = 0x07,
  // True when lhs is unsigned greater than rhs.
  IREE_VM_ISA_INTEGER_COMPARE_UGT = 0x08,
  // True when lhs is unsigned greater than or equal to rhs.
  IREE_VM_ISA_INTEGER_COMPARE_UGE = 0x09,
};

enum {
  IREE_VM_ISA_INTEGER_COMPARE_EQ_SINCE_MINOR = 0,
  IREE_VM_ISA_INTEGER_COMPARE_NE_SINCE_MINOR = 0,
  IREE_VM_ISA_INTEGER_COMPARE_SLT_SINCE_MINOR = 0,
  IREE_VM_ISA_INTEGER_COMPARE_SLE_SINCE_MINOR = 0,
  IREE_VM_ISA_INTEGER_COMPARE_SGT_SINCE_MINOR = 0,
  IREE_VM_ISA_INTEGER_COMPARE_SGE_SINCE_MINOR = 0,
  IREE_VM_ISA_INTEGER_COMPARE_ULT_SINCE_MINOR = 0,
  IREE_VM_ISA_INTEGER_COMPARE_ULE_SINCE_MINOR = 0,
  IREE_VM_ISA_INTEGER_COMPARE_UGT_SINCE_MINOR = 0,
  IREE_VM_ISA_INTEGER_COMPARE_UGE_SINCE_MINOR = 0,
};

// Page 0x00, opcode 0x40: Adds low 32-bit patterns modulo 2^32.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Result value-register ordinal.
  uint8_t dst_v8;
  // Operand value-register ordinal.
  uint8_t lhs_v8;
  // Operand value-register ordinal.
  uint8_t rhs_v8;
} iree_vm_isa_integer_add_i32_record_t;

// Page 0x00, opcode 0x41: Adds complete 64-bit patterns modulo 2^64.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Result value-register ordinal.
  uint8_t dst_v8;
  // Operand value-register ordinal.
  uint8_t lhs_v8;
  // Operand value-register ordinal.
  uint8_t rhs_v8;
} iree_vm_isa_integer_add_i64_record_t;

// Page 0x00, opcode 0x42: Subtracts low 32-bit patterns modulo 2^32.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Result value-register ordinal.
  uint8_t dst_v8;
  // Operand value-register ordinal.
  uint8_t lhs_v8;
  // Operand value-register ordinal.
  uint8_t rhs_v8;
} iree_vm_isa_integer_sub_i32_record_t;

// Page 0x00, opcode 0x43: Subtracts complete 64-bit patterns modulo 2^64.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Result value-register ordinal.
  uint8_t dst_v8;
  // Operand value-register ordinal.
  uint8_t lhs_v8;
  // Operand value-register ordinal.
  uint8_t rhs_v8;
} iree_vm_isa_integer_sub_i64_record_t;

// Page 0x00, opcode 0x44: Multiplies low 32-bit patterns and retains the low 32
// product bits.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Result value-register ordinal.
  uint8_t dst_v8;
  // Operand value-register ordinal.
  uint8_t lhs_v8;
  // Operand value-register ordinal.
  uint8_t rhs_v8;
} iree_vm_isa_integer_mul_i32_record_t;

// Page 0x00, opcode 0x45: Multiplies complete 64-bit patterns and retains the
// low 64 product bits.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Result value-register ordinal.
  uint8_t dst_v8;
  // Operand value-register ordinal.
  uint8_t lhs_v8;
  // Operand value-register ordinal.
  uint8_t rhs_v8;
} iree_vm_isa_integer_mul_i64_record_t;

// Page 0x00, opcode 0x46: Computes the signed 32-bit quotient.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Result value-register ordinal.
  uint8_t dst_v8;
  // Operand value-register ordinal.
  uint8_t lhs_v8;
  // Operand value-register ordinal.
  uint8_t rhs_v8;
} iree_vm_isa_integer_div_s32_record_t;

// Page 0x00, opcode 0x47: Computes the signed 64-bit quotient.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Result value-register ordinal.
  uint8_t dst_v8;
  // Operand value-register ordinal.
  uint8_t lhs_v8;
  // Operand value-register ordinal.
  uint8_t rhs_v8;
} iree_vm_isa_integer_div_s64_record_t;

// Page 0x00, opcode 0x48: Computes the unsigned 32-bit quotient.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Result value-register ordinal.
  uint8_t dst_v8;
  // Operand value-register ordinal.
  uint8_t lhs_v8;
  // Operand value-register ordinal.
  uint8_t rhs_v8;
} iree_vm_isa_integer_div_u32_record_t;

// Page 0x00, opcode 0x49: Computes the unsigned 64-bit quotient.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Result value-register ordinal.
  uint8_t dst_v8;
  // Operand value-register ordinal.
  uint8_t lhs_v8;
  // Operand value-register ordinal.
  uint8_t rhs_v8;
} iree_vm_isa_integer_div_u64_record_t;

// Page 0x00, opcode 0x4A: Computes the signed 32-bit remainder.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Result value-register ordinal.
  uint8_t dst_v8;
  // Operand value-register ordinal.
  uint8_t lhs_v8;
  // Operand value-register ordinal.
  uint8_t rhs_v8;
} iree_vm_isa_integer_rem_s32_record_t;

// Page 0x00, opcode 0x4B: Computes the signed 64-bit remainder.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Result value-register ordinal.
  uint8_t dst_v8;
  // Operand value-register ordinal.
  uint8_t lhs_v8;
  // Operand value-register ordinal.
  uint8_t rhs_v8;
} iree_vm_isa_integer_rem_s64_record_t;

// Page 0x00, opcode 0x4C: Computes the unsigned 32-bit remainder.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Result value-register ordinal.
  uint8_t dst_v8;
  // Operand value-register ordinal.
  uint8_t lhs_v8;
  // Operand value-register ordinal.
  uint8_t rhs_v8;
} iree_vm_isa_integer_rem_u32_record_t;

// Page 0x00, opcode 0x4D: Computes the unsigned 64-bit remainder.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Result value-register ordinal.
  uint8_t dst_v8;
  // Operand value-register ordinal.
  uint8_t lhs_v8;
  // Operand value-register ordinal.
  uint8_t rhs_v8;
} iree_vm_isa_integer_rem_u64_record_t;

// Page 0x00, opcode 0x4E: Computes two's-complement negation modulo 2^32.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Result value-register ordinal.
  uint8_t dst_v8;
  // Operand value-register ordinal.
  uint8_t src_v8;
  // Canonical zero padding.
  uint8_t zero_padding_u8;
} iree_vm_isa_integer_neg_i32_record_t;

// Page 0x00, opcode 0x4F: Computes two's-complement negation modulo 2^64.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Result value-register ordinal.
  uint8_t dst_v8;
  // Operand value-register ordinal.
  uint8_t src_v8;
  // Canonical zero padding.
  uint8_t zero_padding_u8;
} iree_vm_isa_integer_neg_i64_record_t;

// Page 0x00, opcode 0x50: Computes modular signed 32-bit absolute value.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Result value-register ordinal.
  uint8_t dst_v8;
  // Operand value-register ordinal.
  uint8_t src_v8;
  // Canonical zero padding.
  uint8_t zero_padding_u8;
} iree_vm_isa_integer_abs_s32_record_t;

// Page 0x00, opcode 0x51: Computes modular signed 64-bit absolute value.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Result value-register ordinal.
  uint8_t dst_v8;
  // Operand value-register ordinal.
  uint8_t src_v8;
  // Canonical zero padding.
  uint8_t zero_padding_u8;
} iree_vm_isa_integer_abs_s64_record_t;

// Page 0x00, opcode 0x52: Selects the lesser signed two's-complement 32-bit
// operand.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Result value-register ordinal.
  uint8_t dst_v8;
  // Operand value-register ordinal.
  uint8_t lhs_v8;
  // Operand value-register ordinal.
  uint8_t rhs_v8;
} iree_vm_isa_integer_min_s32_record_t;

// Page 0x00, opcode 0x53: Selects the lesser signed two's-complement 64-bit
// operand.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Result value-register ordinal.
  uint8_t dst_v8;
  // Operand value-register ordinal.
  uint8_t lhs_v8;
  // Operand value-register ordinal.
  uint8_t rhs_v8;
} iree_vm_isa_integer_min_s64_record_t;

// Page 0x00, opcode 0x54: Selects the lesser unsigned 32-bit operand.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Result value-register ordinal.
  uint8_t dst_v8;
  // Operand value-register ordinal.
  uint8_t lhs_v8;
  // Operand value-register ordinal.
  uint8_t rhs_v8;
} iree_vm_isa_integer_min_u32_record_t;

// Page 0x00, opcode 0x55: Selects the lesser unsigned 64-bit operand.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Result value-register ordinal.
  uint8_t dst_v8;
  // Operand value-register ordinal.
  uint8_t lhs_v8;
  // Operand value-register ordinal.
  uint8_t rhs_v8;
} iree_vm_isa_integer_min_u64_record_t;

// Page 0x00, opcode 0x56: Selects the greater signed two's-complement 32-bit
// operand.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Result value-register ordinal.
  uint8_t dst_v8;
  // Operand value-register ordinal.
  uint8_t lhs_v8;
  // Operand value-register ordinal.
  uint8_t rhs_v8;
} iree_vm_isa_integer_max_s32_record_t;

// Page 0x00, opcode 0x57: Selects the greater signed two's-complement 64-bit
// operand.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Result value-register ordinal.
  uint8_t dst_v8;
  // Operand value-register ordinal.
  uint8_t lhs_v8;
  // Operand value-register ordinal.
  uint8_t rhs_v8;
} iree_vm_isa_integer_max_s64_record_t;

// Page 0x00, opcode 0x58: Selects the greater unsigned 32-bit operand.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Result value-register ordinal.
  uint8_t dst_v8;
  // Operand value-register ordinal.
  uint8_t lhs_v8;
  // Operand value-register ordinal.
  uint8_t rhs_v8;
} iree_vm_isa_integer_max_u32_record_t;

// Page 0x00, opcode 0x59: Selects the greater unsigned 64-bit operand.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Result value-register ordinal.
  uint8_t dst_v8;
  // Operand value-register ordinal.
  uint8_t lhs_v8;
  // Operand value-register ordinal.
  uint8_t rhs_v8;
} iree_vm_isa_integer_max_u64_record_t;

// Page 0x00, opcode 0x5A: Computes bitwise AND over the low 32 bits.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Result value-register ordinal.
  uint8_t dst_v8;
  // Operand value-register ordinal.
  uint8_t lhs_v8;
  // Operand value-register ordinal.
  uint8_t rhs_v8;
} iree_vm_isa_integer_and_i32_record_t;

// Page 0x00, opcode 0x5B: Computes bitwise AND over all 64 bits.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Result value-register ordinal.
  uint8_t dst_v8;
  // Operand value-register ordinal.
  uint8_t lhs_v8;
  // Operand value-register ordinal.
  uint8_t rhs_v8;
} iree_vm_isa_integer_and_i64_record_t;

// Page 0x00, opcode 0x5C: Computes bitwise OR over the low 32 bits.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Result value-register ordinal.
  uint8_t dst_v8;
  // Operand value-register ordinal.
  uint8_t lhs_v8;
  // Operand value-register ordinal.
  uint8_t rhs_v8;
} iree_vm_isa_integer_or_i32_record_t;

// Page 0x00, opcode 0x5D: Computes bitwise OR over all 64 bits.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Result value-register ordinal.
  uint8_t dst_v8;
  // Operand value-register ordinal.
  uint8_t lhs_v8;
  // Operand value-register ordinal.
  uint8_t rhs_v8;
} iree_vm_isa_integer_or_i64_record_t;

// Page 0x00, opcode 0x5E: Computes bitwise XOR over the low 32 bits.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Result value-register ordinal.
  uint8_t dst_v8;
  // Operand value-register ordinal.
  uint8_t lhs_v8;
  // Operand value-register ordinal.
  uint8_t rhs_v8;
} iree_vm_isa_integer_xor_i32_record_t;

// Page 0x00, opcode 0x5F: Computes bitwise XOR over all 64 bits.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Result value-register ordinal.
  uint8_t dst_v8;
  // Operand value-register ordinal.
  uint8_t lhs_v8;
  // Operand value-register ordinal.
  uint8_t rhs_v8;
} iree_vm_isa_integer_xor_i64_record_t;

// Page 0x00, opcode 0x60: Shifts low 32 bits left by the count's low five bits.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Result value-register ordinal.
  uint8_t dst_v8;
  // Operand value-register ordinal.
  uint8_t lhs_v8;
  // Operand value-register ordinal.
  uint8_t rhs_v8;
} iree_vm_isa_integer_shift_left_i32_record_t;

// Page 0x00, opcode 0x61: Shifts all 64 bits left by the count's low six bits.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Result value-register ordinal.
  uint8_t dst_v8;
  // Operand value-register ordinal.
  uint8_t lhs_v8;
  // Operand value-register ordinal.
  uint8_t rhs_v8;
} iree_vm_isa_integer_shift_left_i64_record_t;

// Page 0x00, opcode 0x62: Sign-fills low 32 bits right by the count's low five
// bits.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Result value-register ordinal.
  uint8_t dst_v8;
  // Operand value-register ordinal.
  uint8_t lhs_v8;
  // Operand value-register ordinal.
  uint8_t rhs_v8;
} iree_vm_isa_integer_shift_right_s32_record_t;

// Page 0x00, opcode 0x63: Sign-fills all 64 bits right by the count's low six
// bits.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Result value-register ordinal.
  uint8_t dst_v8;
  // Operand value-register ordinal.
  uint8_t lhs_v8;
  // Operand value-register ordinal.
  uint8_t rhs_v8;
} iree_vm_isa_integer_shift_right_s64_record_t;

// Page 0x00, opcode 0x64: Logically shifts low 32 bits right by the count's low
// five bits.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Result value-register ordinal.
  uint8_t dst_v8;
  // Operand value-register ordinal.
  uint8_t lhs_v8;
  // Operand value-register ordinal.
  uint8_t rhs_v8;
} iree_vm_isa_integer_shift_right_u32_record_t;

// Page 0x00, opcode 0x65: Logically shifts all 64 bits right by the count's low
// six bits.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Result value-register ordinal.
  uint8_t dst_v8;
  // Operand value-register ordinal.
  uint8_t lhs_v8;
  // Operand value-register ordinal.
  uint8_t rhs_v8;
} iree_vm_isa_integer_shift_right_u64_record_t;

// Page 0x00, opcode 0x66: Rotates low 32 bits left by the count's low five
// bits.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Result value-register ordinal.
  uint8_t dst_v8;
  // Operand value-register ordinal.
  uint8_t lhs_v8;
  // Operand value-register ordinal.
  uint8_t rhs_v8;
} iree_vm_isa_integer_rotate_left_i32_record_t;

// Page 0x00, opcode 0x67: Rotates all 64 bits left by the count's low six bits.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Result value-register ordinal.
  uint8_t dst_v8;
  // Operand value-register ordinal.
  uint8_t lhs_v8;
  // Operand value-register ordinal.
  uint8_t rhs_v8;
} iree_vm_isa_integer_rotate_left_i64_record_t;

// Page 0x00, opcode 0x68: Rotates low 32 bits right by the count's low five
// bits.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Result value-register ordinal.
  uint8_t dst_v8;
  // Operand value-register ordinal.
  uint8_t lhs_v8;
  // Operand value-register ordinal.
  uint8_t rhs_v8;
} iree_vm_isa_integer_rotate_right_i32_record_t;

// Page 0x00, opcode 0x69: Rotates all 64 bits right by the count's low six
// bits.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Result value-register ordinal.
  uint8_t dst_v8;
  // Operand value-register ordinal.
  uint8_t lhs_v8;
  // Operand value-register ordinal.
  uint8_t rhs_v8;
} iree_vm_isa_integer_rotate_right_i64_record_t;

// Page 0x00, opcode 0x6A: Counts leading zeroes in the low 32 bits.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Result value-register ordinal.
  uint8_t dst_v8;
  // Operand value-register ordinal.
  uint8_t src_v8;
  // Canonical zero padding.
  uint8_t zero_padding_u8;
} iree_vm_isa_integer_count_leading_zeros_i32_record_t;

// Page 0x00, opcode 0x6B: Counts leading zeroes in all 64 bits.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Result value-register ordinal.
  uint8_t dst_v8;
  // Operand value-register ordinal.
  uint8_t src_v8;
  // Canonical zero padding.
  uint8_t zero_padding_u8;
} iree_vm_isa_integer_count_leading_zeros_i64_record_t;

// Page 0x00, opcode 0x6C: Counts trailing zeroes in the low 32 bits.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Result value-register ordinal.
  uint8_t dst_v8;
  // Operand value-register ordinal.
  uint8_t src_v8;
  // Canonical zero padding.
  uint8_t zero_padding_u8;
} iree_vm_isa_integer_count_trailing_zeros_i32_record_t;

// Page 0x00, opcode 0x6D: Counts trailing zeroes in all 64 bits.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Result value-register ordinal.
  uint8_t dst_v8;
  // Operand value-register ordinal.
  uint8_t src_v8;
  // Canonical zero padding.
  uint8_t zero_padding_u8;
} iree_vm_isa_integer_count_trailing_zeros_i64_record_t;

// Page 0x00, opcode 0x6E: Counts one bits in the low 32 bits.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Result value-register ordinal.
  uint8_t dst_v8;
  // Operand value-register ordinal.
  uint8_t src_v8;
  // Canonical zero padding.
  uint8_t zero_padding_u8;
} iree_vm_isa_integer_popcount_i32_record_t;

// Page 0x00, opcode 0x6F: Counts one bits in all 64 bits.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Result value-register ordinal.
  uint8_t dst_v8;
  // Operand value-register ordinal.
  uint8_t src_v8;
  // Canonical zero padding.
  uint8_t zero_padding_u8;
} iree_vm_isa_integer_popcount_i64_record_t;

// Page 0x00, opcode 0x70: Evaluates one signed, unsigned, or equality i32
// predicate.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Result value-register ordinal.
  uint8_t dst_v8;
  // Operand value-register ordinal.
  uint8_t lhs_v8;
  // Operand value-register ordinal.
  uint8_t rhs_v8;
  // Closed operation selector.
  uint8_t predicate_u8;
  // Canonical zero padding.
  uint8_t zero_padding_u8[3];
} iree_vm_isa_integer_compare_i32_record_t;

// Page 0x00, opcode 0x71: Evaluates one signed, unsigned, or equality i64
// predicate.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Result value-register ordinal.
  uint8_t dst_v8;
  // Operand value-register ordinal.
  uint8_t lhs_v8;
  // Operand value-register ordinal.
  uint8_t rhs_v8;
  // Closed operation selector.
  uint8_t predicate_u8;
  // Canonical zero padding.
  uint8_t zero_padding_u8[3];
} iree_vm_isa_integer_compare_i64_record_t;

// Page 0x00, opcode 0x72: Combines base, scaled index, and signed offset modulo
// 2^32.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Result value-register ordinal.
  uint8_t dst_v8;
  // Operand value-register ordinal.
  uint8_t base_v8;
  // Operand value-register ordinal.
  uint8_t index_v8;
  // Arbitrary unsigned scale including zero and non-powers-of-two.
  uint8_t scale_u8;
  // Canonical zero padding.
  uint8_t zero_padding_u8;
  // Signed little-endian two's-complement affine offset.
  int16_t offset_i16;
} iree_vm_isa_integer_lea_i32_record_t;

// Page 0x00, opcode 0x73: Combines base, scaled index, and signed offset modulo
// 2^64.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Result value-register ordinal.
  uint8_t dst_v8;
  // Operand value-register ordinal.
  uint8_t base_v8;
  // Operand value-register ordinal.
  uint8_t index_v8;
  // Arbitrary unsigned scale including zero and non-powers-of-two.
  uint8_t scale_u8;
  // Canonical zero padding.
  uint8_t zero_padding_u8;
  // Signed little-endian two's-complement affine offset.
  int16_t offset_i16;
} iree_vm_isa_integer_lea_i64_record_t;

// Page 0x00, opcode 0x74: Computes exact unsigned u32 ceiling division by a
// power of two.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Result value-register ordinal.
  uint8_t dst_v8;
  // Operand value-register ordinal.
  uint8_t src_v8;
  // Base-two logarithm of the nonzero divisor.
  uint8_t log2_u8;
} iree_vm_isa_integer_ceildiv_pow2_u32_record_t;

// Page 0x00, opcode 0x75: Computes exact unsigned u64 ceiling division by a
// power of two.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Result value-register ordinal.
  uint8_t dst_v8;
  // Operand value-register ordinal.
  uint8_t src_v8;
  // Base-two logarithm of the nonzero divisor.
  uint8_t log2_u8;
} iree_vm_isa_integer_ceildiv_pow2_u64_record_t;

// Page 0x00, opcode 0x76: Concatenates each source carrier's low field_width
// bits into one at-most-64-bit stream and writes consecutive result-width
// carriers.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Result value-register ordinal.
  uint8_t result_base_v8;
  // Operand value-register ordinal.
  uint8_t source_base_v8;
  // Logical field width in bits.
  uint8_t field_width_u8;
  // Nonzero consecutive source-register count.
  uint8_t source_count_u8;
  // Nonzero consecutive result-register count.
  uint8_t result_count_u8;
  // Source carrier width in bits.
  uint8_t source_width_u8;
  // Result carrier width in bits.
  uint8_t result_width_u8;
} iree_vm_isa_integer_bitstream_pack_record_t;

// Page 0x00, opcode 0x77: Concatenates complete low-width source carriers,
// extracts field_width-bit fields, and zero-extends each into a result carrier.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Result value-register ordinal.
  uint8_t result_base_v8;
  // Operand value-register ordinal.
  uint8_t source_base_v8;
  // Logical field width in bits.
  uint8_t field_width_u8;
  // Nonzero consecutive source-register count.
  uint8_t source_count_u8;
  // Nonzero consecutive result-register count.
  uint8_t result_count_u8;
  // Source carrier width in bits.
  uint8_t source_width_u8;
  // Result carrier width in bits.
  uint8_t result_width_u8;
} iree_vm_isa_integer_bitstream_unpack_u_record_t;

// Page 0x00, opcode 0x78: Concatenates complete low-width source carriers,
// extracts field_width-bit fields, and sign-extends each into a result carrier.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Result value-register ordinal.
  uint8_t result_base_v8;
  // Operand value-register ordinal.
  uint8_t source_base_v8;
  // Logical field width in bits.
  uint8_t field_width_u8;
  // Nonzero consecutive source-register count.
  uint8_t source_count_u8;
  // Nonzero consecutive result-register count.
  uint8_t result_count_u8;
  // Source carrier width in bits.
  uint8_t source_width_u8;
  // Result carrier width in bits.
  uint8_t result_width_u8;
} iree_vm_isa_integer_bitstream_unpack_s_record_t;

#endif  // IREE_VM_BYTECODE_GENERATED_WIRE_CORE_INTEGER_H_
// clang-format on

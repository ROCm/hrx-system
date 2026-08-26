// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_BYTECODE_INTERPRETER_INTEGER_H_
#define IREE_VM_BYTECODE_INTERPRETER_INTEGER_H_

#include "iree/base/api.h"
#include "iree/vm/bytecode/wire/core/integer.h"

// Executes verified Core integer records against a physical value bank.
// Module-load verification guarantees that every register ordinal and closed
// selector is valid. Sources are always captured before the destination is
// written so that any source may alias the destination.
static inline IREE_ATTRIBUTE_ALWAYS_INLINE uint64_t
iree_vm_bytecode_integer_compare_bits(uint64_t lhs, uint64_t rhs,
                                      uint64_t sign_bit, uint8_t predicate) {
  const uint64_t is_equal = lhs == rhs;
  if (predicate <= IREE_VM_ISA_INTEGER_COMPARE_NE) {
    return is_equal ^ predicate;
  }

  // Signed integer ordering is unsigned ordering with both sign bits flipped.
  // The selector pairs then differ only by strict/inclusive and less/greater.
  const uint64_t order_sign_bit =
      predicate < IREE_VM_ISA_INTEGER_COMPARE_ULT ? sign_bit : 0;
  lhs ^= order_sign_bit;
  rhs ^= order_sign_bit;
  const uint8_t relation = (predicate - IREE_VM_ISA_INTEGER_COMPARE_SLT) & 3;
  const uint64_t strict_result = (relation & 2) != 0 ? lhs > rhs : lhs < rhs;
  return strict_result | (((relation & 1) != 0) & is_equal);
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE void
iree_vm_bytecode_execute_integer_compare_i32(
    const iree_vm_isa_integer_compare_i32_record_t* record, uint64_t* values) {
  const uint32_t lhs = (uint32_t)values[record->lhs_v8];
  const uint32_t rhs = (uint32_t)values[record->rhs_v8];
  values[record->dst_v8] = iree_vm_bytecode_integer_compare_bits(
      lhs, rhs, UINT32_C(0x80000000), record->predicate_u8);
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE void
iree_vm_bytecode_execute_integer_compare_i64(
    const iree_vm_isa_integer_compare_i64_record_t* record, uint64_t* values) {
  const uint64_t lhs = values[record->lhs_v8];
  const uint64_t rhs = values[record->rhs_v8];
  values[record->dst_v8] = iree_vm_bytecode_integer_compare_bits(
      lhs, rhs, UINT64_C(0x8000000000000000), record->predicate_u8);
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE void
iree_vm_bytecode_execute_integer_lea_i32(
    const iree_vm_isa_integer_lea_i32_record_t* record, uint64_t* values) {
  const uint32_t base = (uint32_t)values[record->base_v8];
  const uint32_t index = (uint32_t)values[record->index_v8];
  const uint32_t offset = (uint32_t)(int32_t)record->offset_i16;
  values[record->dst_v8] = base + index * (uint32_t)record->scale_u8 + offset;
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE void
iree_vm_bytecode_execute_integer_lea_i64(
    const iree_vm_isa_integer_lea_i64_record_t* record, uint64_t* values) {
  const uint64_t base = values[record->base_v8];
  const uint64_t index = values[record->index_v8];
  const uint64_t offset = (uint64_t)(int64_t)record->offset_i16;
  values[record->dst_v8] = base + index * (uint64_t)record->scale_u8 + offset;
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE void
iree_vm_bytecode_execute_integer_ceildiv_pow2_u32(
    const iree_vm_isa_integer_ceildiv_pow2_u32_record_t* record,
    uint64_t* values) {
  const uint32_t source = (uint32_t)values[record->src_v8];
  const uint32_t mask = (UINT32_C(1) << record->log2_u8) - UINT32_C(1);
  values[record->dst_v8] = (source >> record->log2_u8) + ((source & mask) != 0);
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE void
iree_vm_bytecode_execute_integer_ceildiv_pow2_u64(
    const iree_vm_isa_integer_ceildiv_pow2_u64_record_t* record,
    uint64_t* values) {
  const uint64_t source = values[record->src_v8];
  const uint64_t mask = (UINT64_C(1) << record->log2_u8) - UINT64_C(1);
  values[record->dst_v8] = (source >> record->log2_u8) + ((source & mask) != 0);
}

#endif  // IREE_VM_BYTECODE_INTERPRETER_INTEGER_H_

// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_BYTECODE_INTERPRETER_INTEGER_H_
#define IREE_VM_BYTECODE_INTERPRETER_INTEGER_H_

#include "iree/base/api.h"
#include "iree/base/internal/math.h"
#include "iree/vm/bytecode/wire/core.h"

// Executes verified Core integer records against a physical value bank.
// Module-load verification guarantees that every register ordinal and closed
// selector is valid. Sources are always captured before the destination is
// written so that any source may alias the destination.

// Terminal failure produced by one total integer division instruction.
typedef enum iree_vm_bytecode_integer_division_failure_e {
  IREE_VM_BYTECODE_INTEGER_DIVISION_FAILURE_NONE = 0,
  IREE_VM_BYTECODE_INTEGER_DIVISION_FAILURE_DIVIDE_BY_ZERO = 1,
  IREE_VM_BYTECODE_INTEGER_DIVISION_FAILURE_SIGNED_OVERFLOW = 2,
} iree_vm_bytecode_integer_division_failure_t;

static inline IREE_ATTRIBUTE_ALWAYS_INLINE uint64_t
iree_vm_bytecode_integer_compare_bits(uint64_t lhs, uint64_t rhs,
                                      uint64_t sign_bit, uint8_t predicate) {
  const uint64_t is_equal = lhs == rhs;
  if (predicate <= IREE_VM_BYTECODE_INTEGER_COMPARE_NE) {
    return is_equal ^ predicate;
  }

  // Signed integer ordering is unsigned ordering with both sign bits flipped.
  // The selector pairs then differ only by strict/inclusive and less/greater.
  const uint64_t order_sign_bit =
      predicate < IREE_VM_BYTECODE_INTEGER_COMPARE_ULT ? sign_bit : 0;
  lhs ^= order_sign_bit;
  rhs ^= order_sign_bit;
  const uint8_t relation =
      (predicate - IREE_VM_BYTECODE_INTEGER_COMPARE_SLT) & 3;
  const uint64_t strict_result = (relation & 2) != 0 ? lhs > rhs : lhs < rhs;
  return strict_result | (((relation & 1) != 0) & is_equal);
}

#define IREE_VM_BYTECODE_INTEGER_EXECUTE_CONCAT_(name, kind, width) \
  iree_vm_bytecode_execute_integer_##name##_##kind##width
#define IREE_VM_BYTECODE_INTEGER_EXECUTE_EXPAND_(name, kind, width) \
  IREE_VM_BYTECODE_INTEGER_EXECUTE_CONCAT_(name, kind, width)
#define IREE_VM_BYTECODE_INTEGER_EXECUTE(name, kind)   \
  IREE_VM_BYTECODE_INTEGER_EXECUTE_EXPAND_(name, kind, \
                                           IREE_VM_BYTECODE_INTEGER_WIDTH)
#define IREE_VM_BYTECODE_INTEGER_RECORD_CONCAT_(name, kind, width) \
  iree_vm_bytecode_integer_##name##_##kind##width##_t
#define IREE_VM_BYTECODE_INTEGER_RECORD_EXPAND_(name, kind, width) \
  IREE_VM_BYTECODE_INTEGER_RECORD_CONCAT_(name, kind, width)
#define IREE_VM_BYTECODE_INTEGER_RECORD(name, kind)   \
  IREE_VM_BYTECODE_INTEGER_RECORD_EXPAND_(name, kind, \
                                          IREE_VM_BYTECODE_INTEGER_WIDTH)
#define IREE_VM_BYTECODE_INTEGER_PRIVATE_CONCAT_(name, width) \
  iree_vm_bytecode_integer_##name##_u##width
#define IREE_VM_BYTECODE_INTEGER_PRIVATE_EXPAND_(name, width) \
  IREE_VM_BYTECODE_INTEGER_PRIVATE_CONCAT_(name, width)
#define IREE_VM_BYTECODE_INTEGER_PRIVATE(name) \
  IREE_VM_BYTECODE_INTEGER_PRIVATE_EXPAND_(name, IREE_VM_BYTECODE_INTEGER_WIDTH)

#define IREE_VM_BYTECODE_INTEGER_WIDTH 32
#define IREE_VM_BYTECODE_INTEGER_TYPE uint32_t
#define IREE_VM_BYTECODE_INTEGER_SIGNED_TYPE int32_t
#define IREE_VM_BYTECODE_INTEGER_ZERO UINT32_C(0)
#define IREE_VM_BYTECODE_INTEGER_ONE UINT32_C(1)
#define IREE_VM_BYTECODE_INTEGER_MAX UINT32_MAX
#define IREE_VM_BYTECODE_INTEGER_SIGN_MASK UINT32_C(0x80000000)
#define IREE_VM_BYTECODE_INTEGER_SHIFT_MASK 31u
#define IREE_VM_BYTECODE_INTEGER_CLZ iree_math_count_leading_zeros_u32
#define IREE_VM_BYTECODE_INTEGER_CTZ iree_math_count_trailing_zeros_u32
#define IREE_VM_BYTECODE_INTEGER_POPCOUNT iree_math_count_ones_u32
#include "iree/vm/bytecode/interpreter_integer_impl.inl"
#undef IREE_VM_BYTECODE_INTEGER_POPCOUNT
#undef IREE_VM_BYTECODE_INTEGER_CTZ
#undef IREE_VM_BYTECODE_INTEGER_CLZ
#undef IREE_VM_BYTECODE_INTEGER_SHIFT_MASK
#undef IREE_VM_BYTECODE_INTEGER_SIGN_MASK
#undef IREE_VM_BYTECODE_INTEGER_MAX
#undef IREE_VM_BYTECODE_INTEGER_ONE
#undef IREE_VM_BYTECODE_INTEGER_ZERO
#undef IREE_VM_BYTECODE_INTEGER_SIGNED_TYPE
#undef IREE_VM_BYTECODE_INTEGER_TYPE
#undef IREE_VM_BYTECODE_INTEGER_WIDTH

#define IREE_VM_BYTECODE_INTEGER_WIDTH 64
#define IREE_VM_BYTECODE_INTEGER_TYPE uint64_t
#define IREE_VM_BYTECODE_INTEGER_SIGNED_TYPE int64_t
#define IREE_VM_BYTECODE_INTEGER_ZERO UINT64_C(0)
#define IREE_VM_BYTECODE_INTEGER_ONE UINT64_C(1)
#define IREE_VM_BYTECODE_INTEGER_MAX UINT64_MAX
#define IREE_VM_BYTECODE_INTEGER_SIGN_MASK UINT64_C(0x8000000000000000)
#define IREE_VM_BYTECODE_INTEGER_SHIFT_MASK 63u
#define IREE_VM_BYTECODE_INTEGER_CLZ iree_math_count_leading_zeros_u64
#define IREE_VM_BYTECODE_INTEGER_CTZ iree_math_count_trailing_zeros_u64
#define IREE_VM_BYTECODE_INTEGER_POPCOUNT iree_math_count_ones_u64
#include "iree/vm/bytecode/interpreter_integer_impl.inl"
#undef IREE_VM_BYTECODE_INTEGER_POPCOUNT
#undef IREE_VM_BYTECODE_INTEGER_CTZ
#undef IREE_VM_BYTECODE_INTEGER_CLZ
#undef IREE_VM_BYTECODE_INTEGER_SHIFT_MASK
#undef IREE_VM_BYTECODE_INTEGER_SIGN_MASK
#undef IREE_VM_BYTECODE_INTEGER_MAX
#undef IREE_VM_BYTECODE_INTEGER_ONE
#undef IREE_VM_BYTECODE_INTEGER_ZERO
#undef IREE_VM_BYTECODE_INTEGER_SIGNED_TYPE
#undef IREE_VM_BYTECODE_INTEGER_TYPE
#undef IREE_VM_BYTECODE_INTEGER_WIDTH

#undef IREE_VM_BYTECODE_INTEGER_PRIVATE
#undef IREE_VM_BYTECODE_INTEGER_PRIVATE_EXPAND_
#undef IREE_VM_BYTECODE_INTEGER_PRIVATE_CONCAT_
#undef IREE_VM_BYTECODE_INTEGER_RECORD
#undef IREE_VM_BYTECODE_INTEGER_RECORD_EXPAND_
#undef IREE_VM_BYTECODE_INTEGER_RECORD_CONCAT_
#undef IREE_VM_BYTECODE_INTEGER_EXECUTE
#undef IREE_VM_BYTECODE_INTEGER_EXECUTE_EXPAND_
#undef IREE_VM_BYTECODE_INTEGER_EXECUTE_CONCAT_

static inline IREE_ATTRIBUTE_ALWAYS_INLINE uint64_t
iree_vm_bytecode_integer_low_mask(uint8_t width) {
  return width == 64 ? UINT64_MAX : (UINT64_C(1) << width) - UINT64_C(1);
}

typedef enum iree_vm_bytecode_integer_bitstream_mode_e {
  IREE_VM_BYTECODE_INTEGER_BITSTREAM_MODE_PACK = 0,
  IREE_VM_BYTECODE_INTEGER_BITSTREAM_MODE_UNPACK_U = 1,
  IREE_VM_BYTECODE_INTEGER_BITSTREAM_MODE_UNPACK_S = 2,
} iree_vm_bytecode_integer_bitstream_mode_t;

static inline IREE_ATTRIBUTE_ALWAYS_INLINE void
iree_vm_bytecode_execute_integer_bitstream(
    uint8_t result_base, uint8_t source_base, uint8_t field_width,
    uint8_t source_count, uint8_t result_count, uint8_t source_width,
    uint8_t result_width, iree_vm_bytecode_integer_bitstream_mode_t mode,
    uint64_t* values) {
  const bool is_pack = mode == IREE_VM_BYTECODE_INTEGER_BITSTREAM_MODE_PACK;
  const bool is_signed =
      mode == IREE_VM_BYTECODE_INTEGER_BITSTREAM_MODE_UNPACK_S;
  uint64_t stream = 0;
  const uint8_t source_field_width = is_pack ? field_width : source_width;
  const uint64_t source_mask =
      iree_vm_bytecode_integer_low_mask(source_field_width);
  for (uint8_t i = 0; i < source_count; ++i) {
    stream |= (values[source_base + i] & source_mask)
              << ((uint32_t)i * source_field_width);
  }

  const uint8_t result_field_width = is_pack ? result_width : field_width;
  const uint64_t result_field_mask =
      iree_vm_bytecode_integer_low_mask(result_field_width);
  const uint64_t result_carrier_mask =
      iree_vm_bytecode_integer_low_mask(result_width);
  for (uint8_t i = 0; i < result_count; ++i) {
    uint64_t field =
        (stream >> ((uint32_t)i * result_field_width)) & result_field_mask;
    if (!is_pack && is_signed && field_width < 64) {
      const uint64_t sign_bit = UINT64_C(1) << (field_width - 1);
      field = (field ^ sign_bit) - sign_bit;
    }
    values[result_base + i] = field & result_carrier_mask;
  }
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE void
iree_vm_bytecode_execute_integer_bitstream_pack(
    const iree_vm_bytecode_integer_bitstream_pack_t* record, uint64_t* values) {
  iree_vm_bytecode_execute_integer_bitstream(
      record->result_base_v8, record->source_base_v8, record->field_width_u8,
      record->source_count_u8, record->result_count_u8, record->source_width_u8,
      record->result_width_u8, IREE_VM_BYTECODE_INTEGER_BITSTREAM_MODE_PACK,
      values);
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE void
iree_vm_bytecode_execute_integer_bitstream_unpack_u(
    const iree_vm_bytecode_integer_bitstream_unpack_u_t* record,
    uint64_t* values) {
  iree_vm_bytecode_execute_integer_bitstream(
      record->result_base_v8, record->source_base_v8, record->field_width_u8,
      record->source_count_u8, record->result_count_u8, record->source_width_u8,
      record->result_width_u8, IREE_VM_BYTECODE_INTEGER_BITSTREAM_MODE_UNPACK_U,
      values);
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE void
iree_vm_bytecode_execute_integer_bitstream_unpack_s(
    const iree_vm_bytecode_integer_bitstream_unpack_s_t* record,
    uint64_t* values) {
  iree_vm_bytecode_execute_integer_bitstream(
      record->result_base_v8, record->source_base_v8, record->field_width_u8,
      record->source_count_u8, record->result_count_u8, record->source_width_u8,
      record->result_width_u8, IREE_VM_BYTECODE_INTEGER_BITSTREAM_MODE_UNPACK_S,
      values);
}

#endif  // IREE_VM_BYTECODE_INTERPRETER_INTEGER_H_

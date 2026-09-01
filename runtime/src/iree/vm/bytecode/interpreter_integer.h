// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_BYTECODE_INTERPRETER_INTEGER_H_
#define IREE_VM_BYTECODE_INTERPRETER_INTEGER_H_

#include "iree/base/api.h"
#include "iree/base/internal/math.h"
#include "iree/vm/bytecode/wire/core/integer.h"

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

// Returns the unsigned magnitude of the two's-complement bit pattern |value|.
static inline IREE_ATTRIBUTE_ALWAYS_INLINE uint32_t
iree_vm_bytecode_integer_magnitude_u32(uint32_t value) {
  const uint32_t sign_mask = UINT32_C(0) - (value >> 31);
  return (value ^ sign_mask) - sign_mask;
}

// Returns the unsigned magnitude of the two's-complement bit pattern |value|.
static inline IREE_ATTRIBUTE_ALWAYS_INLINE uint64_t
iree_vm_bytecode_integer_magnitude_u64(uint64_t value) {
  const uint64_t sign_mask = UINT64_C(0) - (value >> 63);
  return (value ^ sign_mask) - sign_mask;
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE
    iree_vm_bytecode_integer_division_failure_t
    iree_vm_bytecode_execute_integer_div_s32(
        const iree_vm_isa_integer_div_s32_record_t* record, uint64_t* values) {
  const uint32_t lhs = (uint32_t)values[record->lhs_v8];
  const uint32_t rhs = (uint32_t)values[record->rhs_v8];
  if (rhs == 0) {
    return IREE_VM_BYTECODE_INTEGER_DIVISION_FAILURE_DIVIDE_BY_ZERO;
  }
  if (lhs == UINT32_C(0x80000000) && rhs == UINT32_MAX) {
    return IREE_VM_BYTECODE_INTEGER_DIVISION_FAILURE_SIGNED_OVERFLOW;
  }
  const uint32_t quotient = iree_vm_bytecode_integer_magnitude_u32(lhs) /
                            iree_vm_bytecode_integer_magnitude_u32(rhs);
  const uint32_t sign_mask = UINT32_C(0) - ((lhs ^ rhs) >> 31);
  values[record->dst_v8] = (quotient ^ sign_mask) - sign_mask;
  return IREE_VM_BYTECODE_INTEGER_DIVISION_FAILURE_NONE;
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE
    iree_vm_bytecode_integer_division_failure_t
    iree_vm_bytecode_execute_integer_div_s64(
        const iree_vm_isa_integer_div_s64_record_t* record, uint64_t* values) {
  const uint64_t lhs = values[record->lhs_v8];
  const uint64_t rhs = values[record->rhs_v8];
  if (rhs == 0) {
    return IREE_VM_BYTECODE_INTEGER_DIVISION_FAILURE_DIVIDE_BY_ZERO;
  }
  if (lhs == UINT64_C(0x8000000000000000) && rhs == UINT64_MAX) {
    return IREE_VM_BYTECODE_INTEGER_DIVISION_FAILURE_SIGNED_OVERFLOW;
  }
  const uint64_t quotient = iree_vm_bytecode_integer_magnitude_u64(lhs) /
                            iree_vm_bytecode_integer_magnitude_u64(rhs);
  const uint64_t sign_mask = UINT64_C(0) - ((lhs ^ rhs) >> 63);
  values[record->dst_v8] = (quotient ^ sign_mask) - sign_mask;
  return IREE_VM_BYTECODE_INTEGER_DIVISION_FAILURE_NONE;
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE
    iree_vm_bytecode_integer_division_failure_t
    iree_vm_bytecode_execute_integer_div_u32(
        const iree_vm_isa_integer_div_u32_record_t* record, uint64_t* values) {
  const uint32_t lhs = (uint32_t)values[record->lhs_v8];
  const uint32_t rhs = (uint32_t)values[record->rhs_v8];
  if (rhs == 0) {
    return IREE_VM_BYTECODE_INTEGER_DIVISION_FAILURE_DIVIDE_BY_ZERO;
  }
  values[record->dst_v8] = lhs / rhs;
  return IREE_VM_BYTECODE_INTEGER_DIVISION_FAILURE_NONE;
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE
    iree_vm_bytecode_integer_division_failure_t
    iree_vm_bytecode_execute_integer_div_u64(
        const iree_vm_isa_integer_div_u64_record_t* record, uint64_t* values) {
  const uint64_t lhs = values[record->lhs_v8];
  const uint64_t rhs = values[record->rhs_v8];
  if (rhs == 0) {
    return IREE_VM_BYTECODE_INTEGER_DIVISION_FAILURE_DIVIDE_BY_ZERO;
  }
  values[record->dst_v8] = lhs / rhs;
  return IREE_VM_BYTECODE_INTEGER_DIVISION_FAILURE_NONE;
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE
    iree_vm_bytecode_integer_division_failure_t
    iree_vm_bytecode_execute_integer_rem_s32(
        const iree_vm_isa_integer_rem_s32_record_t* record, uint64_t* values) {
  const uint32_t lhs = (uint32_t)values[record->lhs_v8];
  const uint32_t rhs = (uint32_t)values[record->rhs_v8];
  if (rhs == 0) {
    return IREE_VM_BYTECODE_INTEGER_DIVISION_FAILURE_DIVIDE_BY_ZERO;
  }
  const uint32_t remainder = iree_vm_bytecode_integer_magnitude_u32(lhs) %
                             iree_vm_bytecode_integer_magnitude_u32(rhs);
  const uint32_t sign_mask = UINT32_C(0) - (lhs >> 31);
  values[record->dst_v8] = (remainder ^ sign_mask) - sign_mask;
  return IREE_VM_BYTECODE_INTEGER_DIVISION_FAILURE_NONE;
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE
    iree_vm_bytecode_integer_division_failure_t
    iree_vm_bytecode_execute_integer_rem_s64(
        const iree_vm_isa_integer_rem_s64_record_t* record, uint64_t* values) {
  const uint64_t lhs = values[record->lhs_v8];
  const uint64_t rhs = values[record->rhs_v8];
  if (rhs == 0) {
    return IREE_VM_BYTECODE_INTEGER_DIVISION_FAILURE_DIVIDE_BY_ZERO;
  }
  const uint64_t remainder = iree_vm_bytecode_integer_magnitude_u64(lhs) %
                             iree_vm_bytecode_integer_magnitude_u64(rhs);
  const uint64_t sign_mask = UINT64_C(0) - (lhs >> 63);
  values[record->dst_v8] = (remainder ^ sign_mask) - sign_mask;
  return IREE_VM_BYTECODE_INTEGER_DIVISION_FAILURE_NONE;
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE
    iree_vm_bytecode_integer_division_failure_t
    iree_vm_bytecode_execute_integer_rem_u32(
        const iree_vm_isa_integer_rem_u32_record_t* record, uint64_t* values) {
  const uint32_t lhs = (uint32_t)values[record->lhs_v8];
  const uint32_t rhs = (uint32_t)values[record->rhs_v8];
  if (rhs == 0) {
    return IREE_VM_BYTECODE_INTEGER_DIVISION_FAILURE_DIVIDE_BY_ZERO;
  }
  values[record->dst_v8] = lhs % rhs;
  return IREE_VM_BYTECODE_INTEGER_DIVISION_FAILURE_NONE;
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE
    iree_vm_bytecode_integer_division_failure_t
    iree_vm_bytecode_execute_integer_rem_u64(
        const iree_vm_isa_integer_rem_u64_record_t* record, uint64_t* values) {
  const uint64_t lhs = values[record->lhs_v8];
  const uint64_t rhs = values[record->rhs_v8];
  if (rhs == 0) {
    return IREE_VM_BYTECODE_INTEGER_DIVISION_FAILURE_DIVIDE_BY_ZERO;
  }
  values[record->dst_v8] = lhs % rhs;
  return IREE_VM_BYTECODE_INTEGER_DIVISION_FAILURE_NONE;
}

// Performs a total arithmetic right shift over a two's-complement bit pattern.
static inline IREE_ATTRIBUTE_ALWAYS_INLINE uint32_t
iree_vm_bytecode_integer_shift_right_s32(uint32_t source, uint32_t count) {
  count &= 31;
  const uint32_t sign_mask = UINT32_C(0) - (source >> 31);
  const uint32_t fill_mask = ~(UINT32_MAX >> count);
  return (source >> count) | (sign_mask & fill_mask);
}

// Performs a total arithmetic right shift over a two's-complement bit pattern.
static inline IREE_ATTRIBUTE_ALWAYS_INLINE uint64_t
iree_vm_bytecode_integer_shift_right_s64(uint64_t source, uint32_t count) {
  count &= 63;
  const uint64_t sign_mask = UINT64_C(0) - (source >> 63);
  const uint64_t fill_mask = ~(UINT64_MAX >> count);
  return (source >> count) | (sign_mask & fill_mask);
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE void
iree_vm_bytecode_execute_integer_shift_left_i32(
    const iree_vm_isa_integer_shift_left_i32_record_t* record,
    uint64_t* values) {
  const uint32_t source = (uint32_t)values[record->lhs_v8];
  const uint32_t count = (uint32_t)values[record->rhs_v8] & 31;
  values[record->dst_v8] = source << count;
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE void
iree_vm_bytecode_execute_integer_shift_left_i64(
    const iree_vm_isa_integer_shift_left_i64_record_t* record,
    uint64_t* values) {
  const uint64_t source = values[record->lhs_v8];
  const uint32_t count = (uint32_t)(values[record->rhs_v8] & 63);
  values[record->dst_v8] = source << count;
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE void
iree_vm_bytecode_execute_integer_shift_right_s32(
    const iree_vm_isa_integer_shift_right_s32_record_t* record,
    uint64_t* values) {
  const uint32_t source = (uint32_t)values[record->lhs_v8];
  const uint32_t count = (uint32_t)values[record->rhs_v8];
  values[record->dst_v8] =
      iree_vm_bytecode_integer_shift_right_s32(source, count);
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE void
iree_vm_bytecode_execute_integer_shift_right_s64(
    const iree_vm_isa_integer_shift_right_s64_record_t* record,
    uint64_t* values) {
  const uint64_t source = values[record->lhs_v8];
  const uint32_t count = (uint32_t)values[record->rhs_v8];
  values[record->dst_v8] =
      iree_vm_bytecode_integer_shift_right_s64(source, count);
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE void
iree_vm_bytecode_execute_integer_shift_right_u32(
    const iree_vm_isa_integer_shift_right_u32_record_t* record,
    uint64_t* values) {
  const uint32_t source = (uint32_t)values[record->lhs_v8];
  const uint32_t count = (uint32_t)values[record->rhs_v8] & 31;
  values[record->dst_v8] = source >> count;
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE void
iree_vm_bytecode_execute_integer_shift_right_u64(
    const iree_vm_isa_integer_shift_right_u64_record_t* record,
    uint64_t* values) {
  const uint64_t source = values[record->lhs_v8];
  const uint32_t count = (uint32_t)(values[record->rhs_v8] & 63);
  values[record->dst_v8] = source >> count;
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE void
iree_vm_bytecode_execute_integer_rotate_left_i32(
    const iree_vm_isa_integer_rotate_left_i32_record_t* record,
    uint64_t* values) {
  const uint32_t source = (uint32_t)values[record->lhs_v8];
  const uint32_t count = (uint32_t)values[record->rhs_v8];
  values[record->dst_v8] = iree_math_rotl_u32(source, count);
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE void
iree_vm_bytecode_execute_integer_rotate_left_i64(
    const iree_vm_isa_integer_rotate_left_i64_record_t* record,
    uint64_t* values) {
  const uint64_t source = values[record->lhs_v8];
  const uint32_t count = (uint32_t)values[record->rhs_v8];
  values[record->dst_v8] = iree_math_rotl_u64(source, count);
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE void
iree_vm_bytecode_execute_integer_rotate_right_i32(
    const iree_vm_isa_integer_rotate_right_i32_record_t* record,
    uint64_t* values) {
  const uint32_t source = (uint32_t)values[record->lhs_v8];
  const uint32_t count = (uint32_t)values[record->rhs_v8];
  values[record->dst_v8] = iree_math_rotr_u32(source, count);
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE void
iree_vm_bytecode_execute_integer_rotate_right_i64(
    const iree_vm_isa_integer_rotate_right_i64_record_t* record,
    uint64_t* values) {
  const uint64_t source = values[record->lhs_v8];
  const uint32_t count = (uint32_t)values[record->rhs_v8];
  values[record->dst_v8] = iree_math_rotr_u64(source, count);
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE void
iree_vm_bytecode_execute_integer_count_leading_zeros_i32(
    const iree_vm_isa_integer_count_leading_zeros_i32_record_t* record,
    uint64_t* values) {
  const uint32_t source = (uint32_t)values[record->src_v8];
  values[record->dst_v8] = iree_math_count_leading_zeros_u32(source);
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE void
iree_vm_bytecode_execute_integer_count_leading_zeros_i64(
    const iree_vm_isa_integer_count_leading_zeros_i64_record_t* record,
    uint64_t* values) {
  const uint64_t source = values[record->src_v8];
  values[record->dst_v8] = iree_math_count_leading_zeros_u64(source);
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE void
iree_vm_bytecode_execute_integer_count_trailing_zeros_i32(
    const iree_vm_isa_integer_count_trailing_zeros_i32_record_t* record,
    uint64_t* values) {
  const uint32_t source = (uint32_t)values[record->src_v8];
  values[record->dst_v8] =
      source == 0 ? 32 : iree_math_count_trailing_zeros_u32(source);
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE void
iree_vm_bytecode_execute_integer_count_trailing_zeros_i64(
    const iree_vm_isa_integer_count_trailing_zeros_i64_record_t* record,
    uint64_t* values) {
  const uint64_t source = values[record->src_v8];
  values[record->dst_v8] =
      source == 0 ? 64 : iree_math_count_trailing_zeros_u64(source);
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE void
iree_vm_bytecode_execute_integer_popcount_i32(
    const iree_vm_isa_integer_popcount_i32_record_t* record, uint64_t* values) {
  const uint32_t source = (uint32_t)values[record->src_v8];
  values[record->dst_v8] = iree_math_count_ones_u32(source);
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE void
iree_vm_bytecode_execute_integer_popcount_i64(
    const iree_vm_isa_integer_popcount_i64_record_t* record, uint64_t* values) {
  const uint64_t source = values[record->src_v8];
  values[record->dst_v8] = iree_math_count_ones_u64(source);
}

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
    const iree_vm_isa_integer_bitstream_pack_record_t* record,
    uint64_t* values) {
  iree_vm_bytecode_execute_integer_bitstream(
      record->result_base_v8, record->source_base_v8, record->field_width_u8,
      record->source_count_u8, record->result_count_u8, record->source_width_u8,
      record->result_width_u8, IREE_VM_BYTECODE_INTEGER_BITSTREAM_MODE_PACK,
      values);
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE void
iree_vm_bytecode_execute_integer_bitstream_unpack_u(
    const iree_vm_isa_integer_bitstream_unpack_u_record_t* record,
    uint64_t* values) {
  iree_vm_bytecode_execute_integer_bitstream(
      record->result_base_v8, record->source_base_v8, record->field_width_u8,
      record->source_count_u8, record->result_count_u8, record->source_width_u8,
      record->result_width_u8, IREE_VM_BYTECODE_INTEGER_BITSTREAM_MODE_UNPACK_U,
      values);
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE void
iree_vm_bytecode_execute_integer_bitstream_unpack_s(
    const iree_vm_isa_integer_bitstream_unpack_s_record_t* record,
    uint64_t* values) {
  iree_vm_bytecode_execute_integer_bitstream(
      record->result_base_v8, record->source_base_v8, record->field_width_u8,
      record->source_count_u8, record->result_count_u8, record->source_width_u8,
      record->result_width_u8, IREE_VM_BYTECODE_INTEGER_BITSTREAM_MODE_UNPACK_S,
      values);
}

#endif  // IREE_VM_BYTECODE_INTERPRETER_INTEGER_H_

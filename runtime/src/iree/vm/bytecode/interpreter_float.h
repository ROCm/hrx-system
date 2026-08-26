// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_BYTECODE_INTERPRETER_FLOAT_H_
#define IREE_VM_BYTECODE_INTERPRETER_FLOAT_H_

#include <math.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/vm/bytecode/wire/core/float.h"

// Executes verified Core floating records against a physical value bank.
// Invocation drive segments establish the architectural floating environment
// before reaching these helpers. Raw-bit classification precedes every numeric
// comparison so signaling NaNs cannot reach host comparison instructions.

static inline IREE_ATTRIBUTE_ALWAYS_INLINE float
iree_vm_bytecode_float_f32_from_bits(uint32_t bits) {
  float value = 0.0f;
  memcpy(&value, &bits, sizeof(value));
  return value;
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE uint32_t
iree_vm_bytecode_float_f32_to_bits(float value) {
  uint32_t bits = 0;
  memcpy(&bits, &value, sizeof(bits));
  return bits;
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE double
iree_vm_bytecode_float_f64_from_bits(uint64_t bits) {
  double value = 0.0;
  memcpy(&value, &bits, sizeof(value));
  return value;
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE uint64_t
iree_vm_bytecode_float_f64_to_bits(double value) {
  uint64_t bits = 0;
  memcpy(&bits, &value, sizeof(bits));
  return bits;
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE bool
iree_vm_bytecode_float_f32_is_nan(uint32_t bits) {
  return (bits & UINT32_C(0x7FFFFFFF)) > UINT32_C(0x7F800000);
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE bool
iree_vm_bytecode_float_f64_is_nan(uint64_t bits) {
  return (bits & UINT64_C(0x7FFFFFFFFFFFFFFF)) > UINT64_C(0x7FF0000000000000);
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE bool
iree_vm_bytecode_float_f32_is_zero(uint32_t bits) {
  return (bits & UINT32_C(0x7FFFFFFF)) == 0;
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE bool
iree_vm_bytecode_float_f64_is_zero(uint64_t bits) {
  return (bits & UINT64_C(0x7FFFFFFFFFFFFFFF)) == 0;
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE void
iree_vm_bytecode_execute_float_add_f32(
    const iree_vm_isa_float_add_f32_record_t* record, uint64_t* values) {
  const float lhs =
      iree_vm_bytecode_float_f32_from_bits((uint32_t)values[record->lhs_v8]);
  const float rhs =
      iree_vm_bytecode_float_f32_from_bits((uint32_t)values[record->rhs_v8]);
  values[record->dst_v8] = iree_vm_bytecode_float_f32_to_bits(lhs + rhs);
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE void
iree_vm_bytecode_execute_float_add_f64(
    const iree_vm_isa_float_add_f64_record_t* record, uint64_t* values) {
  const double lhs =
      iree_vm_bytecode_float_f64_from_bits(values[record->lhs_v8]);
  const double rhs =
      iree_vm_bytecode_float_f64_from_bits(values[record->rhs_v8]);
  values[record->dst_v8] = iree_vm_bytecode_float_f64_to_bits(lhs + rhs);
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE void
iree_vm_bytecode_execute_float_sub_f32(
    const iree_vm_isa_float_sub_f32_record_t* record, uint64_t* values) {
  const float lhs =
      iree_vm_bytecode_float_f32_from_bits((uint32_t)values[record->lhs_v8]);
  const float rhs =
      iree_vm_bytecode_float_f32_from_bits((uint32_t)values[record->rhs_v8]);
  values[record->dst_v8] = iree_vm_bytecode_float_f32_to_bits(lhs - rhs);
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE void
iree_vm_bytecode_execute_float_sub_f64(
    const iree_vm_isa_float_sub_f64_record_t* record, uint64_t* values) {
  const double lhs =
      iree_vm_bytecode_float_f64_from_bits(values[record->lhs_v8]);
  const double rhs =
      iree_vm_bytecode_float_f64_from_bits(values[record->rhs_v8]);
  values[record->dst_v8] = iree_vm_bytecode_float_f64_to_bits(lhs - rhs);
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE void
iree_vm_bytecode_execute_float_mul_f32(
    const iree_vm_isa_float_mul_f32_record_t* record, uint64_t* values) {
  const float lhs =
      iree_vm_bytecode_float_f32_from_bits((uint32_t)values[record->lhs_v8]);
  const float rhs =
      iree_vm_bytecode_float_f32_from_bits((uint32_t)values[record->rhs_v8]);
  values[record->dst_v8] = iree_vm_bytecode_float_f32_to_bits(lhs * rhs);
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE void
iree_vm_bytecode_execute_float_mul_f64(
    const iree_vm_isa_float_mul_f64_record_t* record, uint64_t* values) {
  const double lhs =
      iree_vm_bytecode_float_f64_from_bits(values[record->lhs_v8]);
  const double rhs =
      iree_vm_bytecode_float_f64_from_bits(values[record->rhs_v8]);
  values[record->dst_v8] = iree_vm_bytecode_float_f64_to_bits(lhs * rhs);
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE void
iree_vm_bytecode_execute_float_div_f32(
    const iree_vm_isa_float_div_f32_record_t* record, uint64_t* values) {
  const float lhs =
      iree_vm_bytecode_float_f32_from_bits((uint32_t)values[record->lhs_v8]);
  const float rhs =
      iree_vm_bytecode_float_f32_from_bits((uint32_t)values[record->rhs_v8]);
  values[record->dst_v8] = iree_vm_bytecode_float_f32_to_bits(lhs / rhs);
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE void
iree_vm_bytecode_execute_float_div_f64(
    const iree_vm_isa_float_div_f64_record_t* record, uint64_t* values) {
  const double lhs =
      iree_vm_bytecode_float_f64_from_bits(values[record->lhs_v8]);
  const double rhs =
      iree_vm_bytecode_float_f64_from_bits(values[record->rhs_v8]);
  values[record->dst_v8] = iree_vm_bytecode_float_f64_to_bits(lhs / rhs);
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE void
iree_vm_bytecode_execute_float_rem_f32(
    const iree_vm_isa_float_rem_f32_record_t* record, uint64_t* values) {
  const float lhs =
      iree_vm_bytecode_float_f32_from_bits((uint32_t)values[record->lhs_v8]);
  const float rhs =
      iree_vm_bytecode_float_f32_from_bits((uint32_t)values[record->rhs_v8]);
  values[record->dst_v8] = iree_vm_bytecode_float_f32_to_bits(fmodf(lhs, rhs));
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE void
iree_vm_bytecode_execute_float_rem_f64(
    const iree_vm_isa_float_rem_f64_record_t* record, uint64_t* values) {
  const double lhs =
      iree_vm_bytecode_float_f64_from_bits(values[record->lhs_v8]);
  const double rhs =
      iree_vm_bytecode_float_f64_from_bits(values[record->rhs_v8]);
  values[record->dst_v8] = iree_vm_bytecode_float_f64_to_bits(fmod(lhs, rhs));
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE void
iree_vm_bytecode_execute_float_neg_f32(
    const iree_vm_isa_float_neg_f32_record_t* record, uint64_t* values) {
  const uint32_t source = (uint32_t)values[record->src_v8];
  values[record->dst_v8] = source ^ UINT32_C(0x80000000);
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE void
iree_vm_bytecode_execute_float_neg_f64(
    const iree_vm_isa_float_neg_f64_record_t* record, uint64_t* values) {
  const uint64_t source = values[record->src_v8];
  values[record->dst_v8] = source ^ UINT64_C(0x8000000000000000);
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE void
iree_vm_bytecode_execute_float_abs_f32(
    const iree_vm_isa_float_abs_f32_record_t* record, uint64_t* values) {
  const uint32_t source = (uint32_t)values[record->src_v8];
  values[record->dst_v8] = source & UINT32_C(0x7FFFFFFF);
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE void
iree_vm_bytecode_execute_float_abs_f64(
    const iree_vm_isa_float_abs_f64_record_t* record, uint64_t* values) {
  const uint64_t source = values[record->src_v8];
  values[record->dst_v8] = source & UINT64_C(0x7FFFFFFFFFFFFFFF);
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE uint32_t
iree_vm_bytecode_float_minmax_f32_bits(uint32_t lhs_bits, uint32_t rhs_bits,
                                       uint8_t selector) {
  const bool lhs_nan = iree_vm_bytecode_float_f32_is_nan(lhs_bits);
  const bool rhs_nan = iree_vm_bytecode_float_f32_is_nan(rhs_bits);
  const bool selects_number = selector >= IREE_VM_ISA_FLOAT_MINMAX_MINNUM;
  const bool selects_maximum = (selector & 1) != 0;
  if (lhs_nan || rhs_nan) {
    if (selects_number && lhs_nan != rhs_nan) {
      return lhs_nan ? rhs_bits : lhs_bits;
    }
    return UINT32_C(0x7FC00000);
  }
  if (iree_vm_bytecode_float_f32_is_zero(lhs_bits) &&
      iree_vm_bytecode_float_f32_is_zero(rhs_bits)) {
    return selects_maximum ? lhs_bits & rhs_bits : lhs_bits | rhs_bits;
  }
  const float lhs = iree_vm_bytecode_float_f32_from_bits(lhs_bits);
  const float rhs = iree_vm_bytecode_float_f32_from_bits(rhs_bits);
  return selects_maximum ? (lhs > rhs ? lhs_bits : rhs_bits)
                         : (lhs < rhs ? lhs_bits : rhs_bits);
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE uint64_t
iree_vm_bytecode_float_minmax_f64_bits(uint64_t lhs_bits, uint64_t rhs_bits,
                                       uint8_t selector) {
  const bool lhs_nan = iree_vm_bytecode_float_f64_is_nan(lhs_bits);
  const bool rhs_nan = iree_vm_bytecode_float_f64_is_nan(rhs_bits);
  const bool selects_number = selector >= IREE_VM_ISA_FLOAT_MINMAX_MINNUM;
  const bool selects_maximum = (selector & 1) != 0;
  if (lhs_nan || rhs_nan) {
    if (selects_number && lhs_nan != rhs_nan) {
      return lhs_nan ? rhs_bits : lhs_bits;
    }
    return UINT64_C(0x7FF8000000000000);
  }
  if (iree_vm_bytecode_float_f64_is_zero(lhs_bits) &&
      iree_vm_bytecode_float_f64_is_zero(rhs_bits)) {
    return selects_maximum ? lhs_bits & rhs_bits : lhs_bits | rhs_bits;
  }
  const double lhs = iree_vm_bytecode_float_f64_from_bits(lhs_bits);
  const double rhs = iree_vm_bytecode_float_f64_from_bits(rhs_bits);
  return selects_maximum ? (lhs > rhs ? lhs_bits : rhs_bits)
                         : (lhs < rhs ? lhs_bits : rhs_bits);
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE void
iree_vm_bytecode_execute_float_minmax_f32(
    const iree_vm_isa_float_minmax_f32_record_t* record, uint64_t* values) {
  const uint32_t lhs = (uint32_t)values[record->lhs_v8];
  const uint32_t rhs = (uint32_t)values[record->rhs_v8];
  values[record->dst_v8] =
      iree_vm_bytecode_float_minmax_f32_bits(lhs, rhs, record->selector_u8);
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE void
iree_vm_bytecode_execute_float_minmax_f64(
    const iree_vm_isa_float_minmax_f64_record_t* record, uint64_t* values) {
  const uint64_t lhs = values[record->lhs_v8];
  const uint64_t rhs = values[record->rhs_v8];
  values[record->dst_v8] =
      iree_vm_bytecode_float_minmax_f64_bits(lhs, rhs, record->selector_u8);
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE bool
iree_vm_bytecode_float_compare_relation_f32(uint8_t relation, float lhs,
                                            float rhs) {
  switch (relation) {
    case 0:
      return lhs == rhs;
    case 1:
      return lhs > rhs;
    case 2:
      return lhs >= rhs;
    case 3:
      return lhs < rhs;
    case 4:
      return lhs <= rhs;
    default:
      return lhs != rhs;
  }
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE bool
iree_vm_bytecode_float_compare_relation_f64(uint8_t relation, double lhs,
                                            double rhs) {
  switch (relation) {
    case 0:
      return lhs == rhs;
    case 1:
      return lhs > rhs;
    case 2:
      return lhs >= rhs;
    case 3:
      return lhs < rhs;
    case 4:
      return lhs <= rhs;
    default:
      return lhs != rhs;
  }
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE uint64_t
iree_vm_bytecode_float_compare_f32_bits(uint32_t lhs_bits, uint32_t rhs_bits,
                                        uint8_t predicate) {
  const bool ordered = !iree_vm_bytecode_float_f32_is_nan(lhs_bits) &&
                       !iree_vm_bytecode_float_f32_is_nan(rhs_bits);
  if (!ordered) return predicate >= IREE_VM_ISA_FLOAT_COMPARE_UEQ;
  if (predicate == IREE_VM_ISA_FLOAT_COMPARE_ORD) return 1;
  if (predicate == IREE_VM_ISA_FLOAT_COMPARE_UNO) return 0;
  const uint8_t relation = predicate >= IREE_VM_ISA_FLOAT_COMPARE_UEQ
                               ? predicate - IREE_VM_ISA_FLOAT_COMPARE_UEQ
                               : predicate;
  return iree_vm_bytecode_float_compare_relation_f32(
      relation, iree_vm_bytecode_float_f32_from_bits(lhs_bits),
      iree_vm_bytecode_float_f32_from_bits(rhs_bits));
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE uint64_t
iree_vm_bytecode_float_compare_f64_bits(uint64_t lhs_bits, uint64_t rhs_bits,
                                        uint8_t predicate) {
  const bool ordered = !iree_vm_bytecode_float_f64_is_nan(lhs_bits) &&
                       !iree_vm_bytecode_float_f64_is_nan(rhs_bits);
  if (!ordered) return predicate >= IREE_VM_ISA_FLOAT_COMPARE_UEQ;
  if (predicate == IREE_VM_ISA_FLOAT_COMPARE_ORD) return 1;
  if (predicate == IREE_VM_ISA_FLOAT_COMPARE_UNO) return 0;
  const uint8_t relation = predicate >= IREE_VM_ISA_FLOAT_COMPARE_UEQ
                               ? predicate - IREE_VM_ISA_FLOAT_COMPARE_UEQ
                               : predicate;
  return iree_vm_bytecode_float_compare_relation_f64(
      relation, iree_vm_bytecode_float_f64_from_bits(lhs_bits),
      iree_vm_bytecode_float_f64_from_bits(rhs_bits));
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE void
iree_vm_bytecode_execute_float_compare_f32(
    const iree_vm_isa_float_compare_f32_record_t* record, uint64_t* values) {
  const uint32_t lhs = (uint32_t)values[record->lhs_v8];
  const uint32_t rhs = (uint32_t)values[record->rhs_v8];
  values[record->dst_v8] =
      iree_vm_bytecode_float_compare_f32_bits(lhs, rhs, record->predicate_u8);
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE void
iree_vm_bytecode_execute_float_compare_f64(
    const iree_vm_isa_float_compare_f64_record_t* record, uint64_t* values) {
  const uint64_t lhs = values[record->lhs_v8];
  const uint64_t rhs = values[record->rhs_v8];
  values[record->dst_v8] =
      iree_vm_bytecode_float_compare_f64_bits(lhs, rhs, record->predicate_u8);
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE uint64_t
iree_vm_bytecode_float_classify_f32_bits(uint32_t bits, uint8_t selector) {
  const uint32_t magnitude = bits & UINT32_C(0x7FFFFFFF);
  if (selector == IREE_VM_ISA_FLOAT_CLASSIFY_ISNAN) {
    return magnitude > UINT32_C(0x7F800000);
  }
  if (selector == IREE_VM_ISA_FLOAT_CLASSIFY_ISINF) {
    return magnitude == UINT32_C(0x7F800000);
  }
  return magnitude < UINT32_C(0x7F800000);
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE uint64_t
iree_vm_bytecode_float_classify_f64_bits(uint64_t bits, uint8_t selector) {
  const uint64_t magnitude = bits & UINT64_C(0x7FFFFFFFFFFFFFFF);
  if (selector == IREE_VM_ISA_FLOAT_CLASSIFY_ISNAN) {
    return magnitude > UINT64_C(0x7FF0000000000000);
  }
  if (selector == IREE_VM_ISA_FLOAT_CLASSIFY_ISINF) {
    return magnitude == UINT64_C(0x7FF0000000000000);
  }
  return magnitude < UINT64_C(0x7FF0000000000000);
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE void
iree_vm_bytecode_execute_float_classify_f32(
    const iree_vm_isa_float_classify_f32_record_t* record, uint64_t* values) {
  const uint32_t source = (uint32_t)values[record->src_v8];
  values[record->dst_v8] =
      iree_vm_bytecode_float_classify_f32_bits(source, record->selector_u8);
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE void
iree_vm_bytecode_execute_float_classify_f64(
    const iree_vm_isa_float_classify_f64_record_t* record, uint64_t* values) {
  const uint64_t source = values[record->src_v8];
  values[record->dst_v8] =
      iree_vm_bytecode_float_classify_f64_bits(source, record->selector_u8);
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE uint32_t
iree_vm_bytecode_float_clamp_f32_bits(uint32_t value_bits, uint32_t lower_bits,
                                      uint32_t upper_bits, uint8_t mode) {
  if (mode == IREE_VM_ISA_FLOAT_CLAMP_ORDERED) {
    uint32_t result_bits = value_bits;
    if (!iree_vm_bytecode_float_f32_is_nan(result_bits) &&
        !iree_vm_bytecode_float_f32_is_nan(lower_bits) &&
        iree_vm_bytecode_float_f32_from_bits(result_bits) <
            iree_vm_bytecode_float_f32_from_bits(lower_bits)) {
      result_bits = lower_bits;
    }
    if (!iree_vm_bytecode_float_f32_is_nan(result_bits) &&
        !iree_vm_bytecode_float_f32_is_nan(upper_bits) &&
        iree_vm_bytecode_float_f32_from_bits(result_bits) >
            iree_vm_bytecode_float_f32_from_bits(upper_bits)) {
      result_bits = upper_bits;
    }
    return result_bits;
  }
  const uint8_t maximum_selector = mode == IREE_VM_ISA_FLOAT_CLAMP_NUMBER
                                       ? IREE_VM_ISA_FLOAT_MINMAX_MAXNUM
                                       : IREE_VM_ISA_FLOAT_MINMAX_MAXIMUM;
  const uint8_t minimum_selector = mode == IREE_VM_ISA_FLOAT_CLAMP_NUMBER
                                       ? IREE_VM_ISA_FLOAT_MINMAX_MINNUM
                                       : IREE_VM_ISA_FLOAT_MINMAX_MINIMUM;
  const uint32_t maximum = iree_vm_bytecode_float_minmax_f32_bits(
      value_bits, lower_bits, maximum_selector);
  return iree_vm_bytecode_float_minmax_f32_bits(maximum, upper_bits,
                                                minimum_selector);
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE uint64_t
iree_vm_bytecode_float_clamp_f64_bits(uint64_t value_bits, uint64_t lower_bits,
                                      uint64_t upper_bits, uint8_t mode) {
  if (mode == IREE_VM_ISA_FLOAT_CLAMP_ORDERED) {
    uint64_t result_bits = value_bits;
    if (!iree_vm_bytecode_float_f64_is_nan(result_bits) &&
        !iree_vm_bytecode_float_f64_is_nan(lower_bits) &&
        iree_vm_bytecode_float_f64_from_bits(result_bits) <
            iree_vm_bytecode_float_f64_from_bits(lower_bits)) {
      result_bits = lower_bits;
    }
    if (!iree_vm_bytecode_float_f64_is_nan(result_bits) &&
        !iree_vm_bytecode_float_f64_is_nan(upper_bits) &&
        iree_vm_bytecode_float_f64_from_bits(result_bits) >
            iree_vm_bytecode_float_f64_from_bits(upper_bits)) {
      result_bits = upper_bits;
    }
    return result_bits;
  }
  const uint8_t maximum_selector = mode == IREE_VM_ISA_FLOAT_CLAMP_NUMBER
                                       ? IREE_VM_ISA_FLOAT_MINMAX_MAXNUM
                                       : IREE_VM_ISA_FLOAT_MINMAX_MAXIMUM;
  const uint8_t minimum_selector = mode == IREE_VM_ISA_FLOAT_CLAMP_NUMBER
                                       ? IREE_VM_ISA_FLOAT_MINMAX_MINNUM
                                       : IREE_VM_ISA_FLOAT_MINMAX_MINIMUM;
  const uint64_t maximum = iree_vm_bytecode_float_minmax_f64_bits(
      value_bits, lower_bits, maximum_selector);
  return iree_vm_bytecode_float_minmax_f64_bits(maximum, upper_bits,
                                                minimum_selector);
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE void
iree_vm_bytecode_execute_float_clamp_f32(
    const iree_vm_isa_float_clamp_f32_record_t* record, uint64_t* values) {
  const uint32_t value = (uint32_t)values[record->value_v8];
  const uint32_t lower = (uint32_t)values[record->lower_v8];
  const uint32_t upper = (uint32_t)values[record->upper_v8];
  values[record->dst_v8] = iree_vm_bytecode_float_clamp_f32_bits(
      value, lower, upper, record->mode_u8);
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE void
iree_vm_bytecode_execute_float_clamp_f64(
    const iree_vm_isa_float_clamp_f64_record_t* record, uint64_t* values) {
  const uint64_t value = values[record->value_v8];
  const uint64_t lower = values[record->lower_v8];
  const uint64_t upper = values[record->upper_v8];
  values[record->dst_v8] = iree_vm_bytecode_float_clamp_f64_bits(
      value, lower, upper, record->mode_u8);
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE void
iree_vm_bytecode_execute_float_copysign_f32(
    const iree_vm_isa_float_copysign_f32_record_t* record, uint64_t* values) {
  const uint32_t lhs = (uint32_t)values[record->lhs_v8];
  const uint32_t rhs = (uint32_t)values[record->rhs_v8];
  values[record->dst_v8] =
      (lhs & UINT32_C(0x7FFFFFFF)) | (rhs & UINT32_C(0x80000000));
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE void
iree_vm_bytecode_execute_float_copysign_f64(
    const iree_vm_isa_float_copysign_f64_record_t* record, uint64_t* values) {
  const uint64_t lhs = values[record->lhs_v8];
  const uint64_t rhs = values[record->rhs_v8];
  values[record->dst_v8] = (lhs & UINT64_C(0x7FFFFFFFFFFFFFFF)) |
                           (rhs & UINT64_C(0x8000000000000000));
}

#endif  // IREE_VM_BYTECODE_INTERPRETER_FLOAT_H_

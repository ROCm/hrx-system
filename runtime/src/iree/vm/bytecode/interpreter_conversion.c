// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/bytecode/interpreter_conversion.h"

#include "iree/base/internal/math.h"

typedef enum iree_vm_bytecode_float_kind_e {
  IREE_VM_BYTECODE_FLOAT_KIND_FINITE = 0,
  IREE_VM_BYTECODE_FLOAT_KIND_INFINITY,
  IREE_VM_BYTECODE_FLOAT_KIND_NAN,
} iree_vm_bytecode_float_kind_t;

enum iree_vm_bytecode_float_format_flag_bits_e {
  // The maximum exponent field contains infinities and NaNs.
  IREE_VM_BYTECODE_FLOAT_FORMAT_FLAG_IEEE_SPECIALS = 1u << 0,
  // The maximum exponent remains finite except for an all-ones NaN mantissa.
  IREE_VM_BYTECODE_FLOAT_FORMAT_FLAG_FINITE_NAN = 1u << 1,
};
typedef uint8_t iree_vm_bytecode_float_format_flags_t;

typedef struct iree_vm_bytecode_float_format_t {
  // Total encoded width including the sign bit.
  uint8_t bit_count;
  // Encoded exponent width.
  uint8_t exponent_bit_count;
  // Encoded mantissa width excluding the implicit leading bit.
  uint8_t mantissa_bit_count;
  // Arithmetic exponent bias.
  int16_t exponent_bias;
  // Special-value encoding flags.
  iree_vm_bytecode_float_format_flags_t flags;
} iree_vm_bytecode_float_format_t;

typedef struct iree_vm_bytecode_float_value_t {
  // Unsigned finite significand, including any implicit leading bit.
  uint64_t significand;
  // Power-of-two scale applied to |significand|.
  int16_t scale;
  // Nonzero when the mathematical value is negative.
  bool is_negative;
  // Finite, infinity, or NaN classification.
  iree_vm_bytecode_float_kind_t kind;
} iree_vm_bytecode_float_value_t;

static const iree_vm_bytecode_float_format_t iree_vm_bytecode_float_f8e4m3fn = {
    .bit_count = 8,
    .exponent_bit_count = 4,
    .mantissa_bit_count = 3,
    .exponent_bias = 7,
    .flags = IREE_VM_BYTECODE_FLOAT_FORMAT_FLAG_FINITE_NAN,
};
static const iree_vm_bytecode_float_format_t iree_vm_bytecode_float_f8e5m2 = {
    .bit_count = 8,
    .exponent_bit_count = 5,
    .mantissa_bit_count = 2,
    .exponent_bias = 15,
    .flags = IREE_VM_BYTECODE_FLOAT_FORMAT_FLAG_IEEE_SPECIALS,
};
static const iree_vm_bytecode_float_format_t iree_vm_bytecode_float_f16 = {
    .bit_count = 16,
    .exponent_bit_count = 5,
    .mantissa_bit_count = 10,
    .exponent_bias = 15,
    .flags = IREE_VM_BYTECODE_FLOAT_FORMAT_FLAG_IEEE_SPECIALS,
};
static const iree_vm_bytecode_float_format_t iree_vm_bytecode_float_bf16 = {
    .bit_count = 16,
    .exponent_bit_count = 8,
    .mantissa_bit_count = 7,
    .exponent_bias = 127,
    .flags = IREE_VM_BYTECODE_FLOAT_FORMAT_FLAG_IEEE_SPECIALS,
};
static const iree_vm_bytecode_float_format_t iree_vm_bytecode_float_f32 = {
    .bit_count = 32,
    .exponent_bit_count = 8,
    .mantissa_bit_count = 23,
    .exponent_bias = 127,
    .flags = IREE_VM_BYTECODE_FLOAT_FORMAT_FLAG_IEEE_SPECIALS,
};
static const iree_vm_bytecode_float_format_t iree_vm_bytecode_float_f64 = {
    .bit_count = 64,
    .exponent_bit_count = 11,
    .mantissa_bit_count = 52,
    .exponent_bias = 1023,
    .flags = IREE_VM_BYTECODE_FLOAT_FORMAT_FLAG_IEEE_SPECIALS,
};

static uint64_t iree_vm_bytecode_round_shift_right_even(uint64_t value,
                                                        int shift) {
  if (shift <= 0) return value << -shift;
  if (shift > 64) return 0;
  if (shift == 64) {
    return value > UINT64_C(0x8000000000000000) ? 1 : 0;
  }
  const uint64_t quotient = value >> shift;
  const uint64_t remainder_mask = (UINT64_C(1) << shift) - 1;
  const uint64_t remainder = value & remainder_mask;
  const uint64_t midpoint = UINT64_C(1) << (shift - 1);
  return quotient + (remainder > midpoint ||
                     (remainder == midpoint && (quotient & UINT64_C(1)) != 0));
}

static iree_vm_bytecode_float_value_t iree_vm_bytecode_float_decode(
    uint64_t bits, iree_vm_bytecode_float_format_t format) {
  const uint64_t mantissa_mask = (UINT64_C(1) << format.mantissa_bit_count) - 1;
  const uint64_t exponent_mask = (UINT64_C(1) << format.exponent_bit_count) - 1;
  const uint64_t mantissa = bits & mantissa_mask;
  const uint64_t exponent = (bits >> format.mantissa_bit_count) & exponent_mask;
  iree_vm_bytecode_float_value_t value = {
      .significand = mantissa,
      .scale = (int16_t)(1 - format.exponent_bias - format.mantissa_bit_count),
      .is_negative = ((bits >> (format.bit_count - 1)) & 1) != 0,
      .kind = IREE_VM_BYTECODE_FLOAT_KIND_FINITE,
  };
  if (exponent == exponent_mask) {
    if (iree_any_bit_set(format.flags,
                         IREE_VM_BYTECODE_FLOAT_FORMAT_FLAG_FINITE_NAN)) {
      if (mantissa == mantissa_mask) {
        value.kind = IREE_VM_BYTECODE_FLOAT_KIND_NAN;
        value.significand = 0;
        return value;
      }
    } else {
      value.kind = mantissa == 0 ? IREE_VM_BYTECODE_FLOAT_KIND_INFINITY
                                 : IREE_VM_BYTECODE_FLOAT_KIND_NAN;
      value.significand = 0;
      return value;
    }
  }
  if (exponent != 0) {
    value.significand |= UINT64_C(1) << format.mantissa_bit_count;
    value.scale = (int16_t)((int)exponent - format.exponent_bias -
                            format.mantissa_bit_count);
  }
  return value;
}

static uint64_t iree_vm_bytecode_float_encode(
    iree_vm_bytecode_float_value_t value,
    iree_vm_bytecode_float_format_t format) {
  const uint64_t sign = (uint64_t)value.is_negative << (format.bit_count - 1);
  const uint64_t exponent_mask = (UINT64_C(1) << format.exponent_bit_count) - 1;
  const uint64_t mantissa_mask = (UINT64_C(1) << format.mantissa_bit_count) - 1;
  if (value.kind == IREE_VM_BYTECODE_FLOAT_KIND_NAN) {
    const uint64_t quiet_mantissa =
        iree_any_bit_set(format.flags,
                         IREE_VM_BYTECODE_FLOAT_FORMAT_FLAG_FINITE_NAN)
            ? mantissa_mask
            : UINT64_C(1) << (format.mantissa_bit_count - 1);
    return (exponent_mask << format.mantissa_bit_count) | quiet_mantissa;
  }
  if (value.kind == IREE_VM_BYTECODE_FLOAT_KIND_INFINITY) {
    if (iree_any_bit_set(format.flags,
                         IREE_VM_BYTECODE_FLOAT_FORMAT_FLAG_FINITE_NAN)) {
      return sign | (exponent_mask << format.mantissa_bit_count) |
             (mantissa_mask - 1);
    }
    return sign | (exponent_mask << format.mantissa_bit_count);
  }
  if (value.significand == 0) return sign;

  int exponent =
      63 - iree_math_count_leading_zeros_u64(value.significand) + value.scale;
  const int minimum_normal_exponent = 1 - format.exponent_bias;
  const int maximum_normal_exponent =
      (int)exponent_mask - format.exponent_bias -
      (iree_any_bit_set(format.flags,
                        IREE_VM_BYTECODE_FLOAT_FORMAT_FLAG_IEEE_SPECIALS)
           ? 1
           : 0);
  uint64_t encoded_exponent = 0;
  uint64_t encoded_mantissa = 0;
  if (exponent < minimum_normal_exponent) {
    const int shift =
        minimum_normal_exponent - format.mantissa_bit_count - value.scale;
    encoded_mantissa =
        iree_vm_bytecode_round_shift_right_even(value.significand, shift);
    if (encoded_mantissa == (UINT64_C(1) << format.mantissa_bit_count)) {
      encoded_exponent = 1;
      encoded_mantissa = 0;
    }
  } else {
    const int leading_bit =
        63 - iree_math_count_leading_zeros_u64(value.significand);
    uint64_t rounded_significand = iree_vm_bytecode_round_shift_right_even(
        value.significand, leading_bit - format.mantissa_bit_count);
    if (rounded_significand ==
        (UINT64_C(1) << (format.mantissa_bit_count + 1))) {
      rounded_significand >>= 1;
      ++exponent;
    }
    if (exponent > maximum_normal_exponent) {
      if (iree_any_bit_set(format.flags,
                           IREE_VM_BYTECODE_FLOAT_FORMAT_FLAG_FINITE_NAN)) {
        return sign | (exponent_mask << format.mantissa_bit_count) |
               (mantissa_mask - 1);
      }
      return sign | (exponent_mask << format.mantissa_bit_count);
    }
    encoded_exponent = (uint64_t)(exponent + format.exponent_bias);
    encoded_mantissa =
        rounded_significand - (UINT64_C(1) << format.mantissa_bit_count);
    if (iree_any_bit_set(format.flags,
                         IREE_VM_BYTECODE_FLOAT_FORMAT_FLAG_FINITE_NAN) &&
        encoded_exponent == exponent_mask &&
        encoded_mantissa == mantissa_mask) {
      encoded_mantissa = mantissa_mask - 1;
    }
  }
  return sign | (encoded_exponent << format.mantissa_bit_count) |
         encoded_mantissa;
}

static uint64_t iree_vm_bytecode_float_convert(
    uint64_t bits, iree_vm_bytecode_float_format_t source_format,
    iree_vm_bytecode_float_format_t target_format) {
  return iree_vm_bytecode_float_encode(
      iree_vm_bytecode_float_decode(bits, source_format), target_format);
}

static iree_vm_bytecode_float_format_t iree_vm_bytecode_narrow_float_format(
    uint8_t ordinal) {
  switch (ordinal) {
    case 0:
      return iree_vm_bytecode_float_f8e4m3fn;
    case 1:
      return iree_vm_bytecode_float_f8e5m2;
    case 2:
      return iree_vm_bytecode_float_f16;
    default:
      return iree_vm_bytecode_float_bf16;
  }
}

static iree_vm_bytecode_float_value_t iree_vm_bytecode_integer_decode(
    uint64_t bits, bool is_signed, uint8_t bit_count) {
  if (bit_count == 32) bits = (uint32_t)bits;
  const uint64_t sign_mask = UINT64_C(1) << (bit_count - 1);
  const bool is_negative = is_signed && (bits & sign_mask) != 0;
  return (iree_vm_bytecode_float_value_t){
      .significand =
          is_negative ? (~bits + 1) & (sign_mask | (sign_mask - 1)) : bits,
      .scale = 0,
      .is_negative = is_negative,
      .kind = IREE_VM_BYTECODE_FLOAT_KIND_FINITE,
  };
}

static uint64_t iree_vm_bytecode_float_truncate_magnitude(
    iree_vm_bytecode_float_value_t value) {
  if (value.scale >= 0) return value.significand << value.scale;
  const int shift = -value.scale;
  return shift >= 64 ? 0 : value.significand >> shift;
}

void iree_vm_bytecode_execute_conversion_float_extend(
    const iree_vm_isa_conversion_float_extend_record_t* record,
    uint64_t* values) {
  const uint64_t source = values[record->src_v8];
  values[record->dst_v8] = iree_vm_bytecode_float_convert(
      source, iree_vm_bytecode_narrow_float_format(record->selector_u8),
      iree_vm_bytecode_float_f32);
}

void iree_vm_bytecode_execute_conversion_float_truncate(
    const iree_vm_isa_conversion_float_truncate_record_t* record,
    uint64_t* values) {
  const uint64_t source = values[record->src_v8];
  const iree_vm_bytecode_float_format_t source_format =
      record->selector_u8 < IREE_VM_ISA_FLOAT_TRUNCATE_F64_TO_F8E4M3
          ? iree_vm_bytecode_float_f32
          : iree_vm_bytecode_float_f64;
  values[record->dst_v8] = iree_vm_bytecode_float_convert(
      source, source_format,
      iree_vm_bytecode_narrow_float_format(record->selector_u8 & 3));
}

void iree_vm_bytecode_execute_conversion_float_width(
    const iree_vm_isa_conversion_float_width_record_t* record,
    uint64_t* values) {
  const uint64_t source = values[record->src_v8];
  values[record->dst_v8] =
      record->selector_u8 == IREE_VM_ISA_FLOAT_WIDTH_F32_TO_F64
          ? iree_vm_bytecode_float_convert(source, iree_vm_bytecode_float_f32,
                                           iree_vm_bytecode_float_f64)
          : iree_vm_bytecode_float_convert(source, iree_vm_bytecode_float_f64,
                                           iree_vm_bytecode_float_f32);
}

void iree_vm_bytecode_execute_conversion_integer_to_float(
    const iree_vm_isa_conversion_integer_to_float_record_t* record,
    uint64_t* values) {
  const uint8_t selector = record->selector_u8;
  const bool is_bf16 = selector >= IREE_VM_ISA_INTEGER_TO_FLOAT_S32_TO_BF16;
  const uint8_t pair_selector = is_bf16 ? selector - 8 : selector;
  const bool is_signed = (pair_selector & 1) == 0;
  const uint8_t source_bit_count =
      pair_selector >= (is_bf16 ? 2 : IREE_VM_ISA_INTEGER_TO_FLOAT_S64_TO_F32)
          ? 64
          : 32;
  iree_vm_bytecode_float_format_t target_format = iree_vm_bytecode_float_bf16;
  if (!is_bf16) {
    target_format = (pair_selector & 2) != 0 ? iree_vm_bytecode_float_f64
                                             : iree_vm_bytecode_float_f32;
  }
  const iree_vm_bytecode_float_value_t source = iree_vm_bytecode_integer_decode(
      values[record->src_v8], is_signed, source_bit_count);
  values[record->dst_v8] = iree_vm_bytecode_float_encode(source, target_format);
}

iree_vm_bytecode_conversion_failure_t
iree_vm_bytecode_execute_conversion_float_to_integer(
    const iree_vm_isa_conversion_float_to_integer_record_t* record,
    uint64_t* values) {
  const uint8_t selector = record->selector_u8;
  const bool source_is_f64 =
      selector >= IREE_VM_ISA_FLOAT_TO_INTEGER_F64_TO_S32;
  const uint8_t destination_selector = selector & 3;
  const bool destination_is_signed = (destination_selector & 1) == 0;
  const uint8_t destination_bit_count = destination_selector < 2 ? 32 : 64;
  const iree_vm_bytecode_float_value_t source = iree_vm_bytecode_float_decode(
      values[record->src_v8],
      source_is_f64 ? iree_vm_bytecode_float_f64 : iree_vm_bytecode_float_f32);
  if (source.kind == IREE_VM_BYTECODE_FLOAT_KIND_NAN) {
    return IREE_VM_BYTECODE_CONVERSION_FAILURE_NAN;
  }
  if (source.kind == IREE_VM_BYTECODE_FLOAT_KIND_INFINITY) {
    return IREE_VM_BYTECODE_CONVERSION_FAILURE_OUT_OF_RANGE;
  }
  if (source.significand == 0) {
    values[record->dst_v8] = 0;
    return IREE_VM_BYTECODE_CONVERSION_FAILURE_NONE;
  }

  const int exponent =
      63 - iree_math_count_leading_zeros_u64(source.significand) + source.scale;
  uint64_t magnitude = 0;
  if (destination_is_signed) {
    if (source.is_negative) {
      if (exponent > destination_bit_count - 1) {
        return IREE_VM_BYTECODE_CONVERSION_FAILURE_OUT_OF_RANGE;
      }
      magnitude = iree_vm_bytecode_float_truncate_magnitude(source);
      if (magnitude > (UINT64_C(1) << (destination_bit_count - 1))) {
        return IREE_VM_BYTECODE_CONVERSION_FAILURE_OUT_OF_RANGE;
      }
    } else if (exponent >= destination_bit_count - 1) {
      return IREE_VM_BYTECODE_CONVERSION_FAILURE_OUT_OF_RANGE;
    } else {
      magnitude = iree_vm_bytecode_float_truncate_magnitude(source);
    }
  } else {
    if (source.is_negative) {
      if (exponent >= 0) {
        return IREE_VM_BYTECODE_CONVERSION_FAILURE_OUT_OF_RANGE;
      }
      magnitude = 0;
    } else if (exponent >= destination_bit_count) {
      return IREE_VM_BYTECODE_CONVERSION_FAILURE_OUT_OF_RANGE;
    } else {
      magnitude = iree_vm_bytecode_float_truncate_magnitude(source);
    }
  }

  uint64_t result = source.is_negative ? 0 - magnitude : magnitude;
  if (destination_bit_count == 32) result = (uint32_t)result;
  values[record->dst_v8] = result;
  return IREE_VM_BYTECODE_CONVERSION_FAILURE_NONE;
}

// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/bytecode/interpreter_conversion.h"

#include "iree/base/internal/math.h"
#include "iree/vm/bytecode/interpreter/data.inl"

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

typedef uint8_t iree_vm_bytecode_float_format_ordinal_t;
enum iree_vm_bytecode_float_format_ordinal_e {
  IREE_VM_BYTECODE_FLOAT_FORMAT_F8E4M3 = 0,
  IREE_VM_BYTECODE_FLOAT_FORMAT_F8E5M2,
  IREE_VM_BYTECODE_FLOAT_FORMAT_F16,
  IREE_VM_BYTECODE_FLOAT_FORMAT_BF16,
  IREE_VM_BYTECODE_FLOAT_FORMAT_F32,
  IREE_VM_BYTECODE_FLOAT_FORMAT_F64,
  IREE_VM_BYTECODE_FLOAT_FORMAT_COUNT,
};

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

static const iree_vm_bytecode_float_format_t iree_vm_bytecode_float_formats[] =
    {
        {8, 4, 3, 7, IREE_VM_BYTECODE_FLOAT_FORMAT_FLAG_FINITE_NAN},
        {8, 5, 2, 15, IREE_VM_BYTECODE_FLOAT_FORMAT_FLAG_IEEE_SPECIALS},
        {16, 5, 10, 15, IREE_VM_BYTECODE_FLOAT_FORMAT_FLAG_IEEE_SPECIALS},
        {16, 8, 7, 127, IREE_VM_BYTECODE_FLOAT_FORMAT_FLAG_IEEE_SPECIALS},
        {32, 8, 23, 127, IREE_VM_BYTECODE_FLOAT_FORMAT_FLAG_IEEE_SPECIALS},
        {64, 11, 52, 1023, IREE_VM_BYTECODE_FLOAT_FORMAT_FLAG_IEEE_SPECIALS},
};
static_assert(IREE_ARRAYSIZE(iree_vm_bytecode_float_formats) ==
                  IREE_VM_BYTECODE_FLOAT_FORMAT_COUNT,
              "every interpreter float format must have a descriptor");

typedef struct iree_vm_bytecode_integer_conversion_t {
  // Number of source bits consumed from the value cell.
  uint8_t source_bit_count;
  // Number of low result bits published to the value cell.
  uint8_t destination_bit_count;
  // Whether the source sign bit is extended through the destination.
  bool is_signed;
} iree_vm_bytecode_integer_conversion_t;

#define IREE_VM_BYTECODE_INTEGER_CONVERSION_ROW(                       \
    name, ordinal, source_bit_count, destination_bit_count, is_signed) \
  {source_bit_count, destination_bit_count, is_signed},
static const iree_vm_bytecode_integer_conversion_t
    iree_vm_bytecode_integer_conversions[] = {
        IREE_VM_BYTECODE_INTEGER_CONVERSION_ROWS(
            IREE_VM_BYTECODE_INTEGER_CONVERSION_ROW)};
#undef IREE_VM_BYTECODE_INTEGER_CONVERSION_ROW
static_assert(IREE_ARRAYSIZE(iree_vm_bytecode_integer_conversions) ==
                  IREE_VM_BYTECODE_INTEGER_CONVERSION_COUNT,
              "every integer conversion selector must have a descriptor");

typedef struct iree_vm_bytecode_float_conversion_t {
  // Source floating-point storage format.
  iree_vm_bytecode_float_format_ordinal_t source_format;
  // Destination floating-point storage format.
  iree_vm_bytecode_float_format_ordinal_t destination_format;
} iree_vm_bytecode_float_conversion_t;

#define IREE_VM_BYTECODE_FLOAT_CONVERSION_ROW(name, ordinal, source, \
                                              destination)           \
  {IREE_VM_BYTECODE_FLOAT_FORMAT_##source,                           \
   IREE_VM_BYTECODE_FLOAT_FORMAT_##destination},
#define IREE_VM_BYTECODE_DEFINE_FLOAT_CONVERSION_TABLE(name, NAME)       \
  static const iree_vm_bytecode_float_conversion_t                       \
      iree_vm_bytecode_##name##_conversions[] = {                        \
          IREE_VM_BYTECODE_##NAME##_ROWS(                                \
              IREE_VM_BYTECODE_FLOAT_CONVERSION_ROW)};                   \
  static_assert(IREE_ARRAYSIZE(iree_vm_bytecode_##name##_conversions) == \
                    IREE_VM_BYTECODE_##NAME##_COUNT,                     \
                "every float conversion selector must have a descriptor")
IREE_VM_BYTECODE_DEFINE_FLOAT_CONVERSION_TABLE(float_extend, FLOAT_EXTEND);
IREE_VM_BYTECODE_DEFINE_FLOAT_CONVERSION_TABLE(float_truncate, FLOAT_TRUNCATE);
IREE_VM_BYTECODE_DEFINE_FLOAT_CONVERSION_TABLE(float_width, FLOAT_WIDTH);
#undef IREE_VM_BYTECODE_DEFINE_FLOAT_CONVERSION_TABLE
#undef IREE_VM_BYTECODE_FLOAT_CONVERSION_ROW

typedef struct iree_vm_bytecode_integer_float_conversion_t {
  // Integer operand or result width in bits.
  uint8_t integer_bit_count;
  // Floating-point operand or result storage format.
  iree_vm_bytecode_float_format_ordinal_t float_format;
  // Whether the integer is interpreted as signed two's complement.
  bool is_signed;
} iree_vm_bytecode_integer_float_conversion_t;

#define IREE_VM_BYTECODE_INTEGER_TO_FLOAT_ROW(                 \
    name, ordinal, integer_bit_count, is_signed, float_format) \
  {integer_bit_count, IREE_VM_BYTECODE_FLOAT_FORMAT_##float_format, is_signed},
static const iree_vm_bytecode_integer_float_conversion_t
    iree_vm_bytecode_integer_to_float_conversions[] = {
        IREE_VM_BYTECODE_INTEGER_TO_FLOAT_ROWS(
            IREE_VM_BYTECODE_INTEGER_TO_FLOAT_ROW)};
#undef IREE_VM_BYTECODE_INTEGER_TO_FLOAT_ROW
static_assert(IREE_ARRAYSIZE(iree_vm_bytecode_integer_to_float_conversions) ==
                  IREE_VM_BYTECODE_INTEGER_TO_FLOAT_COUNT,
              "every integer-to-float selector must have a descriptor");

#define IREE_VM_BYTECODE_FLOAT_TO_INTEGER_ROW(name, ordinal, float_format,  \
                                              integer_bit_count, is_signed) \
  {integer_bit_count, IREE_VM_BYTECODE_FLOAT_FORMAT_##float_format, is_signed},
static const iree_vm_bytecode_integer_float_conversion_t
    iree_vm_bytecode_float_to_integer_conversions[] = {
        IREE_VM_BYTECODE_FLOAT_TO_INTEGER_ROWS(
            IREE_VM_BYTECODE_FLOAT_TO_INTEGER_ROW)};
#undef IREE_VM_BYTECODE_FLOAT_TO_INTEGER_ROW
static_assert(IREE_ARRAYSIZE(iree_vm_bytecode_float_to_integer_conversions) ==
                  IREE_VM_BYTECODE_FLOAT_TO_INTEGER_COUNT,
              "every float-to-integer selector must have a descriptor");

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

static uint64_t iree_vm_bytecode_integer_low_bits(uint64_t value,
                                                  uint8_t bit_count) {
  return value & (UINT64_MAX >> (64 - bit_count));
}

void iree_vm_bytecode_execute_conversion_integer(
    const iree_vm_bytecode_conversion_integer_t* record, uint64_t* values) {
  const iree_vm_bytecode_integer_conversion_t* conversion =
      &iree_vm_bytecode_integer_conversions[record->selector_u8];
  uint64_t result = iree_vm_bytecode_integer_low_bits(
      values[record->source_v8], conversion->source_bit_count);
  if (conversion->is_signed) {
    const uint64_t sign_bit = UINT64_C(1) << (conversion->source_bit_count - 1);
    result = (result ^ sign_bit) - sign_bit;
  }
  values[record->destination_v8] = iree_vm_bytecode_integer_low_bits(
      result, conversion->destination_bit_count);
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
    const iree_vm_bytecode_conversion_float_extend_t* record,
    uint64_t* values) {
  const iree_vm_bytecode_float_conversion_t* conversion =
      &iree_vm_bytecode_float_extend_conversions[record->selector_u8];
  const uint64_t source = values[record->source_v8];
  values[record->destination_v8] = iree_vm_bytecode_float_convert(
      source, iree_vm_bytecode_float_formats[conversion->source_format],
      iree_vm_bytecode_float_formats[conversion->destination_format]);
}

void iree_vm_bytecode_execute_conversion_float_truncate(
    const iree_vm_bytecode_conversion_float_truncate_t* record,
    uint64_t* values) {
  const iree_vm_bytecode_float_conversion_t* conversion =
      &iree_vm_bytecode_float_truncate_conversions[record->selector_u8];
  const uint64_t source = values[record->source_v8];
  values[record->destination_v8] = iree_vm_bytecode_float_convert(
      source, iree_vm_bytecode_float_formats[conversion->source_format],
      iree_vm_bytecode_float_formats[conversion->destination_format]);
}

void iree_vm_bytecode_execute_conversion_float_width(
    const iree_vm_bytecode_conversion_float_width_t* record, uint64_t* values) {
  const iree_vm_bytecode_float_conversion_t* conversion =
      &iree_vm_bytecode_float_width_conversions[record->selector_u8];
  const uint64_t source = values[record->source_v8];
  values[record->destination_v8] = iree_vm_bytecode_float_convert(
      source, iree_vm_bytecode_float_formats[conversion->source_format],
      iree_vm_bytecode_float_formats[conversion->destination_format]);
}

void iree_vm_bytecode_execute_conversion_integer_to_float(
    const iree_vm_bytecode_conversion_integer_to_float_t* record,
    uint64_t* values) {
  const iree_vm_bytecode_integer_float_conversion_t* conversion =
      &iree_vm_bytecode_integer_to_float_conversions[record->selector_u8];
  const iree_vm_bytecode_float_value_t source = iree_vm_bytecode_integer_decode(
      values[record->source_v8], conversion->is_signed,
      conversion->integer_bit_count);
  values[record->destination_v8] = iree_vm_bytecode_float_encode(
      source, iree_vm_bytecode_float_formats[conversion->float_format]);
}

iree_vm_bytecode_conversion_failure_t
iree_vm_bytecode_execute_conversion_float_to_integer(
    const iree_vm_bytecode_conversion_float_to_integer_t* record,
    uint64_t* values) {
  const iree_vm_bytecode_integer_float_conversion_t* conversion =
      &iree_vm_bytecode_float_to_integer_conversions[record->selector_u8];
  const iree_vm_bytecode_float_value_t source = iree_vm_bytecode_float_decode(
      values[record->source_v8],
      iree_vm_bytecode_float_formats[conversion->float_format]);
  if (source.kind == IREE_VM_BYTECODE_FLOAT_KIND_NAN) {
    return IREE_VM_BYTECODE_CONVERSION_FAILURE_NAN;
  }
  if (source.kind == IREE_VM_BYTECODE_FLOAT_KIND_INFINITY) {
    return IREE_VM_BYTECODE_CONVERSION_FAILURE_OUT_OF_RANGE;
  }
  if (source.significand == 0) {
    values[record->destination_v8] = 0;
    return IREE_VM_BYTECODE_CONVERSION_FAILURE_NONE;
  }

  const int exponent =
      63 - iree_math_count_leading_zeros_u64(source.significand) + source.scale;
  uint64_t magnitude = 0;
  if (conversion->is_signed) {
    if (source.is_negative) {
      if (exponent > conversion->integer_bit_count - 1) {
        return IREE_VM_BYTECODE_CONVERSION_FAILURE_OUT_OF_RANGE;
      }
      magnitude = iree_vm_bytecode_float_truncate_magnitude(source);
      if (magnitude > (UINT64_C(1) << (conversion->integer_bit_count - 1))) {
        return IREE_VM_BYTECODE_CONVERSION_FAILURE_OUT_OF_RANGE;
      }
    } else if (exponent >= conversion->integer_bit_count - 1) {
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
    } else if (exponent >= conversion->integer_bit_count) {
      return IREE_VM_BYTECODE_CONVERSION_FAILURE_OUT_OF_RANGE;
    } else {
      magnitude = iree_vm_bytecode_float_truncate_magnitude(source);
    }
  }

  uint64_t result = source.is_negative ? 0 - magnitude : magnitude;
  values[record->destination_v8] =
      iree_vm_bytecode_integer_low_bits(result, conversion->integer_bit_count);
  return IREE_VM_BYTECODE_CONVERSION_FAILURE_NONE;
}

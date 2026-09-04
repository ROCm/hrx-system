// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Instantiates exact-width floating-point record semantics. The including
// header defines the floating type, bit representation, constants, and width
// suffix. All width selection happens in the C preprocessor; generated machine
// code has no runtime width branch.

static inline IREE_ATTRIBUTE_ALWAYS_INLINE IREE_VM_BYTECODE_FLOAT_TYPE
IREE_VM_BYTECODE_FLOAT_PRIVATE(from_bits)(
    IREE_VM_BYTECODE_FLOAT_BITS_TYPE bits) {
  IREE_VM_BYTECODE_FLOAT_TYPE value = IREE_VM_BYTECODE_FLOAT_LITERAL(0.0);
  memcpy(&value, &bits, sizeof(value));
  return value;
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE IREE_VM_BYTECODE_FLOAT_BITS_TYPE
IREE_VM_BYTECODE_FLOAT_PRIVATE(to_bits)(IREE_VM_BYTECODE_FLOAT_TYPE value) {
  IREE_VM_BYTECODE_FLOAT_BITS_TYPE bits = 0;
  memcpy(&bits, &value, sizeof(bits));
  return bits;
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE bool IREE_VM_BYTECODE_FLOAT_PRIVATE(
    is_nan)(IREE_VM_BYTECODE_FLOAT_BITS_TYPE bits) {
  return (bits & IREE_VM_BYTECODE_FLOAT_MAGNITUDE_MASK) >
         IREE_VM_BYTECODE_FLOAT_INFINITY_BITS;
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE bool IREE_VM_BYTECODE_FLOAT_PRIVATE(
    is_zero)(IREE_VM_BYTECODE_FLOAT_BITS_TYPE bits) {
  return (bits & IREE_VM_BYTECODE_FLOAT_MAGNITUDE_MASK) == 0;
}

#define IREE_VM_BYTECODE_DEFINE_FLOAT_BINARY(name, expression)                 \
  static inline IREE_ATTRIBUTE_ALWAYS_INLINE void                              \
  IREE_VM_BYTECODE_FLOAT_EXECUTE(name)(                                        \
      const IREE_VM_BYTECODE_FLOAT_RECORD(name) * record, uint64_t* values) {  \
    const IREE_VM_BYTECODE_FLOAT_TYPE lhs = IREE_VM_BYTECODE_FLOAT_PRIVATE(    \
        from_bits)((IREE_VM_BYTECODE_FLOAT_BITS_TYPE)values[record->left_v8]); \
    const IREE_VM_BYTECODE_FLOAT_TYPE rhs =                                    \
        IREE_VM_BYTECODE_FLOAT_PRIVATE(from_bits)(                             \
            (IREE_VM_BYTECODE_FLOAT_BITS_TYPE)values[record->right_v8]);       \
    values[record->destination_v8] =                                           \
        IREE_VM_BYTECODE_FLOAT_PRIVATE(to_bits)(expression);                   \
  }

IREE_VM_BYTECODE_DEFINE_FLOAT_BINARY(add, lhs + rhs)
IREE_VM_BYTECODE_DEFINE_FLOAT_BINARY(sub, lhs - rhs)
IREE_VM_BYTECODE_DEFINE_FLOAT_BINARY(mul, lhs* rhs)
IREE_VM_BYTECODE_DEFINE_FLOAT_BINARY(div, lhs / rhs)
IREE_VM_BYTECODE_DEFINE_FLOAT_BINARY(rem, IREE_VM_BYTECODE_FLOAT_FMOD(lhs, rhs))

#undef IREE_VM_BYTECODE_DEFINE_FLOAT_BINARY

static inline IREE_ATTRIBUTE_ALWAYS_INLINE void IREE_VM_BYTECODE_FLOAT_EXECUTE(
    neg)(const IREE_VM_BYTECODE_FLOAT_RECORD(neg) * record, uint64_t* values) {
  const IREE_VM_BYTECODE_FLOAT_BITS_TYPE source =
      (IREE_VM_BYTECODE_FLOAT_BITS_TYPE)values[record->source_v8];
  values[record->destination_v8] = source ^ IREE_VM_BYTECODE_FLOAT_SIGN_MASK;
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE void IREE_VM_BYTECODE_FLOAT_EXECUTE(
    abs)(const IREE_VM_BYTECODE_FLOAT_RECORD(abs) * record, uint64_t* values) {
  const IREE_VM_BYTECODE_FLOAT_BITS_TYPE source =
      (IREE_VM_BYTECODE_FLOAT_BITS_TYPE)values[record->source_v8];
  values[record->destination_v8] =
      source & IREE_VM_BYTECODE_FLOAT_MAGNITUDE_MASK;
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE IREE_VM_BYTECODE_FLOAT_BITS_TYPE
IREE_VM_BYTECODE_FLOAT_BITS(minmax)(IREE_VM_BYTECODE_FLOAT_BITS_TYPE lhs_bits,
                                    IREE_VM_BYTECODE_FLOAT_BITS_TYPE rhs_bits,
                                    uint8_t selector) {
  const bool lhs_nan = IREE_VM_BYTECODE_FLOAT_PRIVATE(is_nan)(lhs_bits);
  const bool rhs_nan = IREE_VM_BYTECODE_FLOAT_PRIVATE(is_nan)(rhs_bits);
  const bool selects_number = selector >= IREE_VM_BYTECODE_FLOAT_MINMAX_MINNUM;
  const bool selects_maximum = (selector & 1) != 0;
  if (lhs_nan || rhs_nan) {
    if (selects_number && lhs_nan != rhs_nan) {
      return lhs_nan ? rhs_bits : lhs_bits;
    }
    return IREE_VM_BYTECODE_FLOAT_QUIET_NAN_BITS;
  }
  if (IREE_VM_BYTECODE_FLOAT_PRIVATE(is_zero)(lhs_bits) &&
      IREE_VM_BYTECODE_FLOAT_PRIVATE(is_zero)(rhs_bits)) {
    return selects_maximum ? lhs_bits & rhs_bits : lhs_bits | rhs_bits;
  }
  const IREE_VM_BYTECODE_FLOAT_TYPE lhs =
      IREE_VM_BYTECODE_FLOAT_PRIVATE(from_bits)(lhs_bits);
  const IREE_VM_BYTECODE_FLOAT_TYPE rhs =
      IREE_VM_BYTECODE_FLOAT_PRIVATE(from_bits)(rhs_bits);
  return selects_maximum ? (lhs > rhs ? lhs_bits : rhs_bits)
                         : (lhs < rhs ? lhs_bits : rhs_bits);
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE void IREE_VM_BYTECODE_FLOAT_EXECUTE(
    minmax)(const IREE_VM_BYTECODE_FLOAT_RECORD(minmax) * record,
            uint64_t* values) {
  const IREE_VM_BYTECODE_FLOAT_BITS_TYPE lhs =
      (IREE_VM_BYTECODE_FLOAT_BITS_TYPE)values[record->left_v8];
  const IREE_VM_BYTECODE_FLOAT_BITS_TYPE rhs =
      (IREE_VM_BYTECODE_FLOAT_BITS_TYPE)values[record->right_v8];
  values[record->destination_v8] =
      IREE_VM_BYTECODE_FLOAT_BITS(minmax)(lhs, rhs, record->selector_u8);
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE bool IREE_VM_BYTECODE_FLOAT_PRIVATE(
    compare_relation)(uint8_t relation, IREE_VM_BYTECODE_FLOAT_TYPE lhs,
                      IREE_VM_BYTECODE_FLOAT_TYPE rhs) {
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

static inline IREE_ATTRIBUTE_ALWAYS_INLINE uint64_t IREE_VM_BYTECODE_FLOAT_BITS(
    compare)(IREE_VM_BYTECODE_FLOAT_BITS_TYPE lhs_bits,
             IREE_VM_BYTECODE_FLOAT_BITS_TYPE rhs_bits, uint8_t predicate) {
  const bool ordered = !IREE_VM_BYTECODE_FLOAT_PRIVATE(is_nan)(lhs_bits) &&
                       !IREE_VM_BYTECODE_FLOAT_PRIVATE(is_nan)(rhs_bits);
  if (!ordered) return predicate >= IREE_VM_BYTECODE_FLOAT_COMPARE_UEQ;
  if (predicate == IREE_VM_BYTECODE_FLOAT_COMPARE_ORD) return 1;
  if (predicate == IREE_VM_BYTECODE_FLOAT_COMPARE_UNO) return 0;
  const uint8_t relation = predicate >= IREE_VM_BYTECODE_FLOAT_COMPARE_UEQ
                               ? predicate - IREE_VM_BYTECODE_FLOAT_COMPARE_UEQ
                               : predicate;
  return IREE_VM_BYTECODE_FLOAT_PRIVATE(compare_relation)(
      relation, IREE_VM_BYTECODE_FLOAT_PRIVATE(from_bits)(lhs_bits),
      IREE_VM_BYTECODE_FLOAT_PRIVATE(from_bits)(rhs_bits));
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE void IREE_VM_BYTECODE_FLOAT_EXECUTE(
    compare)(const IREE_VM_BYTECODE_FLOAT_RECORD(compare) * record,
             uint64_t* values) {
  const IREE_VM_BYTECODE_FLOAT_BITS_TYPE lhs =
      (IREE_VM_BYTECODE_FLOAT_BITS_TYPE)values[record->left_v8];
  const IREE_VM_BYTECODE_FLOAT_BITS_TYPE rhs =
      (IREE_VM_BYTECODE_FLOAT_BITS_TYPE)values[record->right_v8];
  values[record->destination_v8] =
      IREE_VM_BYTECODE_FLOAT_BITS(compare)(lhs, rhs, record->predicate_u8);
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE uint64_t IREE_VM_BYTECODE_FLOAT_BITS(
    classify)(IREE_VM_BYTECODE_FLOAT_BITS_TYPE bits, uint8_t selector) {
  const IREE_VM_BYTECODE_FLOAT_BITS_TYPE magnitude =
      bits & IREE_VM_BYTECODE_FLOAT_MAGNITUDE_MASK;
  if (selector == IREE_VM_BYTECODE_FLOAT_CLASSIFY_ISNAN) {
    return magnitude > IREE_VM_BYTECODE_FLOAT_INFINITY_BITS;
  }
  if (selector == IREE_VM_BYTECODE_FLOAT_CLASSIFY_ISINF) {
    return magnitude == IREE_VM_BYTECODE_FLOAT_INFINITY_BITS;
  }
  return magnitude < IREE_VM_BYTECODE_FLOAT_INFINITY_BITS;
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE void IREE_VM_BYTECODE_FLOAT_EXECUTE(
    classify)(const IREE_VM_BYTECODE_FLOAT_RECORD(classify) * record,
              uint64_t* values) {
  const IREE_VM_BYTECODE_FLOAT_BITS_TYPE source =
      (IREE_VM_BYTECODE_FLOAT_BITS_TYPE)values[record->source_v8];
  values[record->destination_v8] =
      IREE_VM_BYTECODE_FLOAT_BITS(classify)(source, record->selector_u8);
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE IREE_VM_BYTECODE_FLOAT_BITS_TYPE
IREE_VM_BYTECODE_FLOAT_BITS(clamp)(IREE_VM_BYTECODE_FLOAT_BITS_TYPE value_bits,
                                   IREE_VM_BYTECODE_FLOAT_BITS_TYPE lower_bits,
                                   IREE_VM_BYTECODE_FLOAT_BITS_TYPE upper_bits,
                                   uint8_t mode) {
  if (mode == IREE_VM_BYTECODE_FLOAT_CLAMP_ORDERED) {
    IREE_VM_BYTECODE_FLOAT_BITS_TYPE result_bits = value_bits;
    if (!IREE_VM_BYTECODE_FLOAT_PRIVATE(is_nan)(result_bits) &&
        !IREE_VM_BYTECODE_FLOAT_PRIVATE(is_nan)(lower_bits) &&
        IREE_VM_BYTECODE_FLOAT_PRIVATE(from_bits)(result_bits) <
            IREE_VM_BYTECODE_FLOAT_PRIVATE(from_bits)(lower_bits)) {
      result_bits = lower_bits;
    }
    if (!IREE_VM_BYTECODE_FLOAT_PRIVATE(is_nan)(result_bits) &&
        !IREE_VM_BYTECODE_FLOAT_PRIVATE(is_nan)(upper_bits) &&
        IREE_VM_BYTECODE_FLOAT_PRIVATE(from_bits)(result_bits) >
            IREE_VM_BYTECODE_FLOAT_PRIVATE(from_bits)(upper_bits)) {
      result_bits = upper_bits;
    }
    return result_bits;
  }
  const uint8_t maximum_selector = mode == IREE_VM_BYTECODE_FLOAT_CLAMP_NUMBER
                                       ? IREE_VM_BYTECODE_FLOAT_MINMAX_MAXNUM
                                       : IREE_VM_BYTECODE_FLOAT_MINMAX_MAXIMUM;
  const uint8_t minimum_selector = mode == IREE_VM_BYTECODE_FLOAT_CLAMP_NUMBER
                                       ? IREE_VM_BYTECODE_FLOAT_MINMAX_MINNUM
                                       : IREE_VM_BYTECODE_FLOAT_MINMAX_MINIMUM;
  const IREE_VM_BYTECODE_FLOAT_BITS_TYPE maximum = IREE_VM_BYTECODE_FLOAT_BITS(
      minmax)(value_bits, lower_bits, maximum_selector);
  return IREE_VM_BYTECODE_FLOAT_BITS(minmax)(maximum, upper_bits,
                                             minimum_selector);
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE void IREE_VM_BYTECODE_FLOAT_EXECUTE(
    clamp)(const IREE_VM_BYTECODE_FLOAT_RECORD(clamp) * record,
           uint64_t* values) {
  const IREE_VM_BYTECODE_FLOAT_BITS_TYPE value =
      (IREE_VM_BYTECODE_FLOAT_BITS_TYPE)values[record->value_v8];
  const IREE_VM_BYTECODE_FLOAT_BITS_TYPE lower =
      (IREE_VM_BYTECODE_FLOAT_BITS_TYPE)values[record->lower_v8];
  const IREE_VM_BYTECODE_FLOAT_BITS_TYPE upper =
      (IREE_VM_BYTECODE_FLOAT_BITS_TYPE)values[record->upper_v8];
  values[record->destination_v8] =
      IREE_VM_BYTECODE_FLOAT_BITS(clamp)(value, lower, upper, record->mode_u8);
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE void IREE_VM_BYTECODE_FLOAT_EXECUTE(
    copysign)(const IREE_VM_BYTECODE_FLOAT_RECORD(copysign) * record,
              uint64_t* values) {
  const IREE_VM_BYTECODE_FLOAT_BITS_TYPE lhs =
      (IREE_VM_BYTECODE_FLOAT_BITS_TYPE)values[record->left_v8];
  const IREE_VM_BYTECODE_FLOAT_BITS_TYPE rhs =
      (IREE_VM_BYTECODE_FLOAT_BITS_TYPE)values[record->right_v8];
  values[record->destination_v8] =
      (lhs & IREE_VM_BYTECODE_FLOAT_MAGNITUDE_MASK) |
      (rhs & IREE_VM_BYTECODE_FLOAT_SIGN_MASK);
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE void IREE_VM_BYTECODE_FLOAT_EXECUTE(
    math_unary)(const IREE_VM_BYTECODE_FLOAT_RECORD(math_unary) * record,
                uint64_t* values) {
  values[record->destination_v8] = IREE_VM_BYTECODE_FLOAT_MATH(unary)(
      record->selector_u8,
      (IREE_VM_BYTECODE_FLOAT_BITS_TYPE)values[record->source_v8]);
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE void IREE_VM_BYTECODE_FLOAT_EXECUTE(
    math_binary)(const IREE_VM_BYTECODE_FLOAT_RECORD(math_binary) * record,
                 uint64_t* values) {
  values[record->destination_v8] = IREE_VM_BYTECODE_FLOAT_MATH(binary)(
      record->selector_u8,
      (IREE_VM_BYTECODE_FLOAT_BITS_TYPE)values[record->left_v8],
      (IREE_VM_BYTECODE_FLOAT_BITS_TYPE)values[record->right_v8]);
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE void IREE_VM_BYTECODE_FLOAT_EXECUTE(
    math_ternary)(const IREE_VM_BYTECODE_FLOAT_RECORD(math_ternary) * record,
                  uint64_t* values) {
  values[record->destination_v8] = IREE_VM_BYTECODE_FLOAT_MATH(ternary)(
      record->selector_u8,
      (IREE_VM_BYTECODE_FLOAT_BITS_TYPE)values[record->a_v8],
      (IREE_VM_BYTECODE_FLOAT_BITS_TYPE)values[record->b_v8],
      (IREE_VM_BYTECODE_FLOAT_BITS_TYPE)values[record->c_v8]);
}

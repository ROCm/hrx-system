// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Instantiates exact-width integer record semantics. The including header
// defines the unsigned carrier type, constants, math primitives, and width
// suffix. All width selection happens in the C preprocessor; generated machine
// code has no runtime width branch.

#define IREE_VM_BYTECODE_DEFINE_INTEGER_BINARY(kind, name, expression) \
  static inline IREE_ATTRIBUTE_ALWAYS_INLINE void                      \
  IREE_VM_BYTECODE_INTEGER_EXECUTE(name, kind)(                        \
      const IREE_VM_BYTECODE_INTEGER_RECORD(name, kind) * record,      \
      uint64_t* values) {                                              \
    const IREE_VM_BYTECODE_INTEGER_TYPE left =                         \
        (IREE_VM_BYTECODE_INTEGER_TYPE)values[record->left_v8];        \
    const IREE_VM_BYTECODE_INTEGER_TYPE right =                        \
        (IREE_VM_BYTECODE_INTEGER_TYPE)values[record->right_v8];       \
    values[record->destination_v8] =                                   \
        (IREE_VM_BYTECODE_INTEGER_TYPE)(expression);                   \
  }

IREE_VM_BYTECODE_DEFINE_INTEGER_BINARY(i, add, left + right)
IREE_VM_BYTECODE_DEFINE_INTEGER_BINARY(i, sub, left - right)
IREE_VM_BYTECODE_DEFINE_INTEGER_BINARY(i, mul, left* right)
IREE_VM_BYTECODE_DEFINE_INTEGER_BINARY(
    s, min,
    (left ^ IREE_VM_BYTECODE_INTEGER_SIGN_MASK) <=
            (right ^ IREE_VM_BYTECODE_INTEGER_SIGN_MASK)
        ? left
        : right)
IREE_VM_BYTECODE_DEFINE_INTEGER_BINARY(u, min, left <= right ? left : right)
IREE_VM_BYTECODE_DEFINE_INTEGER_BINARY(
    s, max,
    (left ^ IREE_VM_BYTECODE_INTEGER_SIGN_MASK) >=
            (right ^ IREE_VM_BYTECODE_INTEGER_SIGN_MASK)
        ? left
        : right)
IREE_VM_BYTECODE_DEFINE_INTEGER_BINARY(u, max, left >= right ? left : right)
IREE_VM_BYTECODE_DEFINE_INTEGER_BINARY(i, and, left& right)
IREE_VM_BYTECODE_DEFINE_INTEGER_BINARY(i, or, left | right)
IREE_VM_BYTECODE_DEFINE_INTEGER_BINARY(i, xor, left ^ right)

#undef IREE_VM_BYTECODE_DEFINE_INTEGER_BINARY

#define IREE_VM_BYTECODE_DEFINE_INTEGER_UNARY(kind, name, expression) \
  static inline IREE_ATTRIBUTE_ALWAYS_INLINE void                     \
  IREE_VM_BYTECODE_INTEGER_EXECUTE(name, kind)(                       \
      const IREE_VM_BYTECODE_INTEGER_RECORD(name, kind) * record,     \
      uint64_t* values) {                                             \
    const IREE_VM_BYTECODE_INTEGER_TYPE source =                      \
        (IREE_VM_BYTECODE_INTEGER_TYPE)values[record->source_v8];     \
    values[record->destination_v8] =                                  \
        (IREE_VM_BYTECODE_INTEGER_TYPE)(expression);                  \
  }

IREE_VM_BYTECODE_DEFINE_INTEGER_UNARY(i, neg,
                                      IREE_VM_BYTECODE_INTEGER_ZERO - source)
IREE_VM_BYTECODE_DEFINE_INTEGER_UNARY(s, abs,
                                      (source &
                                       IREE_VM_BYTECODE_INTEGER_SIGN_MASK)
                                          ? IREE_VM_BYTECODE_INTEGER_ZERO -
                                                source
                                          : source)

#undef IREE_VM_BYTECODE_DEFINE_INTEGER_UNARY

// Returns the unsigned magnitude of the two's-complement bit pattern |value|.
static inline IREE_ATTRIBUTE_ALWAYS_INLINE IREE_VM_BYTECODE_INTEGER_TYPE
IREE_VM_BYTECODE_INTEGER_PRIVATE(magnitude)(
    IREE_VM_BYTECODE_INTEGER_TYPE value) {
  const IREE_VM_BYTECODE_INTEGER_TYPE sign_mask =
      IREE_VM_BYTECODE_INTEGER_ZERO -
      (value >> (IREE_VM_BYTECODE_INTEGER_WIDTH - 1));
  return (value ^ sign_mask) - sign_mask;
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE
    iree_vm_bytecode_integer_division_failure_t
    IREE_VM_BYTECODE_INTEGER_EXECUTE(div, s)(
        const IREE_VM_BYTECODE_INTEGER_RECORD(div, s) * record,
        uint64_t* values) {
  const IREE_VM_BYTECODE_INTEGER_TYPE lhs =
      (IREE_VM_BYTECODE_INTEGER_TYPE)values[record->left_v8];
  const IREE_VM_BYTECODE_INTEGER_TYPE rhs =
      (IREE_VM_BYTECODE_INTEGER_TYPE)values[record->right_v8];
  if (rhs == 0) {
    return IREE_VM_BYTECODE_INTEGER_DIVISION_FAILURE_DIVIDE_BY_ZERO;
  }
  if (lhs == IREE_VM_BYTECODE_INTEGER_SIGN_MASK &&
      rhs == IREE_VM_BYTECODE_INTEGER_MAX) {
    return IREE_VM_BYTECODE_INTEGER_DIVISION_FAILURE_SIGNED_OVERFLOW;
  }
  const IREE_VM_BYTECODE_INTEGER_TYPE quotient =
      IREE_VM_BYTECODE_INTEGER_PRIVATE(magnitude)(lhs) /
      IREE_VM_BYTECODE_INTEGER_PRIVATE(magnitude)(rhs);
  const IREE_VM_BYTECODE_INTEGER_TYPE sign_mask =
      IREE_VM_BYTECODE_INTEGER_ZERO -
      ((lhs ^ rhs) >> (IREE_VM_BYTECODE_INTEGER_WIDTH - 1));
  values[record->destination_v8] = (quotient ^ sign_mask) - sign_mask;
  return IREE_VM_BYTECODE_INTEGER_DIVISION_FAILURE_NONE;
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE
    iree_vm_bytecode_integer_division_failure_t
    IREE_VM_BYTECODE_INTEGER_EXECUTE(div, u)(
        const IREE_VM_BYTECODE_INTEGER_RECORD(div, u) * record,
        uint64_t* values) {
  const IREE_VM_BYTECODE_INTEGER_TYPE lhs =
      (IREE_VM_BYTECODE_INTEGER_TYPE)values[record->left_v8];
  const IREE_VM_BYTECODE_INTEGER_TYPE rhs =
      (IREE_VM_BYTECODE_INTEGER_TYPE)values[record->right_v8];
  if (rhs == 0) {
    return IREE_VM_BYTECODE_INTEGER_DIVISION_FAILURE_DIVIDE_BY_ZERO;
  }
  values[record->destination_v8] = lhs / rhs;
  return IREE_VM_BYTECODE_INTEGER_DIVISION_FAILURE_NONE;
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE
    iree_vm_bytecode_integer_division_failure_t
    IREE_VM_BYTECODE_INTEGER_EXECUTE(rem, s)(
        const IREE_VM_BYTECODE_INTEGER_RECORD(rem, s) * record,
        uint64_t* values) {
  const IREE_VM_BYTECODE_INTEGER_TYPE lhs =
      (IREE_VM_BYTECODE_INTEGER_TYPE)values[record->left_v8];
  const IREE_VM_BYTECODE_INTEGER_TYPE rhs =
      (IREE_VM_BYTECODE_INTEGER_TYPE)values[record->right_v8];
  if (rhs == 0) {
    return IREE_VM_BYTECODE_INTEGER_DIVISION_FAILURE_DIVIDE_BY_ZERO;
  }
  const IREE_VM_BYTECODE_INTEGER_TYPE remainder =
      IREE_VM_BYTECODE_INTEGER_PRIVATE(magnitude)(lhs) %
      IREE_VM_BYTECODE_INTEGER_PRIVATE(magnitude)(rhs);
  const IREE_VM_BYTECODE_INTEGER_TYPE sign_mask =
      IREE_VM_BYTECODE_INTEGER_ZERO -
      (lhs >> (IREE_VM_BYTECODE_INTEGER_WIDTH - 1));
  values[record->destination_v8] = (remainder ^ sign_mask) - sign_mask;
  return IREE_VM_BYTECODE_INTEGER_DIVISION_FAILURE_NONE;
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE
    iree_vm_bytecode_integer_division_failure_t
    IREE_VM_BYTECODE_INTEGER_EXECUTE(rem, u)(
        const IREE_VM_BYTECODE_INTEGER_RECORD(rem, u) * record,
        uint64_t* values) {
  const IREE_VM_BYTECODE_INTEGER_TYPE lhs =
      (IREE_VM_BYTECODE_INTEGER_TYPE)values[record->left_v8];
  const IREE_VM_BYTECODE_INTEGER_TYPE rhs =
      (IREE_VM_BYTECODE_INTEGER_TYPE)values[record->right_v8];
  if (rhs == 0) {
    return IREE_VM_BYTECODE_INTEGER_DIVISION_FAILURE_DIVIDE_BY_ZERO;
  }
  values[record->destination_v8] = lhs % rhs;
  return IREE_VM_BYTECODE_INTEGER_DIVISION_FAILURE_NONE;
}

// Performs a total arithmetic right shift over a two's-complement bit pattern.
static inline IREE_ATTRIBUTE_ALWAYS_INLINE IREE_VM_BYTECODE_INTEGER_TYPE
IREE_VM_BYTECODE_INTEGER_PRIVATE(shift_right_s)(
    IREE_VM_BYTECODE_INTEGER_TYPE source, uint32_t count) {
  count &= IREE_VM_BYTECODE_INTEGER_SHIFT_MASK;
  const IREE_VM_BYTECODE_INTEGER_TYPE sign_mask =
      IREE_VM_BYTECODE_INTEGER_ZERO -
      (source >> (IREE_VM_BYTECODE_INTEGER_WIDTH - 1));
  const IREE_VM_BYTECODE_INTEGER_TYPE fill_mask =
      ~(IREE_VM_BYTECODE_INTEGER_MAX >> count);
  return (source >> count) | (sign_mask & fill_mask);
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE void
IREE_VM_BYTECODE_INTEGER_EXECUTE(shift_left, i)(
    const IREE_VM_BYTECODE_INTEGER_RECORD(shift_left, i) * record,
    uint64_t* values) {
  const IREE_VM_BYTECODE_INTEGER_TYPE source =
      (IREE_VM_BYTECODE_INTEGER_TYPE)values[record->left_v8];
  const uint32_t count =
      (uint32_t)values[record->right_v8] & IREE_VM_BYTECODE_INTEGER_SHIFT_MASK;
  values[record->destination_v8] = source << count;
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE void
IREE_VM_BYTECODE_INTEGER_EXECUTE(shift_right, s)(
    const IREE_VM_BYTECODE_INTEGER_RECORD(shift_right, s) * record,
    uint64_t* values) {
  const IREE_VM_BYTECODE_INTEGER_TYPE source =
      (IREE_VM_BYTECODE_INTEGER_TYPE)values[record->left_v8];
  const uint32_t count = (uint32_t)values[record->right_v8];
  values[record->destination_v8] =
      IREE_VM_BYTECODE_INTEGER_PRIVATE(shift_right_s)(source, count);
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE void
IREE_VM_BYTECODE_INTEGER_EXECUTE(shift_right, u)(
    const IREE_VM_BYTECODE_INTEGER_RECORD(shift_right, u) * record,
    uint64_t* values) {
  const IREE_VM_BYTECODE_INTEGER_TYPE source =
      (IREE_VM_BYTECODE_INTEGER_TYPE)values[record->left_v8];
  const uint32_t count =
      (uint32_t)values[record->right_v8] & IREE_VM_BYTECODE_INTEGER_SHIFT_MASK;
  values[record->destination_v8] = source >> count;
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE void
IREE_VM_BYTECODE_INTEGER_EXECUTE(rotate_left, i)(
    const IREE_VM_BYTECODE_INTEGER_RECORD(rotate_left, i) * record,
    uint64_t* values) {
  const IREE_VM_BYTECODE_INTEGER_TYPE source =
      (IREE_VM_BYTECODE_INTEGER_TYPE)values[record->left_v8];
  const uint32_t count =
      (uint32_t)values[record->right_v8] & IREE_VM_BYTECODE_INTEGER_SHIFT_MASK;
  values[record->destination_v8] =
      (source << count) | (source >> ((IREE_VM_BYTECODE_INTEGER_WIDTH - count) &
                                      IREE_VM_BYTECODE_INTEGER_SHIFT_MASK));
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE void
IREE_VM_BYTECODE_INTEGER_EXECUTE(rotate_right, i)(
    const IREE_VM_BYTECODE_INTEGER_RECORD(rotate_right, i) * record,
    uint64_t* values) {
  const IREE_VM_BYTECODE_INTEGER_TYPE source =
      (IREE_VM_BYTECODE_INTEGER_TYPE)values[record->left_v8];
  const uint32_t count =
      (uint32_t)values[record->right_v8] & IREE_VM_BYTECODE_INTEGER_SHIFT_MASK;
  values[record->destination_v8] =
      (source >> count) | (source << ((IREE_VM_BYTECODE_INTEGER_WIDTH - count) &
                                      IREE_VM_BYTECODE_INTEGER_SHIFT_MASK));
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE void
IREE_VM_BYTECODE_INTEGER_EXECUTE(count_leading_zeros, i)(
    const IREE_VM_BYTECODE_INTEGER_RECORD(count_leading_zeros, i) * record,
    uint64_t* values) {
  const IREE_VM_BYTECODE_INTEGER_TYPE source =
      (IREE_VM_BYTECODE_INTEGER_TYPE)values[record->source_v8];
  values[record->destination_v8] = IREE_VM_BYTECODE_INTEGER_CLZ(source);
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE void
IREE_VM_BYTECODE_INTEGER_EXECUTE(count_trailing_zeros, i)(
    const IREE_VM_BYTECODE_INTEGER_RECORD(count_trailing_zeros, i) * record,
    uint64_t* values) {
  const IREE_VM_BYTECODE_INTEGER_TYPE source =
      (IREE_VM_BYTECODE_INTEGER_TYPE)values[record->source_v8];
  values[record->destination_v8] = source == 0
                                       ? IREE_VM_BYTECODE_INTEGER_WIDTH
                                       : IREE_VM_BYTECODE_INTEGER_CTZ(source);
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE void
IREE_VM_BYTECODE_INTEGER_EXECUTE(popcount, i)(
    const IREE_VM_BYTECODE_INTEGER_RECORD(popcount, i) * record,
    uint64_t* values) {
  const IREE_VM_BYTECODE_INTEGER_TYPE source =
      (IREE_VM_BYTECODE_INTEGER_TYPE)values[record->source_v8];
  values[record->destination_v8] = IREE_VM_BYTECODE_INTEGER_POPCOUNT(source);
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE void
IREE_VM_BYTECODE_INTEGER_EXECUTE(compare, i)(
    const IREE_VM_BYTECODE_INTEGER_RECORD(compare, i) * record,
    uint64_t* values) {
  const IREE_VM_BYTECODE_INTEGER_TYPE lhs =
      (IREE_VM_BYTECODE_INTEGER_TYPE)values[record->left_v8];
  const IREE_VM_BYTECODE_INTEGER_TYPE rhs =
      (IREE_VM_BYTECODE_INTEGER_TYPE)values[record->right_v8];
  values[record->destination_v8] = iree_vm_bytecode_integer_compare_bits(
      lhs, rhs, IREE_VM_BYTECODE_INTEGER_SIGN_MASK, record->predicate_u8);
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE void
IREE_VM_BYTECODE_INTEGER_EXECUTE(lea, i)(
    const IREE_VM_BYTECODE_INTEGER_RECORD(lea, i) * record, uint64_t* values) {
  const IREE_VM_BYTECODE_INTEGER_TYPE base =
      (IREE_VM_BYTECODE_INTEGER_TYPE)values[record->base_v8];
  const IREE_VM_BYTECODE_INTEGER_TYPE index =
      (IREE_VM_BYTECODE_INTEGER_TYPE)values[record->index_v8];
  const IREE_VM_BYTECODE_INTEGER_TYPE offset =
      (IREE_VM_BYTECODE_INTEGER_TYPE)((IREE_VM_BYTECODE_INTEGER_SIGNED_TYPE)
                                          record->offset_i16);
  values[record->destination_v8] =
      base + index * (IREE_VM_BYTECODE_INTEGER_TYPE)record->scale_u8 + offset;
}

static inline IREE_ATTRIBUTE_ALWAYS_INLINE void
IREE_VM_BYTECODE_INTEGER_EXECUTE(ceildiv_pow2, u)(
    const IREE_VM_BYTECODE_INTEGER_RECORD(ceildiv_pow2, u) * record,
    uint64_t* values) {
  const IREE_VM_BYTECODE_INTEGER_TYPE source =
      (IREE_VM_BYTECODE_INTEGER_TYPE)values[record->source_v8];
  const IREE_VM_BYTECODE_INTEGER_TYPE mask =
      (IREE_VM_BYTECODE_INTEGER_ONE << record->log2_u8) -
      IREE_VM_BYTECODE_INTEGER_ONE;
  values[record->destination_v8] =
      (source >> record->log2_u8) + ((source & mask) != 0);
}

// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Numeric storage interpretations shared by testbench reference generation and
// numerical expectations.

#ifndef LOOM_TOOLING_TESTBENCH_NUMERIC_VALUE_H_
#define LOOM_TOOLING_TESTBENCH_NUMERIC_VALUE_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "loom/ir/scalar_type.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_testbench_numeric_kind_e {
  // Invalid or unset numeric kind.
  LOOM_TESTBENCH_NUMERIC_NONE = 0,
  // 8-bit E4M3 finite-only float.
  LOOM_TESTBENCH_NUMERIC_F8E4M3 = 1,
  // 8-bit E5M2 float.
  LOOM_TESTBENCH_NUMERIC_F8E5M2 = 2,
  // IEEE f16.
  LOOM_TESTBENCH_NUMERIC_F16 = 3,
  // bfloat16.
  LOOM_TESTBENCH_NUMERIC_BF16 = 4,
  // IEEE f32.
  LOOM_TESTBENCH_NUMERIC_F32 = 5,
  // IEEE f64.
  LOOM_TESTBENCH_NUMERIC_F64 = 6,
  // Signed 8-bit integer.
  LOOM_TESTBENCH_NUMERIC_I8 = 7,
  // Unsigned 8-bit integer.
  LOOM_TESTBENCH_NUMERIC_U8 = 8,
  // Signed 16-bit integer.
  LOOM_TESTBENCH_NUMERIC_I16 = 9,
  // Unsigned 16-bit integer.
  LOOM_TESTBENCH_NUMERIC_U16 = 10,
  // Signed 32-bit integer.
  LOOM_TESTBENCH_NUMERIC_I32 = 11,
  // Unsigned 32-bit integer.
  LOOM_TESTBENCH_NUMERIC_U32 = 12,
  // Signed 64-bit integer.
  LOOM_TESTBENCH_NUMERIC_I64 = 13,
  // Unsigned 64-bit integer.
  LOOM_TESTBENCH_NUMERIC_U64 = 14,
  LOOM_TESTBENCH_NUMERIC_COUNT_,
} loom_testbench_numeric_kind_t;

typedef enum loom_testbench_numeric_domain_e {
  // Binary floating-point values.
  LOOM_TESTBENCH_NUMERIC_DOMAIN_FLOAT = 0,
  // Fixed-width integer values.
  LOOM_TESTBENCH_NUMERIC_DOMAIN_INTEGER = 1,
} loom_testbench_numeric_domain_t;

// Loads one dense element exactly into binary64. Primitive floats and supported
// integers are all decoded according to their declared storage interpretation.
typedef double (*loom_testbench_numeric_load_f64_fn_t)(const uint8_t* data);

// Loads one dense integer element exactly into a signed 64-bit carrier.
typedef int64_t (*loom_testbench_numeric_load_i64_fn_t)(const uint8_t* data);

// Loads one dense integer element exactly into an unsigned 64-bit carrier.
typedef uint64_t (*loom_testbench_numeric_load_u64_fn_t)(const uint8_t* data);

typedef struct loom_testbench_numeric_type_t {
  // Stable interpretation identity.
  loom_testbench_numeric_kind_t kind;
  // Text spelling used by reference-oracle contracts and diagnostics.
  const char* name;
  // Canonical HAL storage element type.
  iree_hal_element_type_t hal_element_type;
  // Dense storage bytes occupied by one element.
  uint8_t byte_count;
  // Numerical relation family.
  loom_testbench_numeric_domain_t domain;
  // Fixed decoder into binary64.
  loom_testbench_numeric_load_f64_fn_t load_f64;
  // Fixed exact integer decoder, or NULL for floating-point types.
  loom_testbench_numeric_load_i64_fn_t load_i64;
  // Fixed exact unsigned integer decoder, or NULL when not applicable.
  loom_testbench_numeric_load_u64_fn_t load_u64;
  // Smallest signed i64 accepted by integer reference generation.
  int64_t minimum_i64;
  // Largest signed i64 accepted by integer reference generation.
  int64_t maximum_i64;
  // Smallest positive normal value for floating-point types, or zero otherwise.
  double minimum_normal_f64;
} loom_testbench_numeric_type_t;

// Returns the descriptor for |kind|, or NULL for an invalid/unset kind.
const loom_testbench_numeric_type_t* loom_testbench_numeric_type(
    loom_testbench_numeric_kind_t kind);

// Resolves a reference-oracle contract spelling to a numeric descriptor.
const loom_testbench_numeric_type_t* loom_testbench_numeric_type_parse(
    iree_string_view_t value);

// Resolves a supported HAL dense element type to its default interpretation.
const loom_testbench_numeric_type_t*
loom_testbench_numeric_type_from_hal_element_type(
    iree_hal_element_type_t element_type);

// Resolves a supported first-class Loom scalar type to its default
// interpretation. Integer scalar types use signed interpretation.
const loom_testbench_numeric_type_t*
loom_testbench_numeric_type_from_scalar_type(loom_scalar_type_t scalar_type);

// Resolves an i8/i16/i32/i64 scalar to its signed interpretation.
const loom_testbench_numeric_type_t*
loom_testbench_numeric_type_from_signed_scalar_type(
    loom_scalar_type_t scalar_type);

// Resolves an i8/i16/i32/i64 scalar to its unsigned interpretation.
const loom_testbench_numeric_type_t*
loom_testbench_numeric_type_from_unsigned_scalar_type(
    loom_scalar_type_t scalar_type);

// Returns true when |element_type| has the storage width and representation
// required by |numeric_type|. Signless HAL integers admit either interpretation
// of the same width.
bool loom_testbench_numeric_type_matches_storage(
    const loom_testbench_numeric_type_t* numeric_type,
    iree_hal_element_type_t element_type);

// Returns true when |numeric_type| has a floating-point interpretation.
static inline bool loom_testbench_numeric_type_is_float(
    const loom_testbench_numeric_type_t* numeric_type) {
  return numeric_type->domain == LOOM_TESTBENCH_NUMERIC_DOMAIN_FLOAT;
}

// Returns true when |numeric_type| has an integer interpretation.
static inline bool loom_testbench_numeric_type_is_integer(
    const loom_testbench_numeric_type_t* numeric_type) {
  return numeric_type->domain == LOOM_TESTBENCH_NUMERIC_DOMAIN_INTEGER;
}

// Loads element |element_offset| from dense storage into binary64.
static inline double loom_testbench_numeric_load_f64_at(
    const loom_testbench_numeric_type_t* numeric_type, const uint8_t* data,
    iree_host_size_t element_offset) {
  return numeric_type->load_f64(data +
                                element_offset * numeric_type->byte_count);
}

// Loads integer element |element_offset| from dense storage exactly.
static inline int64_t loom_testbench_numeric_load_i64_at(
    const loom_testbench_numeric_type_t* numeric_type, const uint8_t* data,
    iree_host_size_t element_offset) {
  return numeric_type->load_i64(data +
                                element_offset * numeric_type->byte_count);
}

// Loads unsigned integer element |element_offset| from dense storage exactly.
static inline uint64_t loom_testbench_numeric_load_u64_at(
    const loom_testbench_numeric_type_t* numeric_type, const uint8_t* data,
    iree_host_size_t element_offset) {
  return numeric_type->load_u64(data +
                                element_offset * numeric_type->byte_count);
}

// Returns true when |value| can feed integer reference generation.
static inline bool loom_testbench_numeric_type_contains_i64(
    const loom_testbench_numeric_type_t* numeric_type, int64_t value) {
  return loom_testbench_numeric_type_is_integer(numeric_type) &&
         value >= numeric_type->minimum_i64 &&
         value <= numeric_type->maximum_i64;
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_TESTBENCH_NUMERIC_VALUE_H_

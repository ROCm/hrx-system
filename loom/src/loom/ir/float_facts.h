// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Width-aware floating-point value fact construction and transfer.

#ifndef LOOM_IR_FLOAT_FACTS_H_
#define LOOM_IR_FLOAT_FACTS_H_

#include "loom/ir/facts.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef float (*loom_float_unary_f32_fn_t)(float input);
typedef double (*loom_float_unary_f64_fn_t)(double input);
typedef float (*loom_float_unary_data_f32_fn_t)(float input,
                                                const void* user_data);
typedef double (*loom_float_unary_data_f64_fn_t)(double input,
                                                 const void* user_data);
typedef float (*loom_float_binary_f32_fn_t)(float lhs, float rhs);
typedef double (*loom_float_binary_f64_fn_t)(double lhs, double rhs);
typedef float (*loom_float_ternary_f32_fn_t)(float a, float b, float c);
typedef double (*loom_float_ternary_f64_fn_t)(double a, double b, double c);

typedef enum loom_float_minmax_kind_e {
  LOOM_FLOAT_MINMAX_MINIMUM = 0,
  LOOM_FLOAT_MINMAX_MAXIMUM = 1,
  LOOM_FLOAT_MINMAX_MINNUM = 2,
  LOOM_FLOAT_MINMAX_MAXNUM = 3,
} loom_float_minmax_kind_t;

typedef enum loom_float_clamp_kind_e {
  LOOM_FLOAT_CLAMP_ORDERED = 0,
  LOOM_FLOAT_CLAMP_NUMBER = 1,
  LOOM_FLOAT_CLAMP_IEEE = 2,
} loom_float_clamp_kind_t;

typedef enum loom_float_turns_kind_e {
  LOOM_FLOAT_TURNS_SIN = 0,
  LOOM_FLOAT_TURNS_COS = 1,
} loom_float_turns_kind_t;

typedef enum loom_float_integer_conversion_kind_e {
  // Signed destination interpretation used by scalar.fptosi.
  LOOM_FLOAT_INTEGER_CONVERSION_SIGNED = 0,
  // Unsigned destination interpretation used by scalar.fptoui.
  LOOM_FLOAT_INTEGER_CONVERSION_UNSIGNED = 1,
} loom_float_integer_conversion_kind_t;

// Returns exact floating-point facts after rounding |value| to |scalar_type|.
// The compact fact payload stores the rounded value as a host double; callers
// must continue to provide the declared scalar type when interpreting it.
loom_value_facts_t loom_value_facts_exact_float(loom_scalar_type_t scalar_type,
                                                double value);

// Returns facts for the raw bit pattern of |scalar_type|, ignoring bits above
// its declared width. Non-NaN values are represented exactly. NaN payloads
// produce known-NaN facts because the compact fact representation intentionally
// does not retain NaN payload bits.
bool loom_value_facts_from_float_bits(loom_scalar_type_t scalar_type,
                                      uint64_t bits,
                                      loom_value_facts_t* out_facts);

// Returns a non-materializable fact proving that a floating-point value is NaN.
// This represents raw NaN payloads that cannot be retained in the compact fact
// payload, such as signaling NaNs discovered through integer bitcasts.
loom_value_facts_t loom_value_facts_known_nan(void);

// Extracts an exact floating-point value interpreted as |scalar_type|.
bool loom_value_facts_as_exact_float(loom_scalar_type_t scalar_type,
                                     loom_value_facts_t facts,
                                     double* out_value);

// Evaluates logistic using selected-width arithmetic and a sign-split formula
// that preserves representable negative tails. Exact logistic, SiLU, and
// logistic GELU fact transfers share these helpers so their semantics agree.
float loom_float_logistic_f32(float input);
double loom_float_logistic_f64(double input);

// Applies a width-specific operation and rounds once to |scalar_type|.
void loom_value_facts_eval_float_unary(loom_scalar_type_t scalar_type,
                                       const loom_value_facts_t* input,
                                       loom_float_unary_f32_fn_t f32_fn,
                                       loom_float_unary_f64_fn_t f64_fn,
                                       loom_value_facts_t* out_facts);
void loom_value_facts_eval_float_unary_data(
    loom_scalar_type_t scalar_type, const loom_value_facts_t* input,
    loom_float_unary_data_f32_fn_t f32_fn,
    loom_float_unary_data_f64_fn_t f64_fn, const void* user_data,
    loom_value_facts_t* out_facts);
void loom_value_facts_eval_float_binary(loom_scalar_type_t scalar_type,
                                        const loom_value_facts_t* lhs,
                                        const loom_value_facts_t* rhs,
                                        loom_float_binary_f32_fn_t f32_fn,
                                        loom_float_binary_f64_fn_t f64_fn,
                                        loom_value_facts_t* out_facts);
void loom_value_facts_eval_float_ternary(loom_scalar_type_t scalar_type,
                                         const loom_value_facts_t* a,
                                         const loom_value_facts_t* b,
                                         const loom_value_facts_t* c,
                                         loom_float_ternary_f32_fn_t f32_fn,
                                         loom_float_ternary_f64_fn_t f64_fn,
                                         loom_value_facts_t* out_facts);

// Evaluates sine or cosine over turns with exact quarter-turn range reduction.
// Finite inputs preserve periodicity and produce exact cardinal values with
// OpenCL-compatible signed zeros. NaN and infinity produce arithmetic NaN.
void loom_value_facts_eval_float_turns(loom_scalar_type_t scalar_type,
                                       loom_float_turns_kind_t kind,
                                       const loom_value_facts_t* input,
                                       loom_value_facts_t* out_facts);

// Applies the exact payload-preserving sign transforms. NaN results remain
// non-materializable because the compact fact payload cannot retain their bits.
void loom_value_facts_eval_float_negate(loom_scalar_type_t scalar_type,
                                        const loom_value_facts_t* input,
                                        loom_value_facts_t* out_facts);
void loom_value_facts_eval_float_abs(loom_scalar_type_t scalar_type,
                                     const loom_value_facts_t* input,
                                     loom_value_facts_t* out_facts);
void loom_value_facts_eval_float_copysign(loom_scalar_type_t scalar_type,
                                          const loom_value_facts_t* magnitude,
                                          const loom_value_facts_t* sign,
                                          loom_value_facts_t* out_facts);

// Applies structural IEEE minimum/maximum semantics without relying on host
// fmin/fmax behavior for signaling NaNs or opposite signed zeros.
void loom_value_facts_eval_float_minmax(loom_scalar_type_t scalar_type,
                                        loom_float_minmax_kind_t kind,
                                        const loom_value_facts_t* lhs,
                                        const loom_value_facts_t* rhs,
                                        loom_value_facts_t* out_facts);

// Applies one of Loom's three explicit clamp compositions.
void loom_value_facts_eval_float_clamp(loom_scalar_type_t scalar_type,
                                       loom_float_clamp_kind_t kind,
                                       const loom_value_facts_t* value,
                                       const loom_value_facts_t* lower,
                                       const loom_value_facts_t* upper,
                                       loom_value_facts_t* out_facts);

// Truncates an exact floating-point value toward zero and, when representable
// under |kind|, returns its canonical |result_type| fact representation. I1
// results use 0/1; wider integers use their sign-extended bit pattern. NaN,
// infinity, and out-of-range values produce unknown facts without executing an
// undefined host floating-point-to-integer conversion.
void loom_value_facts_eval_float_to_integer(
    loom_scalar_type_t source_type, loom_scalar_type_t result_type,
    loom_float_integer_conversion_kind_t kind,
    const loom_value_facts_t* source_facts,
    loom_value_facts_t* out_result_facts);

// Reinterprets an exact scalar fact through equal-width scalar types. Raw NaN
// bit patterns become known-NaN facts instead of materializable constants so a
// fold cannot erase their payload.
void loom_value_facts_eval_scalar_bitcast(
    loom_scalar_type_t source_type, loom_scalar_type_t result_type,
    const loom_value_facts_t* source_facts,
    loom_value_facts_t* out_result_facts);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_IR_FLOAT_FACTS_H_

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
  // Interpret the integer side of a conversion as signed.
  LOOM_FLOAT_INTEGER_CONVERSION_SIGNED = 0,
  // Interpret the integer side of a conversion as unsigned raw bits.
  LOOM_FLOAT_INTEGER_CONVERSION_UNSIGNED = 1,
} loom_float_integer_conversion_kind_t;

// Subnormal-handling permissions carried by a floating-point conversion.
typedef struct loom_float_conversion_policy_t {
  // Whether a subnormal input may be replaced by its corresponding signed zero.
  bool may_flush_input_subnormal;
  // Whether a subnormal rounded result may be replaced by its signed zero.
  bool may_flush_result_subnormal;
} loom_float_conversion_policy_t;

// Returns exact floating-point facts after rounding |value| to |scalar_type|.
// The compact fact payload stores the rounded value as a host double; callers
// must continue to provide the declared scalar type when interpreting it.
loom_value_facts_t loom_value_facts_exact_float(loom_scalar_type_t scalar_type,
                                                double value);

// Returns facts bounding every result obtained by rounding a finite interval
// to |scalar_type|. Finite rounded endpoints are retained in the compact fact
// payload. A destination overflow keeps the guaranteed non-NaN class but drops
// the interval because its result may be infinite.
loom_value_facts_t loom_value_facts_make_float_range(
    loom_scalar_type_t scalar_type, double lo, double hi);

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

// Adds authored non-NaN/non-infinite result promises to floating-point facts.
// The two promises are independent; together they imply FINITE but say nothing
// about exact destination formats, signed zeros, or subnormals. Existing
// numeric payloads and execution distribution remain unchanged. Known
// nonfinite answers retain their ordinary classification even if a promise
// is violated, so the fact lattice never contains conflicting class bits.
void loom_value_facts_assume_float_class(loom_value_facts_t* facts,
                                         bool not_nan, bool not_inf);

// Evaluates a floating-width conversion under |policy|. Each policy field adds
// the flushed value to the permitted result set; it does not require flushing.
// All other rounding and special-value behavior remains exact.
void loom_value_facts_eval_float_conversion(
    loom_scalar_type_t source_type, loom_scalar_type_t result_type,
    loom_float_conversion_policy_t policy,
    const loom_value_facts_t* input_facts,
    loom_value_facts_t* out_result_facts);

// Extracts an exact floating-point value interpreted as |scalar_type|.
bool loom_value_facts_as_exact_float(loom_scalar_type_t scalar_type,
                                     loom_value_facts_t facts,
                                     double* out_value);

// Extracts the mathematical base-two exponent of an exact finite nonzero
// floating-point value whose magnitude is a power of two. Both positive and
// negative values are accepted; the sign does not affect |out_exponent|.
bool loom_value_facts_as_exact_power_of_two_float(
    loom_scalar_type_t scalar_type, loom_value_facts_t facts,
    int32_t* out_exponent);

// Extracts inclusive finite interval endpoints interpreted as |scalar_type|.
// Returns false for class-only facts and non-finite exact values.
bool loom_value_facts_as_float_range(loom_scalar_type_t scalar_type,
                                     loom_value_facts_t facts, double* out_lo,
                                     double* out_hi);

// Returns true when a retained finite interval lies wholly within
// [minimum_value, maximum_value].
bool loom_value_facts_float_range_within(loom_scalar_type_t scalar_type,
                                         loom_value_facts_t facts,
                                         double minimum_value,
                                         double maximum_value);

// Drops a retained finite interval while preserving float-class and execution
// distribution facts. Used by widening when bounds continue to grow.
void loom_value_facts_drop_float_range(loom_value_facts_t* facts);

// Returns the exact floating-point value encoded in its declared scalar
// format. NaN facts without a retained payload are not exact bit patterns.
bool loom_value_facts_as_exact_float_bits(loom_scalar_type_t scalar_type,
                                          loom_value_facts_t facts,
                                          uint64_t* out_bits);

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

// Evaluates a fused multiply-add with one rounding to |scalar_type|. Narrow
// results retain the contribution of addends lost in F32/F64 intermediates.
// Exact operand payloads have already been rounded to their declared type.
// Requires non-trapping round-to-nearest-even host arithmetic with subnormals
// preserved. The pass interpreter establishes this environment for compiler
// evaluation, including the host F32/F64 arithmetic evaluators.
void loom_value_facts_eval_float_fma(loom_scalar_type_t scalar_type,
                                     const loom_value_facts_t* a,
                                     const loom_value_facts_t* b,
                                     const loom_value_facts_t* c,
                                     loom_value_facts_t* out_facts);

// Evaluates sine or cosine over turns with exact quarter-turn range reduction.
// Finite inputs preserve periodicity and produce exact cardinal values with
// OpenCL-compatible signed zeros. NaN and infinity produce arithmetic NaN. The
// caller must establish non-trapping round-to-nearest-even arithmetic with
// input and output subnormals preserved.
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

// Converts an integer fact interval to a floating-point interval. The source
// interval is first constrained to its declared bit width. Unsigned conversion
// interprets negative integer facts as raw two's-complement payloads and falls
// back to the full unsigned domain only when an interval crosses zero.
void loom_value_facts_eval_integer_to_float(
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

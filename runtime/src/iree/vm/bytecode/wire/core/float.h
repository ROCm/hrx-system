// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// GENERATED FILE: DO NOT EDIT.
// Regenerate with:
//   iree-bazel-run //runtime/src/iree/vm/bytecode/spec:generate
// Natural-layout instruction records for core.family.float.
// Multi-byte fields are little-endian and naturally aligned.
// clang-format off
#ifndef IREE_VM_BYTECODE_GENERATED_WIRE_CORE_FLOAT_H_
#define IREE_VM_BYTECODE_GENERATED_WIRE_CORE_FLOAT_H_

#include <stdint.h>

// Selects IEEE minimum/maximum or number-selecting minnum/maxnum. Numeric
// ordering governs ordinary values; minima choose -0 and maxima choose +0 from
// opposite signed zeros.
typedef uint8_t iree_vm_isa_float_minmax_t;
enum {
  // Returns the numeric minimum and an arithmetic NaN if either operand is NaN.
  IREE_VM_ISA_FLOAT_MINMAX_MINIMUM = 0x00,
  // Returns the numeric maximum and an arithmetic NaN if either operand is NaN.
  IREE_VM_ISA_FLOAT_MINMAX_MAXIMUM = 0x01,
  // Returns the sole numeric operand bit-for-bit, or an arithmetic NaN when
  // both are NaN.
  IREE_VM_ISA_FLOAT_MINMAX_MINNUM = 0x02,
  // Returns the sole numeric operand bit-for-bit, or an arithmetic NaN when
  // both are NaN.
  IREE_VM_ISA_FLOAT_MINMAX_MAXNUM = 0x03,
};

enum {
  IREE_VM_ISA_FLOAT_MINMAX_MINIMUM_SINCE_MINOR = 0,
  IREE_VM_ISA_FLOAT_MINMAX_MAXIMUM_SINCE_MINOR = 0,
  IREE_VM_ISA_FLOAT_MINMAX_MINNUM_SINCE_MINOR = 0,
  IREE_VM_ISA_FLOAT_MINMAX_MAXNUM_SINCE_MINOR = 0,
};

// Selects one unary operation under the floating family profile. Width-matched
// host math uses the named f32/f64 function; staged operations round every
// named assignment at the selected width.
typedef uint8_t iree_vm_isa_float_math_unary_t;
enum {
  // Returns the width-matched exponential exp(x).
  IREE_VM_ISA_FLOAT_MATH_UNARY_EXP = 0x00,
  // Returns the width-matched base-two exponential 2^x.
  IREE_VM_ISA_FLOAT_MATH_UNARY_EXP2 = 0x01,
  // Returns the width-matched exp(x)-1 operation.
  IREE_VM_ISA_FLOAT_MATH_UNARY_EXPM1 = 0x02,
  // Returns the width-matched natural logarithm.
  IREE_VM_ISA_FLOAT_MATH_UNARY_LOG = 0x03,
  // Returns the width-matched base-two logarithm.
  IREE_VM_ISA_FLOAT_MATH_UNARY_LOG2 = 0x04,
  // Returns the width-matched base-ten logarithm.
  IREE_VM_ISA_FLOAT_MATH_UNARY_LOG10 = 0x05,
  // Returns the width-matched log(1+x) operation.
  IREE_VM_ISA_FLOAT_MATH_UNARY_LOG1P = 0x06,
  // Returns correctly rounded selected-width square root.
  IREE_VM_ISA_FLOAT_MATH_UNARY_SQRT = 0x07,
  // Rounds sqrt(x), then rounds selected-width 1/root without contraction.
  IREE_VM_ISA_FLOAT_MATH_UNARY_RSQRT = 0x08,
  // Returns the width-matched cube root.
  IREE_VM_ISA_FLOAT_MATH_UNARY_CBRT = 0x09,
  // Returns width-matched sine with x in radians.
  IREE_VM_ISA_FLOAT_MATH_UNARY_SIN = 0x0A,
  // Returns width-matched cosine with x in radians.
  IREE_VM_ISA_FLOAT_MATH_UNARY_COS = 0x0B,
  // Returns sin(2*pi*x) using exact quarter-turn reduction and structural
  // cardinal results.
  IREE_VM_ISA_FLOAT_MATH_UNARY_SINTURNS = 0x0C,
  // Returns cos(2*pi*x) using exact quarter-turn reduction and structural
  // cardinal results.
  IREE_VM_ISA_FLOAT_MATH_UNARY_COSTURNS = 0x0D,
  // Returns width-matched tangent with x in radians.
  IREE_VM_ISA_FLOAT_MATH_UNARY_TAN = 0x0E,
  // Returns the width-matched inverse sine.
  IREE_VM_ISA_FLOAT_MATH_UNARY_ASIN = 0x0F,
  // Returns the width-matched inverse cosine.
  IREE_VM_ISA_FLOAT_MATH_UNARY_ACOS = 0x10,
  // Returns the width-matched inverse tangent.
  IREE_VM_ISA_FLOAT_MATH_UNARY_ATAN = 0x11,
  // Returns the width-matched hyperbolic sine.
  IREE_VM_ISA_FLOAT_MATH_UNARY_SINH = 0x12,
  // Returns the width-matched hyperbolic cosine.
  IREE_VM_ISA_FLOAT_MATH_UNARY_COSH = 0x13,
  // Returns the width-matched hyperbolic tangent.
  IREE_VM_ISA_FLOAT_MATH_UNARY_TANH = 0x14,
  // Returns the width-matched inverse hyperbolic sine.
  IREE_VM_ISA_FLOAT_MATH_UNARY_ASINH = 0x15,
  // Returns the width-matched inverse hyperbolic cosine.
  IREE_VM_ISA_FLOAT_MATH_UNARY_ACOSH = 0x16,
  // Returns the width-matched inverse hyperbolic tangent.
  IREE_VM_ISA_FLOAT_MATH_UNARY_ATANH = 0x17,
  // Returns the width-matched error function.
  IREE_VM_ISA_FLOAT_MATH_UNARY_ERF = 0x18,
  // Returns the width-matched complementary error function.
  IREE_VM_ISA_FLOAT_MATH_UNARY_ERFC = 0x19,
  // Uses the stable sign split: 1/(1+exp(-x)) for x>=0 and exp(x)/(1+exp(x))
  // otherwise.
  IREE_VM_ISA_FLOAT_MATH_UNARY_LOGISTIC = 0x1A,
  // Rounds logistic(x), then rounds x*logistic(x).
  IREE_VM_ISA_FLOAT_MATH_UNARY_SILU = 0x1B,
  // Computes ordered_max(x,+0)+log1p(exp(-abs(x))) in selected-width stages.
  IREE_VM_ISA_FLOAT_MATH_UNARY_SOFTPLUS = 0x1C,
  // Returns the integral value toward positive infinity.
  IREE_VM_ISA_FLOAT_MATH_UNARY_CEIL = 0x1D,
  // Returns the integral value toward negative infinity.
  IREE_VM_ISA_FLOAT_MATH_UNARY_FLOOR = 0x1E,
  // Returns the nearest integral value with halfway cases away from zero.
  IREE_VM_ISA_FLOAT_MATH_UNARY_ROUND = 0x1F,
  // Returns the nearest integral value with halfway cases to even.
  IREE_VM_ISA_FLOAT_MATH_UNARY_ROUNDEVEN = 0x20,
  // Returns the integral value toward zero.
  IREE_VM_ISA_FLOAT_MATH_UNARY_TRUNC = 0x21,
  // Returns +0 for NaNs and either zero, -1 for negative nonzero values, and +1
  // otherwise.
  IREE_VM_ISA_FLOAT_MATH_UNARY_SIGN = 0x22,
  // Computes (0.5*x)*(1+erf(x*inverse_sqrt2)) in selected-width stages.
  IREE_VM_ISA_FLOAT_MATH_UNARY_GELU_ERF = 0x23,
  // Computes (0.5*x)*(1+tanh(sqrt_2_over_pi*(x+cubic_coefficient*x^3))) in
  // selected-width stages.
  IREE_VM_ISA_FLOAT_MATH_UNARY_GELU_TANH = 0x24,
};

enum {
  IREE_VM_ISA_FLOAT_MATH_UNARY_EXP_SINCE_MINOR = 0,
  IREE_VM_ISA_FLOAT_MATH_UNARY_EXP2_SINCE_MINOR = 0,
  IREE_VM_ISA_FLOAT_MATH_UNARY_EXPM1_SINCE_MINOR = 0,
  IREE_VM_ISA_FLOAT_MATH_UNARY_LOG_SINCE_MINOR = 0,
  IREE_VM_ISA_FLOAT_MATH_UNARY_LOG2_SINCE_MINOR = 0,
  IREE_VM_ISA_FLOAT_MATH_UNARY_LOG10_SINCE_MINOR = 0,
  IREE_VM_ISA_FLOAT_MATH_UNARY_LOG1P_SINCE_MINOR = 0,
  IREE_VM_ISA_FLOAT_MATH_UNARY_SQRT_SINCE_MINOR = 0,
  IREE_VM_ISA_FLOAT_MATH_UNARY_RSQRT_SINCE_MINOR = 0,
  IREE_VM_ISA_FLOAT_MATH_UNARY_CBRT_SINCE_MINOR = 0,
  IREE_VM_ISA_FLOAT_MATH_UNARY_SIN_SINCE_MINOR = 0,
  IREE_VM_ISA_FLOAT_MATH_UNARY_COS_SINCE_MINOR = 0,
  IREE_VM_ISA_FLOAT_MATH_UNARY_SINTURNS_SINCE_MINOR = 0,
  IREE_VM_ISA_FLOAT_MATH_UNARY_COSTURNS_SINCE_MINOR = 0,
  IREE_VM_ISA_FLOAT_MATH_UNARY_TAN_SINCE_MINOR = 0,
  IREE_VM_ISA_FLOAT_MATH_UNARY_ASIN_SINCE_MINOR = 0,
  IREE_VM_ISA_FLOAT_MATH_UNARY_ACOS_SINCE_MINOR = 0,
  IREE_VM_ISA_FLOAT_MATH_UNARY_ATAN_SINCE_MINOR = 0,
  IREE_VM_ISA_FLOAT_MATH_UNARY_SINH_SINCE_MINOR = 0,
  IREE_VM_ISA_FLOAT_MATH_UNARY_COSH_SINCE_MINOR = 0,
  IREE_VM_ISA_FLOAT_MATH_UNARY_TANH_SINCE_MINOR = 0,
  IREE_VM_ISA_FLOAT_MATH_UNARY_ASINH_SINCE_MINOR = 0,
  IREE_VM_ISA_FLOAT_MATH_UNARY_ACOSH_SINCE_MINOR = 0,
  IREE_VM_ISA_FLOAT_MATH_UNARY_ATANH_SINCE_MINOR = 0,
  IREE_VM_ISA_FLOAT_MATH_UNARY_ERF_SINCE_MINOR = 0,
  IREE_VM_ISA_FLOAT_MATH_UNARY_ERFC_SINCE_MINOR = 0,
  IREE_VM_ISA_FLOAT_MATH_UNARY_LOGISTIC_SINCE_MINOR = 0,
  IREE_VM_ISA_FLOAT_MATH_UNARY_SILU_SINCE_MINOR = 0,
  IREE_VM_ISA_FLOAT_MATH_UNARY_SOFTPLUS_SINCE_MINOR = 0,
  IREE_VM_ISA_FLOAT_MATH_UNARY_CEIL_SINCE_MINOR = 0,
  IREE_VM_ISA_FLOAT_MATH_UNARY_FLOOR_SINCE_MINOR = 0,
  IREE_VM_ISA_FLOAT_MATH_UNARY_ROUND_SINCE_MINOR = 0,
  IREE_VM_ISA_FLOAT_MATH_UNARY_ROUNDEVEN_SINCE_MINOR = 0,
  IREE_VM_ISA_FLOAT_MATH_UNARY_TRUNC_SINCE_MINOR = 0,
  IREE_VM_ISA_FLOAT_MATH_UNARY_SIGN_SINCE_MINOR = 0,
  IREE_VM_ISA_FLOAT_MATH_UNARY_GELU_ERF_SINCE_MINOR = 0,
  IREE_VM_ISA_FLOAT_MATH_UNARY_GELU_TANH_SINCE_MINOR = 0,
};

// Selects one binary operation under the floating family profile; staged
// operations round every named assignment at selected width.
typedef uint8_t iree_vm_isa_float_math_binary_t;
enum {
  // Returns the width-matched pow(lhs, rhs) operation.
  IREE_VM_ISA_FLOAT_MATH_BINARY_POW = 0x00,
  // Returns the width-matched atan2(lhs, rhs) operation.
  IREE_VM_ISA_FLOAT_MATH_BINARY_ATAN2 = 0x01,
  // Treats lhs as x and rhs as scale, then computes x*logistic(scale*x) in
  // three stages.
  IREE_VM_ISA_FLOAT_MATH_BINARY_GELU_LOGISTIC = 0x02,
};

enum {
  IREE_VM_ISA_FLOAT_MATH_BINARY_POW_SINCE_MINOR = 0,
  IREE_VM_ISA_FLOAT_MATH_BINARY_ATAN2_SINCE_MINOR = 0,
  IREE_VM_ISA_FLOAT_MATH_BINARY_GELU_LOGISTIC_SINCE_MINOR = 0,
};

// Selects one ternary operation under the floating family profile.
typedef uint8_t iree_vm_isa_float_math_ternary_t;
enum {
  // Computes infinitely precise a*b+c and rounds once to selected width.
  IREE_VM_ISA_FLOAT_MATH_TERNARY_FMA = 0x00,
};

enum {
  IREE_VM_ISA_FLOAT_MATH_TERNARY_FMA_SINCE_MINOR = 0,
};

// Selects an ordered or unordered IEEE predicate. Ordered predicates require
// both operands to be non-NaN; unordered predicates are true when either
// operand is NaN. Signed zeros compare equal.
typedef uint8_t iree_vm_isa_float_compare_t;
enum {
  // True when neither operand is NaN and lhs equals rhs.
  IREE_VM_ISA_FLOAT_COMPARE_OEQ = 0x00,
  // True when neither operand is NaN and lhs is greater than rhs.
  IREE_VM_ISA_FLOAT_COMPARE_OGT = 0x01,
  // True when neither operand is NaN and lhs is at least rhs.
  IREE_VM_ISA_FLOAT_COMPARE_OGE = 0x02,
  // True when neither operand is NaN and lhs is less than rhs.
  IREE_VM_ISA_FLOAT_COMPARE_OLT = 0x03,
  // True when neither operand is NaN and lhs is at most rhs.
  IREE_VM_ISA_FLOAT_COMPARE_OLE = 0x04,
  // True when neither operand is NaN and lhs differs from rhs.
  IREE_VM_ISA_FLOAT_COMPARE_ONE = 0x05,
  // True when neither operand is NaN.
  IREE_VM_ISA_FLOAT_COMPARE_ORD = 0x06,
  // True when either operand is NaN or lhs equals rhs.
  IREE_VM_ISA_FLOAT_COMPARE_UEQ = 0x07,
  // True when either operand is NaN or lhs is greater than rhs.
  IREE_VM_ISA_FLOAT_COMPARE_UGT = 0x08,
  // True when either operand is NaN or lhs is at least rhs.
  IREE_VM_ISA_FLOAT_COMPARE_UGE = 0x09,
  // True when either operand is NaN or lhs is less than rhs.
  IREE_VM_ISA_FLOAT_COMPARE_ULT = 0x0A,
  // True when either operand is NaN or lhs is at most rhs.
  IREE_VM_ISA_FLOAT_COMPARE_ULE = 0x0B,
  // True when either operand is NaN or lhs differs from rhs.
  IREE_VM_ISA_FLOAT_COMPARE_UNE = 0x0C,
  // True when either operand is NaN.
  IREE_VM_ISA_FLOAT_COMPARE_UNO = 0x0D,
};

enum {
  IREE_VM_ISA_FLOAT_COMPARE_OEQ_SINCE_MINOR = 0,
  IREE_VM_ISA_FLOAT_COMPARE_OGT_SINCE_MINOR = 0,
  IREE_VM_ISA_FLOAT_COMPARE_OGE_SINCE_MINOR = 0,
  IREE_VM_ISA_FLOAT_COMPARE_OLT_SINCE_MINOR = 0,
  IREE_VM_ISA_FLOAT_COMPARE_OLE_SINCE_MINOR = 0,
  IREE_VM_ISA_FLOAT_COMPARE_ONE_SINCE_MINOR = 0,
  IREE_VM_ISA_FLOAT_COMPARE_ORD_SINCE_MINOR = 0,
  IREE_VM_ISA_FLOAT_COMPARE_UEQ_SINCE_MINOR = 0,
  IREE_VM_ISA_FLOAT_COMPARE_UGT_SINCE_MINOR = 0,
  IREE_VM_ISA_FLOAT_COMPARE_UGE_SINCE_MINOR = 0,
  IREE_VM_ISA_FLOAT_COMPARE_ULT_SINCE_MINOR = 0,
  IREE_VM_ISA_FLOAT_COMPARE_ULE_SINCE_MINOR = 0,
  IREE_VM_ISA_FLOAT_COMPARE_UNE_SINCE_MINOR = 0,
  IREE_VM_ISA_FLOAT_COMPARE_UNO_SINCE_MINOR = 0,
};

// Selects a raw exponent/significand classification without floating
// arithmetic.
typedef uint8_t iree_vm_isa_float_classify_t;
enum {
  // True for quiet or signaling NaN payloads.
  IREE_VM_ISA_FLOAT_CLASSIFY_ISNAN = 0x00,
  // True for either signed infinity and false for NaNs.
  IREE_VM_ISA_FLOAT_CLASSIFY_ISINF = 0x01,
  // True for zero, subnormal, and normal finite payloads.
  IREE_VM_ISA_FLOAT_CLASSIFY_ISFINITE = 0x02,
};

enum {
  IREE_VM_ISA_FLOAT_CLASSIFY_ISNAN_SINCE_MINOR = 0,
  IREE_VM_ISA_FLOAT_CLASSIFY_ISINF_SINCE_MINOR = 0,
  IREE_VM_ISA_FLOAT_CLASSIFY_ISFINITE_SINCE_MINOR = 0,
};

// Selects one exact selected-width clamp composition. All modes are defined
// when lower exceeds upper.
typedef uint8_t iree_vm_isa_float_clamp_t;
enum {
  // Starts with value, selects lower when result<lower, then upper when
  // result>upper; NaN comparisons are false.
  IREE_VM_ISA_FLOAT_CLAMP_ORDERED = 0x00,
  // Computes minnum(maxnum(value, lower), upper).
  IREE_VM_ISA_FLOAT_CLAMP_NUMBER = 0x01,
  // Computes minimum(maximum(value, lower), upper).
  IREE_VM_ISA_FLOAT_CLAMP_IEEE = 0x02,
};

enum {
  IREE_VM_ISA_FLOAT_CLAMP_ORDERED_SINCE_MINOR = 0,
  IREE_VM_ISA_FLOAT_CLAMP_NUMBER_SINCE_MINOR = 0,
  IREE_VM_ISA_FLOAT_CLAMP_IEEE_SINCE_MINOR = 0,
};

// Page 0x00, opcode 0x80: Adds two f32 values with one selected-width rounding.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Result value-register ordinal.
  uint8_t dst_v8;
  // Operand value-register ordinal.
  uint8_t lhs_v8;
  // Operand value-register ordinal.
  uint8_t rhs_v8;
} iree_vm_isa_float_add_f32_record_t;

// Page 0x00, opcode 0x81: Adds two f64 values with one selected-width rounding.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Result value-register ordinal.
  uint8_t dst_v8;
  // Operand value-register ordinal.
  uint8_t lhs_v8;
  // Operand value-register ordinal.
  uint8_t rhs_v8;
} iree_vm_isa_float_add_f64_record_t;

// Page 0x00, opcode 0x82: Subtracts two f32 values with one selected-width
// rounding.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Result value-register ordinal.
  uint8_t dst_v8;
  // Operand value-register ordinal.
  uint8_t lhs_v8;
  // Operand value-register ordinal.
  uint8_t rhs_v8;
} iree_vm_isa_float_sub_f32_record_t;

// Page 0x00, opcode 0x83: Subtracts two f64 values with one selected-width
// rounding.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Result value-register ordinal.
  uint8_t dst_v8;
  // Operand value-register ordinal.
  uint8_t lhs_v8;
  // Operand value-register ordinal.
  uint8_t rhs_v8;
} iree_vm_isa_float_sub_f64_record_t;

// Page 0x00, opcode 0x84: Multiplies two f32 values with one selected-width
// rounding.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Result value-register ordinal.
  uint8_t dst_v8;
  // Operand value-register ordinal.
  uint8_t lhs_v8;
  // Operand value-register ordinal.
  uint8_t rhs_v8;
} iree_vm_isa_float_mul_f32_record_t;

// Page 0x00, opcode 0x85: Multiplies two f64 values with one selected-width
// rounding.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Result value-register ordinal.
  uint8_t dst_v8;
  // Operand value-register ordinal.
  uint8_t lhs_v8;
  // Operand value-register ordinal.
  uint8_t rhs_v8;
} iree_vm_isa_float_mul_f64_record_t;

// Page 0x00, opcode 0x86: Divides two f32 values with IEEE non-stop semantics.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Result value-register ordinal.
  uint8_t dst_v8;
  // Operand value-register ordinal.
  uint8_t lhs_v8;
  // Operand value-register ordinal.
  uint8_t rhs_v8;
} iree_vm_isa_float_div_f32_record_t;

// Page 0x00, opcode 0x87: Divides two f64 values with IEEE non-stop semantics.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Result value-register ordinal.
  uint8_t dst_v8;
  // Operand value-register ordinal.
  uint8_t lhs_v8;
  // Operand value-register ordinal.
  uint8_t rhs_v8;
} iree_vm_isa_float_div_f64_record_t;

// Page 0x00, opcode 0x88: Computes width-matched f32 fmod: x-trunc(x/y)*y, with
// the dividend's sign including zero.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Result value-register ordinal.
  uint8_t dst_v8;
  // Operand value-register ordinal.
  uint8_t lhs_v8;
  // Operand value-register ordinal.
  uint8_t rhs_v8;
} iree_vm_isa_float_rem_f32_record_t;

// Page 0x00, opcode 0x89: Computes width-matched f64 fmod: x-trunc(x/y)*y, with
// the dividend's sign including zero.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Result value-register ordinal.
  uint8_t dst_v8;
  // Operand value-register ordinal.
  uint8_t lhs_v8;
  // Operand value-register ordinal.
  uint8_t rhs_v8;
} iree_vm_isa_float_rem_f64_record_t;

// Page 0x00, opcode 0x8A: Toggles the raw f32 sign bit.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Result value-register ordinal.
  uint8_t dst_v8;
  // Operand value-register ordinal.
  uint8_t src_v8;
  // Canonical zero padding.
  uint8_t zero_padding_u8;
} iree_vm_isa_float_neg_f32_record_t;

// Page 0x00, opcode 0x8B: Toggles the raw f64 sign bit.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Result value-register ordinal.
  uint8_t dst_v8;
  // Operand value-register ordinal.
  uint8_t src_v8;
  // Canonical zero padding.
  uint8_t zero_padding_u8;
} iree_vm_isa_float_neg_f64_record_t;

// Page 0x00, opcode 0x8C: Clears the raw f32 sign bit.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Result value-register ordinal.
  uint8_t dst_v8;
  // Operand value-register ordinal.
  uint8_t src_v8;
  // Canonical zero padding.
  uint8_t zero_padding_u8;
} iree_vm_isa_float_abs_f32_record_t;

// Page 0x00, opcode 0x8D: Clears the raw f64 sign bit.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Result value-register ordinal.
  uint8_t dst_v8;
  // Operand value-register ordinal.
  uint8_t src_v8;
  // Canonical zero padding.
  uint8_t zero_padding_u8;
} iree_vm_isa_float_abs_f64_record_t;

// Page 0x00, opcode 0x8E: Evaluates f32 IEEE minimum/maximum or
// number-selecting minnum/maxnum with explicit NaN and signed-zero rules.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Result value-register ordinal.
  uint8_t dst_v8;
  // Operand value-register ordinal.
  uint8_t lhs_v8;
  // Operand value-register ordinal.
  uint8_t rhs_v8;
  // Closed float.minmax operation selector.
  uint8_t selector_u8;
  // Canonical zero padding.
  uint8_t zero_padding_u8[3];
} iree_vm_isa_float_minmax_f32_record_t;

// Page 0x00, opcode 0x8F: Evaluates f64 IEEE minimum/maximum or
// number-selecting minnum/maxnum with explicit NaN and signed-zero rules.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Result value-register ordinal.
  uint8_t dst_v8;
  // Operand value-register ordinal.
  uint8_t lhs_v8;
  // Operand value-register ordinal.
  uint8_t rhs_v8;
  // Closed float.minmax operation selector.
  uint8_t selector_u8;
  // Canonical zero padding.
  uint8_t zero_padding_u8[3];
} iree_vm_isa_float_minmax_f64_record_t;

// Page 0x00, opcode 0x90: Evaluates one ordered or unordered raw-payload f32
// predicate.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Result value-register ordinal.
  uint8_t dst_v8;
  // Operand value-register ordinal.
  uint8_t lhs_v8;
  // Operand value-register ordinal.
  uint8_t rhs_v8;
  // Closed float.compare operation selector.
  uint8_t predicate_u8;
  // Canonical zero padding.
  uint8_t zero_padding_u8[3];
} iree_vm_isa_float_compare_f32_record_t;

// Page 0x00, opcode 0x91: Evaluates one ordered or unordered raw-payload f64
// predicate.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Result value-register ordinal.
  uint8_t dst_v8;
  // Operand value-register ordinal.
  uint8_t lhs_v8;
  // Operand value-register ordinal.
  uint8_t rhs_v8;
  // Closed float.compare operation selector.
  uint8_t predicate_u8;
  // Canonical zero padding.
  uint8_t zero_padding_u8[3];
} iree_vm_isa_float_compare_f64_record_t;

// Page 0x00, opcode 0x92: Classifies one raw f32 payload as NaN, infinity, or
// finite.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Result value-register ordinal.
  uint8_t dst_v8;
  // Operand value-register ordinal.
  uint8_t src_v8;
  // Closed float.classify operation selector.
  uint8_t selector_u8;
} iree_vm_isa_float_classify_f32_record_t;

// Page 0x00, opcode 0x93: Classifies one raw f64 payload as NaN, infinity, or
// finite.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Result value-register ordinal.
  uint8_t dst_v8;
  // Operand value-register ordinal.
  uint8_t src_v8;
  // Closed float.classify operation selector.
  uint8_t selector_u8;
} iree_vm_isa_float_classify_f64_record_t;

// Page 0x00, opcode 0x94: Clamps one f32 payload with explicit NaN policy.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Result value-register ordinal.
  uint8_t dst_v8;
  // Operand value-register ordinal.
  uint8_t value_v8;
  // Operand value-register ordinal.
  uint8_t lower_v8;
  // Operand value-register ordinal.
  uint8_t upper_v8;
  // Closed float.clamp operation selector.
  uint8_t mode_u8;
  // Canonical zero padding.
  uint16_t zero_padding_u16;
} iree_vm_isa_float_clamp_f32_record_t;

// Page 0x00, opcode 0x95: Clamps one f64 payload with explicit NaN policy.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Result value-register ordinal.
  uint8_t dst_v8;
  // Operand value-register ordinal.
  uint8_t value_v8;
  // Operand value-register ordinal.
  uint8_t lower_v8;
  // Operand value-register ordinal.
  uint8_t upper_v8;
  // Closed float.clamp operation selector.
  uint8_t mode_u8;
  // Canonical zero padding.
  uint16_t zero_padding_u16;
} iree_vm_isa_float_clamp_f64_record_t;

// Page 0x00, opcode 0x96: Copies rhs's raw f32 sign bit onto every non-sign bit
// from lhs, preserving signaling and noncanonical payloads.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Result value-register ordinal.
  uint8_t dst_v8;
  // Operand value-register ordinal.
  uint8_t lhs_v8;
  // Operand value-register ordinal.
  uint8_t rhs_v8;
} iree_vm_isa_float_copysign_f32_record_t;

// Page 0x00, opcode 0x97: Copies rhs's raw f64 sign bit onto every non-sign bit
// from lhs, preserving signaling and noncanonical payloads.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Result value-register ordinal.
  uint8_t dst_v8;
  // Operand value-register ordinal.
  uint8_t lhs_v8;
  // Operand value-register ordinal.
  uint8_t rhs_v8;
} iree_vm_isa_float_copysign_f64_record_t;

// Page 0x00, opcode 0x98: Evaluates one closed unary f32 math operation.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Result value-register ordinal.
  uint8_t dst_v8;
  // Operand value-register ordinal.
  uint8_t src_v8;
  // Closed float.math.unary operation selector.
  uint8_t selector_u8;
} iree_vm_isa_float_math_unary_f32_record_t;

// Page 0x00, opcode 0x99: Evaluates one closed unary f64 math operation.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Result value-register ordinal.
  uint8_t dst_v8;
  // Operand value-register ordinal.
  uint8_t src_v8;
  // Closed float.math.unary operation selector.
  uint8_t selector_u8;
} iree_vm_isa_float_math_unary_f64_record_t;

// Page 0x00, opcode 0x9A: Evaluates f32 pow, atan2, or staged scaled-logistic
// GELU.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Result value-register ordinal.
  uint8_t dst_v8;
  // Operand value-register ordinal.
  uint8_t lhs_v8;
  // Operand value-register ordinal.
  uint8_t rhs_v8;
  // Closed float.math.binary operation selector.
  uint8_t selector_u8;
  // Canonical zero padding.
  uint8_t zero_padding_u8[3];
} iree_vm_isa_float_math_binary_f32_record_t;

// Page 0x00, opcode 0x9B: Evaluates f64 pow, atan2, or staged scaled-logistic
// GELU.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Result value-register ordinal.
  uint8_t dst_v8;
  // Operand value-register ordinal.
  uint8_t lhs_v8;
  // Operand value-register ordinal.
  uint8_t rhs_v8;
  // Closed float.math.binary operation selector.
  uint8_t selector_u8;
  // Canonical zero padding.
  uint8_t zero_padding_u8[3];
} iree_vm_isa_float_math_binary_f64_record_t;

// Page 0x00, opcode 0x9C: Computes one fused f32 multiply-add.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Result value-register ordinal.
  uint8_t dst_v8;
  // Operand value-register ordinal.
  uint8_t a_v8;
  // Operand value-register ordinal.
  uint8_t b_v8;
  // Operand value-register ordinal.
  uint8_t c_v8;
  // Closed float.math.ternary operation selector.
  uint8_t selector_u8;
  // Canonical zero padding.
  uint16_t zero_padding_u16;
} iree_vm_isa_float_math_ternary_f32_record_t;

// Page 0x00, opcode 0x9D: Computes one fused f64 multiply-add.
typedef struct {
  // Page-local instruction opcode.
  uint8_t opcode_u8;
  // Result value-register ordinal.
  uint8_t dst_v8;
  // Operand value-register ordinal.
  uint8_t a_v8;
  // Operand value-register ordinal.
  uint8_t b_v8;
  // Operand value-register ordinal.
  uint8_t c_v8;
  // Closed float.math.ternary operation selector.
  uint8_t selector_u8;
  // Canonical zero padding.
  uint16_t zero_padding_u16;
} iree_vm_isa_float_math_ternary_f64_record_t;

#endif  // IREE_VM_BYTECODE_GENERATED_WIRE_CORE_FLOAT_H_
// clang-format on

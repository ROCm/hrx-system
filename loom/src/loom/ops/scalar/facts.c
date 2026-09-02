// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Fact implementations for the scalar dialect.
//
// Each fact inference function computes output facts from operand facts. For
// exact inputs, this IS constant folding. For range inputs, it
// propagates range and divisibility information.

#include "loom/ir/facts.h"

#include <math.h>

#include "iree/base/internal/math.h"
#include "loom/ir/float_facts.h"
#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/scalar/compare.h"
#include "loom/ops/scalar/ops.h"
#include "loom/util/predicate_facts.h"

//===----------------------------------------------------------------------===//
// Macros for mechanical fact inference functions
//===----------------------------------------------------------------------===//

static void loom_scalar_expand_result_facts_to_domain_on_overflow(
    const loom_module_t* module, const loom_op_t* op,
    loom_value_facts_t* facts) {
  loom_type_t result_type =
      loom_module_value_type(module, loom_op_results(op)[0]);
  int64_t domain_lo = 0;
  int64_t domain_hi = 0;
  if (loom_value_facts_scalar_type_domain(loom_type_element_type(result_type),
                                          &domain_lo, &domain_hi)) {
    if (facts->range_lo < domain_lo || facts->range_hi > domain_hi) {
      const uint32_t preserved_flags =
          facts->flags &
          (LOOM_VALUE_FACT_UNIFORM_SCOPE_MASK | LOOM_VALUE_FACT_LANE_VARYING |
           LOOM_VALUE_FACT_LANE_PREDICATE | LOOM_VALUE_FACT_SUBGROUP_LANE_MASK);
      const int64_t known_divisor = facts->known_divisor;
      *facts = loom_value_facts_make(domain_lo, domain_hi, known_divisor);
      facts->flags |= preserved_flags;
    }
  }
}

#define BINARY_FACTS(name, transfer_fn)                                  \
  iree_status_t name(loom_fact_context_t* context,                       \
                     const loom_module_t* module, const loom_op_t* op,   \
                     const loom_value_facts_t* operand_facts,            \
                     loom_value_facts_t* result_facts) {                 \
    transfer_fn(&operand_facts[0], &operand_facts[1], &result_facts[0]); \
    return iree_ok_status();                                             \
  }

#define UNARY_FACTS(name, transfer_fn)                                 \
  iree_status_t name(loom_fact_context_t* context,                     \
                     const loom_module_t* module, const loom_op_t* op, \
                     const loom_value_facts_t* operand_facts,          \
                     loom_value_facts_t* result_facts) {               \
    transfer_fn(&operand_facts[0], &result_facts[0]);                  \
    return iree_ok_status();                                           \
  }

static loom_scalar_type_t loom_scalar_result_element_type(
    const loom_module_t* module, const loom_op_t* op) {
  return loom_type_element_type(
      loom_module_value_type(module, loom_op_const_results(op)[0]));
}

#define FLOAT_BINARY_FACTS(name, f32_fn, f64_fn)                        \
  iree_status_t name(loom_fact_context_t* context,                      \
                     const loom_module_t* module, const loom_op_t* op,  \
                     const loom_value_facts_t* operand_facts,           \
                     loom_value_facts_t* result_facts) {                \
    loom_value_facts_eval_float_binary(                                 \
        loom_scalar_result_element_type(module, op), &operand_facts[0], \
        &operand_facts[1], f32_fn, f64_fn, &result_facts[0]);           \
    return iree_ok_status();                                            \
  }

#define FLOAT_UNARY_FACTS(name, f32_fn, f64_fn)                         \
  iree_status_t name(loom_fact_context_t* context,                      \
                     const loom_module_t* module, const loom_op_t* op,  \
                     const loom_value_facts_t* operand_facts,           \
                     loom_value_facts_t* result_facts) {                \
    loom_value_facts_eval_float_unary(                                  \
        loom_scalar_result_element_type(module, op), &operand_facts[0], \
        f32_fn, f64_fn, &result_facts[0]);                              \
    return iree_ok_status();                                            \
  }

static float add_f32(float a, float b) { return a + b; }
static double add_f64(double a, double b) { return a + b; }
static float sub_f32(float a, float b) { return a - b; }
static double sub_f64(double a, double b) { return a - b; }
static float mul_f32(float a, float b) { return a * b; }
static double mul_f64(double a, double b) { return a * b; }
static float div_f32(float a, float b) { return a / b; }
static double div_f64(double a, double b) { return a / b; }
static float rsqrt_f32(float x) { return 1.0f / sqrtf(x); }
static double rsqrt_f64(double x) { return 1.0 / sqrt(x); }
static float roundeven_f32(float x) { return nearbyintf(x); }
static double roundeven_f64(double x) { return nearbyint(x); }
static float silu_f32(float x) { return x * loom_float_logistic_f32(x); }
static double silu_f64(double x) { return x * loom_float_logistic_f64(x); }
static float softplus_f32(float x) {
  return log1pf(expf(-fabsf(x))) + fmaxf(x, 0.0f);
}
static double softplus_f64(double x) {
  return log1p(exp(-fabs(x))) + fmax(x, 0.0);
}
static float gelu_erf_f32(float x) {
  const float inverse_sqrt2 = 0.70710678118654752440f;
  return 0.5f * x * (1.0f + erff(x * inverse_sqrt2));
}
static double gelu_erf_f64(double x) {
  const double inverse_sqrt2 = 0.70710678118654752440;
  return 0.5 * x * (1.0 + erf(x * inverse_sqrt2));
}
static float gelu_tanh_f32(float x) {
  const float sqrt_2_over_pi = 0.79788456080286535588f;
  return 0.5f * x *
         (1.0f + tanhf(sqrt_2_over_pi * (x + 0.044715f * x * x * x)));
}
static double gelu_tanh_f64(double x) {
  const double sqrt_2_over_pi = 0.79788456080286535588;
  return 0.5 * x * (1.0 + tanh(sqrt_2_over_pi * (x + 0.044715 * x * x * x)));
}
static float gelu_logistic_f32(float x, float scale) {
  return x * loom_float_logistic_f32(scale * x);
}
static double gelu_logistic_f64(double x, double scale) {
  return x * loom_float_logistic_f64(scale * x);
}

//===----------------------------------------------------------------------===//
// scalar.constant
//===----------------------------------------------------------------------===//

iree_status_t loom_scalar_constant_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  loom_attribute_t attr = loom_op_attrs(op)[0];
  loom_value_id_t result_id = loom_scalar_constant_result(op);
  loom_type_t result_type = loom_module_value_type(module, result_id);
  loom_scalar_type_t result_element_type = loom_type_element_type(result_type);
  if (loom_scalar_type_is_float(result_element_type)) {
    result_facts[0] = loom_value_facts_exact_float(result_element_type,
                                                   loom_attr_as_f64(attr));
  } else {
    result_facts[0] = loom_value_facts_exact_i64(loom_attr_as_i64(attr));
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Integer arithmetic
//===----------------------------------------------------------------------===//

BINARY_FACTS(loom_scalar_addi_facts, loom_value_facts_addi)
BINARY_FACTS(loom_scalar_subi_facts, loom_value_facts_subi)
BINARY_FACTS(loom_scalar_muli_facts, loom_value_facts_muli)
BINARY_FACTS(loom_scalar_divsi_facts, loom_value_facts_divsi)
BINARY_FACTS(loom_scalar_divui_facts, loom_value_facts_divui)
BINARY_FACTS(loom_scalar_remsi_facts, loom_value_facts_remsi)
BINARY_FACTS(loom_scalar_remui_facts, loom_value_facts_remui)
UNARY_FACTS(loom_scalar_negi_facts, loom_value_facts_negi)
UNARY_FACTS(loom_scalar_absi_facts, loom_value_facts_absi)
BINARY_FACTS(loom_scalar_minsi_facts, loom_value_facts_minsi)
BINARY_FACTS(loom_scalar_maxsi_facts, loom_value_facts_maxsi)
BINARY_FACTS(loom_scalar_minui_facts, loom_value_facts_minui)
BINARY_FACTS(loom_scalar_maxui_facts, loom_value_facts_maxui)

// fmai has 3 operands: a*b + c.
iree_status_t loom_scalar_fmai_facts(loom_fact_context_t* context,
                                     const loom_module_t* module,
                                     const loom_op_t* op,
                                     const loom_value_facts_t* operand_facts,
                                     loom_value_facts_t* result_facts) {
  loom_value_facts_fmai(&operand_facts[0], &operand_facts[1], &operand_facts[2],
                        &result_facts[0]);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Float arithmetic
//===----------------------------------------------------------------------===//

FLOAT_BINARY_FACTS(loom_scalar_addf_facts, add_f32, add_f64)
FLOAT_BINARY_FACTS(loom_scalar_subf_facts, sub_f32, sub_f64)
FLOAT_BINARY_FACTS(loom_scalar_mulf_facts, mul_f32, mul_f64)
FLOAT_BINARY_FACTS(loom_scalar_divf_facts, div_f32, div_f64)
FLOAT_BINARY_FACTS(loom_scalar_remf_facts, fmodf, fmod)

iree_status_t loom_scalar_negf_facts(loom_fact_context_t* context,
                                     const loom_module_t* module,
                                     const loom_op_t* op,
                                     const loom_value_facts_t* operand_facts,
                                     loom_value_facts_t* result_facts) {
  loom_value_facts_eval_float_negate(
      loom_scalar_result_element_type(module, op), &operand_facts[0],
      &result_facts[0]);
  return iree_ok_status();
}

iree_status_t loom_scalar_absf_facts(loom_fact_context_t* context,
                                     const loom_module_t* module,
                                     const loom_op_t* op,
                                     const loom_value_facts_t* operand_facts,
                                     loom_value_facts_t* result_facts) {
  loom_value_facts_eval_float_abs(loom_scalar_result_element_type(module, op),
                                  &operand_facts[0], &result_facts[0]);
  return iree_ok_status();
}

#define FLOAT_MINMAX_FACTS(name, kind)                                        \
  iree_status_t name(loom_fact_context_t* context,                            \
                     const loom_module_t* module, const loom_op_t* op,        \
                     const loom_value_facts_t* operand_facts,                 \
                     loom_value_facts_t* result_facts) {                      \
    loom_value_facts_eval_float_minmax(                                       \
        loom_scalar_result_element_type(module, op), kind, &operand_facts[0], \
        &operand_facts[1], &result_facts[0]);                                 \
    return iree_ok_status();                                                  \
  }

FLOAT_MINMAX_FACTS(loom_scalar_minimumf_facts, LOOM_FLOAT_MINMAX_MINIMUM)
FLOAT_MINMAX_FACTS(loom_scalar_maximumf_facts, LOOM_FLOAT_MINMAX_MAXIMUM)
FLOAT_MINMAX_FACTS(loom_scalar_minnumf_facts, LOOM_FLOAT_MINMAX_MINNUM)
FLOAT_MINMAX_FACTS(loom_scalar_maxnumf_facts, LOOM_FLOAT_MINMAX_MAXNUM)

iree_status_t loom_scalar_copysignf_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  loom_value_facts_eval_float_copysign(
      loom_scalar_result_element_type(module, op), &operand_facts[0],
      &operand_facts[1], &result_facts[0]);
  return iree_ok_status();
}

iree_status_t loom_scalar_clampf_facts(loom_fact_context_t* context,
                                       const loom_module_t* module,
                                       const loom_op_t* op,
                                       const loom_value_facts_t* operand_facts,
                                       loom_value_facts_t* result_facts) {
  loom_float_clamp_kind_t kind = LOOM_FLOAT_CLAMP_ORDERED;
  switch (loom_scalar_clampf_mode(op)) {
    case LOOM_SCALAR_CLAMPF_MODE_ORDERED:
      kind = LOOM_FLOAT_CLAMP_ORDERED;
      break;
    case LOOM_SCALAR_CLAMPF_MODE_NUMBER:
      kind = LOOM_FLOAT_CLAMP_NUMBER;
      break;
    case LOOM_SCALAR_CLAMPF_MODE_IEEE:
      kind = LOOM_FLOAT_CLAMP_IEEE;
      break;
    case LOOM_SCALAR_CLAMPF_MODE_COUNT_:
      result_facts[0] = loom_value_facts_unknown();
      return iree_ok_status();
  }
  loom_value_facts_eval_float_clamp(loom_scalar_result_element_type(module, op),
                                    kind, &operand_facts[0], &operand_facts[1],
                                    &operand_facts[2], &result_facts[0]);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Math functions
//===----------------------------------------------------------------------===//

FLOAT_UNARY_FACTS(loom_scalar_expf_facts, expf, exp)
FLOAT_UNARY_FACTS(loom_scalar_exp2f_facts, exp2f, exp2)
FLOAT_UNARY_FACTS(loom_scalar_expm1f_facts, expm1f, expm1)
FLOAT_UNARY_FACTS(loom_scalar_logf_facts, logf, log)
FLOAT_UNARY_FACTS(loom_scalar_log2f_facts, log2f, log2)
FLOAT_UNARY_FACTS(loom_scalar_log10f_facts, log10f, log10)
FLOAT_UNARY_FACTS(loom_scalar_log1pf_facts, log1pf, log1p)
FLOAT_BINARY_FACTS(loom_scalar_powf_facts, powf, pow)
FLOAT_UNARY_FACTS(loom_scalar_sqrtf_facts, sqrtf, sqrt)
FLOAT_UNARY_FACTS(loom_scalar_rsqrtf_facts, rsqrt_f32, rsqrt_f64)
FLOAT_UNARY_FACTS(loom_scalar_cbrtf_facts, cbrtf, cbrt)
FLOAT_UNARY_FACTS(loom_scalar_sinf_facts, sinf, sin)
FLOAT_UNARY_FACTS(loom_scalar_cosf_facts, cosf, cos)

static iree_status_t loom_scalar_turnsf_facts(
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts, loom_value_facts_t* result_facts,
    loom_float_turns_kind_t kind) {
  loom_value_facts_eval_float_turns(loom_scalar_result_element_type(module, op),
                                    kind, &operand_facts[0], &result_facts[0]);
  return iree_ok_status();
}

iree_status_t loom_scalar_sinturnsf_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  return loom_scalar_turnsf_facts(module, op, operand_facts, result_facts,
                                  LOOM_FLOAT_TURNS_SIN);
}

iree_status_t loom_scalar_costurnsf_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  return loom_scalar_turnsf_facts(module, op, operand_facts, result_facts,
                                  LOOM_FLOAT_TURNS_COS);
}

FLOAT_UNARY_FACTS(loom_scalar_tanf_facts, tanf, tan)
FLOAT_UNARY_FACTS(loom_scalar_asinf_facts, asinf, asin)
FLOAT_UNARY_FACTS(loom_scalar_acosf_facts, acosf, acos)
FLOAT_UNARY_FACTS(loom_scalar_atanf_facts, atanf, atan)
FLOAT_BINARY_FACTS(loom_scalar_atan2f_facts, atan2f, atan2)
FLOAT_UNARY_FACTS(loom_scalar_sinhf_facts, sinhf, sinh)
FLOAT_UNARY_FACTS(loom_scalar_coshf_facts, coshf, cosh)
FLOAT_UNARY_FACTS(loom_scalar_tanhf_facts, tanhf, tanh)
FLOAT_UNARY_FACTS(loom_scalar_asinhf_facts, asinhf, asinh)
FLOAT_UNARY_FACTS(loom_scalar_acoshf_facts, acoshf, acosh)
FLOAT_UNARY_FACTS(loom_scalar_atanhf_facts, atanhf, atanh)
FLOAT_UNARY_FACTS(loom_scalar_erff_facts, erff, erf)
FLOAT_UNARY_FACTS(loom_scalar_erfcf_facts, erfcf, erfc)
FLOAT_UNARY_FACTS(loom_scalar_logisticf_facts, loom_float_logistic_f32,
                  loom_float_logistic_f64)
FLOAT_UNARY_FACTS(loom_scalar_siluf_facts, silu_f32, silu_f64)
FLOAT_UNARY_FACTS(loom_scalar_softplusf_facts, softplus_f32, softplus_f64)

iree_status_t loom_scalar_geluf_facts(loom_fact_context_t* context,
                                      const loom_module_t* module,
                                      const loom_op_t* op,
                                      const loom_value_facts_t* operand_facts,
                                      loom_value_facts_t* result_facts) {
  loom_scalar_type_t scalar_type = loom_scalar_result_element_type(module, op);
  switch (loom_scalar_geluf_variant(op)) {
    case LOOM_SCALAR_GELUF_VARIANT_ERF:
      loom_value_facts_eval_float_unary(scalar_type, &operand_facts[0],
                                        gelu_erf_f32, gelu_erf_f64,
                                        &result_facts[0]);
      return iree_ok_status();
    case LOOM_SCALAR_GELUF_VARIANT_TANH:
      loom_value_facts_eval_float_unary(scalar_type, &operand_facts[0],
                                        gelu_tanh_f32, gelu_tanh_f64,
                                        &result_facts[0]);
      return iree_ok_status();
    case LOOM_SCALAR_GELUF_VARIANT_LOGISTIC: {
      loom_attribute_t scale_attr = loom_op_attrs(op)[1];
      if (loom_attr_is_absent(scale_attr)) {
        result_facts[0] = loom_value_facts_unknown();
        return iree_ok_status();
      }
      const double scale = loom_attr_as_f64(scale_attr);
      double input = 0.0;
      if (!loom_value_facts_as_exact_float(scalar_type, operand_facts[0],
                                           &input)) {
        result_facts[0] = loom_value_facts_unknown();
      } else if (scalar_type == LOOM_SCALAR_TYPE_F64) {
        result_facts[0] = loom_value_facts_exact_float(
            scalar_type, gelu_logistic_f64(input, scale));
      } else {
        result_facts[0] = loom_value_facts_exact_float(
            scalar_type, gelu_logistic_f32((float)input, (float)scale));
      }
      loom_value_facts_propagate_unary_distribution(operand_facts[0],
                                                    &result_facts[0]);
      return iree_ok_status();
    }
    case LOOM_SCALAR_GELUF_VARIANT_COUNT_:
      break;
  }
  result_facts[0] = loom_value_facts_unknown();
  return iree_ok_status();
}

// fmaf has 3 operands.
iree_status_t loom_scalar_fmaf_facts(loom_fact_context_t* context,
                                     const loom_module_t* module,
                                     const loom_op_t* op,
                                     const loom_value_facts_t* operand_facts,
                                     loom_value_facts_t* result_facts) {
  loom_value_facts_eval_float_ternary(
      loom_scalar_result_element_type(module, op), &operand_facts[0],
      &operand_facts[1], &operand_facts[2], fmaf, fma, &result_facts[0]);
  return iree_ok_status();
}

FLOAT_UNARY_FACTS(loom_scalar_ceilf_facts, ceilf, ceil)
FLOAT_UNARY_FACTS(loom_scalar_floorf_facts, floorf, floor)
FLOAT_UNARY_FACTS(loom_scalar_roundf_facts, roundf, round)
FLOAT_UNARY_FACTS(loom_scalar_roundevenf_facts, roundeven_f32, roundeven_f64)
FLOAT_UNARY_FACTS(loom_scalar_truncf_facts, truncf, trunc)

//===----------------------------------------------------------------------===//
// Bitwise
//===----------------------------------------------------------------------===//

static bool loom_scalar_result_is_i1(const loom_module_t* module,
                                     loom_value_id_t result_id) {
  loom_type_t type = loom_module_value_type(module, result_id);
  return loom_type_is_scalar(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_I1;
}

static void loom_scalar_mark_i1_bitwise_distribution(
    const loom_module_t* module, loom_value_id_t result_id,
    loom_value_facts_t* facts) {
  if (!loom_scalar_result_is_i1(module, result_id)) {
    return;
  }
  *facts = loom_value_facts_clamp_domain(*facts, 0, 1);
  if (loom_value_facts_is_lane_varying(*facts)) {
    loom_value_facts_mark_lane_predicate(facts);
  }
}

iree_status_t loom_scalar_andi_facts(loom_fact_context_t* context,
                                     const loom_module_t* module,
                                     const loom_op_t* op,
                                     const loom_value_facts_t* operand_facts,
                                     loom_value_facts_t* result_facts) {
  loom_value_facts_andi(&operand_facts[0], &operand_facts[1], &result_facts[0]);
  loom_scalar_mark_i1_bitwise_distribution(module, loom_scalar_andi_result(op),
                                           &result_facts[0]);
  return iree_ok_status();
}

iree_status_t loom_scalar_ori_facts(loom_fact_context_t* context,
                                    const loom_module_t* module,
                                    const loom_op_t* op,
                                    const loom_value_facts_t* operand_facts,
                                    loom_value_facts_t* result_facts) {
  loom_value_facts_ori(&operand_facts[0], &operand_facts[1], &result_facts[0]);
  loom_scalar_mark_i1_bitwise_distribution(module, loom_scalar_ori_result(op),
                                           &result_facts[0]);
  return iree_ok_status();
}

iree_status_t loom_scalar_xori_facts(loom_fact_context_t* context,
                                     const loom_module_t* module,
                                     const loom_op_t* op,
                                     const loom_value_facts_t* operand_facts,
                                     loom_value_facts_t* result_facts) {
  loom_value_facts_xori(&operand_facts[0], &operand_facts[1], &result_facts[0]);
  loom_scalar_mark_i1_bitwise_distribution(module, loom_scalar_xori_result(op),
                                           &result_facts[0]);
  return iree_ok_status();
}

iree_status_t loom_scalar_shli_facts(loom_fact_context_t* context,
                                     const loom_module_t* module,
                                     const loom_op_t* op,
                                     const loom_value_facts_t* operand_facts,
                                     loom_value_facts_t* result_facts) {
  loom_value_facts_shli(&operand_facts[0], &operand_facts[1], &result_facts[0]);
  loom_scalar_expand_result_facts_to_domain_on_overflow(module, op,
                                                        &result_facts[0]);
  return iree_ok_status();
}
BINARY_FACTS(loom_scalar_shrsi_facts, loom_value_facts_shrsi)
BINARY_FACTS(loom_scalar_shrui_facts, loom_value_facts_shrui)

static bool loom_scalar_integer_bitwidth(loom_type_t type,
                                         int32_t* out_bitwidth) {
  if (!loom_type_is_scalar(type)) return false;
  loom_scalar_type_t element_type = loom_type_element_type(type);
  if (!loom_scalar_type_is_integer(element_type)) return false;
  int32_t bitwidth = loom_scalar_type_bitwidth(element_type);
  if (bitwidth <= 0 || bitwidth > 64) return false;
  *out_bitwidth = bitwidth;
  return true;
}

static iree_status_t loom_scalar_bitfield_extract_facts(
    const loom_module_t* module, const loom_value_facts_t* operand_facts,
    loom_value_id_t source_id, loom_value_id_t result_id, int64_t offset,
    int64_t width, bool signed_extract, loom_value_facts_t* result_facts) {
  int32_t source_width = 0;
  int32_t result_width = 0;
  if (!loom_scalar_integer_bitwidth(loom_module_value_type(module, source_id),
                                    &source_width) ||
      !loom_scalar_integer_bitwidth(loom_module_value_type(module, result_id),
                                    &result_width) ||
      result_width < width) {
    result_facts[0] = loom_value_facts_unknown();
    return iree_ok_status();
  }
  bool valid =
      signed_extract
          ? loom_value_facts_extract_signed_bitfield(
                operand_facts[0], source_width, offset, width, &result_facts[0])
          : loom_value_facts_extract_unsigned_bitfield(operand_facts[0],
                                                       source_width, offset,
                                                       width, &result_facts[0]);
  if (!valid) {
    result_facts[0] = loom_value_facts_unknown();
    return iree_ok_status();
  }
  loom_value_facts_propagate_unary_distribution(operand_facts[0],
                                                &result_facts[0]);
  return iree_ok_status();
}

iree_status_t loom_scalar_bitfield_extractu_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  return loom_scalar_bitfield_extract_facts(
      module, operand_facts, loom_scalar_bitfield_extractu_source(op),
      loom_scalar_bitfield_extractu_result(op),
      loom_scalar_bitfield_extractu_offset(op),
      loom_scalar_bitfield_extractu_width(op), /*signed_extract=*/false,
      result_facts);
}

iree_status_t loom_scalar_bitfield_extracts_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  return loom_scalar_bitfield_extract_facts(
      module, operand_facts, loom_scalar_bitfield_extracts_source(op),
      loom_scalar_bitfield_extracts_result(op),
      loom_scalar_bitfield_extracts_offset(op),
      loom_scalar_bitfield_extracts_width(op), /*signed_extract=*/true,
      result_facts);
}

// Rotates: no transfer function yet — exact-only fact inference.
iree_status_t loom_scalar_rotli_facts(loom_fact_context_t* context,
                                      const loom_module_t* module,
                                      const loom_op_t* op,
                                      const loom_value_facts_t* operand_facts,
                                      loom_value_facts_t* result_facts) {
  result_facts[0] = loom_value_facts_unknown();
  return iree_ok_status();
}
iree_status_t loom_scalar_rotri_facts(loom_fact_context_t* context,
                                      const loom_module_t* module,
                                      const loom_op_t* op,
                                      const loom_value_facts_t* operand_facts,
                                      loom_value_facts_t* result_facts) {
  result_facts[0] = loom_value_facts_unknown();
  return iree_ok_status();
}

// Bit counting is bounded by the declared integer width.
#define BIT_COUNT_FACTS(name, result_accessor, fn)                      \
  iree_status_t name(loom_fact_context_t* context,                      \
                     const loom_module_t* module, const loom_op_t* op,  \
                     const loom_value_facts_t* operand_facts,           \
                     loom_value_facts_t* result_facts) {                \
    loom_type_t result_type =                                           \
        loom_module_value_type(module, result_accessor(op));            \
    int32_t bitwidth =                                                  \
        loom_scalar_type_bitwidth(loom_type_element_type(result_type)); \
    if (bitwidth <= 0) {                                                \
      result_facts[0] = loom_value_facts_unknown();                     \
      return iree_ok_status();                                          \
    }                                                                   \
    if (loom_value_facts_is_exact(operand_facts[0])) {                  \
      result_facts[0] = loom_value_facts_exact_i64(                     \
          fn((uint64_t)operand_facts[0].range_lo, bitwidth));           \
    } else {                                                            \
      result_facts[0] = loom_value_facts_make(0, bitwidth, 1);          \
    }                                                                   \
    loom_value_facts_propagate_unary_distribution(operand_facts[0],     \
                                                  &result_facts[0]);    \
    return iree_ok_status();                                            \
  }

BIT_COUNT_FACTS(loom_scalar_ctlzi_facts, loom_scalar_ctlzi_result,
                iree_math_count_leading_zeros_u64_width)
BIT_COUNT_FACTS(loom_scalar_cttzi_facts, loom_scalar_cttzi_result,
                iree_math_count_trailing_zeros_u64_width)
BIT_COUNT_FACTS(loom_scalar_ctpopi_facts, loom_scalar_ctpopi_result,
                iree_math_count_ones_u64_width)

#undef BIT_COUNT_FACTS

//===----------------------------------------------------------------------===//
// Comparison
//===----------------------------------------------------------------------===//

static void loom_scalar_mark_compare_distribution(
    const loom_value_facts_t* operand_facts, loom_value_facts_t* result_facts) {
  loom_value_facts_propagate_binary_distribution(
      operand_facts[0], operand_facts[1], result_facts);
  if (loom_value_facts_is_lane_varying(*result_facts)) {
    loom_value_facts_mark_lane_predicate(result_facts);
  }
}

iree_status_t loom_scalar_cmpi_facts(loom_fact_context_t* context,
                                     const loom_module_t* module,
                                     const loom_op_t* op,
                                     const loom_value_facts_t* operand_facts,
                                     loom_value_facts_t* result_facts) {
  bool result = false;
  if ((loom_scalar_cmpi_lhs(op) == loom_scalar_cmpi_rhs(op) &&
       loom_scalar_cmpi_same_value_result(loom_scalar_cmpi_predicate(op),
                                          &result)) ||
      loom_scalar_cmpi_result_from_facts(loom_scalar_cmpi_predicate(op),
                                         &operand_facts[0], &operand_facts[1],
                                         &result)) {
    result_facts[0] = loom_value_facts_exact_i64(result ? 1 : 0);
    return iree_ok_status();
  }
  result_facts[0] = loom_value_facts_make(0, 1, 1);
  loom_scalar_mark_compare_distribution(operand_facts, &result_facts[0]);
  return iree_ok_status();
}

iree_status_t loom_scalar_cmpf_facts(loom_fact_context_t* context,
                                     const loom_module_t* module,
                                     const loom_op_t* op,
                                     const loom_value_facts_t* operand_facts,
                                     loom_value_facts_t* result_facts) {
  bool result = false;
  if (loom_scalar_cmpf_lhs(op) == loom_scalar_cmpf_rhs(op) &&
      (loom_scalar_cmpf_fastmath(op) & LOOM_SCALAR_FASTMATHFLAGS_NNAN) != 0 &&
      loom_scalar_cmpf_same_value_result(loom_scalar_cmpf_predicate(op),
                                         &result)) {
    result_facts[0] = loom_value_facts_exact_i64(result ? 1 : 0);
    return iree_ok_status();
  }
  loom_scalar_type_t scalar_type = loom_type_element_type(
      loom_module_value_type(module, loom_scalar_cmpf_lhs(op)));
  double lhs = 0.0;
  double rhs = 0.0;
  const bool has_known_nan = loom_value_facts_is_nan(operand_facts[0]) ||
                             loom_value_facts_is_nan(operand_facts[1]);
  bool has_values = has_known_nan;
  if (has_known_nan) {
    lhs = NAN;
    rhs = NAN;
  } else {
    has_values =
        loom_value_facts_as_exact_float(scalar_type, operand_facts[0], &lhs) &&
        loom_value_facts_as_exact_float(scalar_type, operand_facts[1], &rhs);
  }
  if (has_values && loom_scalar_cmpf_exact_result(
                        loom_scalar_cmpf_predicate(op), lhs, rhs, &result)) {
    result_facts[0] = loom_value_facts_exact_i64(result ? 1 : 0);
    return iree_ok_status();
  }
  result_facts[0] = loom_value_facts_make(0, 1, 1);
  loom_scalar_mark_compare_distribution(operand_facts, &result_facts[0]);
  return iree_ok_status();
}

// Float predicate queries fold from exact constants and checked predicate
// facts.
static bool isnan_f64(double v) { return v != v; }
static bool isinf_f64(double v) { return isinf(v) != 0; }
static bool isfinite_f64(double v) { return isfinite(v) != 0; }

iree_status_t loom_scalar_isnanf_facts(loom_fact_context_t* context,
                                       const loom_module_t* module,
                                       const loom_op_t* op,
                                       const loom_value_facts_t* operand_facts,
                                       loom_value_facts_t* result_facts) {
  if (loom_value_facts_is_nan(operand_facts[0])) {
    result_facts[0] = loom_value_facts_exact_i64(1);
    return iree_ok_status();
  }
  if (loom_value_facts_is_not_nan(operand_facts[0])) {
    result_facts[0] = loom_value_facts_exact_i64(0);
    return iree_ok_status();
  }
  loom_scalar_type_t scalar_type = loom_type_element_type(
      loom_module_value_type(module, loom_scalar_isnanf_input(op)));
  double value = 0.0;
  if (!loom_value_facts_as_exact_float(scalar_type, operand_facts[0], &value)) {
    result_facts[0] = loom_value_facts_make(0, 1, 1);
    return iree_ok_status();
  }
  result_facts[0] = loom_value_facts_exact_i64(isnan_f64(value) ? 1 : 0);
  return iree_ok_status();
}

iree_status_t loom_scalar_isinff_facts(loom_fact_context_t* context,
                                       const loom_module_t* module,
                                       const loom_op_t* op,
                                       const loom_value_facts_t* operand_facts,
                                       loom_value_facts_t* result_facts) {
  if (loom_value_facts_is_inf(operand_facts[0])) {
    result_facts[0] = loom_value_facts_exact_i64(1);
    return iree_ok_status();
  }
  if (loom_value_facts_is_nan(operand_facts[0]) ||
      loom_value_facts_is_finite(operand_facts[0])) {
    result_facts[0] = loom_value_facts_exact_i64(0);
    return iree_ok_status();
  }
  if (loom_value_facts_is_not_inf(operand_facts[0])) {
    result_facts[0] = loom_value_facts_exact_i64(0);
    return iree_ok_status();
  }
  loom_scalar_type_t scalar_type = loom_type_element_type(
      loom_module_value_type(module, loom_scalar_isinff_input(op)));
  double value = 0.0;
  if (!loom_value_facts_as_exact_float(scalar_type, operand_facts[0], &value)) {
    result_facts[0] = loom_value_facts_make(0, 1, 1);
    return iree_ok_status();
  }
  result_facts[0] = loom_value_facts_exact_i64(isinf_f64(value) ? 1 : 0);
  return iree_ok_status();
}

iree_status_t loom_scalar_isfinitef_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  if (loom_value_facts_is_nan(operand_facts[0]) ||
      loom_value_facts_is_inf(operand_facts[0])) {
    result_facts[0] = loom_value_facts_exact_i64(0);
    return iree_ok_status();
  }
  if (loom_value_facts_is_finite(operand_facts[0])) {
    result_facts[0] = loom_value_facts_exact_i64(1);
    return iree_ok_status();
  }
  if (loom_value_facts_is_not_nan(operand_facts[0]) &&
      loom_value_facts_is_not_inf(operand_facts[0])) {
    result_facts[0] = loom_value_facts_exact_i64(1);
    return iree_ok_status();
  }
  loom_scalar_type_t scalar_type = loom_type_element_type(
      loom_module_value_type(module, loom_scalar_isfinitef_input(op)));
  double value = 0.0;
  if (!loom_value_facts_as_exact_float(scalar_type, operand_facts[0], &value)) {
    result_facts[0] = loom_value_facts_make(0, 1, 1);
    return iree_ok_status();
  }
  result_facts[0] = loom_value_facts_exact_i64(isfinite_f64(value) ? 1 : 0);
  return iree_ok_status();
}

iree_status_t loom_scalar_signf_facts(loom_fact_context_t* context,
                                      const loom_module_t* module,
                                      const loom_op_t* op,
                                      const loom_value_facts_t* operand_facts,
                                      loom_value_facts_t* result_facts) {
  loom_scalar_type_t scalar_type = loom_scalar_result_element_type(module, op);
  if (loom_value_facts_is_nan(operand_facts[0])) {
    result_facts[0] = loom_value_facts_exact_float(scalar_type, 0.0);
    return iree_ok_status();
  }
  double value = 0.0;
  if (!loom_value_facts_as_exact_float(scalar_type, operand_facts[0], &value)) {
    result_facts[0] = loom_value_facts_unknown();
    return iree_ok_status();
  }
  result_facts[0] =
      loom_value_facts_exact_float(scalar_type, (value > 0.0)   ? 1.0
                                                : (value < 0.0) ? -1.0
                                                                : 0.0);
  return iree_ok_status();
}

iree_status_t loom_scalar_signi_facts(loom_fact_context_t* context,
                                      const loom_module_t* module,
                                      const loom_op_t* op,
                                      const loom_value_facts_t* operand_facts,
                                      loom_value_facts_t* result_facts) {
  if (!loom_value_facts_is_exact(operand_facts[0])) {
    result_facts[0] = loom_value_facts_unknown();
    return iree_ok_status();
  }
  int64_t v = operand_facts[0].range_lo;
  result_facts[0] = loom_value_facts_exact_i64((v > 0) ? 1 : (v < 0) ? -1 : 0);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Conversion
//===----------------------------------------------------------------------===//

iree_status_t loom_scalar_sitofp_facts(loom_fact_context_t* context,
                                       const loom_module_t* module,
                                       const loom_op_t* op,
                                       const loom_value_facts_t* operand_facts,
                                       loom_value_facts_t* result_facts) {
  if (!loom_value_facts_is_exact(operand_facts[0])) {
    result_facts[0] = loom_value_facts_unknown();
    return iree_ok_status();
  }
  result_facts[0] =
      loom_value_facts_exact_float(loom_scalar_result_element_type(module, op),
                                   (double)operand_facts[0].range_lo);
  return iree_ok_status();
}

iree_status_t loom_scalar_uitofp_facts(loom_fact_context_t* context,
                                       const loom_module_t* module,
                                       const loom_op_t* op,
                                       const loom_value_facts_t* operand_facts,
                                       loom_value_facts_t* result_facts) {
  if (!loom_value_facts_is_exact(operand_facts[0])) {
    result_facts[0] = loom_value_facts_unknown();
    return iree_ok_status();
  }
  loom_type_t source_type =
      loom_module_value_type(module, loom_scalar_uitofp_input(op));
  const int32_t source_bit_count =
      loom_scalar_type_bitwidth(loom_type_element_type(source_type));
  uint64_t source_bits = 0;
  if (!loom_value_facts_as_exact_raw_bits(operand_facts[0], source_bit_count,
                                          &source_bits)) {
    result_facts[0] = loom_value_facts_unknown();
    return iree_ok_status();
  }
  result_facts[0] = loom_value_facts_exact_float(
      loom_scalar_result_element_type(module, op), (double)source_bits);
  return iree_ok_status();
}

iree_status_t loom_scalar_fptosi_facts(loom_fact_context_t* context,
                                       const loom_module_t* module,
                                       const loom_op_t* op,
                                       const loom_value_facts_t* operand_facts,
                                       loom_value_facts_t* result_facts) {
  loom_value_facts_eval_float_to_integer(
      loom_type_element_type(
          loom_module_value_type(module, loom_scalar_fptosi_input(op))),
      loom_scalar_result_element_type(module, op),
      LOOM_FLOAT_INTEGER_CONVERSION_SIGNED, &operand_facts[0],
      &result_facts[0]);
  return iree_ok_status();
}

iree_status_t loom_scalar_fptoui_facts(loom_fact_context_t* context,
                                       const loom_module_t* module,
                                       const loom_op_t* op,
                                       const loom_value_facts_t* operand_facts,
                                       loom_value_facts_t* result_facts) {
  loom_value_facts_eval_float_to_integer(
      loom_type_element_type(
          loom_module_value_type(module, loom_scalar_fptoui_input(op))),
      loom_scalar_result_element_type(module, op),
      LOOM_FLOAT_INTEGER_CONVERSION_UNSIGNED, &operand_facts[0],
      &result_facts[0]);
  return iree_ok_status();
}

iree_status_t loom_scalar_extf_facts(loom_fact_context_t* context,
                                     const loom_module_t* module,
                                     const loom_op_t* op,
                                     const loom_value_facts_t* operand_facts,
                                     loom_value_facts_t* result_facts) {
  loom_scalar_type_t source_scalar_type = loom_type_element_type(
      loom_module_value_type(module, loom_scalar_extf_input(op)));
  double value = 0.0;
  if (loom_value_facts_is_nan(operand_facts[0])) {
    result_facts[0] = loom_value_facts_known_nan();
    loom_value_facts_propagate_unary_distribution(operand_facts[0],
                                                  &result_facts[0]);
  } else if (loom_value_facts_as_exact_float(source_scalar_type,
                                             operand_facts[0], &value)) {
    result_facts[0] = loom_value_facts_exact_float(
        loom_scalar_result_element_type(module, op), value);
  } else {
    result_facts[0] = operand_facts[0];
  }
  return iree_ok_status();
}

iree_status_t loom_scalar_fptrunc_facts(loom_fact_context_t* context,
                                        const loom_module_t* module,
                                        const loom_op_t* op,
                                        const loom_value_facts_t* operand_facts,
                                        loom_value_facts_t* result_facts) {
  loom_scalar_type_t source_scalar_type = loom_type_element_type(
      loom_module_value_type(module, loom_scalar_fptrunc_input(op)));
  if (loom_value_facts_is_nan(operand_facts[0])) {
    result_facts[0] = loom_value_facts_known_nan();
    loom_value_facts_propagate_unary_distribution(operand_facts[0],
                                                  &result_facts[0]);
    return iree_ok_status();
  }
  double value = 0.0;
  if (!loom_value_facts_as_exact_float(source_scalar_type, operand_facts[0],
                                       &value)) {
    result_facts[0] = loom_value_facts_unknown();
    return iree_ok_status();
  }
  result_facts[0] = loom_value_facts_exact_float(
      loom_scalar_result_element_type(module, op), value);
  return iree_ok_status();
}

static bool loom_scalar_integer_value_domain(const loom_module_t* module,
                                             loom_value_id_t value_id,
                                             loom_scalar_type_t* out_type,
                                             int64_t* out_lo, int64_t* out_hi) {
  const loom_type_t type = loom_module_value_type(module, value_id);
  if (!loom_type_is_scalar(type)) {
    return false;
  }
  const loom_scalar_type_t scalar_type = loom_type_element_type(type);
  if (!loom_value_facts_scalar_type_domain(scalar_type, out_lo, out_hi)) {
    return false;
  }
  *out_type = scalar_type;
  return true;
}

static loom_value_facts_t loom_scalar_integer_domain_facts(
    loom_value_facts_t input_facts, int64_t result_lo, int64_t result_hi) {
  loom_value_facts_t result_facts =
      loom_value_facts_make(result_lo, result_hi, 1);
  loom_value_facts_propagate_unary_distribution(input_facts, &result_facts);
  return result_facts;
}

static bool loom_scalar_zero_extend_exact_i64(int64_t value, int32_t bitwidth,
                                              int64_t* out_value) {
  if (bitwidth <= 0 || bitwidth >= 63) {
    return false;
  }
  *out_value = (int64_t)iree_math_mask_low_bits_u64((uint64_t)value, bitwidth);
  return true;
}

static bool loom_scalar_truncate_exact_i64(
    int64_t value, loom_scalar_type_t result_scalar_type, int64_t* out_value) {
  const int32_t bitwidth = loom_scalar_type_bitwidth(result_scalar_type);
  if (bitwidth <= 0 || bitwidth > 64) {
    return false;
  }
  const uint64_t masked =
      iree_math_mask_low_bits_u64((uint64_t)value, bitwidth);
  if (result_scalar_type == LOOM_SCALAR_TYPE_I1) {
    *out_value = (int64_t)masked;
    return true;
  }
  if (bitwidth == 64) {
    *out_value = (int64_t)masked;
    return true;
  }
  const uint64_t sign_bit = UINT64_C(1) << (bitwidth - 1);
  const uint64_t sign_extension_mask =
      ~iree_math_mask_low_bits_u64(~0ull, bitwidth);
  *out_value =
      (int64_t)((masked & sign_bit) ? masked | sign_extension_mask : masked);
  return true;
}

iree_status_t loom_scalar_extsi_facts(loom_fact_context_t* context,
                                      const loom_module_t* module,
                                      const loom_op_t* op,
                                      const loom_value_facts_t* operand_facts,
                                      loom_value_facts_t* result_facts) {
  result_facts[0] = operand_facts[0];
  return iree_ok_status();
}

iree_status_t loom_scalar_extui_facts(loom_fact_context_t* context,
                                      const loom_module_t* module,
                                      const loom_op_t* op,
                                      const loom_value_facts_t* operand_facts,
                                      loom_value_facts_t* result_facts) {
  loom_scalar_type_t input_scalar_type = LOOM_SCALAR_TYPE_NONE;
  loom_scalar_type_t result_scalar_type = LOOM_SCALAR_TYPE_NONE;
  int64_t input_lo = 0;
  int64_t input_hi = 0;
  int64_t result_lo = 0;
  int64_t result_hi = 0;
  if (!loom_scalar_integer_value_domain(module, loom_scalar_extui_input(op),
                                        &input_scalar_type, &input_lo,
                                        &input_hi) ||
      !loom_scalar_integer_value_domain(module, loom_scalar_extui_result(op),
                                        &result_scalar_type, &result_lo,
                                        &result_hi)) {
    result_facts[0] = loom_value_facts_unknown();
    return iree_ok_status();
  }
  (void)result_scalar_type;

  const int32_t input_bitwidth = loom_scalar_type_bitwidth(input_scalar_type);
  int64_t exact_value = 0;
  if (loom_value_facts_as_exact_i64(operand_facts[0], &exact_value) &&
      loom_scalar_zero_extend_exact_i64(exact_value, input_bitwidth,
                                        &exact_value)) {
    result_facts[0] = loom_value_facts_exact_i64(exact_value);
    return iree_ok_status();
  }

  loom_value_facts_t input_facts =
      loom_value_facts_clamp_domain(operand_facts[0], input_lo, input_hi);
  if (input_facts.range_lo >= 0) {
    result_facts[0] =
        loom_value_facts_clamp_domain(input_facts, result_lo, result_hi);
    return iree_ok_status();
  }
  if (input_bitwidth > 0 && input_bitwidth < 63 && input_facts.range_hi < 0) {
    const int64_t unsigned_extent = INT64_C(1) << input_bitwidth;
    result_facts[0] = loom_value_facts_make(
        input_facts.range_lo + unsigned_extent,
        input_facts.range_hi + unsigned_extent,
        iree_math_gcd_i64(input_facts.known_divisor, unsigned_extent));
    loom_value_facts_propagate_unary_distribution(operand_facts[0],
                                                  &result_facts[0]);
    result_facts[0] =
        loom_value_facts_clamp_domain(result_facts[0], result_lo, result_hi);
    return iree_ok_status();
  }

  int64_t unsigned_hi = result_hi;
  if (input_bitwidth > 0 && input_bitwidth < 63) {
    const int64_t input_unsigned_hi = (INT64_C(1) << input_bitwidth) - 1;
    unsigned_hi = input_unsigned_hi < result_hi ? input_unsigned_hi : result_hi;
  }
  result_facts[0] =
      loom_scalar_integer_domain_facts(operand_facts[0], 0, unsigned_hi);
  return iree_ok_status();
}

iree_status_t loom_scalar_trunci_facts(loom_fact_context_t* context,
                                       const loom_module_t* module,
                                       const loom_op_t* op,
                                       const loom_value_facts_t* operand_facts,
                                       loom_value_facts_t* result_facts) {
  loom_scalar_type_t input_scalar_type = LOOM_SCALAR_TYPE_NONE;
  loom_scalar_type_t result_scalar_type = LOOM_SCALAR_TYPE_NONE;
  int64_t input_lo = 0;
  int64_t input_hi = 0;
  int64_t result_lo = 0;
  int64_t result_hi = 0;
  if (!loom_scalar_integer_value_domain(module, loom_scalar_trunci_input(op),
                                        &input_scalar_type, &input_lo,
                                        &input_hi) ||
      !loom_scalar_integer_value_domain(module, loom_scalar_trunci_result(op),
                                        &result_scalar_type, &result_lo,
                                        &result_hi)) {
    result_facts[0] = loom_value_facts_unknown();
    return iree_ok_status();
  }
  (void)input_scalar_type;

  int64_t exact_value = 0;
  if (loom_value_facts_as_exact_i64(operand_facts[0], &exact_value) &&
      loom_scalar_truncate_exact_i64(exact_value, result_scalar_type,
                                     &exact_value)) {
    result_facts[0] = loom_value_facts_exact_i64(exact_value);
    return iree_ok_status();
  }

  loom_value_facts_t input_facts =
      loom_value_facts_clamp_domain(operand_facts[0], input_lo, input_hi);
  if (input_facts.range_lo >= result_lo && input_facts.range_hi <= result_hi) {
    result_facts[0] = input_facts;
    return iree_ok_status();
  }
  result_facts[0] =
      loom_scalar_integer_domain_facts(operand_facts[0], result_lo, result_hi);
  return iree_ok_status();
}

iree_status_t loom_scalar_bitcast_facts(loom_fact_context_t* context,
                                        const loom_module_t* module,
                                        const loom_op_t* op,
                                        const loom_value_facts_t* operand_facts,
                                        loom_value_facts_t* result_facts) {
  loom_scalar_type_t source_type = loom_type_element_type(
      loom_module_value_type(module, loom_scalar_bitcast_input(op)));
  loom_scalar_type_t result_type = loom_scalar_result_element_type(module, op);
  loom_value_facts_eval_scalar_bitcast(source_type, result_type,
                                       &operand_facts[0], &result_facts[0]);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// scalar.assume
//===----------------------------------------------------------------------===//

iree_status_t loom_scalar_assume_facts(loom_fact_context_t* context,
                                       const loom_module_t* module,
                                       const loom_op_t* op,
                                       const loom_value_facts_t* operand_facts,
                                       loom_value_facts_t* result_facts) {
  uint16_t fact_count = op->operand_count < op->result_count ? op->operand_count
                                                             : op->result_count;
  for (uint16_t i = 0; i < fact_count; ++i) {
    result_facts[i] = operand_facts[i];
  }
  for (uint16_t i = fact_count; i < op->result_count; ++i) {
    result_facts[i] = loom_value_facts_unknown();
  }
  loom_attribute_t pred_attr = loom_op_attrs(op)[0];
  const loom_predicate_t* predicates = pred_attr.predicate_list;
  uint16_t predicate_count = pred_attr.count;
  loom_value_slice_t values = loom_scalar_assume_values(op);
  return loom_value_fact_table_apply_alias_predicates(
      context->table, values.values, fact_count, predicates, predicate_count,
      result_facts);
}

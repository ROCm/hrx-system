// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Scalar dialect op generation hooks for the test IR generator.
//
// Generates integer and float arithmetic, bitwise, comparison, conversion, and
// constant ops from the scalar dialect.

#include "loom/ops/scalar/ops.h"
#include "loom/testing/gen.h"

//===----------------------------------------------------------------------===//
// Integer binary ops
//===----------------------------------------------------------------------===//

// Function pointer type for integer binary op builders with flags.
typedef iree_status_t (*loom_scalar_int_binary_flagged_fn_t)(
    loom_builder_t* builder, uint8_t instance_flags, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t result_type, loom_location_id_t location,
    loom_op_t** out_op);

// Function pointer type for integer binary op builders without flags.
typedef iree_status_t (*loom_scalar_int_binary_fn_t)(
    loom_builder_t* builder, loom_value_id_t lhs, loom_value_id_t rhs,
    loom_type_t result_type, loom_location_id_t location, loom_op_t** out_op);

static iree_status_t loom_test_gen_hook_scalar_integer_binary(
    const loom_test_gen_hook_context_t* context, void* user_data,
    loom_test_gen_hook_result_t* out_result) {
  (void)user_data;
  loom_value_id_t lhs, rhs;
  loom_type_t type;
  if (!loom_test_gen_values_pick_binary_integer(context->gen, context->values,
                                                &lhs, &rhs, &type)) {
    *out_result = LOOM_TEST_GEN_HOOK_SKIPPED;
    return iree_ok_status();
  }

  // Flagged integer binary ops (addi, subi, muli).
  static const loom_scalar_int_binary_flagged_fn_t flagged_ops[] = {
      loom_scalar_addi_build,
      loom_scalar_subi_build,
      loom_scalar_muli_build,
  };
  // Unflagged integer binary ops.
  static const loom_scalar_int_binary_fn_t unflagged_ops[] = {
      loom_scalar_divsi_build,      loom_scalar_divui_build,
      loom_scalar_remsi_build,      loom_scalar_remui_build,
      loom_scalar_ceildivsi_build,  loom_scalar_ceildivui_build,
      loom_scalar_floordivsi_build, loom_scalar_minsi_build,
      loom_scalar_maxsi_build,      loom_scalar_minui_build,
      loom_scalar_maxui_build,
  };
  static const uint32_t flagged_count = IREE_ARRAYSIZE(flagged_ops);
  static const uint32_t unflagged_count = IREE_ARRAYSIZE(unflagged_ops);
  uint32_t total = flagged_count + unflagged_count;

  loom_op_t* op = NULL;
  uint32_t pick = loom_test_gen_next_range(context->gen, total);
  if (pick < flagged_count) {
    uint8_t flags = 0;
    if (loom_test_gen_next_probability(context->gen, 20)) {
      flags = (uint8_t)loom_test_gen_next_range(
          context->gen, (LOOM_SCALAR_INTOVERFLOWFLAGS_NSW |
                         LOOM_SCALAR_INTOVERFLOWFLAGS_NUW) +
                            1);
    }
    IREE_RETURN_IF_ERROR(flagged_ops[pick](context->builder, flags, lhs, rhs,
                                           type, LOOM_LOCATION_UNKNOWN, &op));
  } else {
    IREE_RETURN_IF_ERROR(unflagged_ops[pick - flagged_count](
        context->builder, lhs, rhs, type, LOOM_LOCATION_UNKNOWN, &op));
  }
  loom_value_id_t result_id = loom_op_results(op)[0];
  loom_test_gen_values_add(context->values, result_id, type);
  *out_result = LOOM_TEST_GEN_HOOK_EMITTED;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Integer bitwise ops
//===----------------------------------------------------------------------===//

static iree_status_t loom_test_gen_hook_scalar_integer_bitwise(
    const loom_test_gen_hook_context_t* context, void* user_data,
    loom_test_gen_hook_result_t* out_result) {
  (void)user_data;
  loom_value_id_t lhs, rhs;
  loom_type_t type;
  if (!loom_test_gen_values_pick_binary_integer(context->gen, context->values,
                                                &lhs, &rhs, &type)) {
    *out_result = LOOM_TEST_GEN_HOOK_SKIPPED;
    return iree_ok_status();
  }

  // shli is the only flagged bitwise op (overflow flags).
  static const loom_scalar_int_binary_fn_t unflagged_ops[] = {
      loom_scalar_andi_build,  loom_scalar_ori_build,   loom_scalar_xori_build,
      loom_scalar_shrsi_build, loom_scalar_shrui_build, loom_scalar_rotli_build,
      loom_scalar_rotri_build,
  };
  static const uint32_t unflagged_count = IREE_ARRAYSIZE(unflagged_ops);
  // +1 for shli.
  uint32_t total = unflagged_count + 1;

  loom_op_t* op = NULL;
  uint32_t pick = loom_test_gen_next_range(context->gen, total);
  if (pick < unflagged_count) {
    IREE_RETURN_IF_ERROR(unflagged_ops[pick](context->builder, lhs, rhs, type,
                                             LOOM_LOCATION_UNKNOWN, &op));
  } else {
    uint8_t flags = 0;
    if (loom_test_gen_next_probability(context->gen, 20)) {
      flags = (uint8_t)loom_test_gen_next_range(
          context->gen, (LOOM_SCALAR_INTOVERFLOWFLAGS_NSW |
                         LOOM_SCALAR_INTOVERFLOWFLAGS_NUW) +
                            1);
    }
    IREE_RETURN_IF_ERROR(loom_scalar_shli_build(
        context->builder, flags, lhs, rhs, type, LOOM_LOCATION_UNKNOWN, &op));
  }
  loom_test_gen_values_add(context->values, loom_op_results(op)[0], type);
  *out_result = LOOM_TEST_GEN_HOOK_EMITTED;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Integer unary ops
//===----------------------------------------------------------------------===//

typedef iree_status_t (*loom_scalar_unary_fn_t)(loom_builder_t* builder,
                                                loom_value_id_t input,
                                                loom_type_t result_type,
                                                loom_location_id_t location,
                                                loom_op_t** out_op);

static iree_status_t loom_test_gen_hook_scalar_integer_unary(
    const loom_test_gen_hook_context_t* context, void* user_data,
    loom_test_gen_hook_result_t* out_result) {
  (void)user_data;
  loom_value_id_t input =
      loom_test_gen_values_pick_integer(context->gen, context->values);
  if (input == LOOM_VALUE_ID_INVALID) {
    *out_result = LOOM_TEST_GEN_HOOK_SKIPPED;
    return iree_ok_status();
  }
  loom_type_t type = loom_test_gen_values_type_of(context->values, input);

  static const loom_scalar_unary_fn_t ops[] = {
      loom_scalar_negi_build,   loom_scalar_absi_build,
      loom_scalar_ctlzi_build,  loom_scalar_cttzi_build,
      loom_scalar_ctpopi_build,
  };
  static const uint32_t count = IREE_ARRAYSIZE(ops);

  loom_op_t* op = NULL;
  uint32_t pick = loom_test_gen_next_range(context->gen, count);
  IREE_RETURN_IF_ERROR(
      ops[pick](context->builder, input, type, LOOM_LOCATION_UNKNOWN, &op));
  loom_test_gen_values_add(context->values, loom_op_results(op)[0], type);
  *out_result = LOOM_TEST_GEN_HOOK_EMITTED;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Float binary ops
//===----------------------------------------------------------------------===//

typedef iree_status_t (*loom_scalar_float_binary_fn_t)(
    loom_builder_t* builder, uint8_t instance_flags, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t result_type, loom_location_id_t location,
    loom_op_t** out_op);

static iree_status_t loom_test_gen_hook_scalar_float_binary(
    const loom_test_gen_hook_context_t* context, void* user_data,
    loom_test_gen_hook_result_t* out_result) {
  (void)user_data;
  loom_value_id_t lhs, rhs;
  loom_type_t type;
  if (!loom_test_gen_values_pick_binary_float(context->gen, context->values,
                                              &lhs, &rhs, &type)) {
    *out_result = LOOM_TEST_GEN_HOOK_SKIPPED;
    return iree_ok_status();
  }

  // copysignf is unflagged; all other float binary ops take fast-math flags.
  static const loom_scalar_float_binary_fn_t flagged_ops[] = {
      loom_scalar_addf_build,     loom_scalar_subf_build,
      loom_scalar_mulf_build,     loom_scalar_divf_build,
      loom_scalar_remf_build,     loom_scalar_minimumf_build,
      loom_scalar_maximumf_build, loom_scalar_minnumf_build,
      loom_scalar_maxnumf_build,  loom_scalar_powf_build,
      loom_scalar_atan2f_build,
  };
  static const uint32_t flagged_count = IREE_ARRAYSIZE(flagged_ops);
  // +1 for copysignf.
  uint32_t total = flagged_count + 1;

  loom_op_t* op = NULL;
  uint32_t pick = loom_test_gen_next_range(context->gen, total);
  if (pick < flagged_count) {
    uint8_t flags = 0;
    if (loom_test_gen_next_probability(context->gen, 15)) {
      flags = (uint8_t)loom_test_gen_next_range(
          context->gen, LOOM_SCALAR_FASTMATHFLAGS_FAST + 1);
    }
    IREE_RETURN_IF_ERROR(flagged_ops[pick](context->builder, flags, lhs, rhs,
                                           type, LOOM_LOCATION_UNKNOWN, &op));
  } else {
    IREE_RETURN_IF_ERROR(loom_scalar_copysignf_build(
        context->builder, lhs, rhs, type, LOOM_LOCATION_UNKNOWN, &op));
  }
  loom_test_gen_values_add(context->values, loom_op_results(op)[0], type);
  *out_result = LOOM_TEST_GEN_HOOK_EMITTED;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Float unary ops
//===----------------------------------------------------------------------===//

typedef iree_status_t (*loom_scalar_float_unary_fn_t)(
    loom_builder_t* builder, uint8_t instance_flags, loom_value_id_t input,
    loom_type_t result_type, loom_location_id_t location, loom_op_t** out_op);

static iree_status_t loom_test_gen_hook_scalar_float_unary(
    const loom_test_gen_hook_context_t* context, void* user_data,
    loom_test_gen_hook_result_t* out_result) {
  (void)user_data;
  loom_value_id_t input =
      loom_test_gen_values_pick_float(context->gen, context->values);
  if (input == LOOM_VALUE_ID_INVALID) {
    *out_result = LOOM_TEST_GEN_HOOK_SKIPPED;
    return iree_ok_status();
  }
  loom_type_t type = loom_test_gen_values_type_of(context->values, input);

  // signf is unflagged; all other float unary ops take fast-math flags.
  static const loom_scalar_float_unary_fn_t flagged_ops[] = {
      loom_scalar_negf_build,   loom_scalar_absf_build,
      loom_scalar_expf_build,   loom_scalar_exp2f_build,
      loom_scalar_expm1f_build, loom_scalar_logf_build,
      loom_scalar_log2f_build,  loom_scalar_log10f_build,
      loom_scalar_log1pf_build, loom_scalar_sqrtf_build,
      loom_scalar_rsqrtf_build, loom_scalar_cbrtf_build,
      loom_scalar_sinf_build,   loom_scalar_cosf_build,
      loom_scalar_tanf_build,   loom_scalar_asinf_build,
      loom_scalar_acosf_build,  loom_scalar_atanf_build,
      loom_scalar_sinhf_build,  loom_scalar_coshf_build,
      loom_scalar_tanhf_build,  loom_scalar_asinhf_build,
      loom_scalar_acoshf_build, loom_scalar_atanhf_build,
      loom_scalar_erff_build,   loom_scalar_erfcf_build,
      loom_scalar_ceilf_build,  loom_scalar_floorf_build,
      loom_scalar_roundf_build, loom_scalar_roundevenf_build,
      loom_scalar_truncf_build,
  };
  static const uint32_t flagged_count = IREE_ARRAYSIZE(flagged_ops);
  // +1 for signf.
  uint32_t total = flagged_count + 1;

  loom_op_t* op = NULL;
  uint32_t pick = loom_test_gen_next_range(context->gen, total);
  if (pick < flagged_count) {
    uint8_t flags = 0;
    if (loom_test_gen_next_probability(context->gen, 15)) {
      flags = (uint8_t)loom_test_gen_next_range(
          context->gen, LOOM_SCALAR_FASTMATHFLAGS_FAST + 1);
    }
    IREE_RETURN_IF_ERROR(flagged_ops[pick](context->builder, flags, input, type,
                                           LOOM_LOCATION_UNKNOWN, &op));
  } else {
    IREE_RETURN_IF_ERROR(loom_scalar_signf_build(context->builder, input, type,
                                                 LOOM_LOCATION_UNKNOWN, &op));
  }
  loom_test_gen_values_add(context->values, loom_op_results(op)[0], type);
  *out_result = LOOM_TEST_GEN_HOOK_EMITTED;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Comparison ops
//===----------------------------------------------------------------------===//

static iree_status_t loom_test_gen_hook_scalar_comparison(
    const loom_test_gen_hook_context_t* context, void* user_data,
    loom_test_gen_hook_result_t* out_result) {
  (void)user_data;
  loom_value_id_t lhs, rhs;
  loom_type_t operand_type;
  loom_type_t i1_type = loom_type_scalar(LOOM_SCALAR_TYPE_I1);

  bool is_integer = loom_test_gen_next_bool(context->gen);
  if (is_integer) {
    if (!loom_test_gen_values_pick_binary_integer(context->gen, context->values,
                                                  &lhs, &rhs, &operand_type)) {
      *out_result = LOOM_TEST_GEN_HOOK_SKIPPED;
      return iree_ok_status();
    }
    uint8_t predicate = (uint8_t)loom_test_gen_next_range(
        context->gen, LOOM_SCALAR_CMPI_PREDICATE_COUNT_);
    loom_op_t* op = NULL;
    IREE_RETURN_IF_ERROR(loom_scalar_cmpi_build(context->builder, predicate,
                                                lhs, rhs, operand_type, i1_type,
                                                LOOM_LOCATION_UNKNOWN, &op));
    loom_test_gen_values_add(context->values, loom_op_results(op)[0], i1_type);
  } else {
    if (!loom_test_gen_values_pick_binary_float(context->gen, context->values,
                                                &lhs, &rhs, &operand_type)) {
      *out_result = LOOM_TEST_GEN_HOOK_SKIPPED;
      return iree_ok_status();
    }
    uint8_t predicate = (uint8_t)loom_test_gen_next_range(
        context->gen, LOOM_SCALAR_CMPF_PREDICATE_COUNT_);
    uint8_t flags = 0;
    if (loom_test_gen_next_probability(context->gen, 15)) {
      flags = (uint8_t)loom_test_gen_next_range(
          context->gen, LOOM_SCALAR_FASTMATHFLAGS_FAST + 1);
    }
    loom_op_t* op = NULL;
    IREE_RETURN_IF_ERROR(loom_scalar_cmpf_build(
        context->builder, flags, predicate, lhs, rhs, operand_type, i1_type,
        LOOM_LOCATION_UNKNOWN, &op));
    loom_test_gen_values_add(context->values, loom_op_results(op)[0], i1_type);
  }
  *out_result = LOOM_TEST_GEN_HOOK_EMITTED;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Conversion ops
//===----------------------------------------------------------------------===//

typedef iree_status_t (*loom_scalar_conversion_fn_t)(
    loom_builder_t* builder, loom_value_id_t input, loom_type_t input_type,
    loom_type_t result_type, loom_location_id_t location, loom_op_t** out_op);

static iree_status_t loom_test_gen_hook_scalar_conversion(
    const loom_test_gen_hook_context_t* context, void* user_data,
    loom_test_gen_hook_result_t* out_result) {
  (void)user_data;
  loom_value_id_t input =
      loom_test_gen_values_pick_any(context->gen, context->values);
  if (input == LOOM_VALUE_ID_INVALID) {
    *out_result = LOOM_TEST_GEN_HOOK_SKIPPED;
    return iree_ok_status();
  }
  loom_type_t input_type = loom_test_gen_values_type_of(context->values, input);
  if (!loom_type_is_scalar(input_type)) {
    *out_result = LOOM_TEST_GEN_HOOK_SKIPPED;
    return iree_ok_status();
  }
  loom_scalar_type_t src_scalar = loom_type_element_type(input_type);

  // Build a table of valid conversions based on the source type.
  struct {
    loom_scalar_conversion_fn_t fn;
    loom_scalar_type_t result;
  } candidates[8];
  uint32_t candidate_count = 0;

  bool src_is_integer = loom_scalar_type_is_integer(src_scalar);
  bool src_is_float = loom_scalar_type_is_float(src_scalar);

  if (src_is_integer) {
    candidates[candidate_count++] =
        (typeof(candidates[0])){loom_scalar_sitofp_build, LOOM_SCALAR_TYPE_F32};
    candidates[candidate_count++] =
        (typeof(candidates[0])){loom_scalar_sitofp_build, LOOM_SCALAR_TYPE_F64};
    if (loom_scalar_type_bitwidth(src_scalar) < 64) {
      candidates[candidate_count++] = (typeof(candidates[0])){
          loom_scalar_extsi_build, LOOM_SCALAR_TYPE_I64};
      candidates[candidate_count++] = (typeof(candidates[0])){
          loom_scalar_extui_build, LOOM_SCALAR_TYPE_I64};
    }
    if (loom_scalar_type_bitwidth(src_scalar) > 8) {
      candidates[candidate_count++] = (typeof(candidates[0])){
          loom_scalar_trunci_build, LOOM_SCALAR_TYPE_I8};
    }
  } else if (src_is_float) {
    candidates[candidate_count++] =
        (typeof(candidates[0])){loom_scalar_fptosi_build, LOOM_SCALAR_TYPE_I32};
    candidates[candidate_count++] =
        (typeof(candidates[0])){loom_scalar_fptoui_build, LOOM_SCALAR_TYPE_I32};
    if (loom_scalar_type_bitwidth(src_scalar) < 64) {
      candidates[candidate_count++] =
          (typeof(candidates[0])){loom_scalar_extf_build, LOOM_SCALAR_TYPE_F64};
    }
    if (loom_scalar_type_bitwidth(src_scalar) > 16) {
      candidates[candidate_count++] = (typeof(candidates[0])){
          loom_scalar_fptrunc_build, LOOM_SCALAR_TYPE_F16};
    }
  }

  if (candidate_count == 0) {
    *out_result = LOOM_TEST_GEN_HOOK_SKIPPED;
    return iree_ok_status();
  }

  uint32_t pick = loom_test_gen_next_range(context->gen, candidate_count);
  loom_type_t result_type = loom_type_scalar(candidates[pick].result);
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(candidates[pick].fn(context->builder, input, input_type,
                                           result_type, LOOM_LOCATION_UNKNOWN,
                                           &op));
  loom_test_gen_values_add(context->values, loom_op_results(op)[0],
                           result_type);
  *out_result = LOOM_TEST_GEN_HOOK_EMITTED;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Constant ops
//===----------------------------------------------------------------------===//

static bool loom_test_gen_type_palette_pick_scalar_constant_type(
    loom_test_gen_t* gen, const loom_test_gen_type_palette_t* palette,
    loom_type_t* out_type) {
  uint32_t matching_weight = 0;
  for (uint16_t i = 0; i < palette->count; ++i) {
    loom_scalar_type_t scalar_type = palette->types[i];
    if (loom_scalar_type_is_integer(scalar_type) ||
        loom_scalar_type_is_float(scalar_type)) {
      matching_weight += palette->weights[i];
    }
  }
  if (matching_weight == 0) return false;

  uint32_t target = loom_test_gen_next_range(gen, matching_weight);
  uint32_t cumulative = 0;
  for (uint16_t i = 0; i < palette->count; ++i) {
    loom_scalar_type_t scalar_type = palette->types[i];
    if (!loom_scalar_type_is_integer(scalar_type) &&
        !loom_scalar_type_is_float(scalar_type)) {
      continue;
    }
    cumulative += palette->weights[i];
    if (target < cumulative) {
      *out_type = loom_type_scalar(scalar_type);
      return true;
    }
  }

  return false;
}

static iree_status_t loom_test_gen_hook_scalar_constant(
    const loom_test_gen_hook_context_t* context, void* user_data,
    loom_test_gen_hook_result_t* out_result) {
  (void)user_data;
  loom_type_t type = loom_type_none();
  if (!loom_test_gen_type_palette_pick_scalar_constant_type(
          context->gen, context->palette, &type)) {
    *out_result = LOOM_TEST_GEN_HOOK_SKIPPED;
    return iree_ok_status();
  }

  loom_attribute_t value = loom_test_gen_constant_attr(
      type, (int64_t)loom_test_gen_next_range(context->gen, 200) - 100);

  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_constant_build(context->builder, value, type,
                                                  LOOM_LOCATION_UNKNOWN, &op));
  loom_test_gen_values_add(context->values, loom_op_results(op)[0], type);
  *out_result = LOOM_TEST_GEN_HOOK_EMITTED;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Hook table
//===----------------------------------------------------------------------===//

static const loom_test_gen_op_hook_t loom_test_gen_scalar_hooks_table[] = {
    {10, loom_test_gen_hook_scalar_integer_binary, NULL, NULL},
    {4, loom_test_gen_hook_scalar_integer_bitwise, NULL, NULL},
    {3, loom_test_gen_hook_scalar_integer_unary, NULL, NULL},
    {8, loom_test_gen_hook_scalar_float_binary, NULL, NULL},
    {6, loom_test_gen_hook_scalar_float_unary, NULL, NULL},
    {4, loom_test_gen_hook_scalar_comparison, NULL, NULL},
    {3, loom_test_gen_hook_scalar_conversion, NULL, NULL},
    {2, loom_test_gen_hook_scalar_constant, NULL, NULL},
};

const loom_test_gen_op_hook_t* loom_test_gen_scalar_hooks(
    iree_host_size_t* out_hook_count) {
  *out_hook_count = sizeof(loom_test_gen_scalar_hooks_table) /
                    sizeof(loom_test_gen_scalar_hooks_table[0]);
  return loom_test_gen_scalar_hooks_table;
}

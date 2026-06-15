// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Vector dialect op generation hooks for the test IR generator.
//
// Generates target-agnostic vector construction, lanewise arithmetic, and
// scalar extraction. Memory and target-specific vector forms are kept out of
// this core hook set so generator tests do not need optional target providers.

#include "loom/ops/vector/ops.h"
#include "loom/testing/gen.h"

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

static loom_type_t loom_test_gen_vector_static_1d_type(
    loom_test_gen_t* gen, loom_scalar_type_t element_type) {
  static const uint64_t kLaneCounts[] = {1, 2, 4, 8};
  uint64_t lane_count =
      kLaneCounts[loom_test_gen_next_range(gen, IREE_ARRAYSIZE(kLaneCounts))];
  return loom_type_shaped_1d(LOOM_TYPE_VECTOR, element_type,
                             loom_dim_pack_static(lane_count), 0);
}

static bool loom_test_gen_vector_element_satisfies(
    loom_type_t type, loom_type_constraint_t constraint) {
  if (!loom_type_is_vector(type)) {
    return false;
  }
  loom_scalar_type_t element_type = loom_type_element_type(type);
  if (element_type == LOOM_SCALAR_TYPE_I1 &&
      constraint == LOOM_TYPE_CONSTRAINT_INTEGER) {
    return false;
  }
  return loom_type_satisfies_constraint(loom_type_scalar(element_type),
                                        constraint);
}

static loom_value_id_t loom_test_gen_values_pick_vector_for_element_constraint(
    loom_test_gen_t* gen, const loom_test_gen_values_t* values,
    loom_type_constraint_t constraint, loom_type_t* out_type) {
  uint16_t candidate_count = 0;
  for (uint16_t i = 0; i < values->count; ++i) {
    if (loom_test_gen_vector_element_satisfies(values->types[i], constraint)) {
      ++candidate_count;
    }
  }
  if (candidate_count == 0) {
    return LOOM_VALUE_ID_INVALID;
  }
  uint32_t pick = loom_test_gen_next_range(gen, candidate_count);
  for (uint16_t i = 0; i < values->count; ++i) {
    if (!loom_test_gen_vector_element_satisfies(values->types[i], constraint)) {
      continue;
    }
    if (pick == 0) {
      *out_type = values->types[i];
      return values->entries[i];
    }
    --pick;
  }
  return LOOM_VALUE_ID_INVALID;
}

//===----------------------------------------------------------------------===//
// vector.splat and vector.from_elements
//===----------------------------------------------------------------------===//

static iree_status_t loom_test_gen_hook_vector_splat(
    const loom_test_gen_hook_context_t* context, void* user_data,
    loom_test_gen_hook_result_t* out_result) {
  (void)user_data;
  loom_type_t scalar_type = {0};
  loom_value_id_t scalar = loom_test_gen_values_pick_for_constraint(
      context->gen, context->values, LOOM_TYPE_CONSTRAINT_SCALAR, &scalar_type);
  if (scalar == LOOM_VALUE_ID_INVALID) {
    *out_result = LOOM_TEST_GEN_HOOK_SKIPPED;
    return iree_ok_status();
  }

  loom_type_t result_type = loom_test_gen_vector_static_1d_type(
      context->gen, loom_type_element_type(scalar_type));
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_splat_build(
      context->builder, scalar, result_type, LOOM_LOCATION_UNKNOWN, &op));
  loom_test_gen_values_add(context->values, loom_op_results(op)[0],
                           result_type);
  *out_result = LOOM_TEST_GEN_HOOK_EMITTED;
  return iree_ok_status();
}

static iree_status_t loom_test_gen_hook_vector_from_elements(
    const loom_test_gen_hook_context_t* context, void* user_data,
    loom_test_gen_hook_result_t* out_result) {
  (void)user_data;
  loom_type_t scalar_type = {0};
  loom_value_id_t first = loom_test_gen_values_pick_for_constraint(
      context->gen, context->values, LOOM_TYPE_CONSTRAINT_SCALAR, &scalar_type);
  if (first == LOOM_VALUE_ID_INVALID) {
    *out_result = LOOM_TEST_GEN_HOOK_SKIPPED;
    return iree_ok_status();
  }

  static const uint64_t kLaneCounts[] = {2, 4};
  uint64_t lane_count = kLaneCounts[loom_test_gen_next_range(
      context->gen, IREE_ARRAYSIZE(kLaneCounts))];
  loom_scalar_type_t element_type = loom_type_element_type(scalar_type);
  loom_type_t result_type = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, element_type, loom_dim_pack_static(lane_count), 0);

  loom_value_id_t elements[4];
  elements[0] = first;
  for (uint64_t i = 1; i < lane_count; ++i) {
    elements[i] = loom_test_gen_values_pick_typed(context->gen, context->values,
                                                  element_type);
    if (elements[i] == LOOM_VALUE_ID_INVALID) {
      elements[i] = first;
    }
  }

  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_vector_from_elements_build(context->builder, elements, lane_count,
                                      result_type, LOOM_LOCATION_UNKNOWN, &op));
  loom_test_gen_values_add(context->values, loom_op_results(op)[0],
                           result_type);
  *out_result = LOOM_TEST_GEN_HOOK_EMITTED;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Lanewise vector arithmetic
//===----------------------------------------------------------------------===//

typedef iree_status_t (*loom_vector_int_binary_flagged_fn_t)(
    loom_builder_t* builder, uint8_t instance_flags, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t result_type, loom_location_id_t location,
    loom_op_t** out_op);

typedef iree_status_t (*loom_vector_int_binary_fn_t)(
    loom_builder_t* builder, loom_value_id_t lhs, loom_value_id_t rhs,
    loom_type_t result_type, loom_location_id_t location, loom_op_t** out_op);

static iree_status_t loom_test_gen_hook_vector_integer_binary(
    const loom_test_gen_hook_context_t* context, void* user_data,
    loom_test_gen_hook_result_t* out_result) {
  (void)user_data;
  loom_type_t type = {0};
  loom_value_id_t lhs = loom_test_gen_values_pick_vector_for_element_constraint(
      context->gen, context->values, LOOM_TYPE_CONSTRAINT_INTEGER, &type);
  if (lhs == LOOM_VALUE_ID_INVALID) {
    *out_result = LOOM_TEST_GEN_HOOK_SKIPPED;
    return iree_ok_status();
  }
  loom_value_id_t rhs =
      loom_test_gen_values_pick_exact_type(context->gen, context->values, type);
  if (rhs == LOOM_VALUE_ID_INVALID) {
    rhs = lhs;
  }

  static const loom_vector_int_binary_flagged_fn_t flagged_ops[] = {
      loom_vector_addi_build,
      loom_vector_subi_build,
      loom_vector_muli_build,
  };
  static const loom_vector_int_binary_fn_t unflagged_ops[] = {
      loom_vector_andi_build,  loom_vector_ori_build,   loom_vector_xori_build,
      loom_vector_minsi_build, loom_vector_maxsi_build, loom_vector_minui_build,
      loom_vector_maxui_build,
  };
  uint32_t flagged_count = IREE_ARRAYSIZE(flagged_ops);
  uint32_t total = flagged_count + IREE_ARRAYSIZE(unflagged_ops);
  uint32_t pick = loom_test_gen_next_range(context->gen, total);

  loom_op_t* op = NULL;
  if (pick < flagged_count) {
    uint8_t flags = 0;
    if (loom_test_gen_next_probability(context->gen, 20)) {
      flags = (uint8_t)loom_test_gen_next_range(
          context->gen, (LOOM_VECTOR_INTOVERFLOWFLAGS_NSW |
                         LOOM_VECTOR_INTOVERFLOWFLAGS_NUW) +
                            1);
    }
    IREE_RETURN_IF_ERROR(flagged_ops[pick](context->builder, flags, lhs, rhs,
                                           type, LOOM_LOCATION_UNKNOWN, &op));
  } else {
    IREE_RETURN_IF_ERROR(unflagged_ops[pick - flagged_count](
        context->builder, lhs, rhs, type, LOOM_LOCATION_UNKNOWN, &op));
  }
  loom_test_gen_values_add(context->values, loom_op_results(op)[0], type);
  *out_result = LOOM_TEST_GEN_HOOK_EMITTED;
  return iree_ok_status();
}

typedef iree_status_t (*loom_vector_float_binary_fn_t)(
    loom_builder_t* builder, uint8_t instance_flags, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t result_type, loom_location_id_t location,
    loom_op_t** out_op);

static iree_status_t loom_test_gen_hook_vector_float_binary(
    const loom_test_gen_hook_context_t* context, void* user_data,
    loom_test_gen_hook_result_t* out_result) {
  (void)user_data;
  loom_type_t type = {0};
  loom_value_id_t lhs = loom_test_gen_values_pick_vector_for_element_constraint(
      context->gen, context->values, LOOM_TYPE_CONSTRAINT_FLOAT, &type);
  if (lhs == LOOM_VALUE_ID_INVALID) {
    *out_result = LOOM_TEST_GEN_HOOK_SKIPPED;
    return iree_ok_status();
  }
  loom_value_id_t rhs =
      loom_test_gen_values_pick_exact_type(context->gen, context->values, type);
  if (rhs == LOOM_VALUE_ID_INVALID) {
    rhs = lhs;
  }

  static const loom_vector_float_binary_fn_t ops[] = {
      loom_vector_addf_build,     loom_vector_subf_build,
      loom_vector_mulf_build,     loom_vector_divf_build,
      loom_vector_minimumf_build, loom_vector_maximumf_build,
      loom_vector_minnumf_build,  loom_vector_maxnumf_build,
  };
  uint8_t flags = 0;
  if (loom_test_gen_next_probability(context->gen, 15)) {
    flags = (uint8_t)loom_test_gen_next_range(
        context->gen, LOOM_VECTOR_FASTMATHFLAGS_FAST + 1);
  }
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(
      ops[loom_test_gen_next_range(context->gen, IREE_ARRAYSIZE(ops))](
          context->builder, flags, lhs, rhs, type, LOOM_LOCATION_UNKNOWN, &op));
  loom_test_gen_values_add(context->values, loom_op_results(op)[0], type);
  *out_result = LOOM_TEST_GEN_HOOK_EMITTED;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// vector.extract
//===----------------------------------------------------------------------===//

static iree_status_t loom_test_gen_hook_vector_extract(
    const loom_test_gen_hook_context_t* context, void* user_data,
    loom_test_gen_hook_result_t* out_result) {
  (void)user_data;
  uint16_t candidate_count = 0;
  for (uint16_t i = 0; i < context->values->count; ++i) {
    loom_type_t type = context->values->types[i];
    uint64_t element_count = 0;
    if (loom_type_is_vector(type) && loom_type_rank(type) == 1 &&
        loom_type_static_element_count(type, &element_count) &&
        element_count > 0) {
      ++candidate_count;
    }
  }
  if (candidate_count == 0) {
    *out_result = LOOM_TEST_GEN_HOOK_SKIPPED;
    return iree_ok_status();
  }

  uint32_t pick = loom_test_gen_next_range(context->gen, candidate_count);
  loom_value_id_t source = LOOM_VALUE_ID_INVALID;
  loom_type_t source_type = {0};
  uint64_t element_count = 0;
  for (uint16_t i = 0; i < context->values->count; ++i) {
    loom_type_t type = context->values->types[i];
    uint64_t candidate_element_count = 0;
    if (!loom_type_is_vector(type) || loom_type_rank(type) != 1 ||
        !loom_type_static_element_count(type, &candidate_element_count) ||
        candidate_element_count == 0) {
      continue;
    }
    if (pick == 0) {
      source = context->values->entries[i];
      source_type = type;
      element_count = candidate_element_count;
      break;
    }
    --pick;
  }
  if (source == LOOM_VALUE_ID_INVALID) {
    *out_result = LOOM_TEST_GEN_HOOK_SKIPPED;
    return iree_ok_status();
  }

  int64_t* static_indices = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(context->builder->arena, 1,
                                                 sizeof(*static_indices),
                                                 (void**)&static_indices));
  static_indices[0] =
      (int64_t)loom_test_gen_next_range(context->gen, (uint32_t)element_count);
  loom_type_t result_type =
      loom_type_scalar(loom_type_element_type(source_type));
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_extract_build(
      context->builder, source, NULL, 0, static_indices, 1, result_type,
      LOOM_LOCATION_UNKNOWN, &op));
  loom_test_gen_values_add(context->values, loom_op_results(op)[0],
                           result_type);
  *out_result = LOOM_TEST_GEN_HOOK_EMITTED;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Hook table
//===----------------------------------------------------------------------===//

static const loom_test_gen_op_hook_t loom_test_gen_vector_hooks_table[] = {
    {6, loom_test_gen_hook_vector_splat, NULL, NULL},
    {3, loom_test_gen_hook_vector_from_elements, NULL, NULL},
    {8, loom_test_gen_hook_vector_integer_binary, NULL, NULL},
    {6, loom_test_gen_hook_vector_float_binary, NULL, NULL},
    {3, loom_test_gen_hook_vector_extract, NULL, NULL},
};

const loom_test_gen_op_hook_t* loom_test_gen_vector_hooks(
    iree_host_size_t* out_hook_count) {
  *out_hook_count = sizeof(loom_test_gen_vector_hooks_table) /
                    sizeof(loom_test_gen_vector_hooks_table[0]);
  return loom_test_gen_vector_hooks_table;
}

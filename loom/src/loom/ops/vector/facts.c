// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Fact implementations for the vector dialect.
//
// Vector facts deliberately summarize register values instead of interpreting
// every lane. Uniform-element facts let construction and lanewise ops preserve
// "all lanes have the same scalar facts"; reductions and dot products can then
// fold to scalar facts without teaching the canonicalizer to materialize vector
// constants. Iota and prefix-mask facts keep structural vector producers
// visible to later lowering/fact consumers without making every pass walk
// vector lanes.

#include <math.h>
#include <stdint.h>
#include <string.h>

#include "iree/base/internal/math.h"
#include "loom/ir/attribute.h"
#include "loom/ir/float_facts.h"
#include "loom/ir/module.h"
#include "loom/ops/combining.h"
#include "loom/ops/encoding/hadamard.h"
#include "loom/ops/encoding/storage.h"
#include "loom/ops/scalar/compare.h"
#include "loom/ops/vector/fragment.h"
#include "loom/ops/vector/ops.h"
#include "loom/util/fact_table.h"
#include "loom/util/numeric_format.h"

#define LOOM_VECTOR_FACT_STATIC_LOOP_LIMIT 1024

typedef void (*loom_vector_unary_transfer_fn_t)(const loom_value_facts_t* input,
                                                loom_value_facts_t* out);
typedef void (*loom_vector_integer_binary_transfer_fn_t)(
    const loom_value_facts_t* lhs, const loom_value_facts_t* rhs,
    loom_value_facts_t* out);
typedef void (*loom_vector_ternary_transfer_fn_t)(const loom_value_facts_t* a,
                                                  const loom_value_facts_t* b,
                                                  const loom_value_facts_t* c,
                                                  loom_value_facts_t* out);
typedef int (*loom_vector_bit_count_fn_t)(uint64_t value, int32_t bitwidth);
typedef void (*loom_vector_float_unary_fact_transfer_fn_t)(
    loom_scalar_type_t scalar_type, const loom_value_facts_t* input,
    const void* user_data, loom_value_facts_t* out);
typedef void (*loom_vector_float_binary_fact_transfer_fn_t)(
    loom_scalar_type_t scalar_type, const loom_value_facts_t* lhs,
    const loom_value_facts_t* rhs, const void* user_data,
    loom_value_facts_t* out);
typedef void (*loom_vector_float_ternary_fact_transfer_fn_t)(
    loom_scalar_type_t scalar_type, const loom_value_facts_t* a,
    const loom_value_facts_t* b, const loom_value_facts_t* c,
    const void* user_data, loom_value_facts_t* out);

//===----------------------------------------------------------------------===//
// Scalar element helpers
//===----------------------------------------------------------------------===//

static loom_scalar_type_t loom_vector_result_element_type(
    const loom_module_t* module, const loom_op_t* op) {
  return loom_type_element_type(
      loom_module_value_type(module, loom_op_const_results(op)[0]));
}

static float loom_vector_add_f32(float lhs, float rhs) { return lhs + rhs; }
static double loom_vector_add_f64(double lhs, double rhs) { return lhs + rhs; }

static float loom_vector_sub_f32(float lhs, float rhs) { return lhs - rhs; }
static double loom_vector_sub_f64(double lhs, double rhs) { return lhs - rhs; }

static float loom_vector_mul_f32(float lhs, float rhs) { return lhs * rhs; }
static double loom_vector_mul_f64(double lhs, double rhs) { return lhs * rhs; }

static float loom_vector_div_f32(float lhs, float rhs) { return lhs / rhs; }
static double loom_vector_div_f64(double lhs, double rhs) { return lhs / rhs; }

static bool loom_vector_facts_query_uniform_element(
    const loom_fact_context_t* context, loom_value_facts_t facts,
    loom_value_facts_t* out_element) {
  loom_value_fact_uniform_element_t uniform = {0};
  if (!loom_value_facts_query_uniform_element(context, facts, &uniform)) {
    return false;
  }
  *out_element = uniform.element;
  return true;
}

static bool loom_vector_facts_query_small_lanes(
    const loom_fact_context_t* context, loom_value_facts_t facts,
    loom_value_fact_small_static_lanes_t* out_lanes) {
  return loom_value_facts_query_small_static_lanes(context, facts, out_lanes);
}

static bool loom_vector_facts_query_iota_lane(
    const loom_fact_context_t* context, loom_value_facts_t facts,
    iree_host_size_t lane, loom_value_facts_t* out_element) {
  loom_value_fact_vector_iota_t iota = {0};
  if (!loom_value_facts_query_vector_iota(context, facts, &iota)) {
    return false;
  }
  int64_t base = 0;
  int64_t step = 0;
  if (!loom_value_facts_as_exact_i64(iota.base, &base) ||
      !loom_value_facts_as_exact_i64(iota.step, &step) ||
      lane > (iree_host_size_t)INT64_MAX) {
    return false;
  }
  int64_t delta = 0;
  int64_t value = 0;
  if (!iree_checked_mul_i64((int64_t)lane, step, &delta) ||
      !iree_checked_add_i64(base, delta, &value)) {
    return false;
  }
  *out_element = loom_value_facts_exact_i64(value);
  return true;
}

static bool loom_vector_facts_query_lane(const loom_fact_context_t* context,
                                         loom_value_facts_t facts,
                                         iree_host_size_t lane,
                                         loom_value_facts_t* out_element) {
  if (loom_vector_facts_query_uniform_element(context, facts, out_element)) {
    return true;
  }
  loom_value_fact_small_static_lanes_t lanes = {0};
  if (!loom_vector_facts_query_small_lanes(context, facts, &lanes) ||
      lane >= lanes.count) {
    return loom_vector_facts_query_iota_lane(context, facts, lane, out_element);
  }
  *out_element = lanes.lanes[lane];
  return true;
}

static bool loom_vector_facts_query_binary_lane_count(
    const loom_fact_context_t* context, loom_value_facts_t lhs,
    loom_value_facts_t rhs, iree_host_size_t* out_lane_count) {
  loom_value_fact_small_static_lanes_t lhs_lanes = {0};
  loom_value_fact_small_static_lanes_t rhs_lanes = {0};
  bool lhs_is_small =
      loom_vector_facts_query_small_lanes(context, lhs, &lhs_lanes);
  bool rhs_is_small =
      loom_vector_facts_query_small_lanes(context, rhs, &rhs_lanes);
  if (lhs_is_small && rhs_is_small) {
    if (lhs_lanes.count != rhs_lanes.count) return false;
    *out_lane_count = lhs_lanes.count;
    return true;
  }
  if (lhs_is_small) {
    *out_lane_count = lhs_lanes.count;
    return true;
  }
  if (rhs_is_small) {
    *out_lane_count = rhs_lanes.count;
    return true;
  }
  return false;
}

static bool loom_vector_facts_query_ternary_lane_count(
    const loom_fact_context_t* context, loom_value_facts_t a,
    loom_value_facts_t b, loom_value_facts_t c,
    iree_host_size_t* out_lane_count) {
  loom_value_fact_small_static_lanes_t lane_sets[3] = {{0}};
  loom_value_facts_t facts[3] = {a, b, c};
  bool found_count = false;
  iree_host_size_t lane_count = 0;
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(facts); ++i) {
    if (!loom_vector_facts_query_small_lanes(context, facts[i],
                                             &lane_sets[i])) {
      continue;
    }
    if (found_count && lane_sets[i].count != lane_count) return false;
    lane_count = lane_sets[i].count;
    found_count = true;
  }
  if (!found_count) return false;
  *out_lane_count = lane_count;
  return true;
}

static loom_value_facts_t loom_vector_attr_element_facts(
    loom_attribute_t attr, loom_scalar_type_t element_type) {
  if (loom_scalar_type_is_float(element_type)) {
    return loom_value_facts_exact_float(element_type, loom_attr_as_f64(attr));
  }
  if (element_type == LOOM_SCALAR_TYPE_I1 && attr.kind == LOOM_ATTR_BOOL) {
    return loom_value_facts_exact_i64(loom_attr_as_bool(attr) ? 1 : 0);
  }
  return loom_value_facts_exact_i64(loom_attr_as_i64(attr));
}

static bool loom_vector_mask_range_exact_lane(int64_t lower_bound,
                                              int64_t upper_bound, int64_t step,
                                              uint64_t lane_ordinal,
                                              bool* out_value) {
  if (lane_ordinal > (uint64_t)INT64_MAX) return false;
  int64_t lane_delta = 0;
  if (!iree_checked_mul_i64((int64_t)lane_ordinal, step, &lane_delta)) {
    return false;
  }
  int64_t lane_value = 0;
  if (!iree_checked_add_i64(lower_bound, lane_delta, &lane_value)) {
    return false;
  }
  *out_value = lane_value < upper_bound;
  return true;
}

static iree_status_t loom_vector_mask_range_exact_static_facts(
    loom_fact_context_t* context, uint64_t lane_count, int64_t lower_bound,
    int64_t upper_bound, int64_t step, loom_value_facts_t* out_facts,
    bool* out_handled) {
  *out_handled = true;
  if (lane_count == 0) {
    return loom_value_facts_make_uniform_element(
        context, loom_value_facts_exact_i64(0), out_facts);
  }

  bool first_value = false;
  if (!loom_vector_mask_range_exact_lane(lower_bound, upper_bound, step, 0,
                                         &first_value)) {
    *out_facts = loom_value_facts_unknown();
    return iree_ok_status();
  }

  bool last_value = first_value;
  if (lane_count > 1) {
    if (!loom_vector_mask_range_exact_lane(lower_bound, upper_bound, step,
                                           lane_count - 1, &last_value)) {
      *out_facts = loom_value_facts_unknown();
      return iree_ok_status();
    }
  }

  if (first_value == last_value) {
    return loom_value_facts_make_uniform_element(
        context, loom_value_facts_exact_i64(first_value ? 1 : 0), out_facts);
  }

  if (lane_count > LOOM_VALUE_FACT_SMALL_STATIC_LANE_LIMIT) {
    *out_handled = false;
    return iree_ok_status();
  }

  loom_value_facts_t lanes[LOOM_VALUE_FACT_SMALL_STATIC_LANE_LIMIT] = {{0}};
  for (uint64_t i = 0; i < lane_count; ++i) {
    bool lane_value = false;
    if (!loom_vector_mask_range_exact_lane(lower_bound, upper_bound, step, i,
                                           &lane_value)) {
      *out_facts = loom_value_facts_unknown();
      return iree_ok_status();
    }
    lanes[i] = loom_value_facts_exact_i64(lane_value ? 1 : 0);
  }

  loom_value_fact_small_static_lanes_t lane_slice = {
      .lanes = lanes,
      .count = (iree_host_size_t)lane_count,
  };
  return loom_value_facts_make_small_static_lanes(context, lane_slice,
                                                  out_facts);
}

static iree_status_t loom_vector_mask_range_bounded_static_facts(
    loom_fact_context_t* context, uint64_t lane_count,
    loom_value_facts_t lower_bound, loom_value_facts_t upper_bound,
    loom_value_facts_t step, loom_value_facts_t* out_facts, bool* out_handled) {
  *out_handled = false;
  if (lane_count == 0) {
    *out_handled = true;
    return loom_value_facts_make_uniform_element(
        context, loom_value_facts_exact_i64(0), out_facts);
  }
  if (lane_count - 1 > (uint64_t)INT64_MAX ||
      loom_value_facts_is_float(lower_bound) ||
      loom_value_facts_is_float(upper_bound) ||
      loom_value_facts_is_float(step)) {
    return iree_ok_status();
  }

  int64_t maximum_lane_delta = 0;
  int64_t maximum_step = iree_max(step.range_hi, 0);
  int64_t maximum_lane_value = 0;
  if (iree_checked_mul_i64((int64_t)(lane_count - 1), maximum_step,
                           &maximum_lane_delta) &&
      iree_checked_add_i64(lower_bound.range_hi, maximum_lane_delta,
                           &maximum_lane_value) &&
      maximum_lane_value < upper_bound.range_lo) {
    *out_handled = true;
    return loom_value_facts_make_uniform_element(
        context, loom_value_facts_exact_i64(1), out_facts);
  }

  int64_t minimum_lane_delta = 0;
  int64_t minimum_step = iree_min(step.range_lo, 0);
  int64_t minimum_lane_value = 0;
  if (iree_checked_mul_i64((int64_t)(lane_count - 1), minimum_step,
                           &minimum_lane_delta) &&
      iree_checked_add_i64(lower_bound.range_lo, minimum_lane_delta,
                           &minimum_lane_value) &&
      minimum_lane_value >= upper_bound.range_hi) {
    *out_handled = true;
    return loom_value_facts_make_uniform_element(
        context, loom_value_facts_exact_i64(0), out_facts);
  }

  return iree_ok_status();
}

static bool loom_vector_facts_exact_i64_is(loom_value_facts_t facts,
                                           int64_t expected) {
  return loom_value_facts_is_exact(facts) &&
         !loom_value_facts_is_float(facts) && facts.range_lo == expected;
}

static bool loom_vector_facts_query_exact_float(loom_scalar_type_t scalar_type,
                                                loom_value_facts_t facts,
                                                double* out_value) {
  return loom_value_facts_as_exact_float(scalar_type, facts, out_value);
}

static bool loom_vector_facts_query_exact_i64(loom_value_facts_t facts,
                                              int64_t* out_value) {
  if (!loom_value_facts_is_exact(facts) || loom_value_facts_is_float(facts)) {
    return false;
  }
  *out_value = facts.range_lo;
  return true;
}

static bool loom_vector_facts_query_exact_i32(loom_value_facts_t facts,
                                              int32_t* out_value) {
  int64_t value = 0;
  if (!loom_vector_facts_query_exact_i64(facts, &value) || value < INT32_MIN ||
      value > INT32_MAX) {
    return false;
  }
  *out_value = (int32_t)value;
  return true;
}

static bool loom_vector_facts_query_exact_u32_bits(loom_value_facts_t facts,
                                                   uint32_t* out_value) {
  int64_t value = 0;
  if (!loom_vector_facts_query_exact_i64(facts, &value) || value < INT32_MIN ||
      value > UINT32_MAX) {
    return false;
  }
  *out_value = (uint32_t)value;
  return true;
}

static bool loom_vector_type_static_lane_count(loom_type_t type,
                                               iree_host_size_t* out_count) {
  uint64_t count = 0;
  if (!loom_type_static_element_count(type, &count) ||
      count > (uint64_t)IREE_HOST_SIZE_MAX) {
    return false;
  }
  *out_count = (iree_host_size_t)count;
  return true;
}

static void loom_vector_static_indices_from_ordinal(loom_type_t type,
                                                    iree_host_size_t ordinal,
                                                    int64_t* indices) {
  uint8_t rank = loom_type_rank(type);
  for (uint8_t reverse_axis = 0; reverse_axis < rank; ++reverse_axis) {
    uint8_t axis = (uint8_t)(rank - reverse_axis - 1);
    uint64_t dimension_size =
        (uint64_t)loom_type_dim_static_size_at(type, axis);
    indices[axis] =
        dimension_size == 0 ? 0 : (int64_t)(ordinal % dimension_size);
    if (dimension_size != 0) ordinal /= dimension_size;
  }
}

static bool loom_vector_static_ordinal_from_indices(
    loom_type_t type, const int64_t* indices, iree_host_size_t* out_ordinal) {
  iree_host_size_t ordinal = 0;
  uint8_t rank = loom_type_rank(type);
  for (uint8_t axis = 0; axis < rank; ++axis) {
    int64_t dimension_size = loom_type_dim_static_size_at(type, axis);
    if (dimension_size < 0 || indices[axis] < 0 ||
        indices[axis] >= dimension_size) {
      return false;
    }
    if (!iree_host_size_checked_mul(ordinal, (iree_host_size_t)dimension_size,
                                    &ordinal) ||
        !iree_host_size_checked_add(ordinal, (iree_host_size_t)indices[axis],
                                    &ordinal)) {
      return false;
    }
  }
  *out_ordinal = ordinal;
  return true;
}

static iree_status_t loom_vector_make_small_static_lane_facts(
    loom_fact_context_t* context, const loom_value_facts_t* lanes,
    iree_host_size_t count, loom_value_facts_t* out_facts) {
  loom_value_fact_small_static_lanes_t lane_slice = {
      .lanes = lanes,
      .count = count,
  };
  return loom_value_facts_make_small_static_lanes(context, lane_slice,
                                                  out_facts);
}

static iree_status_t loom_vector_make_unknown_facts(
    loom_value_facts_t* result_facts) {
  result_facts[0] = loom_value_facts_unknown();
  return iree_ok_status();
}

static iree_status_t loom_vector_make_unknown_result_facts(
    loom_value_facts_t* result_facts, uint16_t result_count) {
  for (uint16_t i = 0; i < result_count; ++i) {
    result_facts[i] = loom_value_facts_unknown();
  }
  return iree_ok_status();
}

static loom_vector_fragment_fact_t
loom_vector_fragment_fact_dense_interpretation(
    loom_vector_fragment_fact_t source) {
  loom_vector_fragment_fact_t dense;
  loom_vector_fragment_fact_initialize(&dense);
  dense.role_flags = source.role_flags;
  dense.shape_rank = source.shape_rank;
  memcpy(dense.shape_value_ids, source.shape_value_ids,
         sizeof(dense.shape_value_ids));
  return dense;
}

static bool loom_vector_fragment_fact_set_shape(
    loom_value_id_t blocks, loom_value_id_t rows, loom_value_id_t columns,
    loom_vector_fragment_fact_t* out_fact) {
  if (rows == LOOM_VALUE_ID_INVALID || columns == LOOM_VALUE_ID_INVALID) {
    return false;
  }
  out_fact->shape_rank = blocks == LOOM_VALUE_ID_INVALID ? 2 : 3;
  if (out_fact->shape_rank == 3) {
    out_fact->shape_value_ids[0] = blocks;
  }
  out_fact->shape_value_ids[out_fact->shape_rank - 2] = rows;
  out_fact->shape_value_ids[out_fact->shape_rank - 1] = columns;
  return true;
}

static bool loom_vector_fragment_facts_match_except_role(
    loom_vector_fragment_fact_t lhs, loom_vector_fragment_fact_t rhs) {
  lhs.role_flags = 0;
  rhs.role_flags = 0;
  return loom_vector_fragment_fact_equal(lhs, rhs);
}

static bool loom_vector_fragment_facts_match_contract_except_native_storage(
    loom_vector_fragment_fact_t lhs, loom_vector_fragment_fact_t rhs) {
  lhs.flags &= ~LOOM_VECTOR_FRAGMENT_FACT_FLAG_HAS_NATIVE_STORAGE;
  rhs.flags &= ~LOOM_VECTOR_FRAGMENT_FACT_FLAG_HAS_NATIVE_STORAGE;
  return loom_vector_fragment_fact_equal(lhs, rhs);
}

static bool loom_vector_fragment_facts_have_compatible_native_storage(
    loom_vector_fragment_fact_t target, loom_vector_fragment_fact_t source) {
  if ((target.role_flags & source.role_flags) == 0) {
    return false;
  }
  target.role_flags = source.role_flags;
  return loom_vector_fragment_facts_match_contract_except_native_storage(
      target, source);
}

static iree_status_t loom_vector_clone_equal_extension(
    loom_value_fact_table_t* target, const loom_value_fact_table_t* source,
    loom_value_facts_t source_facts, loom_value_facts_t* inout_facts) {
  loom_value_facts_t cloned_extension = loom_value_facts_unknown();
  IREE_RETURN_IF_ERROR(loom_value_fact_table_clone_fact(
      target, source, source_facts, &cloned_extension));
  inout_facts->extension_id = cloned_extension.extension_id;
  return iree_ok_status();
}

static iree_status_t loom_vector_make_accumulator_join_fragment(
    loom_fact_context_t* context, loom_vector_fragment_fact_t lhs,
    loom_vector_fragment_fact_t rhs, loom_value_facts_t* inout_facts) {
  bool has_native_storage =
      iree_all_bits_set(lhs.flags,
                        LOOM_VECTOR_FRAGMENT_FACT_FLAG_HAS_NATIVE_STORAGE) &&
      iree_all_bits_set(rhs.flags,
                        LOOM_VECTOR_FRAGMENT_FACT_FLAG_HAS_NATIVE_STORAGE);
  lhs.role_flags = LOOM_VECTOR_FRAGMENT_ROLE_FLAG_INIT |
                   LOOM_VECTOR_FRAGMENT_ROLE_FLAG_RESULT;
  lhs.flags &= ~LOOM_VECTOR_FRAGMENT_FACT_FLAG_HAS_NATIVE_STORAGE;
  if (has_native_storage) {
    lhs.flags |= LOOM_VECTOR_FRAGMENT_FACT_FLAG_HAS_NATIVE_STORAGE;
  }

  loom_value_facts_t fragment_facts = loom_value_facts_unknown();
  IREE_RETURN_IF_ERROR(loom_vector_fragment_fact_make_value_facts(
      context, lhs, &fragment_facts));
  inout_facts->extension_id = fragment_facts.extension_id;
  return iree_ok_status();
}

typedef enum loom_vector_aggregate_fact_kind_e {
  LOOM_VECTOR_AGGREGATE_FACT_NONE = 0,
  LOOM_VECTOR_AGGREGATE_FACT_UNIFORM = 1,
  LOOM_VECTOR_AGGREGATE_FACT_SMALL_STATIC_LANES = 2,
} loom_vector_aggregate_fact_kind_t;

static loom_vector_aggregate_fact_kind_t loom_vector_query_aggregate_extension(
    const loom_fact_context_t* context, loom_value_facts_t facts,
    loom_value_facts_t* out_uniform_element,
    loom_value_fact_small_static_lanes_t* out_small_static_lanes) {
  loom_value_fact_uniform_element_t uniform = {0};
  if (loom_value_facts_query_uniform_element(context, facts, &uniform)) {
    *out_uniform_element = uniform.element;
    return LOOM_VECTOR_AGGREGATE_FACT_UNIFORM;
  }
  if (loom_value_facts_query_small_static_lanes(context, facts,
                                                out_small_static_lanes)) {
    return LOOM_VECTOR_AGGREGATE_FACT_SMALL_STATIC_LANES;
  }
  return LOOM_VECTOR_AGGREGATE_FACT_NONE;
}

static iree_status_t loom_vector_try_join_aggregate_extensions(
    loom_value_fact_table_t* target, const loom_value_fact_table_t* lhs_table,
    loom_value_facts_t lhs, const loom_value_fact_table_t* rhs_table,
    loom_value_facts_t rhs, loom_value_facts_t* inout_facts,
    bool* out_handled) {
  loom_value_facts_t lhs_uniform = loom_value_facts_unknown();
  loom_value_facts_t rhs_uniform = loom_value_facts_unknown();
  loom_value_fact_small_static_lanes_t lhs_lanes = {0};
  loom_value_fact_small_static_lanes_t rhs_lanes = {0};
  const loom_vector_aggregate_fact_kind_t lhs_kind =
      loom_vector_query_aggregate_extension(&lhs_table->context, lhs,
                                            &lhs_uniform, &lhs_lanes);
  const loom_vector_aggregate_fact_kind_t rhs_kind =
      loom_vector_query_aggregate_extension(&rhs_table->context, rhs,
                                            &rhs_uniform, &rhs_lanes);
  *out_handled = lhs_kind != LOOM_VECTOR_AGGREGATE_FACT_NONE ||
                 rhs_kind != LOOM_VECTOR_AGGREGATE_FACT_NONE;
  if (!*out_handled) return iree_ok_status();
  if (lhs_kind == LOOM_VECTOR_AGGREGATE_FACT_NONE ||
      rhs_kind == LOOM_VECTOR_AGGREGATE_FACT_NONE) {
    return iree_ok_status();
  }

  if (lhs_kind == LOOM_VECTOR_AGGREGATE_FACT_UNIFORM &&
      rhs_kind == LOOM_VECTOR_AGGREGATE_FACT_UNIFORM) {
    loom_value_facts_t element = loom_value_facts_unknown();
    loom_value_facts_meet(&lhs_uniform, &rhs_uniform, &element);
    loom_value_facts_t extension = loom_value_facts_unknown();
    IREE_RETURN_IF_ERROR(loom_value_facts_make_uniform_element(
        &target->context, element, &extension));
    inout_facts->extension_id = extension.extension_id;
    return iree_ok_status();
  }

  if (lhs_kind == LOOM_VECTOR_AGGREGATE_FACT_SMALL_STATIC_LANES &&
      rhs_kind == LOOM_VECTOR_AGGREGATE_FACT_SMALL_STATIC_LANES &&
      lhs_lanes.count != rhs_lanes.count) {
    return iree_ok_status();
  }
  const iree_host_size_t lane_count =
      lhs_kind == LOOM_VECTOR_AGGREGATE_FACT_SMALL_STATIC_LANES
          ? lhs_lanes.count
          : rhs_lanes.count;

  loom_value_facts_t joined_lanes[LOOM_VALUE_FACT_SMALL_STATIC_LANE_LIMIT] = {
      {0}};
  for (iree_host_size_t i = 0; i < lane_count; ++i) {
    const loom_value_facts_t lhs_lane =
        lhs_kind == LOOM_VECTOR_AGGREGATE_FACT_UNIFORM ? lhs_uniform
                                                       : lhs_lanes.lanes[i];
    const loom_value_facts_t rhs_lane =
        rhs_kind == LOOM_VECTOR_AGGREGATE_FACT_UNIFORM ? rhs_uniform
                                                       : rhs_lanes.lanes[i];
    loom_value_facts_meet(&lhs_lane, &rhs_lane, &joined_lanes[i]);
  }
  loom_value_fact_small_static_lanes_t joined = {
      .lanes = joined_lanes,
      .count = lane_count,
  };
  loom_value_facts_t extension = loom_value_facts_unknown();
  IREE_RETURN_IF_ERROR(loom_value_facts_make_small_static_lanes(
      &target->context, joined, &extension));
  inout_facts->extension_id = extension.extension_id;
  return iree_ok_status();
}

static iree_status_t loom_vector_try_join_iota_extensions(
    loom_value_fact_table_t* target, const loom_value_fact_table_t* lhs_table,
    loom_value_facts_t lhs, const loom_value_fact_table_t* rhs_table,
    loom_value_facts_t rhs, loom_value_facts_t* inout_facts,
    bool* out_handled) {
  loom_value_fact_vector_iota_t lhs_iota = {0};
  loom_value_fact_vector_iota_t rhs_iota = {0};
  const bool lhs_has_iota =
      loom_value_facts_query_vector_iota(&lhs_table->context, lhs, &lhs_iota);
  const bool rhs_has_iota =
      loom_value_facts_query_vector_iota(&rhs_table->context, rhs, &rhs_iota);
  *out_handled = lhs_has_iota || rhs_has_iota;
  if (!lhs_has_iota || !rhs_has_iota) return iree_ok_status();

  loom_value_fact_vector_iota_t joined = {0};
  loom_value_facts_meet(&lhs_iota.base, &rhs_iota.base, &joined.base);
  loom_value_facts_meet(&lhs_iota.step, &rhs_iota.step, &joined.step);
  loom_value_facts_t extension = loom_value_facts_unknown();
  IREE_RETURN_IF_ERROR(
      loom_value_facts_make_vector_iota(&target->context, joined, &extension));
  inout_facts->extension_id = extension.extension_id;
  return iree_ok_status();
}

static iree_status_t loom_vector_try_join_prefix_mask_extensions(
    loom_value_fact_table_t* target, const loom_value_fact_table_t* lhs_table,
    loom_value_facts_t lhs, const loom_value_fact_table_t* rhs_table,
    loom_value_facts_t rhs, loom_value_facts_t* inout_facts,
    bool* out_handled) {
  loom_value_fact_vector_prefix_mask_t lhs_mask = {0};
  loom_value_fact_vector_prefix_mask_t rhs_mask = {0};
  const bool lhs_has_mask = loom_value_facts_query_vector_prefix_mask(
      &lhs_table->context, lhs, &lhs_mask);
  const bool rhs_has_mask = loom_value_facts_query_vector_prefix_mask(
      &rhs_table->context, rhs, &rhs_mask);
  *out_handled = lhs_has_mask || rhs_has_mask;
  if (!lhs_has_mask || !rhs_has_mask) return iree_ok_status();

  loom_value_fact_vector_prefix_mask_t joined = {0};
  loom_value_facts_meet(&lhs_mask.lower_bound, &rhs_mask.lower_bound,
                        &joined.lower_bound);
  loom_value_facts_meet(&lhs_mask.upper_bound, &rhs_mask.upper_bound,
                        &joined.upper_bound);
  loom_value_facts_meet(&lhs_mask.step, &rhs_mask.step, &joined.step);
  loom_value_facts_t extension = loom_value_facts_unknown();
  IREE_RETURN_IF_ERROR(loom_value_facts_make_vector_prefix_mask(
      &target->context, joined, &extension));
  inout_facts->extension_id = extension.extension_id;
  return iree_ok_status();
}

static iree_status_t loom_vector_join_fragment_extension(
    loom_value_fact_table_t* target, const loom_value_fact_table_t* lhs_table,
    loom_value_facts_t lhs, const loom_value_fact_table_t* rhs_table,
    loom_value_facts_t rhs, loom_value_facts_t* inout_facts) {
  if (loom_value_fact_table_extensions_equal(lhs_table, lhs, rhs_table, rhs)) {
    return loom_vector_clone_equal_extension(target, lhs_table, lhs,
                                             inout_facts);
  }

  loom_vector_fragment_fact_t lhs_fragment;
  loom_vector_fragment_fact_t rhs_fragment;
  if (!loom_vector_fragment_fact_query_value_facts(&lhs_table->context, lhs,
                                                   &lhs_fragment) ||
      !loom_vector_fragment_fact_query_value_facts(&rhs_table->context, rhs,
                                                   &rhs_fragment) ||
      !loom_vector_fragment_facts_match_accumulator_contract(lhs_fragment,
                                                             rhs_fragment)) {
    inout_facts->extension_id = LOOM_VALUE_FACT_EXTENSION_ID_NONE;
    return iree_ok_status();
  }
  return loom_vector_make_accumulator_join_fragment(
      &target->context, lhs_fragment, rhs_fragment, inout_facts);
}

static iree_status_t loom_vector_join_extension(
    loom_value_fact_table_t* target, const loom_value_fact_table_t* lhs_table,
    loom_value_facts_t lhs, const loom_value_fact_table_t* rhs_table,
    loom_value_facts_t rhs, loom_value_facts_t* inout_facts) {
  if (loom_value_fact_table_extensions_equal(lhs_table, lhs, rhs_table, rhs)) {
    return loom_vector_clone_equal_extension(target, lhs_table, lhs,
                                             inout_facts);
  }

  bool handled = false;
  IREE_RETURN_IF_ERROR(loom_vector_try_join_aggregate_extensions(
      target, lhs_table, lhs, rhs_table, rhs, inout_facts, &handled));
  if (handled) return iree_ok_status();
  IREE_RETURN_IF_ERROR(loom_vector_try_join_iota_extensions(
      target, lhs_table, lhs, rhs_table, rhs, inout_facts, &handled));
  if (handled) return iree_ok_status();
  IREE_RETURN_IF_ERROR(loom_vector_try_join_prefix_mask_extensions(
      target, lhs_table, lhs, rhs_table, rhs, inout_facts, &handled));
  if (handled) return iree_ok_status();
  return loom_vector_join_fragment_extension(target, lhs_table, lhs, rhs_table,
                                             rhs, inout_facts);
}

static iree_status_t loom_vector_meet_extension(
    const loom_value_fact_domain_t* domain, const loom_module_t* module,
    loom_type_t type, loom_value_fact_table_t* target,
    const loom_value_fact_table_t* lhs_table, loom_value_facts_t lhs,
    const loom_value_fact_table_t* rhs_table, loom_value_facts_t rhs,
    loom_value_facts_t* inout_facts) {
  (void)domain;
  (void)module;
  (void)type;
  return loom_vector_join_extension(target, lhs_table, lhs, rhs_table, rhs,
                                    inout_facts);
}

static iree_status_t loom_vector_widen_extension(
    const loom_value_fact_domain_t* domain, const loom_module_t* module,
    loom_type_t type, loom_value_fact_table_t* target,
    const loom_value_fact_table_t* previous_table, loom_value_facts_t previous,
    const loom_value_fact_table_t* next_table, loom_value_facts_t next,
    uint32_t iteration, loom_value_facts_t* inout_facts) {
  (void)domain;
  (void)module;
  (void)type;
  (void)iteration;
  // The generic solver calls this only after its two precise meet iterations.
  // Stable extensions still clone below, and matrix fragment roles form a
  // finite join. Aggregate component ranges can otherwise expand on every
  // backedge, so changing aggregate summaries deliberately degrade here.
  return loom_vector_join_fragment_extension(target, previous_table, previous,
                                             next_table, next, inout_facts);
}

const loom_value_fact_domain_t loom_vector_fact_domain = {
    .meet_extension = loom_vector_meet_extension,
    .widen_extension = loom_vector_widen_extension,
};

static iree_status_t loom_vector_try_preserve_lanewise_fragment_facts(
    loom_fact_context_t* context, const loom_value_facts_t* operand_facts,
    uint16_t operand_count, loom_value_facts_t* result_facts,
    bool* out_handled) {
  *out_handled = false;
  loom_vector_fragment_fact_t dense_fragment;
  loom_vector_fragment_fact_initialize(&dense_fragment);
  bool all_accumulator_like = true;
  bool all_native_storage = true;
  for (uint16_t i = 0; i < operand_count; ++i) {
    loom_vector_fragment_fact_t candidate;
    if (!loom_vector_fragment_fact_query_value_facts(context, operand_facts[i],
                                                     &candidate)) {
      continue;
    }
    *out_handled = true;
    if (!loom_vector_fragment_fact_has_matrix_shape(candidate)) {
      result_facts[0] = loom_value_facts_unknown();
      return iree_ok_status();
    }
    all_accumulator_like &=
        loom_vector_fragment_fact_is_accumulator_like(candidate);
    all_native_storage &= iree_all_bits_set(
        candidate.flags, LOOM_VECTOR_FRAGMENT_FACT_FLAG_HAS_NATIVE_STORAGE);
    loom_vector_fragment_fact_t dense_candidate =
        loom_vector_fragment_fact_dense_interpretation(candidate);
    if (loom_vector_fragment_fact_is_unknown(dense_fragment)) {
      dense_fragment = dense_candidate;
      continue;
    }
    if (!loom_vector_fragment_fact_equal(dense_fragment, dense_candidate)) {
      result_facts[0] = loom_value_facts_unknown();
      return iree_ok_status();
    }
  }
  if (!*out_handled) {
    return iree_ok_status();
  }
  if (all_accumulator_like) {
    dense_fragment.role_flags = LOOM_VECTOR_FRAGMENT_ROLE_FLAG_INIT |
                                LOOM_VECTOR_FRAGMENT_ROLE_FLAG_RESULT;
    if (all_native_storage) {
      dense_fragment.flags |= LOOM_VECTOR_FRAGMENT_FACT_FLAG_HAS_NATIVE_STORAGE;
    }
  }
  return loom_vector_fragment_fact_make_value_facts(context, dense_fragment,
                                                    &result_facts[0]);
}

static void loom_vector_fragment_copy_present_auxiliary(
    loom_encoding_auxiliary_view_t source,
    loom_encoding_auxiliary_view_t* out_target) {
  memset(out_target, 0, sizeof(*out_target));
  out_target->present_keys = source.present_keys;
  for (uint8_t i = 0; i < LOOM_ENCODING_AUXILIARY_KEY_COUNT_; ++i) {
    loom_encoding_auxiliary_key_t key = (loom_encoding_auxiliary_key_t)i;
    loom_encoding_auxiliary_key_flags_t key_flag =
        loom_encoding_auxiliary_key_flag(key);
    if (!iree_any_bit_set(source.present_keys, key_flag)) {
      continue;
    }
    out_target->values[key] = source.values[key];
  }
}

static bool loom_vector_fragment_query_storage_schema_from_facts(
    const loom_fact_context_t* context, loom_value_facts_t facts,
    loom_value_fact_storage_schema_t* out_schema) {
  *out_schema = (loom_value_fact_storage_schema_t){0};
  loom_value_fact_encoding_summary_t summary = {0};
  if (!loom_value_facts_query_encoding_summary(context, facts, &summary)) {
    return false;
  }
  if (summary.storage_schema.static_spec_encoding_id == 0 &&
      loom_value_fact_encoded_operand_schema_is_unknown(
          summary.storage_schema.encoded_operand)) {
    return false;
  }
  *out_schema = summary.storage_schema;
  return true;
}

static bool loom_vector_fragment_load_preserves_view_element_type(
    const loom_module_t* module, const loom_op_t* op) {
  loom_type_t view_type =
      loom_module_value_type(module, loom_vector_fragment_load_view(op));
  loom_type_t result_type =
      loom_module_value_type(module, loom_vector_fragment_load_result(op));
  return loom_type_is_view(view_type) && loom_type_is_vector(result_type) &&
         loom_type_element_type(view_type) ==
             loom_type_element_type(result_type);
}

static bool loom_vector_fragment_load_preserves_target_fragment_storage_schema(
    const loom_module_t* module, const loom_op_t* op,
    loom_value_fact_storage_schema_t storage_schema) {
  const loom_value_fact_encoded_operand_schema_t operand =
      storage_schema.encoded_operand;
  if (loom_value_fact_encoded_operand_schema_is_unknown(operand) ||
      !iree_any_bit_set(operand.payload_packing,
                        LOOM_VALUE_FACT_PAYLOAD_PACKING_TARGET_FRAGMENT) ||
      operand.payload_register_count == 0 ||
      operand.payload_element_count == 0) {
    return false;
  }

  const loom_numeric_format_info_t* element_format = NULL;
  if (!loom_numeric_format_info(operand.element_format, &element_format) ||
      element_format->storage_bit_count == 0) {
    return false;
  }

  loom_type_t view_type =
      loom_module_value_type(module, loom_vector_fragment_load_view(op));
  loom_type_t result_type =
      loom_module_value_type(module, loom_vector_fragment_load_result(op));
  if (!loom_type_is_view(view_type) || !loom_type_is_vector(result_type) ||
      !loom_type_is_all_static(result_type)) {
    return false;
  }
  const int32_t view_element_bit_count =
      loom_scalar_type_bitwidth(loom_type_element_type(view_type));
  if (view_element_bit_count != element_format->storage_bit_count) {
    return false;
  }

  uint64_t result_element_count = 0;
  if (!loom_type_static_element_count(result_type, &result_element_count)) {
    return false;
  }
  const int32_t result_element_bit_count =
      loom_scalar_type_bitwidth(loom_type_element_type(result_type));
  if (result_element_bit_count <= 0 ||
      result_element_count > UINT64_MAX / (uint64_t)result_element_bit_count) {
    return false;
  }
  return result_element_count * (uint64_t)result_element_bit_count ==
         (uint64_t)operand.payload_register_count * 32ull;
}

iree_status_t loom_vector_fragment_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  loom_vector_fragment_fact_t fact;
  loom_vector_fragment_fact_initialize(&fact);
  fact.role_flags =
      loom_vector_fragment_fact_role_flags(loom_vector_fragment_role(op));
  if (fact.role_flags == 0) {
    result_facts[0] = loom_value_facts_unknown();
    return iree_ok_status();
  }
  if (!loom_vector_fragment_fact_set_shape(
          loom_vector_fragment_blocks(op), loom_vector_fragment_rows(op),
          loom_vector_fragment_columns(op), &fact)) {
    result_facts[0] = loom_value_facts_unknown();
    return iree_ok_status();
  }

  loom_vector_fragment_parameter_view_t parameters;
  const loom_value_slice_t parameter_values = loom_vector_fragment_params(op);
  iree_string_view_t unknown_key = iree_string_view_empty();
  if (!loom_vector_fragment_parameter_view_resolve(
          module, parameter_values, loom_vector_fragment_param_names(op),
          &parameters, &unknown_key)) {
    result_facts[0] = loom_value_facts_unknown();
    return iree_ok_status();
  }
  loom_vector_fragment_copy_present_auxiliary(parameters.auxiliary,
                                              &fact.auxiliary);

  if (loom_vector_fragment_role(op) == LOOM_VECTOR_ROLE_INIT) {
    fact.flags |= LOOM_VECTOR_FRAGMENT_FACT_FLAG_HAS_NATIVE_STORAGE;
  }

  if (parameters.has_schema) {
    fact.flags |= LOOM_VECTOR_FRAGMENT_FACT_FLAG_HAS_SCHEMA;
    fact.schema_value_id = parameters.schema_value_id;

    loom_value_fact_storage_schema_t storage_schema = {0};
    if (parameters.schema_parameter_ordinal != UINT16_MAX) {
      const iree_host_size_t parameter_operand_offset =
          (iree_host_size_t)(parameter_values.values -
                             loom_op_const_operands(op));
      const iree_host_size_t schema_operand_index =
          parameter_operand_offset + parameters.schema_parameter_ordinal;
      if (schema_operand_index < op->operand_count &&
          loom_vector_fragment_query_storage_schema_from_facts(
              context, operand_facts[schema_operand_index], &storage_schema)) {
        if (storage_schema.static_spec_encoding_id != 0) {
          fact.flags |= LOOM_VECTOR_FRAGMENT_FACT_FLAG_HAS_STATIC_SCHEMA;
          fact.static_schema_encoding_id =
              storage_schema.static_spec_encoding_id;
        }
        fact.encoded_operand = storage_schema.encoded_operand;
      }
    }
  }

  loom_vector_fragment_fact_t data_fragment;
  if (loom_vector_fragment_fact_query_value_facts(context, operand_facts[0],
                                                  &data_fragment) &&
      iree_all_bits_set(data_fragment.flags,
                        LOOM_VECTOR_FRAGMENT_FACT_FLAG_HAS_NATIVE_STORAGE) &&
      loom_vector_fragment_facts_have_compatible_native_storage(
          fact, data_fragment)) {
    fact.flags |= LOOM_VECTOR_FRAGMENT_FACT_FLAG_HAS_NATIVE_STORAGE;
  }

  return loom_vector_fragment_fact_make_value_facts(context, fact,
                                                    &result_facts[0]);
}

iree_status_t loom_vector_fragment_repack_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  (void)module;
  loom_vector_fragment_fact_t fact;
  loom_vector_fragment_fact_initialize(&fact);
  fact.role_flags = loom_vector_fragment_fact_role_flags(
      loom_vector_fragment_repack_role(op));
  if (fact.role_flags == 0) {
    result_facts[0] = loom_value_facts_unknown();
    return iree_ok_status();
  }
  if (!loom_vector_fragment_fact_set_shape(
          loom_vector_fragment_repack_blocks(op),
          loom_vector_fragment_repack_rows(op),
          loom_vector_fragment_repack_columns(op), &fact)) {
    result_facts[0] = loom_value_facts_unknown();
    return iree_ok_status();
  }
  loom_vector_fragment_fact_t source_fact;
  if (loom_vector_fragment_fact_query_value_facts(context, operand_facts[0],
                                                  &source_fact) &&
      iree_all_bits_set(source_fact.flags,
                        LOOM_VECTOR_FRAGMENT_FACT_FLAG_HAS_NATIVE_STORAGE)) {
    fact.flags |= LOOM_VECTOR_FRAGMENT_FACT_FLAG_HAS_NATIVE_STORAGE;
  }
  return loom_vector_fragment_fact_make_value_facts(context, fact,
                                                    &result_facts[0]);
}

iree_status_t loom_vector_fragment_load_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  (void)operand_facts;
  loom_vector_fragment_fact_t fact;
  loom_vector_fragment_fact_initialize(&fact);
  fact.role_flags =
      loom_vector_fragment_fact_role_flags(loom_vector_fragment_load_role(op));
  if (fact.role_flags == 0) {
    result_facts[0] = loom_value_facts_unknown();
    return iree_ok_status();
  }
  if (!loom_vector_fragment_fact_set_shape(
          loom_vector_fragment_load_blocks(op),
          loom_vector_fragment_load_rows(op),
          loom_vector_fragment_load_columns(op), &fact)) {
    result_facts[0] = loom_value_facts_unknown();
    return iree_ok_status();
  }
  fact.flags |= LOOM_VECTOR_FRAGMENT_FACT_FLAG_HAS_NATIVE_STORAGE;

  loom_encoding_auxiliary_view_t auxiliary;
  if (!loom_encoding_auxiliary_view_resolve(
          module, loom_vector_fragment_load_auxiliary(op),
          loom_vector_fragment_load_auxiliary_names(op), &auxiliary, NULL)) {
    result_facts[0] = loom_value_facts_unknown();
    return iree_ok_status();
  }
  loom_vector_fragment_copy_present_auxiliary(auxiliary, &fact.auxiliary);

  const loom_value_id_t view_value_id = loom_vector_fragment_load_view(op);
  loom_value_fact_storage_schema_t storage_schema = {0};
  if (loom_encoding_query_type_storage_schema(
          context, module, loom_module_value_type(module, view_value_id),
          &storage_schema) &&
      !loom_value_fact_encoded_operand_schema_is_unknown(
          storage_schema.encoded_operand) &&
      (loom_vector_fragment_load_preserves_view_element_type(module, op) ||
       loom_vector_fragment_load_preserves_target_fragment_storage_schema(
           module, op, storage_schema))) {
    fact.flags |= LOOM_VECTOR_FRAGMENT_FACT_FLAG_HAS_SCHEMA;
    fact.encoded_operand = storage_schema.encoded_operand;
    if (storage_schema.static_spec_encoding_id != 0) {
      fact.flags |= LOOM_VECTOR_FRAGMENT_FACT_FLAG_HAS_STATIC_SCHEMA;
      fact.static_schema_encoding_id = storage_schema.static_spec_encoding_id;
    }
  }
  IREE_RETURN_IF_ERROR(loom_vector_fragment_fact_make_value_facts(
      context, fact, &result_facts[0]));
  loom_value_facts_t content_facts = {0};
  if (loom_encoding_query_type_storage_content_facts(
          context, module, loom_module_value_type(module, view_value_id),
          &content_facts)) {
    result_facts[0].flags |= content_facts.flags;
  }
  return iree_ok_status();
}

iree_status_t loom_vector_load_facts(loom_fact_context_t* context,
                                     const loom_module_t* module,
                                     const loom_op_t* op,
                                     const loom_value_facts_t* operand_facts,
                                     loom_value_facts_t* result_facts) {
  (void)operand_facts;
  loom_type_t view_type =
      loom_module_value_type(module, loom_vector_load_view(op));
  loom_value_facts_t element_facts = {0};
  if (!loom_encoding_query_type_storage_content_facts(
          context, module, view_type, &element_facts)) {
    result_facts[0] = loom_value_facts_unknown();
    return iree_ok_status();
  }
  return loom_value_facts_make_uniform_element(context, element_facts,
                                               &result_facts[0]);
}

iree_status_t loom_vector_decode_facts(loom_fact_context_t* context,
                                       const loom_module_t* module,
                                       const loom_op_t* op,
                                       const loom_value_facts_t* operand_facts,
                                       loom_value_facts_t* result_facts) {
  loom_value_fact_encoding_summary_t summary = {0};
  if (!loom_value_facts_query_encoding_summary(context, operand_facts[1],
                                               &summary)) {
    return loom_vector_make_unknown_facts(result_facts);
  }
  const loom_type_t result_type =
      loom_module_value_type(module, loom_vector_decode_result(op));
  loom_value_facts_t element_facts = {0};
  if (!loom_encoding_query_storage_schema_content_facts(
          &summary.storage_schema, loom_type_element_type(result_type),
          &element_facts)) {
    return loom_vector_make_unknown_facts(result_facts);
  }
  return loom_value_facts_make_uniform_element(context, element_facts,
                                               &result_facts[0]);
}

static bool loom_vector_integer_element_bitwidth(loom_type_t type,
                                                 int32_t* out_bitwidth) {
  if (!loom_type_is_shaped(type) && !loom_type_is_scalar(type)) return false;
  loom_scalar_type_t element_type = loom_type_element_type(type);
  if (!loom_scalar_type_is_integer(element_type)) return false;
  int32_t bitwidth = loom_scalar_type_bitwidth(element_type);
  if (bitwidth <= 0 || bitwidth > 64) return false;
  *out_bitwidth = bitwidth;
  return true;
}

static loom_value_facts_t loom_vector_make_integer_raw_bit_facts(
    uint64_t raw_bits, loom_scalar_type_t element_type) {
  int32_t bitwidth = loom_scalar_type_bitwidth(element_type);
  if (element_type == LOOM_SCALAR_TYPE_I1) {
    return loom_value_facts_exact_i64((raw_bits & 1) != 0 ? 1 : 0);
  }
  return loom_value_facts_make_signed_raw_bits(raw_bits, bitwidth);
}

static bool loom_vector_unsigned_code_capacity_covers(loom_type_t type,
                                                      int64_t max_code) {
  int32_t bitwidth = 0;
  if (max_code < 0 || !loom_vector_integer_element_bitwidth(type, &bitwidth)) {
    return false;
  }
  if (bitwidth >= 63) return true;
  return (uint64_t)max_code < (UINT64_C(1) << bitwidth);
}

static int32_t loom_vector_extend_integer_field_i32(int64_t value,
                                                    uint8_t bit_count,
                                                    bool is_signed) {
  if (bit_count == 0) return 0;
  if (bit_count > 32) bit_count = 32;
  uint32_t mask =
      bit_count == 32 ? UINT32_MAX : (((uint32_t)1) << bit_count) - 1;
  uint32_t masked = ((uint32_t)value) & mask;
  if (!is_signed) return (int32_t)masked;
  uint32_t sign_bit = ((uint32_t)1) << (bit_count - 1);
  return (int32_t)((masked ^ sign_bit) - sign_bit);
}

typedef struct loom_vector_grouped_dot_shape_t {
  // Number of logical result lanes.
  iree_host_size_t result_lane_count;
  // Static last-axis extent of each source vector.
  iree_host_size_t source_last_extent;
  // Static last-axis extent of the result vector.
  iree_host_size_t result_last_extent;
} loom_vector_grouped_dot_shape_t;

static bool loom_vector_query_grouped_dot_shape(
    loom_type_t source_type, loom_type_t result_type, uint8_t group_size,
    loom_vector_grouped_dot_shape_t* out_shape) {
  uint8_t rank = loom_type_rank(result_type);
  if (rank == 0 || loom_type_rank(source_type) != rank || group_size == 0) {
    return false;
  }

  uint64_t result_lane_count = 0;
  if (!loom_type_static_element_count(result_type, &result_lane_count) ||
      result_lane_count > (uint64_t)IREE_HOST_SIZE_MAX) {
    return false;
  }

  if (loom_type_dim_is_dynamic_at(source_type, rank - 1) ||
      loom_type_dim_is_dynamic_at(result_type, rank - 1)) {
    return false;
  }
  int64_t source_last_extent =
      loom_type_dim_static_size_at(source_type, rank - 1);
  int64_t result_last_extent =
      loom_type_dim_static_size_at(result_type, rank - 1);
  if (source_last_extent < 0 || result_last_extent < 0 ||
      (uint64_t)source_last_extent > (uint64_t)IREE_HOST_SIZE_MAX ||
      (uint64_t)result_last_extent > (uint64_t)IREE_HOST_SIZE_MAX) {
    return false;
  }

  int64_t expected_source_last_extent = 0;
  if (!iree_checked_mul_i64(result_last_extent, (int64_t)group_size,
                            &expected_source_last_extent) ||
      source_last_extent != expected_source_last_extent) {
    return false;
  }

  for (uint8_t axis = 0; axis + 1 < rank; ++axis) {
    if (loom_type_dim_is_dynamic_at(source_type, axis) ||
        loom_type_dim_is_dynamic_at(result_type, axis) ||
        loom_type_dim_static_size_at(source_type, axis) !=
            loom_type_dim_static_size_at(result_type, axis)) {
      return false;
    }
  }

  *out_shape = (loom_vector_grouped_dot_shape_t){
      .result_lane_count = (iree_host_size_t)result_lane_count,
      .source_last_extent = (iree_host_size_t)source_last_extent,
      .result_last_extent = (iree_host_size_t)result_last_extent,
  };
  return true;
}

static bool loom_vector_grouped_dot_source_lane(
    loom_vector_grouped_dot_shape_t shape, iree_host_size_t result_lane,
    uint8_t group_size, uint8_t group_lane, iree_host_size_t* out_source_lane) {
  if (shape.result_last_extent == 0 || group_lane >= group_size) return false;
  iree_host_size_t leading_lane = result_lane / shape.result_last_extent;
  iree_host_size_t result_last_lane = result_lane % shape.result_last_extent;
  iree_host_size_t source_lane = 0;
  if (!iree_host_size_checked_mul(leading_lane, shape.source_last_extent,
                                  &source_lane)) {
    return false;
  }
  iree_host_size_t source_last_lane = 0;
  if (!iree_host_size_checked_mul(result_last_lane, group_size,
                                  &source_last_lane) ||
      !iree_host_size_checked_add(source_last_lane, group_lane,
                                  &source_last_lane) ||
      !iree_host_size_checked_add(source_lane, source_last_lane,
                                  &source_lane)) {
    return false;
  }
  *out_source_lane = source_lane;
  return true;
}

static bool loom_vector_static_last_axis_extent(loom_type_t type,
                                                iree_host_size_t* out_extent) {
  const uint8_t rank = loom_type_rank(type);
  if (rank == 0 || loom_type_dim_is_dynamic_at(type, (uint8_t)(rank - 1))) {
    return false;
  }
  const int64_t extent =
      loom_type_dim_static_size_at(type, (uint8_t)(rank - 1));
  if (extent <= 0 || (uint64_t)extent > (uint64_t)IREE_HOST_SIZE_MAX) {
    return false;
  }
  *out_extent = (iree_host_size_t)extent;
  return true;
}

static bool loom_vector_same_static_lane_count(loom_type_t source_type,
                                               loom_type_t result_type,
                                               iree_host_size_t* out_count) {
  iree_host_size_t source_count = 0;
  iree_host_size_t result_count = 0;
  if (!loom_vector_type_static_lane_count(source_type, &source_count) ||
      !loom_vector_type_static_lane_count(result_type, &result_count) ||
      source_count != result_count) {
    return false;
  }
  *out_count = result_count;
  return true;
}

static bool loom_vector_bitcast_element_facts(
    loom_value_facts_t source_facts, loom_scalar_type_t source_element_type,
    loom_scalar_type_t result_element_type, loom_value_facts_t* out_facts) {
  loom_value_facts_eval_scalar_bitcast(source_element_type, result_element_type,
                                       &source_facts, out_facts);
  return loom_value_facts_is_exact(*out_facts) ||
         loom_value_facts_is_nan(*out_facts);
}

static bool loom_vector_transform_eval_float_binary(
    loom_scalar_type_t scalar_type, loom_value_facts_t lhs,
    loom_value_facts_t rhs, loom_float_binary_f32_fn_t f32_fn,
    loom_float_binary_f64_fn_t f64_fn, loom_value_facts_t* out_result) {
  loom_value_facts_eval_float_binary(scalar_type, &lhs, &rhs, f32_fn, f64_fn,
                                     out_result);
  return loom_value_facts_is_exact(*out_result);
}

static bool loom_vector_transform_hadamard_slice_facts(
    loom_scalar_type_t scalar_type, iree_host_size_t slice_offset,
    iree_host_size_t slice_extent, loom_value_facts_t* lanes) {
  for (iree_host_size_t half_span = 1; half_span < slice_extent;
       half_span <<= 1) {
    const iree_host_size_t span = half_span << 1;
    for (iree_host_size_t base = 0; base < slice_extent; base += span) {
      for (iree_host_size_t lane = 0; lane < half_span; ++lane) {
        const iree_host_size_t lhs_index = slice_offset + base + lane;
        const iree_host_size_t rhs_index = lhs_index + half_span;
        const loom_value_facts_t lhs = lanes[lhs_index];
        const loom_value_facts_t rhs = lanes[rhs_index];
        loom_value_facts_t sum = loom_value_facts_unknown();
        loom_value_facts_t difference = loom_value_facts_unknown();
        if (!loom_vector_transform_eval_float_binary(
                scalar_type, lhs, rhs, loom_vector_add_f32, loom_vector_add_f64,
                &sum) ||
            !loom_vector_transform_eval_float_binary(
                scalar_type, lhs, rhs, loom_vector_sub_f32, loom_vector_sub_f64,
                &difference)) {
          return false;
        }
        lanes[lhs_index] = sum;
        lanes[rhs_index] = difference;
      }
    }
  }
  return true;
}

static iree_status_t loom_vector_transform_hadamard_facts(
    loom_fact_context_t* context,
    const loom_encoding_hadamard_descriptor_t* descriptor,
    loom_type_t source_type, loom_value_facts_t source_facts,
    loom_value_facts_t* result_facts) {
  iree_host_size_t lane_count = 0;
  iree_host_size_t slice_extent = 0;
  if (!loom_vector_type_static_lane_count(source_type, &lane_count) ||
      !loom_vector_static_last_axis_extent(source_type, &slice_extent) ||
      lane_count > LOOM_VALUE_FACT_SMALL_STATIC_LANE_LIMIT ||
      !iree_math_is_power_of_two_i64((int64_t)slice_extent)) {
    return loom_vector_make_unknown_facts(result_facts);
  }

  const loom_scalar_type_t scalar_type = loom_type_element_type(source_type);
  loom_value_facts_t lanes[LOOM_VALUE_FACT_SMALL_STATIC_LANE_LIMIT] = {{0}};
  for (iree_host_size_t lane = 0; lane < lane_count; ++lane) {
    double exact_value = 0.0;
    if (!loom_vector_facts_query_lane(context, source_facts, lane,
                                      &lanes[lane]) ||
        !loom_value_facts_as_exact_float(scalar_type, lanes[lane],
                                         &exact_value)) {
      return loom_vector_make_unknown_facts(result_facts);
    }
  }

  for (iree_host_size_t offset = 0; offset < lane_count;
       offset += slice_extent) {
    if (!loom_vector_transform_hadamard_slice_facts(scalar_type, offset,
                                                    slice_extent, lanes)) {
      return loom_vector_make_unknown_facts(result_facts);
    }
  }

  if (descriptor->normalization ==
      LOOM_ENCODING_TRANSFORM_NORMALIZATION_ORTHONORMAL) {
    const loom_value_facts_t scale = loom_value_facts_exact_float(
        scalar_type, 1.0 / sqrt((double)slice_extent));
    for (iree_host_size_t lane = 0; lane < lane_count; ++lane) {
      loom_value_facts_t scaled = loom_value_facts_unknown();
      if (!loom_vector_transform_eval_float_binary(
              scalar_type, lanes[lane], scale, loom_vector_mul_f32,
              loom_vector_mul_f64, &scaled)) {
        return loom_vector_make_unknown_facts(result_facts);
      }
      lanes[lane] = scaled;
    }
  }

  return loom_vector_make_small_static_lane_facts(context, lanes, lane_count,
                                                  result_facts);
}

static bool loom_vector_dot4i_lhs_is_signed(uint8_t kind) {
  return kind == LOOM_VECTOR_DOT4I_KIND_S8S8 ||
         kind == LOOM_VECTOR_DOT4I_KIND_S8U8;
}

static bool loom_vector_dot4i_rhs_is_signed(uint8_t kind) {
  return kind == LOOM_VECTOR_DOT4I_KIND_S8S8 ||
         kind == LOOM_VECTOR_DOT4I_KIND_U8S8;
}

static bool loom_vector_dot8i4_lhs_is_signed(uint8_t kind) {
  return kind == LOOM_VECTOR_DOT8I4_KIND_S4S4 ||
         kind == LOOM_VECTOR_DOT8I4_KIND_S4U4;
}

static bool loom_vector_dot8i4_rhs_is_signed(uint8_t kind) {
  return kind == LOOM_VECTOR_DOT8I4_KIND_S4S4 ||
         kind == LOOM_VECTOR_DOT8I4_KIND_U4S4;
}

typedef enum loom_vector_dot4f8_format_e {
  LOOM_VECTOR_DOT4F8_FORMAT_FP8,
  LOOM_VECTOR_DOT4F8_FORMAT_BF8,
} loom_vector_dot4f8_format_t;

static bool loom_vector_dot4f8_lhs_format(
    uint8_t kind, loom_vector_dot4f8_format_t* out_format) {
  switch (kind) {
    case LOOM_VECTOR_DOT4F8_KIND_FP8BF8:
    case LOOM_VECTOR_DOT4F8_KIND_FP8FP8:
      *out_format = LOOM_VECTOR_DOT4F8_FORMAT_FP8;
      return true;
    case LOOM_VECTOR_DOT4F8_KIND_BF8FP8:
    case LOOM_VECTOR_DOT4F8_KIND_BF8BF8:
      *out_format = LOOM_VECTOR_DOT4F8_FORMAT_BF8;
      return true;
    case LOOM_VECTOR_DOT4F8_KIND_COUNT_:
      return false;
  }
  return false;
}

static bool loom_vector_dot4f8_rhs_format(
    uint8_t kind, loom_vector_dot4f8_format_t* out_format) {
  switch (kind) {
    case LOOM_VECTOR_DOT4F8_KIND_FP8BF8:
    case LOOM_VECTOR_DOT4F8_KIND_BF8BF8:
      *out_format = LOOM_VECTOR_DOT4F8_FORMAT_BF8;
      return true;
    case LOOM_VECTOR_DOT4F8_KIND_BF8FP8:
    case LOOM_VECTOR_DOT4F8_KIND_FP8FP8:
      *out_format = LOOM_VECTOR_DOT4F8_FORMAT_FP8;
      return true;
    case LOOM_VECTOR_DOT4F8_KIND_COUNT_:
      return false;
  }
  return false;
}

static float loom_vector_dot4f8_decode_field(loom_vector_dot4f8_format_t format,
                                             uint8_t field) {
  switch (format) {
    case LOOM_VECTOR_DOT4F8_FORMAT_FP8:
      return iree_math_f8e4m3fn_to_f32(field);
    case LOOM_VECTOR_DOT4F8_FORMAT_BF8:
      return iree_math_f8e5m2_to_f32(field);
  }
  return NAN;
}

static bool loom_vector_dot4i_apply(uint8_t kind, int64_t lhs_raw,
                                    int64_t rhs_raw, int32_t* accumulator) {
  if (kind >= LOOM_VECTOR_DOT4I_KIND_COUNT_) return false;
  int32_t lhs = loom_vector_extend_integer_field_i32(
      lhs_raw, 8, loom_vector_dot4i_lhs_is_signed(kind));
  int32_t rhs = loom_vector_extend_integer_field_i32(
      rhs_raw, 8, loom_vector_dot4i_rhs_is_signed(kind));
  int32_t next = 0;
  if (!iree_checked_mul_add_i32(*accumulator, lhs, rhs, &next)) return false;
  *accumulator = next;
  return true;
}

static bool loom_vector_dot8i4_apply(uint8_t kind, uint32_t lhs_raw,
                                     uint32_t rhs_raw, int32_t* accumulator) {
  if (kind >= LOOM_VECTOR_DOT8I4_KIND_COUNT_) return false;
  bool lhs_is_signed = loom_vector_dot8i4_lhs_is_signed(kind);
  bool rhs_is_signed = loom_vector_dot8i4_rhs_is_signed(kind);
  for (uint8_t field_ordinal = 0; field_ordinal < 8; ++field_ordinal) {
    uint8_t shift = (uint8_t)(4 * field_ordinal);
    int32_t lhs = loom_vector_extend_integer_field_i32(lhs_raw >> shift, 4,
                                                       lhs_is_signed);
    int32_t rhs = loom_vector_extend_integer_field_i32(rhs_raw >> shift, 4,
                                                       rhs_is_signed);
    int32_t next = 0;
    if (!iree_checked_mul_add_i32(*accumulator, lhs, rhs, &next)) {
      return false;
    }
    *accumulator = next;
  }
  return true;
}

static bool loom_vector_dot4f8_apply(uint8_t kind, uint32_t lhs_raw,
                                     uint32_t rhs_raw, float* accumulator) {
  loom_vector_dot4f8_format_t lhs_format = LOOM_VECTOR_DOT4F8_FORMAT_FP8;
  loom_vector_dot4f8_format_t rhs_format = LOOM_VECTOR_DOT4F8_FORMAT_FP8;
  if (!loom_vector_dot4f8_lhs_format(kind, &lhs_format) ||
      !loom_vector_dot4f8_rhs_format(kind, &rhs_format)) {
    return false;
  }
  for (uint8_t field_ordinal = 0; field_ordinal < 4; ++field_ordinal) {
    uint8_t shift = (uint8_t)(8 * field_ordinal);
    uint8_t lhs_field = (uint8_t)(lhs_raw >> shift);
    uint8_t rhs_field = (uint8_t)(rhs_raw >> shift);
    float lhs = loom_vector_dot4f8_decode_field(lhs_format, lhs_field);
    float rhs = loom_vector_dot4f8_decode_field(rhs_format, rhs_field);
    *accumulator = fmaf(lhs, rhs, *accumulator);
  }
  return true;
}

static float loom_vector_rsqrt_f32(float input) { return 1.0f / sqrtf(input); }
static double loom_vector_rsqrt_f64(double input) { return 1.0 / sqrt(input); }

static float loom_vector_roundeven_f32(float input) {
  return nearbyintf(input);
}
static double loom_vector_roundeven_f64(double input) {
  return nearbyint(input);
}

static float loom_vector_silu_f32(float input) {
  return input * loom_float_logistic_f32(input);
}
static double loom_vector_silu_f64(double input) {
  return input * loom_float_logistic_f64(input);
}

static float loom_vector_softplus_f32(float input) {
  return log1pf(expf(-fabsf(input))) + fmaxf(input, 0.0f);
}
static double loom_vector_softplus_f64(double input) {
  return log1p(exp(-fabs(input))) + fmax(input, 0.0);
}

static float loom_vector_gelu_erf_f32(float input) {
  const float inverse_sqrt2 = 0.70710678118654752440f;
  return 0.5f * input * (1.0f + erff(input * inverse_sqrt2));
}
static double loom_vector_gelu_erf_f64(double input) {
  const double inverse_sqrt2 = 0.70710678118654752440;
  return 0.5 * input * (1.0 + erf(input * inverse_sqrt2));
}

static float loom_vector_gelu_tanh_f32(float input) {
  const float sqrt_2_over_pi = 0.79788456080286535588f;
  return 0.5f * input *
         (1.0f +
          tanhf(sqrt_2_over_pi * (input + 0.044715f * input * input * input)));
}
static double loom_vector_gelu_tanh_f64(double input) {
  const double sqrt_2_over_pi = 0.79788456080286535588;
  return 0.5 * input *
         (1.0 +
          tanh(sqrt_2_over_pi * (input + 0.044715 * input * input * input)));
}

static float loom_vector_gelu_logistic_f32(float input, float scale) {
  return input * loom_float_logistic_f32(scale * input);
}
static double loom_vector_gelu_logistic_f64(double input, double scale) {
  return input * loom_float_logistic_f64(scale * input);
}

static void loom_vector_float_negate_transfer(loom_scalar_type_t scalar_type,
                                              const loom_value_facts_t* input,
                                              const void* user_data,
                                              loom_value_facts_t* out) {
  loom_value_facts_eval_float_negate(scalar_type, input, out);
}

static void loom_vector_float_abs_transfer(loom_scalar_type_t scalar_type,
                                           const loom_value_facts_t* input,
                                           const void* user_data,
                                           loom_value_facts_t* out) {
  loom_value_facts_eval_float_abs(scalar_type, input, out);
}

static void loom_vector_float_turns_transfer(loom_scalar_type_t scalar_type,
                                             const loom_value_facts_t* input,
                                             const void* user_data,
                                             loom_value_facts_t* out) {
  const loom_float_turns_kind_t kind =
      *(const loom_float_turns_kind_t*)user_data;
  loom_value_facts_eval_float_turns(scalar_type, kind, input, out);
}

static void loom_vector_float_minmax_transfer(loom_scalar_type_t scalar_type,
                                              const loom_value_facts_t* lhs,
                                              const loom_value_facts_t* rhs,
                                              const void* user_data,
                                              loom_value_facts_t* out) {
  const loom_float_minmax_kind_t kind =
      *(const loom_float_minmax_kind_t*)user_data;
  loom_value_facts_eval_float_minmax(scalar_type, kind, lhs, rhs, out);
}

static void loom_vector_float_copysign_transfer(loom_scalar_type_t scalar_type,
                                                const loom_value_facts_t* lhs,
                                                const loom_value_facts_t* rhs,
                                                const void* user_data,
                                                loom_value_facts_t* out) {
  loom_value_facts_eval_float_copysign(scalar_type, lhs, rhs, out);
}

static void loom_vector_float_clamp_transfer(loom_scalar_type_t scalar_type,
                                             const loom_value_facts_t* value,
                                             const loom_value_facts_t* lower,
                                             const loom_value_facts_t* upper,
                                             const void* user_data,
                                             loom_value_facts_t* out) {
  const loom_float_clamp_kind_t kind =
      *(const loom_float_clamp_kind_t*)user_data;
  loom_value_facts_eval_float_clamp(scalar_type, kind, value, lower, upper,
                                    out);
}

static void loom_vector_isnanf_transfer(loom_scalar_type_t scalar_type,
                                        const loom_value_facts_t* input,
                                        const void* user_data,
                                        loom_value_facts_t* out) {
  if (loom_value_facts_is_nan(*input)) {
    *out = loom_value_facts_exact_i64(1);
    return;
  }
  if (loom_value_facts_is_not_nan(*input)) {
    *out = loom_value_facts_exact_i64(0);
    return;
  }
  double value = 0.0;
  if (!loom_vector_facts_query_exact_float(scalar_type, *input, &value)) {
    *out = loom_value_facts_make(0, 1, 1);
    return;
  }
  *out = loom_value_facts_exact_i64(isnan(value) ? 1 : 0);
}

static void loom_vector_isinff_transfer(loom_scalar_type_t scalar_type,
                                        const loom_value_facts_t* input,
                                        const void* user_data,
                                        loom_value_facts_t* out) {
  if (loom_value_facts_is_inf(*input)) {
    *out = loom_value_facts_exact_i64(1);
    return;
  }
  if (loom_value_facts_is_nan(*input) || loom_value_facts_is_finite(*input)) {
    *out = loom_value_facts_exact_i64(0);
    return;
  }
  if (loom_value_facts_is_not_inf(*input)) {
    *out = loom_value_facts_exact_i64(0);
    return;
  }
  double value = 0.0;
  if (!loom_vector_facts_query_exact_float(scalar_type, *input, &value)) {
    *out = loom_value_facts_make(0, 1, 1);
    return;
  }
  *out = loom_value_facts_exact_i64(isinf(value) ? 1 : 0);
}

static void loom_vector_isfinitef_transfer(loom_scalar_type_t scalar_type,
                                           const loom_value_facts_t* input,
                                           const void* user_data,
                                           loom_value_facts_t* out) {
  if (loom_value_facts_is_nan(*input) || loom_value_facts_is_inf(*input)) {
    *out = loom_value_facts_exact_i64(0);
    return;
  }
  if (loom_value_facts_is_finite(*input)) {
    *out = loom_value_facts_exact_i64(1);
    return;
  }
  if (loom_value_facts_is_not_nan(*input) &&
      loom_value_facts_is_not_inf(*input)) {
    *out = loom_value_facts_exact_i64(1);
    return;
  }
  double value = 0.0;
  if (!loom_vector_facts_query_exact_float(scalar_type, *input, &value)) {
    *out = loom_value_facts_make(0, 1, 1);
    return;
  }
  *out = loom_value_facts_exact_i64(isfinite(value) ? 1 : 0);
}

static void loom_vector_signf_transfer(loom_scalar_type_t scalar_type,
                                       const loom_value_facts_t* input,
                                       const void* user_data,
                                       loom_value_facts_t* out) {
  if (loom_value_facts_is_nan(*input)) {
    *out = loom_value_facts_exact_float(scalar_type, 0.0);
    return;
  }
  double value = 0.0;
  if (!loom_vector_facts_query_exact_float(scalar_type, *input, &value)) {
    *out = loom_value_facts_unknown();
    return;
  }
  *out = loom_value_facts_exact_float(scalar_type, (value > 0.0)   ? 1.0
                                                   : (value < 0.0) ? -1.0
                                                                   : 0.0);
}

static void loom_vector_signi_transfer(const loom_value_facts_t* input,
                                       loom_value_facts_t* out) {
  if (!loom_value_facts_is_exact(*input)) {
    *out = loom_value_facts_unknown();
    return;
  }
  int64_t value = input->range_lo;
  *out = loom_value_facts_exact_i64((value > 0) ? 1 : (value < 0) ? -1 : 0);
}

static void loom_vector_passthrough_transfer(const loom_value_facts_t* input,
                                             loom_value_facts_t* out) {
  *out = *input;
}

static void loom_vector_sitofp_transfer(loom_scalar_type_t scalar_type,
                                        const loom_value_facts_t* input,
                                        const void* user_data,
                                        loom_value_facts_t* out) {
  int64_t value = 0;
  if (!loom_vector_facts_query_exact_i64(*input, &value)) {
    *out = loom_value_facts_unknown();
    return;
  }
  *out = loom_value_facts_exact_float(scalar_type, (double)value);
}

static void loom_vector_fmai_transfer(const loom_value_facts_t* a,
                                      const loom_value_facts_t* b,
                                      const loom_value_facts_t* c,
                                      loom_value_facts_t* out) {
  loom_value_facts_fmai(a, b, c, out);
}

//===----------------------------------------------------------------------===//
// Construction
//===----------------------------------------------------------------------===//

iree_status_t loom_vector_constant_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  loom_value_id_t result_id = loom_vector_constant_result(op);
  loom_type_t result_type = loom_module_value_type(module, result_id);
  loom_value_facts_t element = loom_vector_attr_element_facts(
      loom_vector_constant_value(op), loom_type_element_type(result_type));
  IREE_RETURN_IF_ERROR(loom_value_facts_make_uniform_element(context, element,
                                                             &result_facts[0]));
  loom_value_facts_mark_cluster_uniform(&result_facts[0]);
  return iree_ok_status();
}

iree_status_t loom_vector_splat_facts(loom_fact_context_t* context,
                                      const loom_module_t* module,
                                      const loom_op_t* op,
                                      const loom_value_facts_t* operand_facts,
                                      loom_value_facts_t* result_facts) {
  IREE_RETURN_IF_ERROR(loom_value_fact_table_define_uniform_element_origin(
      context->table, loom_vector_splat_result(op),
      loom_vector_splat_scalar(op)));
  return loom_value_facts_make_uniform_element(context, operand_facts[0],
                                               &result_facts[0]);
}

iree_status_t loom_vector_broadcast_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  loom_type_t source_type =
      loom_module_value_type(module, loom_vector_broadcast_source(op));
  loom_type_t result_type =
      loom_module_value_type(module, loom_vector_broadcast_result(op));
  loom_value_facts_t uniform_element = {0};
  if (loom_vector_facts_query_uniform_element(context, operand_facts[0],
                                              &uniform_element)) {
    return loom_value_facts_make_uniform_element(context, uniform_element,
                                                 &result_facts[0]);
  }

  uint8_t source_rank = loom_type_rank(source_type);
  uint8_t result_rank = loom_type_rank(result_type);
  iree_host_size_t result_lane_count = 0;
  if (source_rank > result_rank ||
      !loom_vector_type_static_lane_count(result_type, &result_lane_count) ||
      result_lane_count > LOOM_VALUE_FACT_SMALL_STATIC_LANE_LIMIT ||
      !loom_type_is_all_static(source_type)) {
    return loom_vector_make_unknown_facts(result_facts);
  }

  loom_value_facts_t lanes[LOOM_VALUE_FACT_SMALL_STATIC_LANE_LIMIT] = {{0}};
  int64_t result_indices[LOOM_TYPE_MAX_RANK] = {0};
  int64_t source_indices[LOOM_TYPE_MAX_RANK] = {0};
  uint8_t rank_delta = (uint8_t)(result_rank - source_rank);
  for (iree_host_size_t lane = 0; lane < result_lane_count; ++lane) {
    loom_vector_static_indices_from_ordinal(result_type, lane, result_indices);
    for (uint8_t source_axis = 0; source_axis < source_rank; ++source_axis) {
      int64_t source_extent =
          loom_type_dim_static_size_at(source_type, source_axis);
      source_indices[source_axis] =
          source_extent == 1 ? 0 : result_indices[rank_delta + source_axis];
    }
    iree_host_size_t source_lane = 0;
    if (!loom_vector_static_ordinal_from_indices(source_type, source_indices,
                                                 &source_lane) ||
        !loom_vector_facts_query_lane(context, operand_facts[0], source_lane,
                                      &lanes[lane])) {
      return loom_vector_make_unknown_facts(result_facts);
    }
  }
  return loom_vector_make_small_static_lane_facts(
      context, lanes, result_lane_count, &result_facts[0]);
}

iree_status_t loom_vector_from_elements_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  loom_value_slice_t elements = loom_vector_from_elements_elements(op);
  if (elements.count > 0) {
    loom_value_id_t first_element = loom_value_slice_get(elements, 0);
    bool all_same_element = true;
    for (uint16_t i = 1; i < elements.count; ++i) {
      if (loom_value_slice_get(elements, i) != first_element) {
        all_same_element = false;
        break;
      }
    }
    if (all_same_element) {
      IREE_RETURN_IF_ERROR(loom_value_fact_table_define_uniform_element_origin(
          context->table, loom_vector_from_elements_result(op), first_element));
      return loom_value_facts_make_uniform_element(context, operand_facts[0],
                                                   &result_facts[0]);
    }
  }
  loom_value_fact_small_static_lanes_t lanes = {
      .lanes = operand_facts,
      .count = op->operand_count,
  };
  return loom_value_facts_make_small_static_lanes(context, lanes,
                                                  &result_facts[0]);
}

iree_status_t loom_vector_extract_facts(loom_fact_context_t* context,
                                        const loom_module_t* module,
                                        const loom_op_t* op,
                                        const loom_value_facts_t* operand_facts,
                                        loom_value_facts_t* result_facts) {
  loom_type_t source_type =
      loom_module_value_type(module, loom_vector_extract_source(op));
  loom_value_id_t result_id = loom_vector_extract_result(op);
  loom_type_t result_type = loom_module_value_type(module, result_id);
  loom_attribute_t static_indices = loom_vector_extract_static_indices(op);
  if (!loom_type_is_vector(source_type) ||
      static_indices.count > loom_type_rank(source_type)) {
    return loom_vector_make_unknown_facts(result_facts);
  }
  for (uint16_t i = 0; i < static_indices.count; ++i) {
    if (static_indices.i64_array[i] == INT64_MIN ||
        static_indices.i64_array[i] < 0) {
      return loom_vector_make_unknown_facts(result_facts);
    }
  }

  loom_value_facts_t uniform_element = {0};
  if (loom_vector_facts_query_uniform_element(context, operand_facts[0],
                                              &uniform_element)) {
    if (loom_type_is_scalar(result_type) &&
        static_indices.count == loom_type_rank(source_type)) {
      result_facts[0] = uniform_element;
      return iree_ok_status();
    }
    if (loom_type_is_vector(result_type) &&
        static_indices.count + loom_type_rank(result_type) ==
            loom_type_rank(source_type)) {
      return loom_value_facts_make_uniform_element(context, uniform_element,
                                                   &result_facts[0]);
    }
    return loom_vector_make_unknown_facts(result_facts);
  }

  int64_t source_indices[LOOM_TYPE_MAX_RANK] = {0};
  for (uint16_t i = 0; i < static_indices.count; ++i) {
    source_indices[i] = static_indices.i64_array[i];
  }

  if (loom_type_is_scalar(result_type)) {
    if (static_indices.count != loom_type_rank(source_type)) {
      return loom_vector_make_unknown_facts(result_facts);
    }
    iree_host_size_t lane = 0;
    if (!loom_vector_static_ordinal_from_indices(source_type, source_indices,
                                                 &lane) ||
        !loom_vector_facts_query_lane(context, operand_facts[0], lane,
                                      &result_facts[0])) {
      return loom_vector_make_unknown_facts(result_facts);
    }
    return iree_ok_status();
  }

  if (!loom_type_is_vector(result_type) ||
      static_indices.count + loom_type_rank(result_type) !=
          loom_type_rank(source_type)) {
    return loom_vector_make_unknown_facts(result_facts);
  }

  iree_host_size_t result_lane_count = 0;
  if (!loom_vector_type_static_lane_count(result_type, &result_lane_count) ||
      result_lane_count > LOOM_VALUE_FACT_SMALL_STATIC_LANE_LIMIT) {
    return loom_vector_make_unknown_facts(result_facts);
  }

  loom_value_facts_t lanes[LOOM_VALUE_FACT_SMALL_STATIC_LANE_LIMIT] = {{0}};
  int64_t result_indices[LOOM_TYPE_MAX_RANK] = {0};
  for (iree_host_size_t result_lane = 0; result_lane < result_lane_count;
       ++result_lane) {
    loom_vector_static_indices_from_ordinal(result_type, result_lane,
                                            result_indices);
    for (uint8_t axis = 0; axis < loom_type_rank(result_type); ++axis) {
      source_indices[static_indices.count + axis] = result_indices[axis];
    }
    iree_host_size_t source_lane = 0;
    if (!loom_vector_static_ordinal_from_indices(source_type, source_indices,
                                                 &source_lane) ||
        !loom_vector_facts_query_lane(context, operand_facts[0], source_lane,
                                      &lanes[result_lane])) {
      return loom_vector_make_unknown_facts(result_facts);
    }
  }
  return loom_vector_make_small_static_lane_facts(
      context, lanes, result_lane_count, &result_facts[0]);
}

iree_status_t loom_vector_insert_facts(loom_fact_context_t* context,
                                       const loom_module_t* module,
                                       const loom_op_t* op,
                                       const loom_value_facts_t* operand_facts,
                                       loom_value_facts_t* result_facts) {
  loom_type_t value_type =
      loom_module_value_type(module, loom_vector_insert_value(op));
  loom_type_t result_type =
      loom_module_value_type(module, loom_vector_insert_result(op));
  uint8_t result_rank = loom_type_rank(result_type);
  bool value_is_scalar = loom_type_is_scalar(value_type);
  loom_value_facts_t dest_uniform = {0};
  loom_value_facts_t value_uniform = {0};
  if (loom_vector_facts_query_uniform_element(context, operand_facts[1],
                                              &dest_uniform) &&
      (value_is_scalar || loom_vector_facts_query_uniform_element(
                              context, operand_facts[0], &value_uniform))) {
    if (value_is_scalar) {
      value_uniform = operand_facts[0];
    }
    if (loom_value_facts_is_exact(dest_uniform) &&
        loom_value_facts_equal(dest_uniform, value_uniform)) {
      return loom_value_facts_make_uniform_element(context, dest_uniform,
                                                   &result_facts[0]);
    }
  }
  loom_attribute_t static_indices = loom_vector_insert_static_indices(op);
  iree_host_size_t result_lane_count = 0;
  if (static_indices.count > result_rank ||
      !loom_vector_type_static_lane_count(result_type, &result_lane_count) ||
      result_lane_count > LOOM_VALUE_FACT_SMALL_STATIC_LANE_LIMIT) {
    return loom_vector_make_unknown_facts(result_facts);
  }
  for (uint16_t axis = 0; axis < static_indices.count; ++axis) {
    if (static_indices.i64_array[axis] == INT64_MIN ||
        static_indices.i64_array[axis] < 0) {
      return loom_vector_make_unknown_facts(result_facts);
    }
  }

  if (value_is_scalar && static_indices.count != result_rank) {
    return loom_vector_make_unknown_facts(result_facts);
  }
  if (!value_is_scalar &&
      (!loom_type_is_vector(value_type) ||
       static_indices.count + loom_type_rank(value_type) != result_rank ||
       !loom_type_is_all_static(value_type))) {
    return loom_vector_make_unknown_facts(result_facts);
  }

  loom_value_facts_t lanes[LOOM_VALUE_FACT_SMALL_STATIC_LANE_LIMIT] = {{0}};
  int64_t result_indices[LOOM_TYPE_MAX_RANK] = {0};
  int64_t value_indices[LOOM_TYPE_MAX_RANK] = {0};
  for (iree_host_size_t lane = 0; lane < result_lane_count; ++lane) {
    loom_vector_static_indices_from_ordinal(result_type, lane, result_indices);
    bool lane_is_inserted = true;
    for (uint16_t axis = 0; axis < static_indices.count; ++axis) {
      lane_is_inserted = lane_is_inserted &&
                         result_indices[axis] == static_indices.i64_array[axis];
    }
    if (!lane_is_inserted) {
      if (!loom_vector_facts_query_lane(context, operand_facts[1], lane,
                                        &lanes[lane])) {
        return loom_vector_make_unknown_facts(result_facts);
      }
      continue;
    }

    if (value_is_scalar) {
      lanes[lane] = operand_facts[0];
      continue;
    }

    uint8_t value_rank = loom_type_rank(value_type);
    for (uint8_t axis = 0; axis < value_rank; ++axis) {
      value_indices[axis] = result_indices[static_indices.count + axis];
    }
    iree_host_size_t value_lane = 0;
    if (!loom_vector_static_ordinal_from_indices(value_type, value_indices,
                                                 &value_lane) ||
        !loom_vector_facts_query_lane(context, operand_facts[0], value_lane,
                                      &lanes[lane])) {
      return loom_vector_make_unknown_facts(result_facts);
    }
  }
  return loom_vector_make_small_static_lane_facts(
      context, lanes, result_lane_count, &result_facts[0]);
}

iree_status_t loom_vector_iota_facts(loom_fact_context_t* context,
                                     const loom_module_t* module,
                                     const loom_op_t* op,
                                     const loom_value_facts_t* operand_facts,
                                     loom_value_facts_t* result_facts) {
  loom_value_fact_vector_iota_t iota = {
      .base = operand_facts[0],
      .step = operand_facts[1],
  };
  return loom_value_facts_make_vector_iota(context, iota, &result_facts[0]);
}

iree_status_t loom_vector_mask_range_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  loom_type_t result_type =
      loom_module_value_type(module, loom_vector_mask_range_result(op));
  uint64_t lane_count = 0;
  bool has_static_lane_count =
      loom_type_static_element_count(result_type, &lane_count);
  if (loom_value_facts_is_exact(operand_facts[0]) &&
      loom_value_facts_is_exact(operand_facts[1]) &&
      loom_value_facts_is_exact(operand_facts[2]) &&
      !loom_value_facts_is_float(operand_facts[0]) &&
      !loom_value_facts_is_float(operand_facts[1]) &&
      !loom_value_facts_is_float(operand_facts[2])) {
    if (has_static_lane_count) {
      bool handled = false;
      IREE_RETURN_IF_ERROR(loom_vector_mask_range_exact_static_facts(
          context, lane_count, operand_facts[0].range_lo,
          operand_facts[1].range_lo, operand_facts[2].range_lo,
          &result_facts[0], &handled));
      if (handled) return iree_ok_status();
    }

    int64_t lower_bound = operand_facts[0].range_lo;
    int64_t upper_bound = operand_facts[1].range_lo;
    int64_t step = operand_facts[2].range_lo;
    if ((step >= 0 && lower_bound >= upper_bound) ||
        (step <= 0 && lower_bound < upper_bound)) {
      return loom_value_facts_make_uniform_element(
          context,
          loom_value_facts_exact_i64(lower_bound < upper_bound ? 1 : 0),
          &result_facts[0]);
    }
  }
  if (has_static_lane_count) {
    bool handled = false;
    IREE_RETURN_IF_ERROR(loom_vector_mask_range_bounded_static_facts(
        context, lane_count, operand_facts[0], operand_facts[1],
        operand_facts[2], &result_facts[0], &handled));
    if (handled) return iree_ok_status();
  }
  loom_value_fact_vector_prefix_mask_t mask = {
      .lower_bound = operand_facts[0],
      .upper_bound = operand_facts[1],
      .step = operand_facts[2],
  };
  return loom_value_facts_make_vector_prefix_mask(context, mask,
                                                  &result_facts[0]);
}

iree_status_t loom_vector_shuffle_facts(loom_fact_context_t* context,
                                        const loom_module_t* module,
                                        const loom_op_t* op,
                                        const loom_value_facts_t* operand_facts,
                                        loom_value_facts_t* result_facts) {
  loom_value_facts_t uniform = {0};
  if (loom_vector_facts_query_uniform_element(context, operand_facts[0],
                                              &uniform)) {
    return loom_value_facts_make_uniform_element(context, uniform,
                                                 &result_facts[0]);
  }

  loom_attribute_t source_lanes = loom_vector_shuffle_source_lanes(op);
  if (source_lanes.count > LOOM_VALUE_FACT_SMALL_STATIC_LANE_LIMIT) {
    result_facts[0] = loom_value_facts_unknown();
    return iree_ok_status();
  }
  loom_value_facts_t lanes[LOOM_VALUE_FACT_SMALL_STATIC_LANE_LIMIT] = {{0}};
  for (uint16_t i = 0; i < source_lanes.count; ++i) {
    int64_t source_lane = source_lanes.i64_array[i];
    if (source_lane < 0 || !loom_vector_facts_query_lane(
                               context, operand_facts[0],
                               (iree_host_size_t)source_lane, &lanes[i])) {
      result_facts[0] = loom_value_facts_unknown();
      return iree_ok_status();
    }
  }
  loom_value_fact_small_static_lanes_t lane_slice = {
      .lanes = lanes,
      .count = source_lanes.count,
  };
  return loom_value_facts_make_small_static_lanes(context, lane_slice,
                                                  &result_facts[0]);
}

iree_status_t loom_vector_slice_facts(loom_fact_context_t* context,
                                      const loom_module_t* module,
                                      const loom_op_t* op,
                                      const loom_value_facts_t* operand_facts,
                                      loom_value_facts_t* result_facts) {
  loom_type_t source_type =
      loom_module_value_type(module, loom_vector_slice_source(op));
  loom_type_t result_type =
      loom_module_value_type(module, loom_vector_slice_result(op));
  if (loom_type_rank(source_type) != loom_type_rank(result_type)) {
    return loom_vector_make_unknown_facts(result_facts);
  }

  uint8_t rank = loom_type_rank(result_type);
  loom_attribute_t static_offsets = loom_vector_slice_static_offsets(op);
  if (static_offsets.count != rank)
    return loom_vector_make_unknown_facts(result_facts);
  for (uint8_t axis = 0; axis < rank; ++axis) {
    if (static_offsets.i64_array[axis] == INT64_MIN) {
      return loom_vector_make_unknown_facts(result_facts);
    }
  }
  const loom_value_id_t source = loom_vector_slice_source(op);
  const loom_value_id_t result = loom_vector_slice_result(op);
  if (rank == 1 && static_offsets.i64_array[0] >= 0 &&
      static_offsets.i64_array[0] <= UINT32_MAX) {
    loom_value_fact_static_lane_origin_t source_origin = {
        .source_value_id = source,
        .source_lane_offset = 0,
        .source_lane_stride = 1,
    };
    loom_value_fact_static_lane_origin_t existing_origin = {0};
    if (loom_value_fact_table_query_static_lane_origin(
            context->table, module, source, &existing_origin)) {
      source_origin = existing_origin;
    }
    const uint64_t source_lane_offset =
        (uint64_t)source_origin.source_lane_offset +
        (uint64_t)(uint32_t)static_offsets.i64_array[0] *
            (uint64_t)source_origin.source_lane_stride;
    if (source_lane_offset <= UINT32_MAX) {
      IREE_RETURN_IF_ERROR(loom_value_fact_table_define_static_lane_origin(
          context->table, result,
          (loom_value_fact_static_lane_origin_t){
              .source_value_id = source_origin.source_value_id,
              .source_lane_offset = (uint32_t)source_lane_offset,
              .source_lane_stride = source_origin.source_lane_stride,
          }));
    }
  }

  iree_host_size_t result_lane_count = 0;
  if (!loom_vector_type_static_lane_count(result_type, &result_lane_count) ||
      result_lane_count > LOOM_VALUE_FACT_SMALL_STATIC_LANE_LIMIT) {
    return loom_vector_make_unknown_facts(result_facts);
  }

  loom_value_facts_t lanes[LOOM_VALUE_FACT_SMALL_STATIC_LANE_LIMIT] = {{0}};
  int64_t source_indices[LOOM_TYPE_MAX_RANK] = {0};
  for (iree_host_size_t lane = 0; lane < result_lane_count; ++lane) {
    loom_vector_static_indices_from_ordinal(result_type, lane, source_indices);
    for (uint8_t axis = 0; axis < rank; ++axis) {
      if (!iree_checked_add_i64(source_indices[axis],
                                static_offsets.i64_array[axis],
                                &source_indices[axis])) {
        return loom_vector_make_unknown_facts(result_facts);
      }
    }
    iree_host_size_t source_lane = 0;
    if (!loom_vector_static_ordinal_from_indices(source_type, source_indices,
                                                 &source_lane) ||
        !loom_vector_facts_query_lane(context, operand_facts[0], source_lane,
                                      &lanes[lane])) {
      return loom_vector_make_unknown_facts(result_facts);
    }
  }
  return loom_vector_make_small_static_lane_facts(
      context, lanes, result_lane_count, &result_facts[0]);
}

iree_status_t loom_vector_concat_facts(loom_fact_context_t* context,
                                       const loom_module_t* module,
                                       const loom_op_t* op,
                                       const loom_value_facts_t* operand_facts,
                                       loom_value_facts_t* result_facts) {
  loom_type_t result_type =
      loom_module_value_type(module, loom_vector_concat_result(op));
  int64_t axis = loom_vector_concat_axis(op);
  iree_host_size_t result_lane_count = 0;
  if (axis < 0 || axis >= loom_type_rank(result_type) ||
      !loom_vector_type_static_lane_count(result_type, &result_lane_count) ||
      result_lane_count > LOOM_VALUE_FACT_SMALL_STATIC_LANE_LIMIT) {
    return loom_vector_make_unknown_facts(result_facts);
  }

  loom_value_facts_t lanes[LOOM_VALUE_FACT_SMALL_STATIC_LANE_LIMIT] = {{0}};
  int64_t result_indices[LOOM_TYPE_MAX_RANK] = {0};
  int64_t input_indices[LOOM_TYPE_MAX_RANK] = {0};
  const loom_value_id_t* operands = loom_op_const_operands(op);
  for (iree_host_size_t lane = 0; lane < result_lane_count; ++lane) {
    loom_vector_static_indices_from_ordinal(result_type, lane, result_indices);
    memcpy(input_indices, result_indices,
           loom_type_rank(result_type) * sizeof(input_indices[0]));

    int64_t axis_index = result_indices[axis];
    int64_t axis_base = 0;
    bool found_input = false;
    for (uint16_t operand_index = 0; operand_index < op->operand_count;
         ++operand_index) {
      loom_type_t input_type =
          loom_module_value_type(module, operands[operand_index]);
      if (loom_type_rank(input_type) != loom_type_rank(result_type) ||
          loom_type_dim_is_dynamic_at(input_type, (iree_host_size_t)axis)) {
        return loom_vector_make_unknown_facts(result_facts);
      }
      int64_t input_axis_size =
          loom_type_dim_static_size_at(input_type, (iree_host_size_t)axis);
      int64_t next_axis_base = 0;
      if (!iree_checked_add_i64(axis_base, input_axis_size, &next_axis_base)) {
        return loom_vector_make_unknown_facts(result_facts);
      }
      if (axis_index < next_axis_base) {
        input_indices[axis] = axis_index - axis_base;
        iree_host_size_t input_lane = 0;
        if (!loom_vector_static_ordinal_from_indices(input_type, input_indices,
                                                     &input_lane) ||
            !loom_vector_facts_query_lane(context, operand_facts[operand_index],
                                          input_lane, &lanes[lane])) {
          return loom_vector_make_unknown_facts(result_facts);
        }
        found_input = true;
        break;
      }
      axis_base = next_axis_base;
    }
    if (!found_input) return loom_vector_make_unknown_facts(result_facts);
  }
  return loom_vector_make_small_static_lane_facts(
      context, lanes, result_lane_count, &result_facts[0]);
}

iree_status_t loom_vector_transpose_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  loom_value_facts_t uniform_element = {0};
  if (loom_vector_facts_query_uniform_element(context, operand_facts[0],
                                              &uniform_element)) {
    return loom_value_facts_make_uniform_element(context, uniform_element,
                                                 &result_facts[0]);
  }

  loom_type_t source_type =
      loom_module_value_type(module, loom_vector_transpose_source(op));
  loom_type_t result_type =
      loom_module_value_type(module, loom_vector_transpose_result(op));
  loom_attribute_t permutation = loom_vector_transpose_permutation(op);
  uint8_t rank = loom_type_rank(result_type);
  iree_host_size_t result_lane_count = 0;
  if (loom_type_rank(source_type) != rank || permutation.count != rank ||
      !loom_vector_type_static_lane_count(result_type, &result_lane_count) ||
      result_lane_count > LOOM_VALUE_FACT_SMALL_STATIC_LANE_LIMIT ||
      !loom_type_is_all_static(source_type)) {
    return loom_vector_make_unknown_facts(result_facts);
  }

  loom_value_facts_t lanes[LOOM_VALUE_FACT_SMALL_STATIC_LANE_LIMIT] = {{0}};
  int64_t result_indices[LOOM_TYPE_MAX_RANK] = {0};
  int64_t source_indices[LOOM_TYPE_MAX_RANK] = {0};
  for (iree_host_size_t lane = 0; lane < result_lane_count; ++lane) {
    loom_vector_static_indices_from_ordinal(result_type, lane, result_indices);
    for (uint8_t result_axis = 0; result_axis < rank; ++result_axis) {
      int64_t source_axis = permutation.i64_array[result_axis];
      if (source_axis < 0 || source_axis >= rank) {
        return loom_vector_make_unknown_facts(result_facts);
      }
      source_indices[source_axis] = result_indices[result_axis];
    }
    iree_host_size_t source_lane = 0;
    if (!loom_vector_static_ordinal_from_indices(source_type, source_indices,
                                                 &source_lane) ||
        !loom_vector_facts_query_lane(context, operand_facts[0], source_lane,
                                      &lanes[lane])) {
      return loom_vector_make_unknown_facts(result_facts);
    }
  }
  return loom_vector_make_small_static_lane_facts(
      context, lanes, result_lane_count, &result_facts[0]);
}

iree_status_t loom_vector_interleave_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  loom_type_t even_type =
      loom_module_value_type(module, loom_vector_interleave_even(op));
  loom_type_t result_type =
      loom_module_value_type(module, loom_vector_interleave_result(op));
  int64_t axis = loom_vector_interleave_axis(op);
  iree_host_size_t result_lane_count = 0;
  if (axis < 0 || axis >= loom_type_rank(result_type) ||
      loom_type_rank(even_type) != loom_type_rank(result_type) ||
      !loom_vector_type_static_lane_count(result_type, &result_lane_count) ||
      result_lane_count > LOOM_VALUE_FACT_SMALL_STATIC_LANE_LIMIT) {
    return loom_vector_make_unknown_facts(result_facts);
  }

  loom_value_facts_t lanes[LOOM_VALUE_FACT_SMALL_STATIC_LANE_LIMIT] = {{0}};
  int64_t source_indices[LOOM_TYPE_MAX_RANK] = {0};
  for (iree_host_size_t lane = 0; lane < result_lane_count; ++lane) {
    loom_vector_static_indices_from_ordinal(result_type, lane, source_indices);
    uint16_t operand_index = (source_indices[axis] & 1) ? 1 : 0;
    source_indices[axis] /= 2;
    iree_host_size_t source_lane = 0;
    if (!loom_vector_static_ordinal_from_indices(even_type, source_indices,
                                                 &source_lane) ||
        !loom_vector_facts_query_lane(context, operand_facts[operand_index],
                                      source_lane, &lanes[lane])) {
      return loom_vector_make_unknown_facts(result_facts);
    }
  }
  return loom_vector_make_small_static_lane_facts(
      context, lanes, result_lane_count, &result_facts[0]);
}

iree_status_t loom_vector_deinterleave_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  if (op->result_count != 2) {
    return loom_vector_make_unknown_result_facts(result_facts,
                                                 op->result_count);
  }
  const loom_value_id_t source = loom_vector_deinterleave_source(op);
  loom_type_t source_type = loom_module_value_type(module, source);
  int64_t axis = loom_vector_deinterleave_axis(op);
  if (axis < 0 || axis >= loom_type_rank(source_type)) {
    return loom_vector_make_unknown_result_facts(result_facts,
                                                 op->result_count);
  }

  const loom_value_id_t* results = loom_op_const_results(op);
  for (uint16_t result_index = 0; result_index < op->result_count;
       ++result_index) {
    loom_type_t result_type =
        loom_module_value_type(module, results[result_index]);
    iree_host_size_t result_lane_count = 0;
    if (loom_type_rank(result_type) != loom_type_rank(source_type)) {
      return loom_vector_make_unknown_result_facts(result_facts,
                                                   op->result_count);
    }
    if (axis == 0 && loom_type_rank(source_type) == 1) {
      IREE_RETURN_IF_ERROR(loom_value_fact_table_define_static_lane_origin(
          context->table, results[result_index],
          (loom_value_fact_static_lane_origin_t){
              .source_value_id = source,
              .source_lane_offset = result_index,
              .source_lane_stride = 2,
          }));
    }
    if (!loom_vector_type_static_lane_count(result_type, &result_lane_count) ||
        result_lane_count > LOOM_VALUE_FACT_SMALL_STATIC_LANE_LIMIT) {
      result_facts[result_index] = loom_value_facts_unknown();
      continue;
    }

    loom_value_facts_t lanes[LOOM_VALUE_FACT_SMALL_STATIC_LANE_LIMIT] = {{0}};
    int64_t source_indices[LOOM_TYPE_MAX_RANK] = {0};
    for (iree_host_size_t lane = 0; lane < result_lane_count; ++lane) {
      loom_vector_static_indices_from_ordinal(result_type, lane,
                                              source_indices);
      if (!iree_checked_mul_i64(source_indices[axis], 2,
                                &source_indices[axis]) ||
          !iree_checked_add_i64(source_indices[axis], result_index,
                                &source_indices[axis])) {
        return loom_vector_make_unknown_result_facts(result_facts,
                                                     op->result_count);
      }
      iree_host_size_t source_lane = 0;
      if (!loom_vector_static_ordinal_from_indices(source_type, source_indices,
                                                   &source_lane) ||
          !loom_vector_facts_query_lane(context, operand_facts[0], source_lane,
                                        &lanes[lane])) {
        return loom_vector_make_unknown_result_facts(result_facts,
                                                     op->result_count);
      }
    }
    IREE_RETURN_IF_ERROR(loom_vector_make_small_static_lane_facts(
        context, lanes, result_lane_count, &result_facts[result_index]));
  }
  return iree_ok_status();
}

iree_status_t loom_vector_transform_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  loom_encoding_hadamard_descriptor_t descriptor;
  if (!loom_encoding_hadamard_try_read_verified_descriptor(
          module, loom_vector_transform_transform(op), &descriptor)) {
    return loom_vector_make_unknown_facts(result_facts);
  }

  const loom_type_t source_type =
      loom_module_value_type(module, loom_vector_transform_source(op));
  return loom_vector_transform_hadamard_facts(
      context, &descriptor, source_type, operand_facts[0], &result_facts[0]);
}

//===----------------------------------------------------------------------===//
// Lanewise summary propagation
//===----------------------------------------------------------------------===//

static iree_status_t loom_vector_integer_binary_summary_facts(
    loom_fact_context_t* context, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts,
    loom_vector_integer_binary_transfer_fn_t transfer_fn) {
  loom_value_facts_t lhs = {0};
  loom_value_facts_t rhs = {0};
  if (loom_vector_facts_query_uniform_element(context, operand_facts[0],
                                              &lhs) &&
      loom_vector_facts_query_uniform_element(context, operand_facts[1],
                                              &rhs)) {
    loom_value_facts_t element = loom_value_facts_unknown();
    transfer_fn(&lhs, &rhs, &element);
    return loom_value_facts_make_uniform_element(context, element,
                                                 &result_facts[0]);
  }

  iree_host_size_t lane_count = 0;
  if (!loom_vector_facts_query_binary_lane_count(
          context, operand_facts[0], operand_facts[1], &lane_count)) {
    result_facts[0] = loom_value_facts_unknown();
    return iree_ok_status();
  }
  loom_value_facts_t lanes[LOOM_VALUE_FACT_SMALL_STATIC_LANE_LIMIT] = {{0}};
  for (iree_host_size_t i = 0; i < lane_count; ++i) {
    if (!loom_vector_facts_query_lane(context, operand_facts[0], i, &lhs) ||
        !loom_vector_facts_query_lane(context, operand_facts[1], i, &rhs)) {
      result_facts[0] = loom_value_facts_unknown();
      return iree_ok_status();
    }
    transfer_fn(&lhs, &rhs, &lanes[i]);
  }
  loom_value_fact_small_static_lanes_t lane_slice = {
      .lanes = lanes,
      .count = lane_count,
  };
  return loom_value_facts_make_small_static_lanes(context, lane_slice,
                                                  &result_facts[0]);
}

static iree_status_t loom_vector_unary_summary_facts(
    loom_fact_context_t* context, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts, loom_vector_unary_transfer_fn_t fn) {
  loom_value_facts_t input = {0};
  if (loom_vector_facts_query_uniform_element(context, operand_facts[0],
                                              &input)) {
    loom_value_facts_t element = loom_value_facts_unknown();
    fn(&input, &element);
    return loom_value_facts_make_uniform_element(context, element,
                                                 &result_facts[0]);
  }

  loom_value_fact_small_static_lanes_t input_lanes = {0};
  if (!loom_vector_facts_query_small_lanes(context, operand_facts[0],
                                           &input_lanes)) {
    result_facts[0] = loom_value_facts_unknown();
    return iree_ok_status();
  }
  loom_value_facts_t lanes[LOOM_VALUE_FACT_SMALL_STATIC_LANE_LIMIT] = {{0}};
  for (iree_host_size_t i = 0; i < input_lanes.count; ++i) {
    fn(&input_lanes.lanes[i], &lanes[i]);
  }
  loom_value_fact_small_static_lanes_t lane_slice = {
      .lanes = lanes,
      .count = input_lanes.count,
  };
  return loom_value_facts_make_small_static_lanes(context, lane_slice,
                                                  &result_facts[0]);
}

static void loom_vector_bit_count_element_facts(const loom_value_facts_t* input,
                                                int32_t bitwidth,
                                                loom_vector_bit_count_fn_t fn,
                                                loom_value_facts_t* out) {
  if (!loom_value_facts_is_exact(*input)) {
    *out = loom_value_facts_unknown();
    return;
  }
  *out = loom_value_facts_exact_i64(fn((uint64_t)input->range_lo, bitwidth));
}

static iree_status_t loom_vector_bit_count_summary_facts(
    loom_fact_context_t* context, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts, int32_t bitwidth,
    loom_vector_bit_count_fn_t fn) {
  if (bitwidth <= 0) {
    result_facts[0] = loom_value_facts_unknown();
    return iree_ok_status();
  }

  loom_value_facts_t input = {0};
  if (loom_vector_facts_query_uniform_element(context, operand_facts[0],
                                              &input)) {
    loom_value_facts_t element = loom_value_facts_unknown();
    loom_vector_bit_count_element_facts(&input, bitwidth, fn, &element);
    return loom_value_facts_make_uniform_element(context, element,
                                                 &result_facts[0]);
  }

  loom_value_fact_small_static_lanes_t input_lanes = {0};
  if (!loom_vector_facts_query_small_lanes(context, operand_facts[0],
                                           &input_lanes)) {
    result_facts[0] = loom_value_facts_unknown();
    return iree_ok_status();
  }
  loom_value_facts_t lanes[LOOM_VALUE_FACT_SMALL_STATIC_LANE_LIMIT] = {{0}};
  for (iree_host_size_t i = 0; i < input_lanes.count; ++i) {
    loom_vector_bit_count_element_facts(&input_lanes.lanes[i], bitwidth, fn,
                                        &lanes[i]);
  }
  loom_value_fact_small_static_lanes_t lane_slice = {
      .lanes = lanes,
      .count = input_lanes.count,
  };
  return loom_value_facts_make_small_static_lanes(context, lane_slice,
                                                  &result_facts[0]);
}

static iree_status_t loom_vector_float_unary_summary_facts(
    loom_fact_context_t* context, loom_scalar_type_t scalar_type,
    const loom_value_facts_t* operand_facts, loom_value_facts_t* result_facts,
    loom_vector_float_unary_fact_transfer_fn_t transfer_fn,
    const void* user_data) {
  bool fragment_handled = false;
  IREE_RETURN_IF_ERROR(loom_vector_try_preserve_lanewise_fragment_facts(
      context, operand_facts, 1, result_facts, &fragment_handled));
  if (fragment_handled) {
    return iree_ok_status();
  }

  loom_value_facts_t input = {0};
  if (loom_vector_facts_query_uniform_element(context, operand_facts[0],
                                              &input)) {
    loom_value_facts_t element = loom_value_facts_unknown();
    transfer_fn(scalar_type, &input, user_data, &element);
    return loom_value_facts_make_uniform_element(context, element,
                                                 &result_facts[0]);
  }

  loom_value_fact_small_static_lanes_t input_lanes = {0};
  if (!loom_vector_facts_query_small_lanes(context, operand_facts[0],
                                           &input_lanes)) {
    result_facts[0] = loom_value_facts_unknown();
    return iree_ok_status();
  }
  loom_value_facts_t lanes[LOOM_VALUE_FACT_SMALL_STATIC_LANE_LIMIT] = {{0}};
  for (iree_host_size_t i = 0; i < input_lanes.count; ++i) {
    transfer_fn(scalar_type, &input_lanes.lanes[i], user_data, &lanes[i]);
  }
  loom_value_fact_small_static_lanes_t lane_slice = {
      .lanes = lanes,
      .count = input_lanes.count,
  };
  return loom_value_facts_make_small_static_lanes(context, lane_slice,
                                                  &result_facts[0]);
}

static iree_status_t loom_vector_float_classify_summary_facts(
    loom_fact_context_t* context, loom_scalar_type_t scalar_type,
    const loom_value_facts_t* operand_facts, loom_value_facts_t* result_facts,
    loom_vector_float_unary_fact_transfer_fn_t transfer_fn) {
  loom_value_facts_t element = loom_value_facts_unknown();
  transfer_fn(scalar_type, &operand_facts[0], NULL, &element);
  if (loom_value_facts_is_exact(element) &&
      !loom_value_facts_is_float(element)) {
    return loom_value_facts_make_uniform_element(context, element,
                                                 &result_facts[0]);
  }
  return loom_vector_float_unary_summary_facts(
      context, scalar_type, operand_facts, result_facts, transfer_fn, NULL);
}

typedef struct loom_vector_float_unary_math_transfer_t {
  loom_float_unary_f32_fn_t f32_fn;
  loom_float_unary_f64_fn_t f64_fn;
} loom_vector_float_unary_math_transfer_t;

static void loom_vector_float_unary_math_transfer(
    loom_scalar_type_t scalar_type, const loom_value_facts_t* input,
    const void* user_data, loom_value_facts_t* out) {
  const loom_vector_float_unary_math_transfer_t* transfer = user_data;
  loom_value_facts_eval_float_unary(scalar_type, input, transfer->f32_fn,
                                    transfer->f64_fn, out);
}

static iree_status_t loom_vector_float_unary_math_summary_facts(
    loom_fact_context_t* context, loom_scalar_type_t scalar_type,
    const loom_value_facts_t* operand_facts, loom_value_facts_t* result_facts,
    loom_float_unary_f32_fn_t f32_fn, loom_float_unary_f64_fn_t f64_fn) {
  const loom_vector_float_unary_math_transfer_t transfer = {
      .f32_fn = f32_fn,
      .f64_fn = f64_fn,
  };
  return loom_vector_float_unary_summary_facts(
      context, scalar_type, operand_facts, result_facts,
      loom_vector_float_unary_math_transfer, &transfer);
}

typedef struct loom_vector_float_unary_data_math_transfer_t {
  loom_float_unary_data_f32_fn_t f32_fn;
  loom_float_unary_data_f64_fn_t f64_fn;
  const void* user_data;
} loom_vector_float_unary_data_math_transfer_t;

static void loom_vector_float_unary_data_math_transfer(
    loom_scalar_type_t scalar_type, const loom_value_facts_t* input,
    const void* user_data, loom_value_facts_t* out) {
  const loom_vector_float_unary_data_math_transfer_t* transfer = user_data;
  loom_value_facts_eval_float_unary_data(scalar_type, input, transfer->f32_fn,
                                         transfer->f64_fn, transfer->user_data,
                                         out);
}

static iree_status_t loom_vector_float_unary_data_math_summary_facts(
    loom_fact_context_t* context, loom_scalar_type_t scalar_type,
    const loom_value_facts_t* operand_facts, loom_value_facts_t* result_facts,
    loom_float_unary_data_f32_fn_t f32_fn,
    loom_float_unary_data_f64_fn_t f64_fn, const void* user_data) {
  const loom_vector_float_unary_data_math_transfer_t transfer = {
      .f32_fn = f32_fn,
      .f64_fn = f64_fn,
      .user_data = user_data,
  };
  return loom_vector_float_unary_summary_facts(
      context, scalar_type, operand_facts, result_facts,
      loom_vector_float_unary_data_math_transfer, &transfer);
}

static iree_status_t loom_vector_float_binary_summary_facts(
    loom_fact_context_t* context, loom_scalar_type_t scalar_type,
    const loom_value_facts_t* operand_facts, loom_value_facts_t* result_facts,
    loom_vector_float_binary_fact_transfer_fn_t transfer_fn,
    const void* user_data) {
  bool fragment_handled = false;
  IREE_RETURN_IF_ERROR(loom_vector_try_preserve_lanewise_fragment_facts(
      context, operand_facts, 2, result_facts, &fragment_handled));
  if (fragment_handled) {
    return iree_ok_status();
  }

  loom_value_facts_t lhs = {0};
  loom_value_facts_t rhs = {0};
  if (loom_vector_facts_query_uniform_element(context, operand_facts[0],
                                              &lhs) &&
      loom_vector_facts_query_uniform_element(context, operand_facts[1],
                                              &rhs)) {
    loom_value_facts_t element = loom_value_facts_unknown();
    transfer_fn(scalar_type, &lhs, &rhs, user_data, &element);
    return loom_value_facts_make_uniform_element(context, element,
                                                 &result_facts[0]);
  }

  iree_host_size_t lane_count = 0;
  if (!loom_vector_facts_query_binary_lane_count(
          context, operand_facts[0], operand_facts[1], &lane_count)) {
    result_facts[0] = loom_value_facts_unknown();
    return iree_ok_status();
  }
  loom_value_facts_t lanes[LOOM_VALUE_FACT_SMALL_STATIC_LANE_LIMIT] = {{0}};
  for (iree_host_size_t i = 0; i < lane_count; ++i) {
    if (!loom_vector_facts_query_lane(context, operand_facts[0], i, &lhs) ||
        !loom_vector_facts_query_lane(context, operand_facts[1], i, &rhs)) {
      result_facts[0] = loom_value_facts_unknown();
      return iree_ok_status();
    }
    transfer_fn(scalar_type, &lhs, &rhs, user_data, &lanes[i]);
  }
  loom_value_fact_small_static_lanes_t lane_slice = {
      .lanes = lanes,
      .count = lane_count,
  };
  return loom_value_facts_make_small_static_lanes(context, lane_slice,
                                                  &result_facts[0]);
}

typedef struct loom_vector_float_binary_math_transfer_t {
  loom_float_binary_f32_fn_t f32_fn;
  loom_float_binary_f64_fn_t f64_fn;
} loom_vector_float_binary_math_transfer_t;

static void loom_vector_float_binary_math_transfer(
    loom_scalar_type_t scalar_type, const loom_value_facts_t* lhs,
    const loom_value_facts_t* rhs, const void* user_data,
    loom_value_facts_t* out) {
  const loom_vector_float_binary_math_transfer_t* transfer = user_data;
  loom_value_facts_eval_float_binary(scalar_type, lhs, rhs, transfer->f32_fn,
                                     transfer->f64_fn, out);
}

static iree_status_t loom_vector_float_binary_math_summary_facts(
    loom_fact_context_t* context, loom_scalar_type_t scalar_type,
    const loom_value_facts_t* operand_facts, loom_value_facts_t* result_facts,
    loom_float_binary_f32_fn_t f32_fn, loom_float_binary_f64_fn_t f64_fn) {
  const loom_vector_float_binary_math_transfer_t transfer = {
      .f32_fn = f32_fn,
      .f64_fn = f64_fn,
  };
  return loom_vector_float_binary_summary_facts(
      context, scalar_type, operand_facts, result_facts,
      loom_vector_float_binary_math_transfer, &transfer);
}

static iree_status_t loom_vector_ternary_summary_facts(
    loom_fact_context_t* context, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts, loom_vector_ternary_transfer_fn_t fn) {
  bool fragment_handled = false;
  IREE_RETURN_IF_ERROR(loom_vector_try_preserve_lanewise_fragment_facts(
      context, operand_facts, 3, result_facts, &fragment_handled));
  if (fragment_handled) {
    return iree_ok_status();
  }

  loom_value_facts_t a = {0};
  loom_value_facts_t b = {0};
  loom_value_facts_t c = {0};
  if (loom_vector_facts_query_uniform_element(context, operand_facts[0], &a) &&
      loom_vector_facts_query_uniform_element(context, operand_facts[1], &b) &&
      loom_vector_facts_query_uniform_element(context, operand_facts[2], &c)) {
    loom_value_facts_t element = loom_value_facts_unknown();
    fn(&a, &b, &c, &element);
    return loom_value_facts_make_uniform_element(context, element,
                                                 &result_facts[0]);
  }

  iree_host_size_t lane_count = 0;
  if (!loom_vector_facts_query_ternary_lane_count(
          context, operand_facts[0], operand_facts[1], operand_facts[2],
          &lane_count)) {
    result_facts[0] = loom_value_facts_unknown();
    return iree_ok_status();
  }
  loom_value_facts_t lanes[LOOM_VALUE_FACT_SMALL_STATIC_LANE_LIMIT] = {{0}};
  for (iree_host_size_t i = 0; i < lane_count; ++i) {
    if (!loom_vector_facts_query_lane(context, operand_facts[0], i, &a) ||
        !loom_vector_facts_query_lane(context, operand_facts[1], i, &b) ||
        !loom_vector_facts_query_lane(context, operand_facts[2], i, &c)) {
      result_facts[0] = loom_value_facts_unknown();
      return iree_ok_status();
    }
    fn(&a, &b, &c, &lanes[i]);
  }
  loom_value_fact_small_static_lanes_t lane_slice = {
      .lanes = lanes,
      .count = lane_count,
  };
  return loom_value_facts_make_small_static_lanes(context, lane_slice,
                                                  &result_facts[0]);
}

static iree_status_t loom_vector_float_ternary_summary_facts(
    loom_fact_context_t* context, loom_scalar_type_t scalar_type,
    const loom_value_facts_t* operand_facts, loom_value_facts_t* result_facts,
    loom_vector_float_ternary_fact_transfer_fn_t transfer_fn,
    const void* user_data) {
  bool fragment_handled = false;
  IREE_RETURN_IF_ERROR(loom_vector_try_preserve_lanewise_fragment_facts(
      context, operand_facts, 3, result_facts, &fragment_handled));
  if (fragment_handled) return iree_ok_status();

  loom_value_facts_t a = {0};
  loom_value_facts_t b = {0};
  loom_value_facts_t c = {0};
  if (loom_vector_facts_query_uniform_element(context, operand_facts[0], &a) &&
      loom_vector_facts_query_uniform_element(context, operand_facts[1], &b) &&
      loom_vector_facts_query_uniform_element(context, operand_facts[2], &c)) {
    loom_value_facts_t element = loom_value_facts_unknown();
    transfer_fn(scalar_type, &a, &b, &c, user_data, &element);
    return loom_value_facts_make_uniform_element(context, element,
                                                 &result_facts[0]);
  }

  iree_host_size_t lane_count = 0;
  if (!loom_vector_facts_query_ternary_lane_count(
          context, operand_facts[0], operand_facts[1], operand_facts[2],
          &lane_count)) {
    result_facts[0] = loom_value_facts_unknown();
    return iree_ok_status();
  }
  loom_value_facts_t lanes[LOOM_VALUE_FACT_SMALL_STATIC_LANE_LIMIT] = {{0}};
  for (iree_host_size_t i = 0; i < lane_count; ++i) {
    if (!loom_vector_facts_query_lane(context, operand_facts[0], i, &a) ||
        !loom_vector_facts_query_lane(context, operand_facts[1], i, &b) ||
        !loom_vector_facts_query_lane(context, operand_facts[2], i, &c)) {
      result_facts[0] = loom_value_facts_unknown();
      return iree_ok_status();
    }
    transfer_fn(scalar_type, &a, &b, &c, user_data, &lanes[i]);
  }
  loom_value_fact_small_static_lanes_t lane_slice = {
      .lanes = lanes,
      .count = lane_count,
  };
  return loom_value_facts_make_small_static_lanes(context, lane_slice,
                                                  &result_facts[0]);
}

#define LOOM_VECTOR_INTEGER_BINARY_FACTS(name, transfer_fn)            \
  iree_status_t name(loom_fact_context_t* context,                     \
                     const loom_module_t* module, const loom_op_t* op, \
                     const loom_value_facts_t* operand_facts,          \
                     loom_value_facts_t* result_facts) {               \
    return loom_vector_integer_binary_summary_facts(                   \
        context, operand_facts, result_facts, transfer_fn);            \
  }

#define LOOM_VECTOR_FLOAT_BINARY_FACTS(name, f32_fn, f64_fn)                 \
  iree_status_t name(loom_fact_context_t* context,                           \
                     const loom_module_t* module, const loom_op_t* op,       \
                     const loom_value_facts_t* operand_facts,                \
                     loom_value_facts_t* result_facts) {                     \
    return loom_vector_float_binary_math_summary_facts(                      \
        context, loom_vector_result_element_type(module, op), operand_facts, \
        result_facts, f32_fn, f64_fn);                                       \
  }

#define LOOM_VECTOR_FLOAT_UNARY_FACTS(name, f32_fn, f64_fn)                  \
  iree_status_t name(loom_fact_context_t* context,                           \
                     const loom_module_t* module, const loom_op_t* op,       \
                     const loom_value_facts_t* operand_facts,                \
                     loom_value_facts_t* result_facts) {                     \
    return loom_vector_float_unary_math_summary_facts(                       \
        context, loom_vector_result_element_type(module, op), operand_facts, \
        result_facts, f32_fn, f64_fn);                                       \
  }

#define LOOM_VECTOR_UNARY_FACTS(name, fn)                              \
  iree_status_t name(loom_fact_context_t* context,                     \
                     const loom_module_t* module, const loom_op_t* op, \
                     const loom_value_facts_t* operand_facts,          \
                     loom_value_facts_t* result_facts) {               \
    return loom_vector_unary_summary_facts(context, operand_facts,     \
                                           result_facts, fn);          \
  }

#define LOOM_VECTOR_BIT_COUNT_FACTS(name, result_accessor, fn)              \
  iree_status_t name(loom_fact_context_t* context,                          \
                     const loom_module_t* module, const loom_op_t* op,      \
                     const loom_value_facts_t* operand_facts,               \
                     loom_value_facts_t* result_facts) {                    \
    loom_type_t result_type =                                               \
        loom_module_value_type(module, result_accessor(op));                \
    int32_t bitwidth =                                                      \
        loom_scalar_type_bitwidth(loom_type_element_type(result_type));     \
    return loom_vector_bit_count_summary_facts(context, operand_facts,      \
                                               result_facts, bitwidth, fn); \
  }

LOOM_VECTOR_FLOAT_BINARY_FACTS(loom_vector_addf_facts, loom_vector_add_f32,
                               loom_vector_add_f64)
LOOM_VECTOR_FLOAT_BINARY_FACTS(loom_vector_subf_facts, loom_vector_sub_f32,
                               loom_vector_sub_f64)

static iree_status_t loom_vector_try_define_same_lane_origin(
    loom_fact_context_t* context, const loom_module_t* module,
    loom_value_id_t result, loom_value_id_t source) {
  if (context == NULL || context->table == NULL || module == NULL) {
    return iree_ok_status();
  }
  loom_type_t source_type = loom_module_value_type(module, source);
  loom_type_t result_type = loom_module_value_type(module, result);
  iree_host_size_t source_lane_count = 0;
  iree_host_size_t result_lane_count = 0;
  if (!loom_type_is_vector(source_type) || !loom_type_is_vector(result_type) ||
      !loom_vector_type_static_lane_count(source_type, &source_lane_count) ||
      !loom_vector_type_static_lane_count(result_type, &result_lane_count) ||
      source_lane_count != result_lane_count) {
    return iree_ok_status();
  }

  loom_value_fact_static_lane_origin_t source_origin = {
      .source_value_id = source,
      .source_lane_offset = 0,
      .source_lane_stride = 1,
  };
  loom_value_fact_static_lane_origin_t existing_origin = {0};
  if (loom_value_fact_table_query_static_lane_origin(
          context->table, module, source, &existing_origin)) {
    source_origin = existing_origin;
  }
  return loom_value_fact_table_define_static_lane_origin(context->table, result,
                                                         source_origin);
}

static iree_status_t loom_vector_try_define_select_same_lane_origin(
    loom_fact_context_t* context, const loom_module_t* module,
    loom_value_facts_t condition_facts, loom_value_id_t result,
    loom_value_id_t true_value, loom_value_id_t false_value) {
  if (context == NULL || context->table == NULL || module == NULL) {
    return iree_ok_status();
  }

  loom_value_facts_t condition = {0};
  if (loom_vector_facts_query_uniform_element(context, condition_facts,
                                              &condition) &&
      loom_value_facts_is_exact(condition)) {
    const loom_value_id_t selected =
        condition.range_lo ? true_value : false_value;
    return loom_vector_try_define_same_lane_origin(context, module, result,
                                                   selected);
  }
  return iree_ok_status();
}

static iree_status_t loom_vector_try_define_uniform_scale_origin(
    loom_fact_context_t* context, const loom_module_t* module,
    loom_value_id_t result, loom_value_id_t source, loom_value_id_t scale) {
  if (context == NULL || context->table == NULL || module == NULL) {
    return iree_ok_status();
  }
  loom_value_id_t scalar_scale = LOOM_VALUE_ID_INVALID;
  if (!loom_value_fact_table_query_uniform_element_origin(
          context->table, module, scale, &scalar_scale)) {
    return iree_ok_status();
  }
  return loom_value_fact_table_define_uniform_scale_origin(
      context->table, result,
      (loom_value_fact_uniform_scale_origin_t){
          .source_value_id = source,
          .scale_value_id = scalar_scale,
      });
}

iree_status_t loom_vector_mulf_facts(loom_fact_context_t* context,
                                     const loom_module_t* module,
                                     const loom_op_t* op,
                                     const loom_value_facts_t* operand_facts,
                                     loom_value_facts_t* result_facts) {
  IREE_RETURN_IF_ERROR(loom_vector_float_binary_math_summary_facts(
      context, loom_vector_result_element_type(module, op), operand_facts,
      result_facts, loom_vector_mul_f32, loom_vector_mul_f64));
  const loom_value_id_t lhs = loom_vector_mulf_lhs(op);
  const loom_value_id_t rhs = loom_vector_mulf_rhs(op);
  const loom_value_id_t result = loom_vector_mulf_result(op);
  IREE_RETURN_IF_ERROR(loom_vector_try_define_uniform_scale_origin(
      context, module, result, lhs, rhs));
  return loom_vector_try_define_uniform_scale_origin(context, module, result,
                                                     rhs, lhs);
}

LOOM_VECTOR_FLOAT_BINARY_FACTS(loom_vector_divf_facts, loom_vector_div_f32,
                               loom_vector_div_f64)
LOOM_VECTOR_FLOAT_BINARY_FACTS(loom_vector_remf_facts, fmodf, fmod)

iree_status_t loom_vector_negf_facts(loom_fact_context_t* context,
                                     const loom_module_t* module,
                                     const loom_op_t* op,
                                     const loom_value_facts_t* operand_facts,
                                     loom_value_facts_t* result_facts) {
  return loom_vector_float_unary_summary_facts(
      context, loom_vector_result_element_type(module, op), operand_facts,
      result_facts, loom_vector_float_negate_transfer, NULL);
}

iree_status_t loom_vector_absf_facts(loom_fact_context_t* context,
                                     const loom_module_t* module,
                                     const loom_op_t* op,
                                     const loom_value_facts_t* operand_facts,
                                     loom_value_facts_t* result_facts) {
  return loom_vector_float_unary_summary_facts(
      context, loom_vector_result_element_type(module, op), operand_facts,
      result_facts, loom_vector_float_abs_transfer, NULL);
}

static iree_status_t loom_vector_minmaxf_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts, loom_float_minmax_kind_t kind) {
  return loom_vector_float_binary_summary_facts(
      context, loom_vector_result_element_type(module, op), operand_facts,
      result_facts, loom_vector_float_minmax_transfer, &kind);
}

iree_status_t loom_vector_minimumf_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  return loom_vector_minmaxf_facts(context, module, op, operand_facts,
                                   result_facts, LOOM_FLOAT_MINMAX_MINIMUM);
}

iree_status_t loom_vector_maximumf_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  return loom_vector_minmaxf_facts(context, module, op, operand_facts,
                                   result_facts, LOOM_FLOAT_MINMAX_MAXIMUM);
}

iree_status_t loom_vector_minnumf_facts(loom_fact_context_t* context,
                                        const loom_module_t* module,
                                        const loom_op_t* op,
                                        const loom_value_facts_t* operand_facts,
                                        loom_value_facts_t* result_facts) {
  return loom_vector_minmaxf_facts(context, module, op, operand_facts,
                                   result_facts, LOOM_FLOAT_MINMAX_MINNUM);
}

iree_status_t loom_vector_maxnumf_facts(loom_fact_context_t* context,
                                        const loom_module_t* module,
                                        const loom_op_t* op,
                                        const loom_value_facts_t* operand_facts,
                                        loom_value_facts_t* result_facts) {
  return loom_vector_minmaxf_facts(context, module, op, operand_facts,
                                   result_facts, LOOM_FLOAT_MINMAX_MAXNUM);
}

iree_status_t loom_vector_copysignf_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  return loom_vector_float_binary_summary_facts(
      context, loom_vector_result_element_type(module, op), operand_facts,
      result_facts, loom_vector_float_copysign_transfer, NULL);
}

iree_status_t loom_vector_clampf_facts(loom_fact_context_t* context,
                                       const loom_module_t* module,
                                       const loom_op_t* op,
                                       const loom_value_facts_t* operand_facts,
                                       loom_value_facts_t* result_facts) {
  loom_float_clamp_kind_t kind = LOOM_FLOAT_CLAMP_ORDERED;
  switch (loom_vector_clampf_mode(op)) {
    case LOOM_VECTOR_CLAMPF_MODE_ORDERED:
      kind = LOOM_FLOAT_CLAMP_ORDERED;
      break;
    case LOOM_VECTOR_CLAMPF_MODE_NUMBER:
      kind = LOOM_FLOAT_CLAMP_NUMBER;
      break;
    case LOOM_VECTOR_CLAMPF_MODE_IEEE:
      kind = LOOM_FLOAT_CLAMP_IEEE;
      break;
    case LOOM_VECTOR_CLAMPF_MODE_COUNT_:
      result_facts[0] = loom_value_facts_unknown();
      return iree_ok_status();
  }
  return loom_vector_float_ternary_summary_facts(
      context, loom_vector_result_element_type(module, op), operand_facts,
      result_facts, loom_vector_float_clamp_transfer, &kind);
}

LOOM_VECTOR_INTEGER_BINARY_FACTS(loom_vector_addi_facts, loom_value_facts_addi)
LOOM_VECTOR_INTEGER_BINARY_FACTS(loom_vector_subi_facts, loom_value_facts_subi)
LOOM_VECTOR_INTEGER_BINARY_FACTS(loom_vector_muli_facts, loom_value_facts_muli)
LOOM_VECTOR_INTEGER_BINARY_FACTS(loom_vector_divsi_facts,
                                 loom_value_facts_divsi)
LOOM_VECTOR_INTEGER_BINARY_FACTS(loom_vector_divui_facts,
                                 loom_value_facts_divui)
LOOM_VECTOR_INTEGER_BINARY_FACTS(loom_vector_remsi_facts,
                                 loom_value_facts_remsi)
LOOM_VECTOR_INTEGER_BINARY_FACTS(loom_vector_remui_facts,
                                 loom_value_facts_remui)
LOOM_VECTOR_UNARY_FACTS(loom_vector_negi_facts, loom_value_facts_negi)
LOOM_VECTOR_UNARY_FACTS(loom_vector_absi_facts, loom_value_facts_absi)
LOOM_VECTOR_INTEGER_BINARY_FACTS(loom_vector_minsi_facts,
                                 loom_value_facts_minsi)
LOOM_VECTOR_INTEGER_BINARY_FACTS(loom_vector_maxsi_facts,
                                 loom_value_facts_maxsi)
LOOM_VECTOR_INTEGER_BINARY_FACTS(loom_vector_minui_facts,
                                 loom_value_facts_minui)
LOOM_VECTOR_INTEGER_BINARY_FACTS(loom_vector_maxui_facts,
                                 loom_value_facts_maxui)
LOOM_VECTOR_INTEGER_BINARY_FACTS(loom_vector_andi_facts, loom_value_facts_andi)
LOOM_VECTOR_INTEGER_BINARY_FACTS(loom_vector_ori_facts, loom_value_facts_ori)
LOOM_VECTOR_INTEGER_BINARY_FACTS(loom_vector_xori_facts, loom_value_facts_xori)
LOOM_VECTOR_INTEGER_BINARY_FACTS(loom_vector_shli_facts, loom_value_facts_shli)
LOOM_VECTOR_INTEGER_BINARY_FACTS(loom_vector_shrsi_facts,
                                 loom_value_facts_shrsi)
LOOM_VECTOR_INTEGER_BINARY_FACTS(loom_vector_shrui_facts,
                                 loom_value_facts_shrui)
LOOM_VECTOR_BIT_COUNT_FACTS(loom_vector_ctlzi_facts, loom_vector_ctlzi_result,
                            iree_math_count_leading_zeros_u64_width)
LOOM_VECTOR_BIT_COUNT_FACTS(loom_vector_cttzi_facts, loom_vector_cttzi_result,
                            iree_math_count_trailing_zeros_u64_width)
LOOM_VECTOR_BIT_COUNT_FACTS(loom_vector_ctpopi_facts, loom_vector_ctpopi_result,
                            iree_math_count_ones_u64_width)
LOOM_VECTOR_FLOAT_UNARY_FACTS(loom_vector_expf_facts, expf, exp)
LOOM_VECTOR_FLOAT_UNARY_FACTS(loom_vector_exp2f_facts, exp2f, exp2)
LOOM_VECTOR_FLOAT_UNARY_FACTS(loom_vector_expm1f_facts, expm1f, expm1)
LOOM_VECTOR_FLOAT_UNARY_FACTS(loom_vector_logf_facts, logf, log)
LOOM_VECTOR_FLOAT_UNARY_FACTS(loom_vector_log2f_facts, log2f, log2)
LOOM_VECTOR_FLOAT_UNARY_FACTS(loom_vector_log10f_facts, log10f, log10)
LOOM_VECTOR_FLOAT_UNARY_FACTS(loom_vector_log1pf_facts, log1pf, log1p)
LOOM_VECTOR_FLOAT_BINARY_FACTS(loom_vector_powf_facts, powf, pow)
LOOM_VECTOR_FLOAT_UNARY_FACTS(loom_vector_sqrtf_facts, sqrtf, sqrt)
LOOM_VECTOR_FLOAT_UNARY_FACTS(loom_vector_rsqrtf_facts, loom_vector_rsqrt_f32,
                              loom_vector_rsqrt_f64)
LOOM_VECTOR_FLOAT_UNARY_FACTS(loom_vector_cbrtf_facts, cbrtf, cbrt)
LOOM_VECTOR_FLOAT_UNARY_FACTS(loom_vector_sinf_facts, sinf, sin)
LOOM_VECTOR_FLOAT_UNARY_FACTS(loom_vector_cosf_facts, cosf, cos)

static iree_status_t loom_vector_turnsf_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts, loom_float_turns_kind_t kind) {
  return loom_vector_float_unary_summary_facts(
      context, loom_vector_result_element_type(module, op), operand_facts,
      result_facts, loom_vector_float_turns_transfer, &kind);
}

iree_status_t loom_vector_sinturnsf_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  return loom_vector_turnsf_facts(context, module, op, operand_facts,
                                  result_facts, LOOM_FLOAT_TURNS_SIN);
}

iree_status_t loom_vector_costurnsf_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  return loom_vector_turnsf_facts(context, module, op, operand_facts,
                                  result_facts, LOOM_FLOAT_TURNS_COS);
}

LOOM_VECTOR_FLOAT_UNARY_FACTS(loom_vector_tanf_facts, tanf, tan)
LOOM_VECTOR_FLOAT_UNARY_FACTS(loom_vector_asinf_facts, asinf, asin)
LOOM_VECTOR_FLOAT_UNARY_FACTS(loom_vector_acosf_facts, acosf, acos)
LOOM_VECTOR_FLOAT_UNARY_FACTS(loom_vector_atanf_facts, atanf, atan)
LOOM_VECTOR_FLOAT_BINARY_FACTS(loom_vector_atan2f_facts, atan2f, atan2)
LOOM_VECTOR_FLOAT_UNARY_FACTS(loom_vector_sinhf_facts, sinhf, sinh)
LOOM_VECTOR_FLOAT_UNARY_FACTS(loom_vector_coshf_facts, coshf, cosh)
LOOM_VECTOR_FLOAT_UNARY_FACTS(loom_vector_tanhf_facts, tanhf, tanh)
LOOM_VECTOR_FLOAT_UNARY_FACTS(loom_vector_asinhf_facts, asinhf, asinh)
LOOM_VECTOR_FLOAT_UNARY_FACTS(loom_vector_acoshf_facts, acoshf, acosh)
LOOM_VECTOR_FLOAT_UNARY_FACTS(loom_vector_atanhf_facts, atanhf, atanh)
LOOM_VECTOR_FLOAT_UNARY_FACTS(loom_vector_erff_facts, erff, erf)
LOOM_VECTOR_FLOAT_UNARY_FACTS(loom_vector_erfcf_facts, erfcf, erfc)
LOOM_VECTOR_FLOAT_UNARY_FACTS(loom_vector_logisticf_facts,
                              loom_float_logistic_f32, loom_float_logistic_f64)
LOOM_VECTOR_FLOAT_UNARY_FACTS(loom_vector_siluf_facts, loom_vector_silu_f32,
                              loom_vector_silu_f64)
LOOM_VECTOR_FLOAT_UNARY_FACTS(loom_vector_softplusf_facts,
                              loom_vector_softplus_f32,
                              loom_vector_softplus_f64)
LOOM_VECTOR_FLOAT_UNARY_FACTS(loom_vector_ceilf_facts, ceilf, ceil)
LOOM_VECTOR_FLOAT_UNARY_FACTS(loom_vector_floorf_facts, floorf, floor)
LOOM_VECTOR_FLOAT_UNARY_FACTS(loom_vector_roundf_facts, roundf, round)
LOOM_VECTOR_FLOAT_UNARY_FACTS(loom_vector_roundevenf_facts,
                              loom_vector_roundeven_f32,
                              loom_vector_roundeven_f64)
LOOM_VECTOR_FLOAT_UNARY_FACTS(loom_vector_truncf_facts, truncf, trunc)
LOOM_VECTOR_UNARY_FACTS(loom_vector_signi_facts, loom_vector_signi_transfer)
iree_status_t loom_vector_extf_facts(loom_fact_context_t* context,
                                     const loom_module_t* module,
                                     const loom_op_t* op,
                                     const loom_value_facts_t* operand_facts,
                                     loom_value_facts_t* result_facts) {
  bool fragment_handled = false;
  IREE_RETURN_IF_ERROR(loom_vector_try_preserve_lanewise_fragment_facts(
      context, operand_facts, 1, result_facts, &fragment_handled));
  if (!fragment_handled) {
    IREE_RETURN_IF_ERROR(
        loom_vector_unary_summary_facts(context, operand_facts, result_facts,
                                        loom_vector_passthrough_transfer));
  }
  return loom_vector_try_define_same_lane_origin(
      context, module, loom_vector_extf_result(op), loom_vector_extf_input(op));
}

iree_status_t loom_vector_fptrunc_facts(loom_fact_context_t* context,
                                        const loom_module_t* module,
                                        const loom_op_t* op,
                                        const loom_value_facts_t* operand_facts,
                                        loom_value_facts_t* result_facts) {
  bool fragment_handled = false;
  IREE_RETURN_IF_ERROR(loom_vector_try_preserve_lanewise_fragment_facts(
      context, operand_facts, 1, result_facts, &fragment_handled));
  if (!fragment_handled) {
    IREE_RETURN_IF_ERROR(loom_vector_make_unknown_facts(result_facts));
  }
  return loom_vector_try_define_same_lane_origin(context, module,
                                                 loom_vector_fptrunc_result(op),
                                                 loom_vector_fptrunc_input(op));
}

LOOM_VECTOR_UNARY_FACTS(loom_vector_extsi_facts,
                        loom_vector_passthrough_transfer)

static loom_scalar_type_t loom_vector_first_operand_element_type(
    const loom_module_t* module, const loom_op_t* op) {
  return loom_type_element_type(
      loom_module_value_type(module, loom_op_const_operands(op)[0]));
}

#define LOOM_VECTOR_FLOAT_CLASSIFY_FACTS(name, transfer_fn)            \
  iree_status_t name(loom_fact_context_t* context,                     \
                     const loom_module_t* module, const loom_op_t* op, \
                     const loom_value_facts_t* operand_facts,          \
                     loom_value_facts_t* result_facts) {               \
    return loom_vector_float_classify_summary_facts(                   \
        context, loom_vector_first_operand_element_type(module, op),   \
        operand_facts, result_facts, transfer_fn);                     \
  }

LOOM_VECTOR_FLOAT_CLASSIFY_FACTS(loom_vector_isnanf_facts,
                                 loom_vector_isnanf_transfer)
LOOM_VECTOR_FLOAT_CLASSIFY_FACTS(loom_vector_isinff_facts,
                                 loom_vector_isinff_transfer)
LOOM_VECTOR_FLOAT_CLASSIFY_FACTS(loom_vector_isfinitef_facts,
                                 loom_vector_isfinitef_transfer)
LOOM_VECTOR_FLOAT_CLASSIFY_FACTS(loom_vector_signf_facts,
                                 loom_vector_signf_transfer)

iree_status_t loom_vector_sitofp_facts(loom_fact_context_t* context,
                                       const loom_module_t* module,
                                       const loom_op_t* op,
                                       const loom_value_facts_t* operand_facts,
                                       loom_value_facts_t* result_facts) {
  return loom_vector_float_unary_summary_facts(
      context, loom_vector_result_element_type(module, op), operand_facts,
      result_facts, loom_vector_sitofp_transfer, NULL);
}

typedef struct loom_vector_geluf_transfer_t {
  loom_vector_geluf_variant_t variant;
  double scale;
} loom_vector_geluf_transfer_t;

static float loom_vector_geluf_transfer_f32(float input,
                                            const void* user_data) {
  const loom_vector_geluf_transfer_t* transfer = user_data;
  switch (transfer->variant) {
    case LOOM_VECTOR_GELUF_VARIANT_ERF:
      return loom_vector_gelu_erf_f32(input);
    case LOOM_VECTOR_GELUF_VARIANT_TANH:
      return loom_vector_gelu_tanh_f32(input);
    case LOOM_VECTOR_GELUF_VARIANT_LOGISTIC:
      return loom_vector_gelu_logistic_f32(input, (float)transfer->scale);
    case LOOM_VECTOR_GELUF_VARIANT_COUNT_:
      return NAN;
  }
  return NAN;
}

static double loom_vector_geluf_transfer_f64(double input,
                                             const void* user_data) {
  const loom_vector_geluf_transfer_t* transfer = user_data;
  switch (transfer->variant) {
    case LOOM_VECTOR_GELUF_VARIANT_ERF:
      return loom_vector_gelu_erf_f64(input);
    case LOOM_VECTOR_GELUF_VARIANT_TANH:
      return loom_vector_gelu_tanh_f64(input);
    case LOOM_VECTOR_GELUF_VARIANT_LOGISTIC:
      return loom_vector_gelu_logistic_f64(input, transfer->scale);
    case LOOM_VECTOR_GELUF_VARIANT_COUNT_:
      return NAN;
  }
  return NAN;
}

iree_status_t loom_vector_geluf_facts(loom_fact_context_t* context,
                                      const loom_module_t* module,
                                      const loom_op_t* op,
                                      const loom_value_facts_t* operand_facts,
                                      loom_value_facts_t* result_facts) {
  loom_vector_geluf_transfer_t transfer = {
      .variant = loom_vector_geluf_variant(op),
      .scale = 0.0,
  };
  if (transfer.variant == LOOM_VECTOR_GELUF_VARIANT_LOGISTIC) {
    loom_attribute_t scale_attr = loom_op_attrs(op)[1];
    if (loom_attr_is_absent(scale_attr)) {
      result_facts[0] = loom_value_facts_unknown();
      return iree_ok_status();
    }
    transfer.scale = loom_attr_as_f64(scale_attr);
  }
  return loom_vector_float_unary_data_math_summary_facts(
      context, loom_vector_result_element_type(module, op), operand_facts,
      result_facts, loom_vector_geluf_transfer_f32,
      loom_vector_geluf_transfer_f64, &transfer);
}

iree_status_t loom_vector_fmai_facts(loom_fact_context_t* context,
                                     const loom_module_t* module,
                                     const loom_op_t* op,
                                     const loom_value_facts_t* operand_facts,
                                     loom_value_facts_t* result_facts) {
  return loom_vector_ternary_summary_facts(context, operand_facts, result_facts,
                                           loom_vector_fmai_transfer);
}

static void loom_vector_fmaf_transfer(loom_scalar_type_t scalar_type,
                                      const loom_value_facts_t* a,
                                      const loom_value_facts_t* b,
                                      const loom_value_facts_t* c,
                                      const void* user_data,
                                      loom_value_facts_t* out) {
  loom_value_facts_eval_float_ternary(scalar_type, a, b, c, fmaf, fma, out);
}

iree_status_t loom_vector_fmaf_facts(loom_fact_context_t* context,
                                     const loom_module_t* module,
                                     const loom_op_t* op,
                                     const loom_value_facts_t* operand_facts,
                                     loom_value_facts_t* result_facts) {
  return loom_vector_float_ternary_summary_facts(
      context, loom_vector_result_element_type(module, op), operand_facts,
      result_facts, loom_vector_fmaf_transfer, NULL);
}

iree_status_t loom_vector_select_facts(loom_fact_context_t* context,
                                       const loom_module_t* module,
                                       const loom_op_t* op,
                                       const loom_value_facts_t* operand_facts,
                                       loom_value_facts_t* result_facts) {
  IREE_RETURN_IF_ERROR(loom_vector_try_define_select_same_lane_origin(
      context, module, operand_facts[0], loom_vector_select_result(op),
      loom_vector_select_true_value(op), loom_vector_select_false_value(op)));

  if (loom_value_facts_equal(operand_facts[1], operand_facts[2])) {
    result_facts[0] = operand_facts[1];
    return iree_ok_status();
  }
  loom_value_facts_t condition = {0};
  if (loom_vector_facts_query_uniform_element(context, operand_facts[0],
                                              &condition)) {
    if (loom_value_facts_is_exact(condition)) {
      result_facts[0] =
          condition.range_lo ? operand_facts[1] : operand_facts[2];
      return iree_ok_status();
    }
  }

  loom_value_fact_small_static_lanes_t condition_lanes = {0};
  loom_type_t result_type =
      loom_module_value_type(module, loom_vector_select_result(op));
  uint64_t result_lane_count = 0;
  if (!loom_type_static_element_count(result_type, &result_lane_count) ||
      result_lane_count == 0 ||
      result_lane_count > LOOM_VALUE_FACT_SMALL_STATIC_LANE_LIMIT) {
    result_facts[0] = loom_value_facts_unknown();
    return iree_ok_status();
  }
  if (!loom_vector_facts_query_uniform_element(context, operand_facts[0],
                                               &condition) &&
      (!loom_vector_facts_query_small_lanes(context, operand_facts[0],
                                            &condition_lanes) ||
       condition_lanes.count != (iree_host_size_t)result_lane_count)) {
    result_facts[0] = loom_value_facts_unknown();
    return iree_ok_status();
  }
  loom_type_t element_type =
      loom_type_scalar(loom_type_element_type(result_type));
  loom_value_facts_t lanes[LOOM_VALUE_FACT_SMALL_STATIC_LANE_LIMIT] = {{0}};
  for (iree_host_size_t i = 0; i < (iree_host_size_t)result_lane_count; ++i) {
    if (condition_lanes.count != 0) {
      condition = condition_lanes.lanes[i];
    }
    if (!loom_value_facts_is_exact(condition)) {
      loom_value_facts_t true_lane = {0};
      loom_value_facts_t false_lane = {0};
      if (loom_vector_facts_query_lane(context, operand_facts[1], i,
                                       &true_lane) &&
          loom_vector_facts_query_lane(context, operand_facts[2], i,
                                       &false_lane)) {
        IREE_RETURN_IF_ERROR(loom_value_fact_table_meet_for_type(
            context->table, module, element_type, context->table, true_lane,
            context->table, false_lane, &lanes[i]));
      } else {
        lanes[i] = loom_value_facts_unknown();
      }
      continue;
    }
    if (!loom_vector_facts_query_lane(
            context, condition.range_lo ? operand_facts[1] : operand_facts[2],
            i, &lanes[i])) {
      result_facts[0] = loom_value_facts_unknown();
      return iree_ok_status();
    }
  }
  loom_value_fact_small_static_lanes_t lane_slice = {
      .lanes = lanes,
      .count = (iree_host_size_t)result_lane_count,
  };
  return loom_value_facts_make_small_static_lanes(context, lane_slice,
                                                  &result_facts[0]);
}

static loom_value_facts_t loom_vector_boolean_range_facts(void) {
  return loom_value_facts_make(0, 1, 1);
}

iree_status_t loom_vector_cmpi_facts(loom_fact_context_t* context,
                                     const loom_module_t* module,
                                     const loom_op_t* op,
                                     const loom_value_facts_t* operand_facts,
                                     loom_value_facts_t* result_facts) {
  bool result = false;
  uint8_t predicate = loom_vector_cmpi_predicate(op);
  if (loom_vector_cmpi_lhs(op) == loom_vector_cmpi_rhs(op) &&
      loom_scalar_cmpi_same_value_result(predicate, &result)) {
    return loom_value_facts_make_uniform_element(
        context, loom_value_facts_exact_i64(result ? 1 : 0), &result_facts[0]);
  }

  loom_value_facts_t lhs = {0};
  loom_value_facts_t rhs = {0};
  if (loom_vector_facts_query_uniform_element(context, operand_facts[0],
                                              &lhs) &&
      loom_vector_facts_query_uniform_element(context, operand_facts[1],
                                              &rhs)) {
    loom_value_facts_t element = loom_vector_boolean_range_facts();
    if (loom_scalar_cmpi_result_from_facts(predicate, &lhs, &rhs, &result)) {
      element = loom_value_facts_exact_i64(result ? 1 : 0);
    }
    return loom_value_facts_make_uniform_element(context, element,
                                                 &result_facts[0]);
  }

  iree_host_size_t lane_count = 0;
  if (!loom_vector_facts_query_binary_lane_count(
          context, operand_facts[0], operand_facts[1], &lane_count)) {
    return loom_value_facts_make_uniform_element(
        context, loom_vector_boolean_range_facts(), &result_facts[0]);
  }
  loom_value_facts_t lanes[LOOM_VALUE_FACT_SMALL_STATIC_LANE_LIMIT] = {{0}};
  for (iree_host_size_t i = 0; i < lane_count; ++i) {
    if (!loom_vector_facts_query_lane(context, operand_facts[0], i, &lhs) ||
        !loom_vector_facts_query_lane(context, operand_facts[1], i, &rhs)) {
      return loom_value_facts_make_uniform_element(
          context, loom_vector_boolean_range_facts(), &result_facts[0]);
    }
    lanes[i] = loom_vector_boolean_range_facts();
    if (loom_scalar_cmpi_result_from_facts(predicate, &lhs, &rhs, &result)) {
      lanes[i] = loom_value_facts_exact_i64(result ? 1 : 0);
    }
  }
  return loom_vector_make_small_static_lane_facts(context, lanes, lane_count,
                                                  &result_facts[0]);
}

iree_status_t loom_vector_cmpf_facts(loom_fact_context_t* context,
                                     const loom_module_t* module,
                                     const loom_op_t* op,
                                     const loom_value_facts_t* operand_facts,
                                     loom_value_facts_t* result_facts) {
  uint8_t predicate = loom_vector_cmpf_predicate(op);
  const loom_scalar_type_t scalar_type = loom_type_element_type(
      loom_module_value_type(module, loom_vector_cmpf_lhs(op)));
  loom_value_facts_t lhs = {0};
  loom_value_facts_t rhs = {0};
  if (loom_vector_facts_query_uniform_element(context, operand_facts[0],
                                              &lhs) &&
      loom_vector_facts_query_uniform_element(context, operand_facts[1],
                                              &rhs)) {
    double lhs_value = 0.0;
    double rhs_value = 0.0;
    bool result = false;
    loom_value_facts_t element = loom_vector_boolean_range_facts();
    const bool has_known_nan =
        loom_value_facts_is_nan(lhs) || loom_value_facts_is_nan(rhs);
    bool has_values = has_known_nan;
    if (has_known_nan) {
      lhs_value = NAN;
      rhs_value = NAN;
    } else {
      has_values =
          loom_vector_facts_query_exact_float(scalar_type, lhs, &lhs_value) &&
          loom_vector_facts_query_exact_float(scalar_type, rhs, &rhs_value);
    }
    if (has_values && loom_scalar_cmpf_exact_result(predicate, lhs_value,
                                                    rhs_value, &result)) {
      element = loom_value_facts_exact_i64(result ? 1 : 0);
    }
    return loom_value_facts_make_uniform_element(context, element,
                                                 &result_facts[0]);
  }

  iree_host_size_t lane_count = 0;
  if (!loom_vector_facts_query_binary_lane_count(
          context, operand_facts[0], operand_facts[1], &lane_count)) {
    return loom_value_facts_make_uniform_element(
        context, loom_vector_boolean_range_facts(), &result_facts[0]);
  }
  loom_value_facts_t lanes[LOOM_VALUE_FACT_SMALL_STATIC_LANE_LIMIT] = {{0}};
  for (iree_host_size_t i = 0; i < lane_count; ++i) {
    if (!loom_vector_facts_query_lane(context, operand_facts[0], i, &lhs) ||
        !loom_vector_facts_query_lane(context, operand_facts[1], i, &rhs)) {
      return loom_value_facts_make_uniform_element(
          context, loom_vector_boolean_range_facts(), &result_facts[0]);
    }
    double lhs_value = 0.0;
    double rhs_value = 0.0;
    bool result = false;
    lanes[i] = loom_vector_boolean_range_facts();
    const bool has_known_nan =
        loom_value_facts_is_nan(lhs) || loom_value_facts_is_nan(rhs);
    bool has_values = has_known_nan;
    if (has_known_nan) {
      lhs_value = NAN;
      rhs_value = NAN;
    } else {
      has_values =
          loom_vector_facts_query_exact_float(scalar_type, lhs, &lhs_value) &&
          loom_vector_facts_query_exact_float(scalar_type, rhs, &rhs_value);
    }
    if (has_values && loom_scalar_cmpf_exact_result(predicate, lhs_value,
                                                    rhs_value, &result)) {
      lanes[i] = loom_value_facts_exact_i64(result ? 1 : 0);
    }
  }
  return loom_vector_make_small_static_lane_facts(context, lanes, lane_count,
                                                  &result_facts[0]);
}

iree_status_t loom_vector_bitcast_facts(loom_fact_context_t* context,
                                        const loom_module_t* module,
                                        const loom_op_t* op,
                                        const loom_value_facts_t* operand_facts,
                                        loom_value_facts_t* result_facts) {
  loom_type_t source_type =
      loom_module_value_type(module, loom_vector_bitcast_input(op));
  loom_type_t result_type =
      loom_module_value_type(module, loom_vector_bitcast_result(op));
  loom_scalar_type_t source_element_type = loom_type_element_type(source_type);
  loom_scalar_type_t result_element_type = loom_type_element_type(result_type);
  iree_host_size_t lane_count = 0;
  if (!loom_vector_same_static_lane_count(source_type, result_type,
                                          &lane_count)) {
    return loom_vector_make_unknown_facts(result_facts);
  }

  loom_value_facts_t uniform_element = {0};
  loom_value_facts_t bitcast_element = {0};
  if (loom_vector_facts_query_uniform_element(context, operand_facts[0],
                                              &uniform_element) &&
      loom_vector_bitcast_element_facts(uniform_element, source_element_type,
                                        result_element_type,
                                        &bitcast_element)) {
    return loom_value_facts_make_uniform_element(context, bitcast_element,
                                                 &result_facts[0]);
  }

  if (lane_count > LOOM_VALUE_FACT_SMALL_STATIC_LANE_LIMIT) {
    return loom_vector_make_unknown_facts(result_facts);
  }
  loom_value_facts_t lanes[LOOM_VALUE_FACT_SMALL_STATIC_LANE_LIMIT] = {{0}};
  for (iree_host_size_t lane = 0; lane < lane_count; ++lane) {
    loom_value_facts_t source_lane = {0};
    if (!loom_vector_facts_query_lane(context, operand_facts[0], lane,
                                      &source_lane) ||
        !loom_vector_bitcast_element_facts(source_lane, source_element_type,
                                           result_element_type, &lanes[lane])) {
      return loom_vector_make_unknown_facts(result_facts);
    }
  }
  return loom_vector_make_small_static_lane_facts(context, lanes, lane_count,
                                                  &result_facts[0]);
}

//===----------------------------------------------------------------------===//
// Block-quant decode helpers
//===----------------------------------------------------------------------===//

static bool loom_vector_bitfield_extract_element(
    loom_value_facts_t source, int32_t source_width, int64_t offset,
    int64_t width, bool signed_extract, loom_value_facts_t* out_element) {
  return signed_extract ? loom_value_facts_extract_signed_bitfield(
                              source, source_width, offset, width, out_element)
                        : loom_value_facts_extract_unsigned_bitfield(
                              source, source_width, offset, width, out_element);
}

static iree_status_t loom_vector_bitfield_extract_summary_facts(
    loom_fact_context_t* context, loom_type_t source_type,
    loom_type_t result_type, int64_t offset, int64_t width, bool signed_extract,
    const loom_value_facts_t* operand_facts, loom_value_facts_t* result_facts) {
  int32_t source_width = 0;
  int32_t result_width = 0;
  if (!loom_vector_integer_element_bitwidth(source_type, &source_width) ||
      !loom_vector_integer_element_bitwidth(result_type, &result_width) ||
      result_width < width) {
    return loom_vector_make_unknown_facts(result_facts);
  }

  loom_value_facts_t source = {0};
  if (loom_vector_facts_query_uniform_element(context, operand_facts[0],
                                              &source)) {
    loom_value_facts_t element = loom_value_facts_unknown();
    if (!loom_vector_bitfield_extract_element(
            source, source_width, offset, width, signed_extract, &element)) {
      return loom_vector_make_unknown_facts(result_facts);
    }
    return loom_value_facts_make_uniform_element(context, element,
                                                 &result_facts[0]);
  }

  loom_value_fact_small_static_lanes_t source_lanes = {0};
  if (!loom_vector_facts_query_small_lanes(context, operand_facts[0],
                                           &source_lanes)) {
    return loom_vector_make_unknown_facts(result_facts);
  }
  loom_value_facts_t lanes[LOOM_VALUE_FACT_SMALL_STATIC_LANE_LIMIT] = {{0}};
  for (iree_host_size_t i = 0; i < source_lanes.count; ++i) {
    if (!loom_vector_bitfield_extract_element(source_lanes.lanes[i],
                                              source_width, offset, width,
                                              signed_extract, &lanes[i])) {
      return loom_vector_make_unknown_facts(result_facts);
    }
  }
  return loom_vector_make_small_static_lane_facts(
      context, lanes, source_lanes.count, &result_facts[0]);
}

iree_status_t loom_vector_bitfield_extractu_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  loom_type_t source_type =
      loom_module_value_type(module, loom_vector_bitfield_extractu_source(op));
  loom_type_t result_type =
      loom_module_value_type(module, loom_vector_bitfield_extractu_result(op));
  return loom_vector_bitfield_extract_summary_facts(
      context, source_type, result_type,
      loom_vector_bitfield_extractu_offset(op),
      loom_vector_bitfield_extractu_width(op), /*signed_extract=*/false,
      operand_facts, result_facts);
}

iree_status_t loom_vector_bitfield_extracts_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  loom_type_t source_type =
      loom_module_value_type(module, loom_vector_bitfield_extracts_source(op));
  loom_type_t result_type =
      loom_module_value_type(module, loom_vector_bitfield_extracts_result(op));
  return loom_vector_bitfield_extract_summary_facts(
      context, source_type, result_type,
      loom_vector_bitfield_extracts_offset(op),
      loom_vector_bitfield_extracts_width(op), /*signed_extract=*/true,
      operand_facts, result_facts);
}

static bool loom_vector_bitfield_insert_element(
    loom_value_facts_t field, loom_value_facts_t base, int32_t base_width,
    loom_scalar_type_t base_element_type, int64_t offset, int64_t width,
    loom_value_facts_t* out_element) {
  if (offset < 0 || width <= 0 || base_width <= 0 || offset > base_width ||
      width > base_width - offset || width > 64) {
    return false;
  }
  uint64_t field_bits = 0;
  uint64_t base_bits = 0;
  if (!loom_value_facts_as_exact_raw_bits(field, (int32_t)width, &field_bits) ||
      !loom_value_facts_as_exact_raw_bits(base, base_width, &base_bits)) {
    *out_element = loom_value_facts_unknown();
    return true;
  }
  uint64_t field_mask = iree_math_mask_low_bits_u64(UINT64_MAX, (int32_t)width);
  uint64_t target_mask = field_mask << offset;
  uint64_t raw_bits =
      (base_bits & ~target_mask) | ((field_bits & field_mask) << offset);
  *out_element =
      loom_vector_make_integer_raw_bit_facts(raw_bits, base_element_type);
  return true;
}

iree_status_t loom_vector_bitfield_insert_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  loom_type_t base_type =
      loom_module_value_type(module, loom_vector_bitfield_insert_base(op));
  loom_type_t field_type =
      loom_module_value_type(module, loom_vector_bitfield_insert_field(op));
  loom_scalar_type_t base_element_type = loom_type_element_type(base_type);
  int32_t base_width = 0;
  int32_t field_width = 0;
  int64_t offset = loom_vector_bitfield_insert_offset(op);
  int64_t width = loom_vector_bitfield_insert_width(op);
  if (!loom_vector_integer_element_bitwidth(base_type, &base_width) ||
      !loom_vector_integer_element_bitwidth(field_type, &field_width) ||
      field_width < width) {
    return loom_vector_make_unknown_facts(result_facts);
  }

  loom_value_facts_t field = {0};
  loom_value_facts_t base = {0};
  if (loom_vector_facts_query_uniform_element(context, operand_facts[0],
                                              &field) &&
      loom_vector_facts_query_uniform_element(context, operand_facts[1],
                                              &base)) {
    loom_value_facts_t element = loom_value_facts_unknown();
    if (!loom_vector_bitfield_insert_element(field, base, base_width,
                                             base_element_type, offset, width,
                                             &element)) {
      return loom_vector_make_unknown_facts(result_facts);
    }
    return loom_value_facts_make_uniform_element(context, element,
                                                 &result_facts[0]);
  }

  iree_host_size_t lane_count = 0;
  if (!loom_vector_facts_query_binary_lane_count(
          context, operand_facts[0], operand_facts[1], &lane_count)) {
    return loom_vector_make_unknown_facts(result_facts);
  }
  loom_value_facts_t lanes[LOOM_VALUE_FACT_SMALL_STATIC_LANE_LIMIT] = {{0}};
  for (iree_host_size_t i = 0; i < lane_count; ++i) {
    if (!loom_vector_facts_query_lane(context, operand_facts[0], i, &field) ||
        !loom_vector_facts_query_lane(context, operand_facts[1], i, &base) ||
        !loom_vector_bitfield_insert_element(field, base, base_width,
                                             base_element_type, offset, width,
                                             &lanes[i])) {
      return loom_vector_make_unknown_facts(result_facts);
    }
  }
  return loom_vector_make_small_static_lane_facts(context, lanes, lane_count,
                                                  &result_facts[0]);
}

static bool loom_vector_read_logical_bitstream(
    const loom_fact_context_t* context, loom_value_facts_t source_facts,
    loom_type_t source_type, int32_t logical_lane_width, uint64_t bit_position,
    int32_t bit_count, uint64_t* out_bits) {
  if (logical_lane_width <= 0 || logical_lane_width > 64 || bit_count < 0 ||
      bit_count > 64) {
    return false;
  }
  iree_host_size_t source_lane_count = 0;
  if (!loom_vector_type_static_lane_count(source_type, &source_lane_count)) {
    return false;
  }
  if ((uint64_t)source_lane_count > UINT64_MAX / (uint64_t)logical_lane_width) {
    return false;
  }
  uint64_t total_bits =
      (uint64_t)source_lane_count * (uint64_t)logical_lane_width;
  if (bit_position > total_bits ||
      (uint64_t)bit_count > total_bits - bit_position) {
    return false;
  }

  uint64_t bits = 0;
  int32_t destination_shift = 0;
  int32_t remaining_bits = bit_count;
  while (remaining_bits > 0) {
    uint64_t source_lane = bit_position / (uint64_t)logical_lane_width;
    int32_t source_shift =
        (int32_t)(bit_position % (uint64_t)logical_lane_width);
    int32_t piece_bits = logical_lane_width - source_shift < remaining_bits
                             ? logical_lane_width - source_shift
                             : remaining_bits;
    loom_value_facts_t lane_facts = {0};
    uint64_t lane_bits = 0;
    if (source_lane > (uint64_t)IREE_HOST_SIZE_MAX ||
        !loom_vector_facts_query_lane(context, source_facts,
                                      (iree_host_size_t)source_lane,
                                      &lane_facts) ||
        !loom_value_facts_as_exact_raw_bits(lane_facts, logical_lane_width,
                                            &lane_bits)) {
      return false;
    }
    uint64_t piece =
        iree_math_mask_low_bits_u64(lane_bits >> source_shift, piece_bits);
    bits |= piece << destination_shift;
    bit_position += (uint64_t)piece_bits;
    destination_shift += piece_bits;
    remaining_bits -= piece_bits;
  }
  *out_bits = bits;
  return true;
}

iree_status_t loom_vector_bitpack_facts(loom_fact_context_t* context,
                                        const loom_module_t* module,
                                        const loom_op_t* op,
                                        const loom_value_facts_t* operand_facts,
                                        loom_value_facts_t* result_facts) {
  loom_type_t source_type =
      loom_module_value_type(module, loom_vector_bitpack_source(op));
  loom_type_t result_type =
      loom_module_value_type(module, loom_vector_bitpack_result(op));
  loom_scalar_type_t result_element_type = loom_type_element_type(result_type);
  int64_t width = loom_vector_bitpack_width(op);
  int32_t source_width = 0;
  int32_t storage_width = 0;
  iree_host_size_t result_lane_count = 0;
  if (width <= 0 || width > 64 ||
      !loom_vector_integer_element_bitwidth(source_type, &source_width) ||
      source_width < width ||
      !loom_vector_integer_element_bitwidth(result_type, &storage_width) ||
      !loom_vector_type_static_lane_count(result_type, &result_lane_count) ||
      result_lane_count > LOOM_VALUE_FACT_SMALL_STATIC_LANE_LIMIT) {
    return loom_vector_make_unknown_facts(result_facts);
  }

  loom_value_facts_t lanes[LOOM_VALUE_FACT_SMALL_STATIC_LANE_LIMIT] = {{0}};
  for (iree_host_size_t lane = 0; lane < result_lane_count; ++lane) {
    uint64_t bit_position = 0;
    if ((uint64_t)lane > UINT64_MAX / (uint64_t)storage_width) {
      return loom_vector_make_unknown_facts(result_facts);
    }
    bit_position = (uint64_t)lane * (uint64_t)storage_width;
    uint64_t raw_bits = 0;
    if (!loom_vector_read_logical_bitstream(
            context, operand_facts[0], source_type, (int32_t)width,
            bit_position, storage_width, &raw_bits)) {
      return loom_vector_make_unknown_facts(result_facts);
    }
    lanes[lane] =
        loom_vector_make_integer_raw_bit_facts(raw_bits, result_element_type);
  }
  return loom_vector_make_small_static_lane_facts(
      context, lanes, result_lane_count, &result_facts[0]);
}

static iree_status_t loom_vector_bitunpack_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts, bool signed_unpack) {
  loom_value_id_t source = signed_unpack ? loom_vector_bitunpacks_source(op)
                                         : loom_vector_bitunpacku_source(op);
  loom_value_id_t result = signed_unpack ? loom_vector_bitunpacks_result(op)
                                         : loom_vector_bitunpacku_result(op);
  int64_t width = signed_unpack ? loom_vector_bitunpacks_width(op)
                                : loom_vector_bitunpacku_width(op);
  loom_type_t source_type = loom_module_value_type(module, source);
  loom_type_t result_type = loom_module_value_type(module, result);
  int32_t storage_width = 0;
  int32_t result_width = 0;
  if (width <= 0 || width > 64 ||
      !loom_vector_integer_element_bitwidth(source_type, &storage_width) ||
      !loom_vector_integer_element_bitwidth(result_type, &result_width) ||
      result_width < width) {
    return loom_vector_make_unknown_facts(result_facts);
  }

  loom_value_facts_t source_element = {0};
  uint64_t raw_bits = 0;
  if (width == storage_width &&
      loom_vector_facts_query_uniform_element(context, operand_facts[0],
                                              &source_element) &&
      loom_value_facts_as_exact_raw_bits(source_element, storage_width,
                                         &raw_bits)) {
    loom_value_facts_t result_element = {0};
    if (signed_unpack) {
      result_element =
          loom_value_facts_make_signed_raw_bits(raw_bits, (int32_t)width);
    } else if (!loom_value_facts_make_unsigned_raw_bits(
                   raw_bits, (int32_t)width, &result_element)) {
      return loom_vector_make_unknown_facts(result_facts);
    }
    return loom_value_facts_make_uniform_element(context, result_element,
                                                 &result_facts[0]);
  }

  iree_host_size_t result_lane_count = 0;
  if (!loom_vector_type_static_lane_count(result_type, &result_lane_count) ||
      result_lane_count > LOOM_VALUE_FACT_SMALL_STATIC_LANE_LIMIT) {
    return loom_vector_make_unknown_facts(result_facts);
  }

  loom_value_facts_t lanes[LOOM_VALUE_FACT_SMALL_STATIC_LANE_LIMIT] = {{0}};
  const loom_value_facts_t dynamic_lane_facts =
      signed_unpack ? loom_value_facts_make_signed_bit_count_range(width)
                    : loom_value_facts_make_unsigned_bit_count_range(width);
  for (iree_host_size_t lane = 0; lane < result_lane_count; ++lane) {
    uint64_t bit_position = 0;
    if ((uint64_t)lane > UINT64_MAX / (uint64_t)width) {
      return loom_vector_make_unknown_facts(result_facts);
    }
    bit_position = (uint64_t)lane * (uint64_t)width;
    uint64_t raw_bits = 0;
    if (!loom_vector_read_logical_bitstream(
            context, operand_facts[0], source_type, storage_width, bit_position,
            (int32_t)width, &raw_bits)) {
      lanes[lane] = dynamic_lane_facts;
    } else if (signed_unpack) {
      lanes[lane] =
          loom_value_facts_make_signed_raw_bits(raw_bits, (int32_t)width);
    } else if (!loom_value_facts_make_unsigned_raw_bits(
                   raw_bits, (int32_t)width, &lanes[lane])) {
      return loom_vector_make_unknown_facts(result_facts);
    }
  }
  return loom_vector_make_small_static_lane_facts(
      context, lanes, result_lane_count, &result_facts[0]);
}

iree_status_t loom_vector_bitunpacku_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  return loom_vector_bitunpack_facts(context, module, op, operand_facts,
                                     result_facts, /*signed_unpack=*/false);
}

iree_status_t loom_vector_bitunpacks_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  return loom_vector_bitunpack_facts(context, module, op, operand_facts,
                                     result_facts, /*signed_unpack=*/true);
}

static bool loom_vector_table_lookup_exact_index_lane(
    const loom_fact_context_t* context, loom_value_facts_t table_facts,
    loom_type_t table_type, loom_value_facts_t index_facts,
    loom_value_facts_t* out_element) {
  int64_t index = 0;
  if (!loom_vector_facts_query_exact_i64(index_facts, &index) || index < 0) {
    *out_element = loom_value_facts_unknown();
    return true;
  }
  iree_host_size_t table_lane_count = 0;
  if (loom_vector_type_static_lane_count(table_type, &table_lane_count) &&
      (uint64_t)index >= (uint64_t)table_lane_count) {
    return false;
  }
  if ((uint64_t)index > (uint64_t)IREE_HOST_SIZE_MAX ||
      !loom_vector_facts_query_lane(context, table_facts,
                                    (iree_host_size_t)index, out_element)) {
    *out_element = loom_value_facts_unknown();
  }
  return true;
}

iree_status_t loom_vector_table_lookup_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  loom_type_t table_type =
      loom_module_value_type(module, loom_vector_table_lookup_table(op));
  loom_type_t result_type =
      loom_module_value_type(module, loom_vector_table_lookup_result(op));
  if (loom_type_element_type(table_type) !=
      loom_type_element_type(result_type)) {
    return loom_vector_make_unknown_facts(result_facts);
  }
  loom_value_facts_t uniform_index = {0};
  if (loom_vector_facts_query_uniform_element(context, operand_facts[1],
                                              &uniform_index)) {
    loom_value_facts_t element = loom_value_facts_unknown();
    if (!loom_vector_table_lookup_exact_index_lane(
            context, operand_facts[0], table_type, uniform_index, &element)) {
      return loom_vector_make_unknown_facts(result_facts);
    }
    return loom_value_facts_make_uniform_element(context, element,
                                                 &result_facts[0]);
  }

  iree_host_size_t result_lane_count = 0;
  if (!loom_vector_type_static_lane_count(result_type, &result_lane_count) ||
      result_lane_count > LOOM_VALUE_FACT_SMALL_STATIC_LANE_LIMIT) {
    return loom_vector_make_unknown_facts(result_facts);
  }
  loom_value_facts_t lanes[LOOM_VALUE_FACT_SMALL_STATIC_LANE_LIMIT] = {{0}};
  for (iree_host_size_t lane = 0; lane < result_lane_count; ++lane) {
    loom_value_facts_t index_facts = {0};
    if (!loom_vector_facts_query_lane(context, operand_facts[1], lane,
                                      &index_facts) ||
        !loom_vector_table_lookup_exact_index_lane(
            context, operand_facts[0], table_type, index_facts, &lanes[lane])) {
      return loom_vector_make_unknown_facts(result_facts);
    }
  }
  return loom_vector_make_small_static_lane_facts(
      context, lanes, result_lane_count, &result_facts[0]);
}

static bool loom_vector_table_quantize_exact_lane(
    const loom_fact_context_t* context, loom_scalar_type_t scalar_type,
    loom_value_facts_t input, loom_value_facts_t threshold_facts,
    iree_host_size_t threshold_count, uint8_t nan_policy, uint8_t tie_policy,
    loom_value_facts_t* out_element) {
  if (loom_value_facts_is_nan(input)) {
    *out_element = loom_value_facts_exact_i64(
        nan_policy == LOOM_VECTOR_TABLE_QUANTIZE_NAN_MAX
            ? (int64_t)threshold_count
            : 0);
    return true;
  }
  double input_value = 0.0;
  if (!loom_vector_facts_query_exact_float(scalar_type, input, &input_value)) {
    *out_element = loom_value_facts_make(0, (int64_t)threshold_count, 1);
    return true;
  }
  if (isnan(input_value)) {
    *out_element = loom_value_facts_exact_i64(
        nan_policy == LOOM_VECTOR_TABLE_QUANTIZE_NAN_MAX
            ? (int64_t)threshold_count
            : 0);
    return true;
  }

  int64_t code = 0;
  for (iree_host_size_t i = 0; i < threshold_count; ++i) {
    loom_value_facts_t threshold = {0};
    double threshold_value = 0.0;
    if (!loom_vector_facts_query_lane(context, threshold_facts, i,
                                      &threshold) ||
        !loom_vector_facts_query_exact_float(scalar_type, threshold,
                                             &threshold_value)) {
      *out_element = loom_value_facts_unknown();
      return true;
    }
    bool passed = tie_policy == LOOM_VECTOR_TABLE_QUANTIZE_TIE_UPPER
                      ? threshold_value <= input_value
                      : threshold_value < input_value;
    if (passed) ++code;
  }
  *out_element = loom_value_facts_exact_i64(code);
  return true;
}

iree_status_t loom_vector_table_quantize_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  loom_type_t threshold_type =
      loom_module_value_type(module, loom_vector_table_quantize_thresholds(op));
  const loom_scalar_type_t scalar_type = loom_type_element_type(threshold_type);
  loom_type_t result_type =
      loom_module_value_type(module, loom_vector_table_quantize_result(op));
  iree_host_size_t threshold_count = 0;
  if (!loom_vector_type_static_lane_count(threshold_type, &threshold_count) ||
      threshold_count > LOOM_VECTOR_FACT_STATIC_LOOP_LIMIT ||
      threshold_count > (iree_host_size_t)INT64_MAX ||
      !loom_vector_unsigned_code_capacity_covers(result_type,
                                                 (int64_t)threshold_count)) {
    return loom_vector_make_unknown_facts(result_facts);
  }

  uint8_t nan_policy = loom_vector_table_quantize_nan(op);
  uint8_t tie_policy = loom_vector_table_quantize_tie(op);
  loom_value_facts_t uniform_input = {0};
  if (loom_vector_facts_query_uniform_element(context, operand_facts[0],
                                              &uniform_input)) {
    loom_value_facts_t element = loom_value_facts_unknown();
    if (!loom_vector_table_quantize_exact_lane(
            context, scalar_type, uniform_input, operand_facts[1],
            threshold_count, nan_policy, tie_policy, &element)) {
      return loom_vector_make_unknown_facts(result_facts);
    }
    return loom_value_facts_make_uniform_element(context, element,
                                                 &result_facts[0]);
  }

  iree_host_size_t result_lane_count = 0;
  if (!loom_vector_type_static_lane_count(result_type, &result_lane_count) ||
      result_lane_count > LOOM_VALUE_FACT_SMALL_STATIC_LANE_LIMIT) {
    return loom_vector_make_unknown_facts(result_facts);
  }
  loom_value_facts_t lanes[LOOM_VALUE_FACT_SMALL_STATIC_LANE_LIMIT] = {{0}};
  for (iree_host_size_t lane = 0; lane < result_lane_count; ++lane) {
    loom_value_facts_t input = {0};
    if (!loom_vector_facts_query_lane(context, operand_facts[0], lane,
                                      &input) ||
        !loom_vector_table_quantize_exact_lane(
            context, scalar_type, input, operand_facts[1], threshold_count,
            nan_policy, tie_policy, &lanes[lane])) {
      return loom_vector_make_unknown_facts(result_facts);
    }
  }
  return loom_vector_make_small_static_lane_facts(
      context, lanes, result_lane_count, &result_facts[0]);
}

//===----------------------------------------------------------------------===//
// Scalar-producing reductions
//===----------------------------------------------------------------------===//

static bool loom_vector_reduce_apply_integer(
    loom_combining_kind_t kind, const loom_value_facts_t* accumulator,
    const loom_value_facts_t* element, loom_value_facts_t* out) {
  switch (kind) {
    case LOOM_COMBINING_KIND_ADDI:
      loom_value_facts_addi(accumulator, element, out);
      return true;
    case LOOM_COMBINING_KIND_MULI:
      loom_value_facts_muli(accumulator, element, out);
      return true;
    case LOOM_COMBINING_KIND_MINSI:
      loom_value_facts_minsi(accumulator, element, out);
      return true;
    case LOOM_COMBINING_KIND_MAXSI:
      loom_value_facts_maxsi(accumulator, element, out);
      return true;
    case LOOM_COMBINING_KIND_MINUI:
      loom_value_facts_minui(accumulator, element, out);
      return true;
    case LOOM_COMBINING_KIND_MAXUI:
      loom_value_facts_maxui(accumulator, element, out);
      return true;
    case LOOM_COMBINING_KIND_ANDI:
      loom_value_facts_andi(accumulator, element, out);
      return true;
    case LOOM_COMBINING_KIND_ORI:
      loom_value_facts_ori(accumulator, element, out);
      return true;
    case LOOM_COMBINING_KIND_XORI:
      loom_value_facts_xori(accumulator, element, out);
      return true;
    default:
      return false;
  }
}

static bool loom_vector_reduce_apply_float(
    loom_scalar_type_t scalar_type, loom_combining_kind_t kind,
    const loom_value_facts_t* accumulator, const loom_value_facts_t* element,
    loom_value_facts_t* out) {
  switch (kind) {
    case LOOM_COMBINING_KIND_ADDF:
      loom_value_facts_eval_float_binary(scalar_type, accumulator, element,
                                         loom_vector_add_f32,
                                         loom_vector_add_f64, out);
      return true;
    case LOOM_COMBINING_KIND_MULF:
      loom_value_facts_eval_float_binary(scalar_type, accumulator, element,
                                         loom_vector_mul_f32,
                                         loom_vector_mul_f64, out);
      return true;
    case LOOM_COMBINING_KIND_MINIMUMF:
      loom_value_facts_eval_float_minmax(scalar_type, LOOM_FLOAT_MINMAX_MINIMUM,
                                         accumulator, element, out);
      return true;
    case LOOM_COMBINING_KIND_MAXIMUMF:
      loom_value_facts_eval_float_minmax(scalar_type, LOOM_FLOAT_MINMAX_MAXIMUM,
                                         accumulator, element, out);
      return true;
    case LOOM_COMBINING_KIND_MINNUMF:
      loom_value_facts_eval_float_minmax(scalar_type, LOOM_FLOAT_MINMAX_MINNUM,
                                         accumulator, element, out);
      return true;
    case LOOM_COMBINING_KIND_MAXNUMF:
      loom_value_facts_eval_float_minmax(scalar_type, LOOM_FLOAT_MINMAX_MAXNUM,
                                         accumulator, element, out);
      return true;
    default:
      return false;
  }
}

static int64_t loom_vector_facts_integer_all_ones(
    loom_scalar_type_t element_type) {
  return element_type == LOOM_SCALAR_TYPE_I1 ? 1 : -1;
}

static bool loom_vector_reduce_dynamic_identity(loom_combining_kind_t kind,
                                                loom_scalar_type_t element_type,
                                                loom_value_facts_t element,
                                                loom_value_facts_t init,
                                                loom_value_facts_t* out) {
  switch (kind) {
    case LOOM_COMBINING_KIND_ADDI:
    case LOOM_COMBINING_KIND_ORI:
    case LOOM_COMBINING_KIND_XORI:
      if (loom_vector_facts_exact_i64_is(element, 0)) {
        *out = init;
        return true;
      }
      return false;
    case LOOM_COMBINING_KIND_MULI:
      if (loom_vector_facts_exact_i64_is(element, 1)) {
        *out = init;
        return true;
      }
      return false;
    case LOOM_COMBINING_KIND_ANDI:
      if (loom_vector_facts_exact_i64_is(
              element, loom_vector_facts_integer_all_ones(element_type))) {
        *out = init;
        return true;
      }
      return false;
    default:
      return false;
  }
}

static bool loom_vector_reduce_apply_facts(loom_scalar_type_t scalar_type,
                                           loom_combining_kind_t kind,
                                           loom_value_facts_t accumulator,
                                           loom_value_facts_t element,
                                           loom_value_facts_t* out) {
  if (loom_scalar_type_is_float(scalar_type)) {
    return loom_vector_reduce_apply_float(scalar_type, kind, &accumulator,
                                          &element, out);
  }
  return loom_vector_reduce_apply_integer(kind, &accumulator, &element, out);
}

static bool loom_vector_reduce_static_uniform(loom_combining_kind_t kind,
                                              loom_scalar_type_t scalar_type,
                                              uint64_t element_count,
                                              loom_value_facts_t element,
                                              loom_value_facts_t init,
                                              loom_value_facts_t* out) {
  if (element_count == 0) {
    *out = init;
    return true;
  }
  if (element_count > LOOM_VECTOR_FACT_STATIC_LOOP_LIMIT) return false;

  loom_value_facts_t accumulator = init;
  for (uint64_t i = 0; i < element_count; ++i) {
    loom_value_facts_t next = loom_value_facts_unknown();
    if (!loom_vector_reduce_apply_facts(scalar_type, kind, accumulator, element,
                                        &next)) {
      return false;
    }
    accumulator = next;
  }
  *out = accumulator;
  return true;
}

static bool loom_vector_reduce_small_static_lanes(
    loom_combining_kind_t kind, loom_scalar_type_t scalar_type,
    loom_value_fact_small_static_lanes_t lanes, loom_value_facts_t init,
    loom_value_facts_t* out) {
  if (lanes.count == 0) {
    *out = init;
    return true;
  }

  loom_value_facts_t accumulator = init;
  for (iree_host_size_t i = 0; i < lanes.count; ++i) {
    loom_value_facts_t next = loom_value_facts_unknown();
    if (!loom_vector_reduce_apply_facts(scalar_type, kind, accumulator,
                                        lanes.lanes[i], &next)) {
      return false;
    }
    accumulator = next;
  }
  *out = accumulator;
  return true;
}

static iree_status_t loom_vector_reduce_all_lanes_facts(
    loom_fact_context_t* context, loom_type_t input_type,
    loom_combining_kind_t kind, loom_value_facts_t input_facts,
    loom_value_facts_t init_facts, loom_value_facts_t* out_facts) {
  uint64_t element_count = 0;
  if (loom_type_static_element_count(input_type, &element_count) &&
      element_count == 0) {
    *out_facts = init_facts;
    return iree_ok_status();
  }

  loom_value_fact_small_static_lanes_t lanes = {0};
  if (loom_vector_facts_query_small_lanes(context, input_facts, &lanes)) {
    if (loom_vector_reduce_small_static_lanes(
            kind, loom_type_element_type(input_type), lanes, init_facts,
            out_facts)) {
      return iree_ok_status();
    }
    *out_facts = loom_value_facts_unknown();
    return iree_ok_status();
  }

  loom_value_facts_t element = {0};
  if (!loom_vector_facts_query_uniform_element(context, input_facts,
                                               &element)) {
    *out_facts = loom_value_facts_unknown();
    return iree_ok_status();
  }

  if (loom_type_static_element_count(input_type, &element_count)) {
    if (loom_vector_reduce_static_uniform(
            kind, loom_type_element_type(input_type), element_count, element,
            init_facts, out_facts)) {
      return iree_ok_status();
    }
  } else if (loom_vector_reduce_dynamic_identity(
                 kind, loom_type_element_type(input_type), element, init_facts,
                 out_facts)) {
    return iree_ok_status();
  }

  *out_facts = loom_value_facts_unknown();
  return iree_ok_status();
}

static bool loom_vector_reduce_axes_all_source_axes(loom_type_t input_type,
                                                    loom_attribute_t axes) {
  if (axes.count != loom_type_rank(input_type)) return false;
  for (uint16_t i = 0; i < axes.count; ++i) {
    if (axes.i64_array[i] != (int64_t)i) return false;
  }
  return true;
}

static bool loom_vector_reduce_axes_static_element_count(loom_type_t input_type,
                                                         loom_attribute_t axes,
                                                         uint64_t* out_count) {
  uint64_t count = 1;
  for (uint16_t i = 0; i < axes.count; ++i) {
    uint8_t axis = (uint8_t)axes.i64_array[i];
    if (loom_type_dim_is_dynamic_at(input_type, axis)) return false;
    uint64_t dimension_size =
        (uint64_t)loom_type_dim_static_size_at(input_type, axis);
    if (dimension_size != 0 && count > UINT64_MAX / dimension_size) {
      return false;
    }
    count *= dimension_size;
  }
  *out_count = count;
  return true;
}

static void loom_vector_reduce_axes_indices_from_ordinal(
    loom_type_t input_type, loom_attribute_t axes, iree_host_size_t ordinal,
    int64_t* reduced_indices) {
  for (uint16_t reverse_index = 0; reverse_index < axes.count;
       ++reverse_index) {
    uint16_t index = (uint16_t)(axes.count - reverse_index - 1);
    uint8_t axis = (uint8_t)axes.i64_array[index];
    uint64_t dimension_size =
        (uint64_t)loom_type_dim_static_size_at(input_type, axis);
    reduced_indices[index] =
        dimension_size == 0 ? 0 : (int64_t)(ordinal % dimension_size);
    if (dimension_size != 0) ordinal /= dimension_size;
  }
}

static void loom_vector_reduce_axes_source_indices(
    loom_type_t input_type, loom_attribute_t axes,
    const int64_t* result_indices, const int64_t* reduced_indices,
    int64_t* source_indices) {
  uint16_t reduced_index = 0;
  uint8_t result_axis = 0;
  uint8_t input_rank = loom_type_rank(input_type);
  for (uint8_t input_axis = 0; input_axis < input_rank; ++input_axis) {
    if (reduced_index < axes.count &&
        axes.i64_array[reduced_index] == input_axis) {
      source_indices[input_axis] = reduced_indices[reduced_index++];
    } else {
      source_indices[input_axis] = result_indices[result_axis++];
    }
  }
}

static iree_status_t loom_vector_reduce_axes_static_lane_facts(
    loom_fact_context_t* context, loom_type_t input_type,
    loom_type_t result_type, loom_attribute_t axes, loom_combining_kind_t kind,
    const loom_value_facts_t* operand_facts, loom_value_facts_t* out_facts,
    bool* out_handled) {
  *out_handled = false;
  if (!loom_type_is_vector(result_type) ||
      !loom_type_is_all_static(input_type) ||
      !loom_type_is_all_static(result_type)) {
    return iree_ok_status();
  }

  iree_host_size_t result_lane_count = 0;
  if (!loom_vector_type_static_lane_count(result_type, &result_lane_count) ||
      result_lane_count > LOOM_VALUE_FACT_SMALL_STATIC_LANE_LIMIT) {
    return iree_ok_status();
  }

  uint64_t reduced_element_count = 0;
  if (!loom_vector_reduce_axes_static_element_count(input_type, axes,
                                                    &reduced_element_count) ||
      reduced_element_count > LOOM_VECTOR_FACT_STATIC_LOOP_LIMIT) {
    return iree_ok_status();
  }
  *out_handled = true;
  if (reduced_element_count == 0 || result_lane_count == 0) {
    *out_facts = operand_facts[1];
    return iree_ok_status();
  }

  loom_value_facts_t lanes[LOOM_VALUE_FACT_SMALL_STATIC_LANE_LIMIT] = {{0}};
  int64_t result_indices[LOOM_TYPE_MAX_RANK] = {0};
  int64_t reduced_indices[LOOM_TYPE_MAX_RANK] = {0};
  int64_t source_indices[LOOM_TYPE_MAX_RANK] = {0};
  for (iree_host_size_t result_ordinal = 0; result_ordinal < result_lane_count;
       ++result_ordinal) {
    loom_vector_static_indices_from_ordinal(result_type, result_ordinal,
                                            result_indices);

    loom_value_facts_t accumulator = {0};
    if (!loom_vector_facts_query_lane(context, operand_facts[1], result_ordinal,
                                      &accumulator)) {
      *out_facts = loom_value_facts_unknown();
      return iree_ok_status();
    }

    for (uint64_t reduction_ordinal = 0;
         reduction_ordinal < reduced_element_count; ++reduction_ordinal) {
      loom_vector_reduce_axes_indices_from_ordinal(
          input_type, axes, (iree_host_size_t)reduction_ordinal,
          reduced_indices);
      loom_vector_reduce_axes_source_indices(input_type, axes, result_indices,
                                             reduced_indices, source_indices);

      iree_host_size_t source_ordinal = 0;
      if (!loom_vector_static_ordinal_from_indices(input_type, source_indices,
                                                   &source_ordinal)) {
        *out_facts = loom_value_facts_unknown();
        return iree_ok_status();
      }
      loom_value_facts_t element = {0};
      if (!loom_vector_facts_query_lane(context, operand_facts[0],
                                        source_ordinal, &element)) {
        *out_facts = loom_value_facts_unknown();
        return iree_ok_status();
      }
      loom_value_facts_t next = {0};
      if (!loom_vector_reduce_apply_facts(loom_type_element_type(input_type),
                                          kind, accumulator, element, &next)) {
        *out_facts = loom_value_facts_unknown();
        return iree_ok_status();
      }
      accumulator = next;
    }
    lanes[result_ordinal] = accumulator;
  }

  return loom_vector_make_small_static_lane_facts(context, lanes,
                                                  result_lane_count, out_facts);
}

iree_status_t loom_vector_reduce_facts(loom_fact_context_t* context,
                                       const loom_module_t* module,
                                       const loom_op_t* op,
                                       const loom_value_facts_t* operand_facts,
                                       loom_value_facts_t* result_facts) {
  loom_type_t input_type =
      loom_module_value_type(module, loom_vector_reduce_input(op));
  return loom_vector_reduce_all_lanes_facts(
      context, input_type, loom_vector_reduce_kind(op), operand_facts[0],
      operand_facts[1], &result_facts[0]);
}

iree_status_t loom_vector_reduce_axes_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  loom_type_t input_type =
      loom_module_value_type(module, loom_vector_reduce_axes_input(op));
  loom_attribute_t axes = loom_vector_reduce_axes_axes(op);
  loom_combining_kind_t kind = loom_vector_reduce_axes_kind(op);
  if (loom_vector_reduce_axes_all_source_axes(input_type, axes)) {
    return loom_vector_reduce_all_lanes_facts(
        context, input_type, kind, operand_facts[0], operand_facts[1],
        &result_facts[0]);
  }

  loom_type_t result_type =
      loom_module_value_type(module, loom_vector_reduce_axes_result(op));
  if (loom_type_is_vector(result_type) &&
      loom_type_is_all_static(result_type)) {
    iree_host_size_t result_lane_count = 0;
    if (loom_vector_type_static_lane_count(result_type, &result_lane_count) &&
        result_lane_count == 0) {
      result_facts[0] = operand_facts[1];
      return iree_ok_status();
    }
  }

  uint64_t reduced_element_count = 0;
  if (loom_vector_reduce_axes_static_element_count(input_type, axes,
                                                   &reduced_element_count) &&
      reduced_element_count == 0) {
    result_facts[0] = operand_facts[1];
    return iree_ok_status();
  }

  loom_value_facts_t input_element = {0};
  loom_value_facts_t init_element = {0};
  if (loom_vector_facts_query_uniform_element(context, operand_facts[0],
                                              &input_element) &&
      loom_vector_facts_query_uniform_element(context, operand_facts[1],
                                              &init_element)) {
    if (loom_vector_reduce_axes_static_element_count(input_type, axes,
                                                     &reduced_element_count)) {
      loom_value_facts_t reduced_element = {0};
      if (loom_vector_reduce_static_uniform(
              kind, loom_type_element_type(input_type), reduced_element_count,
              input_element, init_element, &reduced_element)) {
        return loom_value_facts_make_uniform_element(context, reduced_element,
                                                     &result_facts[0]);
      }
    } else if (loom_vector_reduce_dynamic_identity(
                   kind, loom_type_element_type(input_type), input_element,
                   operand_facts[1], &result_facts[0])) {
      return iree_ok_status();
    }
  } else if (loom_vector_facts_query_uniform_element(context, operand_facts[0],
                                                     &input_element) &&
             loom_vector_reduce_dynamic_identity(
                 kind, loom_type_element_type(input_type), input_element,
                 operand_facts[1], &result_facts[0])) {
    return iree_ok_status();
  }

  bool handled = false;
  IREE_RETURN_IF_ERROR(loom_vector_reduce_axes_static_lane_facts(
      context, input_type, result_type, axes, kind, operand_facts,
      &result_facts[0], &handled));
  if (handled) {
    return iree_ok_status();
  }

  result_facts[0] = loom_value_facts_unknown();
  return iree_ok_status();
}

iree_status_t loom_vector_mma_facts(loom_fact_context_t* context,
                                    const loom_module_t* module,
                                    const loom_op_t* op,
                                    const loom_value_facts_t* operand_facts,
                                    loom_value_facts_t* result_facts) {
  (void)module;
  (void)op;
  loom_vector_fragment_fact_t init_fragment;
  if (!loom_vector_fragment_fact_query_value_facts(context, operand_facts[2],
                                                   &init_fragment) ||
      !iree_any_bit_set(init_fragment.role_flags,
                        LOOM_VECTOR_FRAGMENT_ROLE_FLAG_INIT |
                            LOOM_VECTOR_FRAGMENT_ROLE_FLAG_RESULT)) {
    result_facts[0] = loom_value_facts_unknown();
    return iree_ok_status();
  }
  init_fragment.role_flags = LOOM_VECTOR_FRAGMENT_ROLE_FLAG_INIT |
                             LOOM_VECTOR_FRAGMENT_ROLE_FLAG_RESULT;
  init_fragment.flags |= LOOM_VECTOR_FRAGMENT_FACT_FLAG_HAS_NATIVE_STORAGE;
  return loom_vector_fragment_fact_make_value_facts(context, init_fragment,
                                                    &result_facts[0]);
}

static bool loom_vector_accumulate_float_fma(
    loom_scalar_type_t scalar_type, loom_value_facts_t lhs,
    loom_value_facts_t rhs, loom_value_facts_t accumulator,
    loom_value_facts_t* out_accumulator) {
  loom_value_facts_eval_float_ternary(scalar_type, &lhs, &rhs, &accumulator,
                                      fmaf, fma, out_accumulator);
  return loom_value_facts_is_exact(*out_accumulator) ||
         loom_value_facts_is_nan(*out_accumulator);
}

iree_status_t loom_vector_dotf_facts(loom_fact_context_t* context,
                                     const loom_module_t* module,
                                     const loom_op_t* op,
                                     const loom_value_facts_t* operand_facts,
                                     loom_value_facts_t* result_facts) {
  uint64_t element_count = 0;
  loom_type_t lhs_type =
      loom_module_value_type(module, loom_vector_dotf_lhs(op));
  const loom_scalar_type_t scalar_type =
      loom_vector_result_element_type(module, op);
  if (loom_type_static_element_count(lhs_type, &element_count) &&
      element_count == 0) {
    result_facts[0] = operand_facts[2];
    return iree_ok_status();
  }

  iree_host_size_t lane_count = 0;
  if (loom_vector_facts_query_binary_lane_count(
          context, operand_facts[0], operand_facts[1], &lane_count)) {
    loom_value_facts_t accumulator = operand_facts[2];
    for (iree_host_size_t i = 0; i < lane_count; ++i) {
      loom_value_facts_t lhs = {0};
      loom_value_facts_t rhs = {0};
      if (!loom_vector_facts_query_lane(context, operand_facts[0], i, &lhs) ||
          !loom_vector_facts_query_lane(context, operand_facts[1], i, &rhs)) {
        result_facts[0] = loom_value_facts_unknown();
        return iree_ok_status();
      }
      loom_value_facts_t next = loom_value_facts_unknown();
      if (!loom_vector_accumulate_float_fma(scalar_type, lhs, rhs, accumulator,
                                            &next)) {
        result_facts[0] = loom_value_facts_unknown();
        return iree_ok_status();
      }
      accumulator = next;
    }
    result_facts[0] = accumulator;
    return iree_ok_status();
  }

  if (!loom_type_static_element_count(lhs_type, &element_count) ||
      element_count > LOOM_VECTOR_FACT_STATIC_LOOP_LIMIT) {
    result_facts[0] = loom_value_facts_unknown();
    return iree_ok_status();
  }

  loom_value_facts_t lhs = {0};
  loom_value_facts_t rhs = {0};
  if (!loom_vector_facts_query_uniform_element(context, operand_facts[0],
                                               &lhs) ||
      !loom_vector_facts_query_uniform_element(context, operand_facts[1],
                                               &rhs)) {
    result_facts[0] = loom_value_facts_unknown();
    return iree_ok_status();
  }

  loom_value_facts_t accumulator = operand_facts[2];
  for (uint64_t i = 0; i < element_count; ++i) {
    loom_value_facts_t next = loom_value_facts_unknown();
    if (!loom_vector_accumulate_float_fma(scalar_type, lhs, rhs, accumulator,
                                          &next)) {
      result_facts[0] = loom_value_facts_unknown();
      return iree_ok_status();
    }
    accumulator = next;
  }
  result_facts[0] = accumulator;
  return iree_ok_status();
}

iree_status_t loom_vector_dot2f_facts(loom_fact_context_t* context,
                                      const loom_module_t* module,
                                      const loom_op_t* op,
                                      const loom_value_facts_t* operand_facts,
                                      loom_value_facts_t* result_facts) {
  const loom_scalar_type_t scalar_type =
      loom_vector_result_element_type(module, op);
  loom_value_facts_t lhs_element = {0};
  loom_value_facts_t rhs_element = {0};
  loom_value_facts_t acc_element = {0};
  if (loom_vector_facts_query_uniform_element(context, operand_facts[0],
                                              &lhs_element) &&
      loom_vector_facts_query_uniform_element(context, operand_facts[1],
                                              &rhs_element) &&
      loom_vector_facts_query_uniform_element(context, operand_facts[2],
                                              &acc_element)) {
    loom_value_facts_t accumulator = acc_element;
    for (uint8_t group_lane = 0; group_lane < 2; ++group_lane) {
      loom_value_facts_t next = loom_value_facts_unknown();
      if (!loom_vector_accumulate_float_fma(scalar_type, lhs_element,
                                            rhs_element, accumulator, &next)) {
        return loom_vector_make_unknown_facts(result_facts);
      }
      accumulator = next;
    }
    return loom_value_facts_make_uniform_element(context, accumulator,
                                                 &result_facts[0]);
  }

  loom_type_t lhs_type =
      loom_module_value_type(module, loom_vector_dot2f_lhs(op));
  loom_type_t result_type =
      loom_module_value_type(module, loom_vector_dot2f_result(op));
  loom_vector_grouped_dot_shape_t shape = {0};
  if (!loom_vector_query_grouped_dot_shape(lhs_type, result_type, 2, &shape) ||
      shape.result_lane_count > LOOM_VALUE_FACT_SMALL_STATIC_LANE_LIMIT) {
    return loom_vector_make_unknown_facts(result_facts);
  }

  loom_value_facts_t lanes[LOOM_VALUE_FACT_SMALL_STATIC_LANE_LIMIT] = {{0}};
  for (iree_host_size_t result_lane = 0; result_lane < shape.result_lane_count;
       ++result_lane) {
    loom_value_facts_t acc = {0};
    if (!loom_vector_facts_query_lane(context, operand_facts[2], result_lane,
                                      &acc)) {
      return loom_vector_make_unknown_facts(result_facts);
    }
    loom_value_facts_t accumulator = acc;
    for (uint8_t group_lane = 0; group_lane < 2; ++group_lane) {
      iree_host_size_t source_lane = 0;
      loom_value_facts_t lhs = {0};
      loom_value_facts_t rhs = {0};
      if (!loom_vector_grouped_dot_source_lane(shape, result_lane, 2,
                                               group_lane, &source_lane) ||
          !loom_vector_facts_query_lane(context, operand_facts[0], source_lane,
                                        &lhs) ||
          !loom_vector_facts_query_lane(context, operand_facts[1], source_lane,
                                        &rhs)) {
        return loom_vector_make_unknown_facts(result_facts);
      }
      loom_value_facts_t next = loom_value_facts_unknown();
      if (!loom_vector_accumulate_float_fma(scalar_type, lhs, rhs, accumulator,
                                            &next)) {
        return loom_vector_make_unknown_facts(result_facts);
      }
      accumulator = next;
    }
    lanes[result_lane] = accumulator;
  }

  loom_value_fact_small_static_lanes_t lane_slice = {
      .lanes = lanes,
      .count = shape.result_lane_count,
  };
  return loom_value_facts_make_small_static_lanes(context, lane_slice,
                                                  &result_facts[0]);
}

iree_status_t loom_vector_dot4i_facts(loom_fact_context_t* context,
                                      const loom_module_t* module,
                                      const loom_op_t* op,
                                      const loom_value_facts_t* operand_facts,
                                      loom_value_facts_t* result_facts) {
  uint8_t kind = loom_vector_dot4i_kind(op);
  loom_value_facts_t lhs_element = {0};
  loom_value_facts_t rhs_element = {0};
  loom_value_facts_t acc_element = {0};
  if (loom_vector_facts_query_uniform_element(context, operand_facts[0],
                                              &lhs_element) &&
      loom_vector_facts_query_uniform_element(context, operand_facts[1],
                                              &rhs_element) &&
      loom_vector_facts_query_uniform_element(context, operand_facts[2],
                                              &acc_element)) {
    int64_t lhs_value = 0;
    int64_t rhs_value = 0;
    int32_t accumulator = 0;
    if (!loom_vector_facts_query_exact_i64(lhs_element, &lhs_value) ||
        !loom_vector_facts_query_exact_i64(rhs_element, &rhs_value) ||
        !loom_vector_facts_query_exact_i32(acc_element, &accumulator)) {
      return loom_vector_make_unknown_facts(result_facts);
    }
    for (uint8_t group_lane = 0; group_lane < 4; ++group_lane) {
      if (!loom_vector_dot4i_apply(kind, lhs_value, rhs_value, &accumulator)) {
        return loom_vector_make_unknown_facts(result_facts);
      }
    }
    return loom_value_facts_make_uniform_element(
        context, loom_value_facts_exact_i64(accumulator), &result_facts[0]);
  }

  loom_type_t lhs_type =
      loom_module_value_type(module, loom_vector_dot4i_lhs(op));
  loom_type_t result_type =
      loom_module_value_type(module, loom_vector_dot4i_result(op));
  loom_vector_grouped_dot_shape_t shape = {0};
  if (!loom_vector_query_grouped_dot_shape(lhs_type, result_type, 4, &shape) ||
      shape.result_lane_count > LOOM_VALUE_FACT_SMALL_STATIC_LANE_LIMIT) {
    return loom_vector_make_unknown_facts(result_facts);
  }

  loom_value_facts_t lanes[LOOM_VALUE_FACT_SMALL_STATIC_LANE_LIMIT] = {{0}};
  for (iree_host_size_t result_lane = 0; result_lane < shape.result_lane_count;
       ++result_lane) {
    loom_value_facts_t acc = {0};
    int32_t accumulator = 0;
    if (!loom_vector_facts_query_lane(context, operand_facts[2], result_lane,
                                      &acc) ||
        !loom_vector_facts_query_exact_i32(acc, &accumulator)) {
      return loom_vector_make_unknown_facts(result_facts);
    }
    for (uint8_t group_lane = 0; group_lane < 4; ++group_lane) {
      iree_host_size_t source_lane = 0;
      loom_value_facts_t lhs = {0};
      loom_value_facts_t rhs = {0};
      int64_t lhs_value = 0;
      int64_t rhs_value = 0;
      if (!loom_vector_grouped_dot_source_lane(shape, result_lane, 4,
                                               group_lane, &source_lane) ||
          !loom_vector_facts_query_lane(context, operand_facts[0], source_lane,
                                        &lhs) ||
          !loom_vector_facts_query_lane(context, operand_facts[1], source_lane,
                                        &rhs) ||
          !loom_vector_facts_query_exact_i64(lhs, &lhs_value) ||
          !loom_vector_facts_query_exact_i64(rhs, &rhs_value) ||
          !loom_vector_dot4i_apply(kind, lhs_value, rhs_value, &accumulator)) {
        return loom_vector_make_unknown_facts(result_facts);
      }
    }
    lanes[result_lane] = loom_value_facts_exact_i64(accumulator);
  }

  loom_value_fact_small_static_lanes_t lane_slice = {
      .lanes = lanes,
      .count = shape.result_lane_count,
  };
  return loom_value_facts_make_small_static_lanes(context, lane_slice,
                                                  &result_facts[0]);
}

iree_status_t loom_vector_dot8i4_facts(loom_fact_context_t* context,
                                       const loom_module_t* module,
                                       const loom_op_t* op,
                                       const loom_value_facts_t* operand_facts,
                                       loom_value_facts_t* result_facts) {
  uint8_t kind = loom_vector_dot8i4_kind(op);
  loom_value_facts_t lhs_element = {0};
  loom_value_facts_t rhs_element = {0};
  loom_value_facts_t acc_element = {0};
  if (loom_vector_facts_query_uniform_element(context, operand_facts[0],
                                              &lhs_element) &&
      loom_vector_facts_query_uniform_element(context, operand_facts[1],
                                              &rhs_element) &&
      loom_vector_facts_query_uniform_element(context, operand_facts[2],
                                              &acc_element)) {
    uint32_t lhs_bits = 0;
    uint32_t rhs_bits = 0;
    int32_t accumulator = 0;
    if (!loom_vector_facts_query_exact_u32_bits(lhs_element, &lhs_bits) ||
        !loom_vector_facts_query_exact_u32_bits(rhs_element, &rhs_bits) ||
        !loom_vector_facts_query_exact_i32(acc_element, &accumulator) ||
        !loom_vector_dot8i4_apply(kind, lhs_bits, rhs_bits, &accumulator)) {
      return loom_vector_make_unknown_facts(result_facts);
    }
    return loom_value_facts_make_uniform_element(
        context, loom_value_facts_exact_i64(accumulator), &result_facts[0]);
  }

  loom_type_t result_type =
      loom_module_value_type(module, loom_vector_dot8i4_result(op));
  uint64_t result_lane_count = 0;
  if (!loom_type_static_element_count(result_type, &result_lane_count) ||
      result_lane_count > LOOM_VALUE_FACT_SMALL_STATIC_LANE_LIMIT) {
    return loom_vector_make_unknown_facts(result_facts);
  }

  loom_value_facts_t lanes[LOOM_VALUE_FACT_SMALL_STATIC_LANE_LIMIT] = {{0}};
  for (iree_host_size_t result_lane = 0;
       result_lane < (iree_host_size_t)result_lane_count; ++result_lane) {
    loom_value_facts_t lhs = {0};
    loom_value_facts_t rhs = {0};
    loom_value_facts_t acc = {0};
    uint32_t lhs_bits = 0;
    uint32_t rhs_bits = 0;
    int32_t accumulator = 0;
    if (!loom_vector_facts_query_lane(context, operand_facts[0], result_lane,
                                      &lhs) ||
        !loom_vector_facts_query_lane(context, operand_facts[1], result_lane,
                                      &rhs) ||
        !loom_vector_facts_query_lane(context, operand_facts[2], result_lane,
                                      &acc) ||
        !loom_vector_facts_query_exact_u32_bits(lhs, &lhs_bits) ||
        !loom_vector_facts_query_exact_u32_bits(rhs, &rhs_bits) ||
        !loom_vector_facts_query_exact_i32(acc, &accumulator) ||
        !loom_vector_dot8i4_apply(kind, lhs_bits, rhs_bits, &accumulator)) {
      return loom_vector_make_unknown_facts(result_facts);
    }
    lanes[result_lane] = loom_value_facts_exact_i64(accumulator);
  }

  loom_value_fact_small_static_lanes_t lane_slice = {
      .lanes = lanes,
      .count = (iree_host_size_t)result_lane_count,
  };
  return loom_value_facts_make_small_static_lanes(context, lane_slice,
                                                  &result_facts[0]);
}

iree_status_t loom_vector_dot4f8_facts(loom_fact_context_t* context,
                                       const loom_module_t* module,
                                       const loom_op_t* op,
                                       const loom_value_facts_t* operand_facts,
                                       loom_value_facts_t* result_facts) {
  uint8_t kind = loom_vector_dot4f8_kind(op);
  loom_value_facts_t lhs_element = {0};
  loom_value_facts_t rhs_element = {0};
  loom_value_facts_t acc_element = {0};
  if (loom_vector_facts_query_uniform_element(context, operand_facts[0],
                                              &lhs_element) &&
      loom_vector_facts_query_uniform_element(context, operand_facts[1],
                                              &rhs_element) &&
      loom_vector_facts_query_uniform_element(context, operand_facts[2],
                                              &acc_element)) {
    uint32_t lhs_bits = 0;
    uint32_t rhs_bits = 0;
    double accumulator_value = 0.0;
    if (!loom_vector_facts_query_exact_u32_bits(lhs_element, &lhs_bits) ||
        !loom_vector_facts_query_exact_u32_bits(rhs_element, &rhs_bits) ||
        !loom_vector_facts_query_exact_float(LOOM_SCALAR_TYPE_F32, acc_element,
                                             &accumulator_value)) {
      return loom_vector_make_unknown_facts(result_facts);
    }
    float accumulator = (float)accumulator_value;
    if (!loom_vector_dot4f8_apply(kind, lhs_bits, rhs_bits, &accumulator)) {
      return loom_vector_make_unknown_facts(result_facts);
    }
    return loom_value_facts_make_uniform_element(
        context,
        loom_value_facts_exact_float(LOOM_SCALAR_TYPE_F32, accumulator),
        &result_facts[0]);
  }

  loom_type_t result_type =
      loom_module_value_type(module, loom_vector_dot4f8_result(op));
  uint64_t result_lane_count = 0;
  if (!loom_type_static_element_count(result_type, &result_lane_count) ||
      result_lane_count > LOOM_VALUE_FACT_SMALL_STATIC_LANE_LIMIT) {
    return loom_vector_make_unknown_facts(result_facts);
  }

  loom_value_facts_t lanes[LOOM_VALUE_FACT_SMALL_STATIC_LANE_LIMIT] = {{0}};
  for (iree_host_size_t result_lane = 0;
       result_lane < (iree_host_size_t)result_lane_count; ++result_lane) {
    loom_value_facts_t lhs = {0};
    loom_value_facts_t rhs = {0};
    loom_value_facts_t acc = {0};
    uint32_t lhs_bits = 0;
    uint32_t rhs_bits = 0;
    double accumulator_value = 0.0;
    if (!loom_vector_facts_query_lane(context, operand_facts[0], result_lane,
                                      &lhs) ||
        !loom_vector_facts_query_lane(context, operand_facts[1], result_lane,
                                      &rhs) ||
        !loom_vector_facts_query_lane(context, operand_facts[2], result_lane,
                                      &acc) ||
        !loom_vector_facts_query_exact_u32_bits(lhs, &lhs_bits) ||
        !loom_vector_facts_query_exact_u32_bits(rhs, &rhs_bits) ||
        !loom_vector_facts_query_exact_float(LOOM_SCALAR_TYPE_F32, acc,
                                             &accumulator_value)) {
      return loom_vector_make_unknown_facts(result_facts);
    }
    float accumulator = (float)accumulator_value;
    if (!loom_vector_dot4f8_apply(kind, lhs_bits, rhs_bits, &accumulator)) {
      return loom_vector_make_unknown_facts(result_facts);
    }
    lanes[result_lane] =
        loom_value_facts_exact_float(LOOM_SCALAR_TYPE_F32, accumulator);
  }

  loom_value_fact_small_static_lanes_t lane_slice = {
      .lanes = lanes,
      .count = (iree_host_size_t)result_lane_count,
  };
  return loom_value_facts_make_small_static_lanes(context, lane_slice,
                                                  &result_facts[0]);
}

#undef LOOM_VECTOR_FLOAT_BINARY_FACTS
#undef LOOM_VECTOR_FLOAT_UNARY_FACTS
#undef LOOM_VECTOR_BIT_COUNT_FACTS
#undef LOOM_VECTOR_INTEGER_BINARY_FACTS
#undef LOOM_VECTOR_UNARY_FACTS

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
#include <string.h>

#include "loom/ir/attribute.h"
#include "loom/ir/module.h"
#include "loom/ops/combining.h"
#include "loom/ops/encoding/numeric_transform.h"
#include "loom/ops/encoding/storage.h"
#include "loom/ops/scalar/compare.h"
#include "loom/ops/vector/fragment.h"
#include "loom/ops/vector/ops.h"
#include "loom/util/fact_table.h"
#include "loom/util/math.h"

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
typedef int64_t (*loom_vector_bit_count_fn_t)(uint64_t value, int32_t bitwidth);

typedef double (*loom_vector_float_unary_transfer_fn_t)(double input);
typedef double (*loom_vector_float_unary_data_transfer_fn_t)(
    double input, const void* user_data);
typedef double (*loom_vector_float_binary_transfer_fn_t)(double lhs,
                                                         double rhs);

//===----------------------------------------------------------------------===//
// Scalar element helpers
//===----------------------------------------------------------------------===//

static double loom_vector_sinturns_f64(double x) {
  return sin(6.28318530717958647692 * x);
}

static double loom_vector_costurns_f64(double x) {
  return cos(6.28318530717958647692 * x);
}

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
  if (!loom_checked_mul_i64((int64_t)lane, step, &delta) ||
      !loom_checked_add_i64(base, delta, &value)) {
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
    return loom_value_facts_exact_f64(loom_attr_as_f64(attr));
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
  if (!loom_checked_mul_i64((int64_t)lane_ordinal, step, &lane_delta)) {
    return false;
  }
  int64_t lane_value = 0;
  if (!loom_checked_add_i64(lower_bound, lane_delta, &lane_value)) {
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

static bool loom_vector_facts_exact_i64_is(loom_value_facts_t facts,
                                           int64_t expected) {
  return loom_value_facts_is_exact(facts) &&
         !loom_value_facts_is_float(facts) && facts.range_lo == expected;
}

static bool loom_vector_facts_query_exact_f64(loom_value_facts_t facts,
                                              double* out_value) {
  if (!loom_value_facts_is_exact(facts) || !loom_value_facts_is_float(facts)) {
    return false;
  }
  *out_value = loom_value_facts_as_f64(facts);
  return true;
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

static bool loom_vector_fragment_fact_is_accumulator_like(
    loom_vector_fragment_fact_t fact) {
  const loom_vector_fragment_role_flags_t accumulator_roles =
      LOOM_VECTOR_FRAGMENT_ROLE_FLAG_INIT |
      LOOM_VECTOR_FRAGMENT_ROLE_FLAG_RESULT;
  return fact.shape_rank == 2 && fact.role_flags != 0 &&
         (fact.role_flags & ~accumulator_roles) == 0;
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

static bool loom_vector_fragment_facts_match_accumulator_contract(
    loom_vector_fragment_fact_t lhs, loom_vector_fragment_fact_t rhs) {
  if (!loom_vector_fragment_fact_is_accumulator_like(lhs) ||
      !loom_vector_fragment_fact_is_accumulator_like(rhs)) {
    return false;
  }
  lhs.role_flags = LOOM_VECTOR_FRAGMENT_ROLE_FLAG_INIT |
                   LOOM_VECTOR_FRAGMENT_ROLE_FLAG_RESULT;
  rhs.role_flags = LOOM_VECTOR_FRAGMENT_ROLE_FLAG_INIT |
                   LOOM_VECTOR_FRAGMENT_ROLE_FLAG_RESULT;
  return loom_vector_fragment_facts_match_contract_except_native_storage(lhs,
                                                                         rhs);
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

static iree_status_t loom_vector_meet_extension(
    const loom_value_fact_domain_t* domain, const loom_module_t* module,
    loom_type_t type, loom_value_fact_table_t* target,
    const loom_value_fact_table_t* lhs_table, loom_value_facts_t lhs,
    const loom_value_fact_table_t* rhs_table, loom_value_facts_t rhs,
    loom_value_facts_t* inout_facts) {
  (void)domain;
  (void)module;
  (void)type;
  return loom_vector_join_fragment_extension(target, lhs_table, lhs, rhs_table,
                                             rhs, inout_facts);
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
  return loom_vector_join_fragment_extension(target, previous_table, previous,
                                             next_table, next, inout_facts);
}

const loom_value_fact_domain_t loom_vector_fact_domain = {
    .meet_extension = loom_vector_meet_extension,
    .widen_extension = loom_vector_widen_extension,
};

static iree_status_t loom_vector_try_preserve_accumulator_fragment_facts(
    loom_fact_context_t* context, const loom_value_facts_t* operand_facts,
    uint16_t operand_count, loom_value_facts_t* result_facts,
    bool* out_handled) {
  *out_handled = false;
  loom_vector_fragment_fact_t fragment;
  loom_vector_fragment_fact_initialize(&fragment);
  for (uint16_t i = 0; i < operand_count; ++i) {
    loom_vector_fragment_fact_t candidate;
    if (!loom_vector_fragment_fact_query_value_facts(context, operand_facts[i],
                                                     &candidate)) {
      continue;
    }
    *out_handled = true;
    if (!loom_vector_fragment_fact_is_accumulator_like(candidate)) {
      result_facts[0] = loom_value_facts_unknown();
      return iree_ok_status();
    }
    if (loom_vector_fragment_fact_is_unknown(fragment)) {
      fragment = candidate;
      continue;
    }
    if (!loom_vector_fragment_facts_match_except_role(fragment, candidate)) {
      result_facts[0] = loom_value_facts_unknown();
      return iree_ok_status();
    }
  }
  if (!*out_handled) {
    return iree_ok_status();
  }
  fragment.role_flags = LOOM_VECTOR_FRAGMENT_ROLE_FLAG_INIT |
                        LOOM_VECTOR_FRAGMENT_ROLE_FLAG_RESULT;
  return loom_vector_fragment_fact_make_value_facts(context, fragment,
                                                    &result_facts[0]);
}

static void loom_vector_fragment_copy_present_auxiliary(
    loom_vector_encoding_auxiliary_view_t source,
    loom_vector_encoding_auxiliary_view_t* out_target) {
  memset(out_target, 0, sizeof(*out_target));
  out_target->present_keys = source.present_keys;
  for (uint8_t i = 0; i < LOOM_VECTOR_ENCODING_AUXILIARY_KEY_COUNT_; ++i) {
    loom_vector_encoding_auxiliary_key_t key =
        (loom_vector_encoding_auxiliary_key_t)i;
    loom_vector_encoding_auxiliary_key_flags_t key_flag =
        loom_vector_encoding_auxiliary_key_flag(key);
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
  fact.shape_rank = 2;
  fact.shape_value_ids[0] = loom_vector_fragment_rows(op);
  fact.shape_value_ids[1] = loom_vector_fragment_columns(op);

  loom_vector_fragment_parameter_view_t parameters;
  iree_string_view_t unknown_key = iree_string_view_empty();
  if (!loom_vector_fragment_parameter_view_resolve(
          module, loom_vector_fragment_params(op),
          loom_vector_fragment_param_names(op), &parameters, &unknown_key)) {
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
      uint16_t schema_operand_index =
          (uint16_t)(3 + parameters.schema_parameter_ordinal);
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
  fact.shape_rank = 2;
  fact.shape_value_ids[0] = loom_vector_fragment_load_rows(op);
  fact.shape_value_ids[1] = loom_vector_fragment_load_columns(op);
  fact.flags |= LOOM_VECTOR_FRAGMENT_FACT_FLAG_HAS_NATIVE_STORAGE;

  const loom_value_id_t view_value_id = loom_vector_fragment_load_view(op);
  loom_value_fact_storage_schema_t storage_schema = {0};
  if (loom_encoding_query_type_storage_schema(
          context, module, loom_module_value_type(module, view_value_id),
          &storage_schema) &&
      !loom_value_fact_encoded_operand_schema_is_unknown(
          storage_schema.encoded_operand)) {
    fact.flags |= LOOM_VECTOR_FRAGMENT_FACT_FLAG_HAS_SCHEMA;
    fact.encoded_operand = storage_schema.encoded_operand;
    if (storage_schema.static_spec_encoding_id != 0) {
      fact.flags |= LOOM_VECTOR_FRAGMENT_FACT_FLAG_HAS_STATIC_SCHEMA;
      fact.static_schema_encoding_id = storage_schema.static_spec_encoding_id;
    }
  }
  return loom_vector_fragment_fact_make_value_facts(context, fact,
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

static bool loom_vector_static_shapes_match(loom_type_t lhs_type,
                                            loom_type_t rhs_type) {
  uint8_t rank = loom_type_rank(lhs_type);
  if (loom_type_rank(rhs_type) != rank) return false;
  for (uint8_t axis = 0; axis < rank; ++axis) {
    if (loom_type_dim_is_dynamic_at(lhs_type, axis) ||
        loom_type_dim_is_dynamic_at(rhs_type, axis) ||
        loom_type_dim_static_size_at(lhs_type, axis) !=
            loom_type_dim_static_size_at(rhs_type, axis)) {
      return false;
    }
  }
  return true;
}

static bool loom_vector_static_leading_shapes_match(loom_type_t lhs_type,
                                                    loom_type_t rhs_type) {
  uint8_t rank = loom_type_rank(lhs_type);
  if (loom_type_rank(rhs_type) != rank || rank == 0) return false;
  for (uint8_t axis = 0; axis + 1 < rank; ++axis) {
    if (loom_type_dim_is_dynamic_at(lhs_type, axis) ||
        loom_type_dim_is_dynamic_at(rhs_type, axis) ||
        loom_type_dim_static_size_at(lhs_type, axis) !=
            loom_type_dim_static_size_at(rhs_type, axis)) {
      return false;
    }
  }
  return true;
}

static bool loom_vector_static_last_axis_extent(loom_type_t type,
                                                iree_host_size_t* out_extent) {
  uint8_t rank = loom_type_rank(type);
  if (rank == 0 || loom_type_dim_is_dynamic_at(type, (uint8_t)(rank - 1))) {
    return false;
  }
  int64_t extent = loom_type_dim_static_size_at(type, (uint8_t)(rank - 1));
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

static bool loom_vector_float_exact_bits(loom_value_facts_t facts,
                                         loom_scalar_type_t scalar_type,
                                         uint64_t* out_bits) {
  double value = 0.0;
  if (!loom_vector_facts_query_exact_f64(facts, &value)) {
    return false;
  }
  if (isnan(value)) {
    return false;
  }
  switch (scalar_type) {
    case LOOM_SCALAR_TYPE_F8E4M3:
      *out_bits = (uint64_t)iree_math_f32_to_f8e4m3fn((float)value);
      return true;
    case LOOM_SCALAR_TYPE_F8E5M2:
      *out_bits = (uint64_t)iree_math_f32_to_f8e5m2((float)value);
      return true;
    case LOOM_SCALAR_TYPE_F16:
      *out_bits = (uint64_t)iree_math_f32_to_f16((float)value);
      return true;
    case LOOM_SCALAR_TYPE_BF16:
      *out_bits = (uint64_t)iree_math_f32_to_bf16((float)value);
      return true;
    case LOOM_SCALAR_TYPE_F32: {
      float narrowed = (float)value;
      uint32_t bits = 0;
      memcpy(&bits, &narrowed, sizeof(bits));
      *out_bits = bits;
      return true;
    }
    case LOOM_SCALAR_TYPE_F64: {
      uint64_t bits = 0;
      memcpy(&bits, &value, sizeof(bits));
      *out_bits = bits;
      return true;
    }
    default:
      return false;
  }
}

static bool loom_vector_fact_exact_bits(loom_value_facts_t facts,
                                        loom_scalar_type_t scalar_type,
                                        uint64_t* out_bits) {
  int32_t bitwidth = loom_scalar_type_bitwidth(scalar_type);
  if (bitwidth <= 0 || bitwidth > 64) {
    return false;
  }
  if (loom_scalar_type_is_float(scalar_type)) {
    return loom_vector_float_exact_bits(facts, scalar_type, out_bits);
  }
  int64_t value = 0;
  if (!loom_vector_facts_query_exact_i64(facts, &value)) {
    return false;
  }
  *out_bits = loom_mask_to_bitwidth_u64((uint64_t)value, bitwidth);
  return true;
}

static bool loom_vector_float_facts_from_bits(uint64_t bits,
                                              loom_scalar_type_t scalar_type,
                                              loom_value_facts_t* out_facts) {
  switch (scalar_type) {
    case LOOM_SCALAR_TYPE_F8E4M3:
      *out_facts = loom_value_facts_exact_f64(
          (double)iree_math_f8e4m3fn_to_f32((uint8_t)bits));
      return true;
    case LOOM_SCALAR_TYPE_F8E5M2:
      *out_facts = loom_value_facts_exact_f64(
          (double)iree_math_f8e5m2_to_f32((uint8_t)bits));
      return true;
    case LOOM_SCALAR_TYPE_F16:
      *out_facts = loom_value_facts_exact_f64(
          (double)iree_math_f16_to_f32((uint16_t)bits));
      return true;
    case LOOM_SCALAR_TYPE_BF16:
      *out_facts = loom_value_facts_exact_f64(
          (double)iree_math_bf16_to_f32((uint16_t)bits));
      return true;
    case LOOM_SCALAR_TYPE_F32: {
      uint32_t narrowed_bits = (uint32_t)bits;
      float value = 0.0f;
      memcpy(&value, &narrowed_bits, sizeof(value));
      *out_facts = loom_value_facts_exact_f64((double)value);
      return true;
    }
    case LOOM_SCALAR_TYPE_F64: {
      double value = 0.0;
      memcpy(&value, &bits, sizeof(value));
      *out_facts = loom_value_facts_exact_f64(value);
      return true;
    }
    default:
      return false;
  }
}

static bool loom_vector_facts_from_exact_bits(uint64_t bits,
                                              loom_scalar_type_t scalar_type,
                                              loom_value_facts_t* out_facts) {
  int32_t bitwidth = loom_scalar_type_bitwidth(scalar_type);
  if (bitwidth <= 0 || bitwidth > 64) {
    return false;
  }
  uint64_t masked = loom_mask_to_bitwidth_u64(bits, bitwidth);
  if (loom_scalar_type_is_float(scalar_type)) {
    return loom_vector_float_facts_from_bits(masked, scalar_type, out_facts);
  }
  *out_facts = loom_vector_make_integer_raw_bit_facts(masked, scalar_type);
  return true;
}

static bool loom_vector_bitcast_element_facts(
    loom_value_facts_t source_facts, loom_scalar_type_t source_element_type,
    loom_scalar_type_t result_element_type, loom_value_facts_t* out_facts) {
  if (loom_scalar_type_bitwidth(source_element_type) !=
      loom_scalar_type_bitwidth(result_element_type)) {
    return false;
  }
  uint64_t bits = 0;
  return loom_vector_fact_exact_bits(source_facts, source_element_type,
                                     &bits) &&
         loom_vector_facts_from_exact_bits(bits, result_element_type,
                                           out_facts);
}

static bool loom_vector_facts_query_lane_or_iota(
    const loom_fact_context_t* context, loom_value_facts_t facts,
    iree_host_size_t lane, loom_value_facts_t* out_element) {
  return loom_vector_facts_query_lane(context, facts, lane, out_element);
}

static bool loom_vector_transform_query_f64_lane(
    const loom_fact_context_t* context, loom_value_facts_t facts,
    iree_host_size_t lane, double* out_value) {
  loom_value_facts_t lane_facts = {0};
  return loom_vector_facts_query_lane_or_iota(context, facts, lane,
                                              &lane_facts) &&
         loom_vector_facts_query_exact_f64(lane_facts, out_value);
}

static bool loom_vector_transform_query_i64_lane(
    const loom_fact_context_t* context, loom_value_facts_t facts,
    iree_host_size_t lane, int64_t* out_value) {
  loom_value_facts_t lane_facts = {0};
  return loom_vector_facts_query_lane_or_iota(context, facts, lane,
                                              &lane_facts) &&
         loom_vector_facts_query_exact_i64(lane_facts, out_value);
}

static bool loom_vector_transform_dynamic_value_facts(
    const loom_fact_context_t* context, loom_value_id_t value_id,
    loom_value_facts_t* out_facts) {
  if (value_id == LOOM_VALUE_ID_INVALID) {
    return false;
  }
  *out_facts = loom_value_fact_table_lookup(context->table, value_id);
  return true;
}

static bool loom_vector_transform_validate_hadamard_descriptor(
    const loom_encoding_numeric_transform_descriptor_t* descriptor) {
  if (descriptor->family == LOOM_ENCODING_NUMERIC_TRANSFORM_FAMILY_HADAMARD) {
    return !loom_encoding_numeric_transform_has_signs(descriptor) &&
           !loom_encoding_numeric_transform_has_permutation(descriptor) &&
           !loom_encoding_numeric_transform_has_matrix(descriptor) &&
           !loom_encoding_numeric_transform_has_seed(descriptor);
  }
  if (descriptor->family ==
      LOOM_ENCODING_NUMERIC_TRANSFORM_FAMILY_HADAMARD_SIGN) {
    bool has_signs = loom_encoding_numeric_transform_has_signs(descriptor);
    bool has_seed = loom_encoding_numeric_transform_has_seed(descriptor);
    return has_signs != has_seed &&
           !loom_encoding_numeric_transform_has_permutation(descriptor) &&
           !loom_encoding_numeric_transform_has_matrix(descriptor);
  }
  if (descriptor->family ==
      LOOM_ENCODING_NUMERIC_TRANSFORM_FAMILY_SIGN_PERMUTE_HADAMARD) {
    return loom_encoding_numeric_transform_has_signs(descriptor) &&
           loom_encoding_numeric_transform_has_permutation(descriptor) &&
           !loom_encoding_numeric_transform_has_matrix(descriptor) &&
           !loom_encoding_numeric_transform_has_seed(descriptor);
  }
  return false;
}

static bool loom_vector_transform_query_sign(
    const loom_fact_context_t* context,
    const loom_encoding_numeric_transform_descriptor_t* descriptor,
    iree_host_size_t lane, bool* out_negate) {
  *out_negate = false;
  if (!loom_encoding_numeric_transform_has_signs(descriptor)) {
    return !loom_encoding_numeric_transform_has_seed(descriptor);
  }
  loom_value_facts_t signs_facts = {0};
  int64_t sign_bit = 0;
  if (!loom_vector_transform_dynamic_value_facts(context, descriptor->signs,
                                                 &signs_facts) ||
      !loom_vector_transform_query_i64_lane(context, signs_facts, lane,
                                            &sign_bit)) {
    return false;
  }
  *out_negate = sign_bit != 0;
  return true;
}

static bool loom_vector_transform_query_permutation_lane(
    const loom_fact_context_t* context,
    const loom_encoding_numeric_transform_descriptor_t* descriptor,
    iree_host_size_t transform_lane, int64_t default_source_last_index,
    iree_host_size_t input_extent, int64_t* out_source_last_index) {
  *out_source_last_index = 0;
  if (!loom_encoding_numeric_transform_has_permutation(descriptor)) {
    *out_source_last_index = default_source_last_index;
    return true;
  }
  loom_value_facts_t permutation_facts = {0};
  if (!loom_vector_transform_dynamic_value_facts(
          context, descriptor->permutation, &permutation_facts) ||
      !loom_vector_transform_query_i64_lane(
          context, permutation_facts, transform_lane, out_source_last_index)) {
    return false;
  }
  return *out_source_last_index >= 0 &&
         (uint64_t)*out_source_last_index < (uint64_t)input_extent;
}

static bool loom_vector_transform_query_seed_sign(
    const loom_fact_context_t* context,
    const loom_encoding_numeric_transform_descriptor_t* descriptor,
    int64_t input_index, bool* out_negate) {
  loom_value_facts_t seed_facts = {0};
  int64_t seed = 0;
  if (input_index < 0 ||
      !loom_vector_transform_dynamic_value_facts(context, descriptor->seed,
                                                 &seed_facts) ||
      !loom_vector_facts_query_exact_i64(seed_facts, &seed)) {
    return false;
  }
  return loom_encoding_numeric_transform_seed_sign_bit(seed, input_index,
                                                       out_negate);
}

static bool loom_vector_transform_hadamard_lane_value(
    const loom_fact_context_t* context,
    const loom_encoding_numeric_transform_descriptor_t* descriptor,
    loom_type_t source_type, loom_value_facts_t source_facts,
    const int64_t* result_indices, iree_host_size_t input_extent,
    double* out_value) {
  uint8_t rank = loom_type_rank(source_type);
  uint8_t last_axis = (uint8_t)(rank - 1);
  int64_t source_indices[LOOM_TYPE_MAX_RANK] = {0};
  memcpy(source_indices, result_indices, rank * sizeof(source_indices[0]));

  double accumulator = 0.0;
  for (iree_host_size_t input_index = 0; input_index < input_extent;
       ++input_index) {
    source_indices[last_axis] = (int64_t)input_index;
    iree_host_size_t transform_lane = 0;
    if (!loom_vector_static_ordinal_from_indices(source_type, source_indices,
                                                 &transform_lane)) {
      return false;
    }

    int64_t source_last_index = 0;
    if (!loom_vector_transform_query_permutation_lane(
            context, descriptor, transform_lane, (int64_t)input_index,
            input_extent, &source_last_index)) {
      return false;
    }
    source_indices[last_axis] = source_last_index;
    iree_host_size_t source_lane = 0;
    if (!loom_vector_static_ordinal_from_indices(source_type, source_indices,
                                                 &source_lane)) {
      return false;
    }

    double term = 0.0;
    if (!loom_vector_transform_query_f64_lane(context, source_facts,
                                              source_lane, &term)) {
      return false;
    }

    bool sign_negates = false;
    if (loom_encoding_numeric_transform_has_seed(descriptor)) {
      if (!loom_vector_transform_query_seed_sign(
              context, descriptor, (int64_t)input_index, &sign_negates)) {
        return false;
      }
    } else if (!loom_vector_transform_query_sign(context, descriptor,
                                                 source_lane, &sign_negates)) {
      return false;
    }
    if (sign_negates) term = -term;

    int64_t output_index = result_indices[last_axis];
    if (loom_count_ones_u64_width(
            (uint64_t)(output_index & (int64_t)input_index), 64) &
        1) {
      term = -term;
    }
    accumulator += term;
    source_indices[last_axis] = (int64_t)input_index;
  }

  if (descriptor->normalization ==
      LOOM_ENCODING_NUMERIC_TRANSFORM_NORMALIZATION_ORTHONORMAL) {
    accumulator *= 1.0 / sqrt((double)input_extent);
  }
  *out_value = accumulator;
  return true;
}

static iree_status_t loom_vector_transform_hadamard_facts(
    loom_fact_context_t* context,
    const loom_encoding_numeric_transform_descriptor_t* descriptor,
    loom_type_t source_type, loom_type_t result_type,
    loom_value_facts_t source_facts, loom_value_facts_t* result_facts) {
  iree_host_size_t result_lane_count = 0;
  iree_host_size_t input_extent = 0;
  if (!loom_vector_transform_validate_hadamard_descriptor(descriptor) ||
      !loom_vector_static_shapes_match(source_type, result_type) ||
      !loom_vector_type_static_lane_count(result_type, &result_lane_count) ||
      !loom_vector_static_last_axis_extent(source_type, &input_extent) ||
      result_lane_count > LOOM_VALUE_FACT_SMALL_STATIC_LANE_LIMIT ||
      input_extent > LOOM_VECTOR_FACT_STATIC_LOOP_LIMIT ||
      !loom_is_power_of_two_i64((int64_t)input_extent)) {
    return loom_vector_make_unknown_facts(result_facts);
  }

  loom_value_facts_t lanes[LOOM_VALUE_FACT_SMALL_STATIC_LANE_LIMIT] = {{0}};
  int64_t result_indices[LOOM_TYPE_MAX_RANK] = {0};
  for (iree_host_size_t result_lane = 0; result_lane < result_lane_count;
       ++result_lane) {
    loom_vector_static_indices_from_ordinal(result_type, result_lane,
                                            result_indices);
    double value = 0.0;
    lanes[result_lane] = loom_value_facts_unknown();
    if (loom_vector_transform_hadamard_lane_value(
            context, descriptor, source_type, source_facts, result_indices,
            input_extent, &value)) {
      lanes[result_lane] = loom_value_facts_exact_f64(value);
    }
  }
  return loom_vector_make_small_static_lane_facts(
      context, lanes, result_lane_count, result_facts);
}

static bool loom_vector_transform_jl_descriptor_is_valid(
    const loom_encoding_numeric_transform_descriptor_t* descriptor) {
  return descriptor->family ==
             LOOM_ENCODING_NUMERIC_TRANSFORM_FAMILY_JL_DENSE &&
         descriptor->normalization ==
             LOOM_ENCODING_NUMERIC_TRANSFORM_NORMALIZATION_NONE &&
         loom_encoding_numeric_transform_has_matrix(descriptor) &&
         !loom_encoding_numeric_transform_has_signs(descriptor) &&
         !loom_encoding_numeric_transform_has_permutation(descriptor) &&
         !loom_encoding_numeric_transform_has_seed(descriptor);
}

static bool loom_vector_transform_jl_lane_value(
    const loom_fact_context_t* context, loom_type_t source_type,
    loom_type_t matrix_type, loom_value_facts_t source_facts,
    loom_value_facts_t matrix_facts, const int64_t* result_indices,
    iree_host_size_t input_extent, double* out_value) {
  uint8_t rank = loom_type_rank(source_type);
  uint8_t last_axis = (uint8_t)(rank - 1);
  int64_t source_indices[LOOM_TYPE_MAX_RANK] = {0};
  int64_t matrix_indices[2] = {
      result_indices[last_axis],
      0,
  };
  memcpy(source_indices, result_indices, rank * sizeof(source_indices[0]));

  double accumulator = 0.0;
  for (iree_host_size_t input_index = 0; input_index < input_extent;
       ++input_index) {
    source_indices[last_axis] = (int64_t)input_index;
    matrix_indices[1] = (int64_t)input_index;

    iree_host_size_t source_lane = 0;
    iree_host_size_t matrix_lane = 0;
    if (!loom_vector_static_ordinal_from_indices(source_type, source_indices,
                                                 &source_lane) ||
        !loom_vector_static_ordinal_from_indices(matrix_type, matrix_indices,
                                                 &matrix_lane)) {
      return false;
    }

    double source_value = 0.0;
    double matrix_value = 0.0;
    if (!loom_vector_transform_query_f64_lane(context, source_facts,
                                              source_lane, &source_value) ||
        !loom_vector_transform_query_f64_lane(context, matrix_facts,
                                              matrix_lane, &matrix_value)) {
      return false;
    }
    accumulator += matrix_value * source_value;
  }
  *out_value = accumulator;
  return true;
}

static iree_status_t loom_vector_transform_jl_dense_facts(
    loom_fact_context_t* context,
    const loom_encoding_numeric_transform_descriptor_t* descriptor,
    const loom_module_t* module, loom_type_t source_type,
    loom_type_t result_type, loom_value_facts_t source_facts,
    loom_value_facts_t* result_facts) {
  iree_host_size_t result_lane_count = 0;
  iree_host_size_t input_extent = 0;
  if (!loom_vector_transform_jl_descriptor_is_valid(descriptor) ||
      !loom_vector_static_leading_shapes_match(source_type, result_type) ||
      !loom_vector_type_static_lane_count(result_type, &result_lane_count) ||
      !loom_vector_static_last_axis_extent(source_type, &input_extent) ||
      result_lane_count > LOOM_VALUE_FACT_SMALL_STATIC_LANE_LIMIT ||
      input_extent > LOOM_VECTOR_FACT_STATIC_LOOP_LIMIT) {
    return loom_vector_make_unknown_facts(result_facts);
  }

  loom_type_t matrix_type = loom_module_value_type(module, descriptor->matrix);
  uint8_t result_last_axis = (uint8_t)(loom_type_rank(result_type) - 1);
  if (!loom_type_is_vector(matrix_type) || loom_type_rank(matrix_type) != 2 ||
      loom_type_dim_is_dynamic_at(matrix_type, 0) ||
      loom_type_dim_is_dynamic_at(matrix_type, 1) ||
      loom_type_dim_is_dynamic_at(result_type, result_last_axis) ||
      loom_type_dim_static_size_at(matrix_type, 0) !=
          loom_type_dim_static_size_at(result_type, result_last_axis) ||
      loom_type_dim_static_size_at(matrix_type, 1) != (int64_t)input_extent) {
    return loom_vector_make_unknown_facts(result_facts);
  }

  loom_value_facts_t matrix_facts = {0};
  if (!loom_vector_transform_dynamic_value_facts(context, descriptor->matrix,
                                                 &matrix_facts)) {
    return loom_vector_make_unknown_facts(result_facts);
  }

  loom_value_facts_t lanes[LOOM_VALUE_FACT_SMALL_STATIC_LANE_LIMIT] = {{0}};
  int64_t result_indices[LOOM_TYPE_MAX_RANK] = {0};
  for (iree_host_size_t result_lane = 0; result_lane < result_lane_count;
       ++result_lane) {
    loom_vector_static_indices_from_ordinal(result_type, result_lane,
                                            result_indices);
    double value = 0.0;
    lanes[result_lane] = loom_value_facts_unknown();
    if (loom_vector_transform_jl_lane_value(
            context, source_type, matrix_type, source_facts, matrix_facts,
            result_indices, input_extent, &value)) {
      lanes[result_lane] = loom_value_facts_exact_f64(value);
    }
  }
  return loom_vector_make_small_static_lane_facts(
      context, lanes, result_lane_count, result_facts);
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

static double loom_vector_dot4f8_decode_field(
    loom_vector_dot4f8_format_t format, uint8_t field) {
  switch (format) {
    case LOOM_VECTOR_DOT4F8_FORMAT_FP8:
      return (double)iree_math_f8e4m3fn_to_f32(field);
    case LOOM_VECTOR_DOT4F8_FORMAT_BF8:
      return (double)iree_math_f8e5m2_to_f32(field);
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
                                     uint32_t rhs_raw, double* accumulator) {
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
    double lhs = loom_vector_dot4f8_decode_field(lhs_format, lhs_field);
    double rhs = loom_vector_dot4f8_decode_field(rhs_format, rhs_field);
    *accumulator = fma(lhs, rhs, *accumulator);
  }
  return true;
}

static double loom_vector_add_f64(double lhs, double rhs) { return lhs + rhs; }

static double loom_vector_sub_f64(double lhs, double rhs) { return lhs - rhs; }

static double loom_vector_mul_f64(double lhs, double rhs) { return lhs * rhs; }

static double loom_vector_div_f64(double lhs, double rhs) { return lhs / rhs; }

static double loom_vector_neg_f64(double input) { return -input; }

static double loom_vector_rsqrt_f64(double input) { return 1.0 / sqrt(input); }

static double loom_vector_roundeven_f64(double input) {
  return nearbyint(input);
}

static double loom_vector_logistic_f64(double input) {
  return 1.0 / (1.0 + exp(-input));
}

static double loom_vector_silu_f64(double input) {
  return input * loom_vector_logistic_f64(input);
}

static double loom_vector_softplus_f64(double input) {
  return log1p(exp(-fabs(input))) + fmax(input, 0.0);
}

static double loom_vector_gelu_erf_f64(double input) {
  const double inverse_sqrt2 = 0.70710678118654752440;
  return 0.5 * input * (1.0 + erf(input * inverse_sqrt2));
}

static double loom_vector_gelu_tanh_f64(double input) {
  const double sqrt_2_over_pi = 0.79788456080286535588;
  return 0.5 * input *
         (1.0 +
          tanh(sqrt_2_over_pi * (input + 0.044715 * input * input * input)));
}

static double loom_vector_gelu_logistic_f64(double input, double scale) {
  return input * loom_vector_logistic_f64(scale * input);
}

static double loom_vector_minimum_f64(double lhs, double rhs) {
  return (isnan(lhs) || isnan(rhs)) ? NAN : fmin(lhs, rhs);
}

static double loom_vector_maximum_f64(double lhs, double rhs) {
  return (isnan(lhs) || isnan(rhs)) ? NAN : fmax(lhs, rhs);
}

static double loom_vector_minnum_f64(double lhs, double rhs) {
  return fmin(lhs, rhs);
}

static double loom_vector_maxnum_f64(double lhs, double rhs) {
  return fmax(lhs, rhs);
}

static double loom_vector_clampf_ordered_f64(double value, double lower,
                                             double upper) {
  double result = value;
  if (result < lower) {
    result = lower;
  }
  if (result > upper) {
    result = upper;
  }
  return result;
}

static double loom_vector_clampf_number_f64(double value, double lower,
                                            double upper) {
  return fmin(fmax(value, lower), upper);
}

static double loom_vector_clampf_ieee_f64(double value, double lower,
                                          double upper) {
  return loom_vector_minimum_f64(loom_vector_maximum_f64(value, lower), upper);
}

static void loom_vector_clampf_transfer(const loom_value_facts_t* value,
                                        const loom_value_facts_t* lower,
                                        const loom_value_facts_t* upper,
                                        double (*fn)(double, double, double),
                                        loom_value_facts_t* out) {
  double value_f64 = 0.0;
  double lower_f64 = 0.0;
  double upper_f64 = 0.0;
  if (!loom_vector_facts_query_exact_f64(*value, &value_f64) ||
      !loom_vector_facts_query_exact_f64(*lower, &lower_f64) ||
      !loom_vector_facts_query_exact_f64(*upper, &upper_f64)) {
    *out = loom_value_facts_unknown();
    return;
  }
  *out = loom_value_facts_exact_f64(fn(value_f64, lower_f64, upper_f64));
}

static void loom_vector_clampf_ordered_transfer(const loom_value_facts_t* value,
                                                const loom_value_facts_t* lower,
                                                const loom_value_facts_t* upper,
                                                loom_value_facts_t* out) {
  loom_vector_clampf_transfer(value, lower, upper,
                              loom_vector_clampf_ordered_f64, out);
}

static void loom_vector_clampf_number_transfer(const loom_value_facts_t* value,
                                               const loom_value_facts_t* lower,
                                               const loom_value_facts_t* upper,
                                               loom_value_facts_t* out) {
  loom_vector_clampf_transfer(value, lower, upper,
                              loom_vector_clampf_number_f64, out);
}

static void loom_vector_clampf_ieee_transfer(const loom_value_facts_t* value,
                                             const loom_value_facts_t* lower,
                                             const loom_value_facts_t* upper,
                                             loom_value_facts_t* out) {
  loom_vector_clampf_transfer(value, lower, upper, loom_vector_clampf_ieee_f64,
                              out);
}

static void loom_vector_isnanf_transfer(const loom_value_facts_t* input,
                                        loom_value_facts_t* out) {
  if (loom_value_facts_is_not_nan(*input)) {
    *out = loom_value_facts_exact_i64(0);
    return;
  }
  double value = 0.0;
  if (!loom_vector_facts_query_exact_f64(*input, &value)) {
    *out = loom_value_facts_make(0, 1, 1);
    return;
  }
  *out = loom_value_facts_exact_i64(isnan(value) ? 1 : 0);
}

static void loom_vector_isinff_transfer(const loom_value_facts_t* input,
                                        loom_value_facts_t* out) {
  if (loom_value_facts_is_finite(*input)) {
    *out = loom_value_facts_exact_i64(0);
    return;
  }
  if (loom_value_facts_is_not_inf(*input)) {
    *out = loom_value_facts_exact_i64(0);
    return;
  }
  double value = 0.0;
  if (!loom_vector_facts_query_exact_f64(*input, &value)) {
    *out = loom_value_facts_make(0, 1, 1);
    return;
  }
  *out = loom_value_facts_exact_i64(isinf(value) ? 1 : 0);
}

static void loom_vector_isfinitef_transfer(const loom_value_facts_t* input,
                                           loom_value_facts_t* out) {
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
  if (!loom_vector_facts_query_exact_f64(*input, &value)) {
    *out = loom_value_facts_make(0, 1, 1);
    return;
  }
  *out = loom_value_facts_exact_i64(isfinite(value) ? 1 : 0);
}

static void loom_vector_signf_transfer(const loom_value_facts_t* input,
                                       loom_value_facts_t* out) {
  double value = 0.0;
  if (!loom_vector_facts_query_exact_f64(*input, &value)) {
    *out = loom_value_facts_unknown();
    return;
  }
  *out = loom_value_facts_exact_f64((value > 0.0)   ? 1.0
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

static void loom_vector_sitofp_transfer(const loom_value_facts_t* input,
                                        loom_value_facts_t* out) {
  int64_t value = 0;
  if (!loom_vector_facts_query_exact_i64(*input, &value)) {
    *out = loom_value_facts_unknown();
    return;
  }
  *out = loom_value_facts_exact_f64((double)value);
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
  return loom_value_facts_make_uniform_element(context, element,
                                               &result_facts[0]);
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
  if (loom_value_facts_is_exact(operand_facts[0]) &&
      loom_value_facts_is_exact(operand_facts[1]) &&
      loom_value_facts_is_exact(operand_facts[2]) &&
      !loom_value_facts_is_float(operand_facts[0]) &&
      !loom_value_facts_is_float(operand_facts[1]) &&
      !loom_value_facts_is_float(operand_facts[2])) {
    loom_type_t result_type =
        loom_module_value_type(module, loom_vector_mask_range_result(op));
    uint64_t lane_count = 0;
    if (loom_type_static_element_count(result_type, &lane_count)) {
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
  iree_host_size_t result_lane_count = 0;
  if (!loom_vector_type_static_lane_count(result_type, &result_lane_count) ||
      result_lane_count > LOOM_VALUE_FACT_SMALL_STATIC_LANE_LIMIT ||
      loom_type_rank(source_type) != loom_type_rank(result_type)) {
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

  loom_value_facts_t lanes[LOOM_VALUE_FACT_SMALL_STATIC_LANE_LIMIT] = {{0}};
  int64_t source_indices[LOOM_TYPE_MAX_RANK] = {0};
  for (iree_host_size_t lane = 0; lane < result_lane_count; ++lane) {
    loom_vector_static_indices_from_ordinal(result_type, lane, source_indices);
    for (uint8_t axis = 0; axis < rank; ++axis) {
      if (!loom_checked_add_i64(source_indices[axis],
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
      if (!loom_checked_add_i64(axis_base, input_axis_size, &next_axis_base)) {
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
      if (!loom_checked_mul_i64(source_indices[axis], 2,
                                &source_indices[axis]) ||
          !loom_checked_add_i64(source_indices[axis], result_index,
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
  loom_encoding_numeric_transform_read_t read =
      loom_encoding_numeric_transform_read_descriptor(
          module, loom_vector_transform_transform(op));
  if (read.code != LOOM_ENCODING_NUMERIC_TRANSFORM_READ_OK) {
    return loom_vector_make_unknown_facts(result_facts);
  }

  loom_type_t source_type =
      loom_module_value_type(module, loom_vector_transform_source(op));
  loom_type_t result_type =
      loom_module_value_type(module, loom_vector_transform_result(op));
  switch (read.descriptor.family) {
    case LOOM_ENCODING_NUMERIC_TRANSFORM_FAMILY_HADAMARD:
    case LOOM_ENCODING_NUMERIC_TRANSFORM_FAMILY_HADAMARD_SIGN:
    case LOOM_ENCODING_NUMERIC_TRANSFORM_FAMILY_SIGN_PERMUTE_HADAMARD:
      return loom_vector_transform_hadamard_facts(
          context, &read.descriptor, source_type, result_type, operand_facts[0],
          &result_facts[0]);
    case LOOM_ENCODING_NUMERIC_TRANSFORM_FAMILY_JL_DENSE:
      return loom_vector_transform_jl_dense_facts(
          context, &read.descriptor, module, source_type, result_type,
          operand_facts[0], &result_facts[0]);
    case LOOM_ENCODING_NUMERIC_TRANSFORM_FAMILY_UNKNOWN:
      return loom_vector_make_unknown_facts(result_facts);
  }
  return loom_vector_make_unknown_facts(result_facts);
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

static iree_status_t loom_vector_float_unary_data_summary_facts(
    loom_fact_context_t* context, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts,
    loom_vector_float_unary_data_transfer_fn_t fn, const void* user_data) {
  bool fragment_handled = false;
  IREE_RETURN_IF_ERROR(loom_vector_try_preserve_accumulator_fragment_facts(
      context, operand_facts, 1, result_facts, &fragment_handled));
  if (fragment_handled) {
    return iree_ok_status();
  }

  loom_value_facts_t input = {0};
  if (loom_vector_facts_query_uniform_element(context, operand_facts[0],
                                              &input)) {
    double input_value = 0.0;
    loom_value_facts_t element = loom_value_facts_unknown();
    if (loom_vector_facts_query_exact_f64(input, &input_value)) {
      element = loom_value_facts_exact_f64(fn(input_value, user_data));
    }
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
    double input_value = 0.0;
    lanes[i] = loom_value_facts_unknown();
    if (loom_vector_facts_query_exact_f64(input_lanes.lanes[i], &input_value)) {
      lanes[i] = loom_value_facts_exact_f64(fn(input_value, user_data));
    }
  }
  loom_value_fact_small_static_lanes_t lane_slice = {
      .lanes = lanes,
      .count = input_lanes.count,
  };
  return loom_value_facts_make_small_static_lanes(context, lane_slice,
                                                  &result_facts[0]);
}

typedef struct loom_vector_float_unary_adapter_t {
  loom_vector_float_unary_transfer_fn_t fn;
} loom_vector_float_unary_adapter_t;

static double loom_vector_float_unary_adapter_transfer(double input,
                                                       const void* user_data) {
  const loom_vector_float_unary_adapter_t* adapter = user_data;
  return adapter->fn(input);
}

static iree_status_t loom_vector_float_unary_summary_facts(
    loom_fact_context_t* context, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts,
    loom_vector_float_unary_transfer_fn_t fn) {
  const loom_vector_float_unary_adapter_t adapter = {
      .fn = fn,
  };
  return loom_vector_float_unary_data_summary_facts(
      context, operand_facts, result_facts,
      loom_vector_float_unary_adapter_transfer, &adapter);
}

static iree_status_t loom_vector_float_binary_summary_facts(
    loom_fact_context_t* context, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts,
    loom_vector_float_binary_transfer_fn_t fn) {
  bool fragment_handled = false;
  IREE_RETURN_IF_ERROR(loom_vector_try_preserve_accumulator_fragment_facts(
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
    double lhs_value = 0.0;
    double rhs_value = 0.0;
    loom_value_facts_t element = loom_value_facts_unknown();
    if (loom_vector_facts_query_exact_f64(lhs, &lhs_value) &&
        loom_vector_facts_query_exact_f64(rhs, &rhs_value)) {
      element = loom_value_facts_exact_f64(fn(lhs_value, rhs_value));
    }
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
    double lhs_value = 0.0;
    double rhs_value = 0.0;
    lanes[i] = loom_value_facts_unknown();
    if (loom_vector_facts_query_exact_f64(lhs, &lhs_value) &&
        loom_vector_facts_query_exact_f64(rhs, &rhs_value)) {
      lanes[i] = loom_value_facts_exact_f64(fn(lhs_value, rhs_value));
    }
  }
  loom_value_fact_small_static_lanes_t lane_slice = {
      .lanes = lanes,
      .count = lane_count,
  };
  return loom_value_facts_make_small_static_lanes(context, lane_slice,
                                                  &result_facts[0]);
}

static iree_status_t loom_vector_ternary_summary_facts(
    loom_fact_context_t* context, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts, loom_vector_ternary_transfer_fn_t fn) {
  bool fragment_handled = false;
  IREE_RETURN_IF_ERROR(loom_vector_try_preserve_accumulator_fragment_facts(
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

#define LOOM_VECTOR_INTEGER_BINARY_FACTS(name, transfer_fn)            \
  iree_status_t name(loom_fact_context_t* context,                     \
                     const loom_module_t* module, const loom_op_t* op, \
                     const loom_value_facts_t* operand_facts,          \
                     loom_value_facts_t* result_facts) {               \
    return loom_vector_integer_binary_summary_facts(                   \
        context, operand_facts, result_facts, transfer_fn);            \
  }

#define LOOM_VECTOR_FLOAT_BINARY_FACTS(name, fn)                          \
  iree_status_t name(loom_fact_context_t* context,                        \
                     const loom_module_t* module, const loom_op_t* op,    \
                     const loom_value_facts_t* operand_facts,             \
                     loom_value_facts_t* result_facts) {                  \
    return loom_vector_float_binary_summary_facts(context, operand_facts, \
                                                  result_facts, fn);      \
  }

#define LOOM_VECTOR_FLOAT_UNARY_FACTS(name, fn)                          \
  iree_status_t name(loom_fact_context_t* context,                       \
                     const loom_module_t* module, const loom_op_t* op,   \
                     const loom_value_facts_t* operand_facts,            \
                     loom_value_facts_t* result_facts) {                 \
    return loom_vector_float_unary_summary_facts(context, operand_facts, \
                                                 result_facts, fn);      \
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

LOOM_VECTOR_FLOAT_BINARY_FACTS(loom_vector_addf_facts, loom_vector_add_f64)
LOOM_VECTOR_FLOAT_BINARY_FACTS(loom_vector_subf_facts, loom_vector_sub_f64)
LOOM_VECTOR_FLOAT_BINARY_FACTS(loom_vector_mulf_facts, loom_vector_mul_f64)
LOOM_VECTOR_FLOAT_BINARY_FACTS(loom_vector_divf_facts, loom_vector_div_f64)
LOOM_VECTOR_FLOAT_BINARY_FACTS(loom_vector_remf_facts, fmod)
LOOM_VECTOR_FLOAT_UNARY_FACTS(loom_vector_negf_facts, loom_vector_neg_f64)
LOOM_VECTOR_FLOAT_UNARY_FACTS(loom_vector_absf_facts, fabs)
LOOM_VECTOR_FLOAT_BINARY_FACTS(loom_vector_minimumf_facts,
                               loom_vector_minimum_f64)
LOOM_VECTOR_FLOAT_BINARY_FACTS(loom_vector_maximumf_facts,
                               loom_vector_maximum_f64)
LOOM_VECTOR_FLOAT_BINARY_FACTS(loom_vector_minnumf_facts,
                               loom_vector_minnum_f64)
LOOM_VECTOR_FLOAT_BINARY_FACTS(loom_vector_maxnumf_facts,
                               loom_vector_maxnum_f64)
LOOM_VECTOR_FLOAT_BINARY_FACTS(loom_vector_copysignf_facts, copysign)

iree_status_t loom_vector_clampf_facts(loom_fact_context_t* context,
                                       const loom_module_t* module,
                                       const loom_op_t* op,
                                       const loom_value_facts_t* operand_facts,
                                       loom_value_facts_t* result_facts) {
  switch (loom_vector_clampf_mode(op)) {
    case LOOM_VECTOR_CLAMPF_MODE_ORDERED:
      return loom_vector_ternary_summary_facts(
          context, operand_facts, result_facts,
          loom_vector_clampf_ordered_transfer);
    case LOOM_VECTOR_CLAMPF_MODE_NUMBER:
      return loom_vector_ternary_summary_facts(
          context, operand_facts, result_facts,
          loom_vector_clampf_number_transfer);
    case LOOM_VECTOR_CLAMPF_MODE_IEEE:
      return loom_vector_ternary_summary_facts(
          context, operand_facts, result_facts,
          loom_vector_clampf_ieee_transfer);
    case LOOM_VECTOR_CLAMPF_MODE_COUNT_:
      break;
  }
  result_facts[0] = loom_value_facts_unknown();
  return iree_ok_status();
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
                            loom_count_leading_zeros_u64_width)
LOOM_VECTOR_BIT_COUNT_FACTS(loom_vector_cttzi_facts, loom_vector_cttzi_result,
                            loom_count_trailing_zeros_u64_width)
LOOM_VECTOR_BIT_COUNT_FACTS(loom_vector_ctpopi_facts, loom_vector_ctpopi_result,
                            loom_count_ones_u64_width)
LOOM_VECTOR_FLOAT_UNARY_FACTS(loom_vector_expf_facts, exp)
LOOM_VECTOR_FLOAT_UNARY_FACTS(loom_vector_exp2f_facts, exp2)
LOOM_VECTOR_FLOAT_UNARY_FACTS(loom_vector_expm1f_facts, expm1)
LOOM_VECTOR_FLOAT_UNARY_FACTS(loom_vector_logf_facts, log)
LOOM_VECTOR_FLOAT_UNARY_FACTS(loom_vector_log2f_facts, log2)
LOOM_VECTOR_FLOAT_UNARY_FACTS(loom_vector_log10f_facts, log10)
LOOM_VECTOR_FLOAT_UNARY_FACTS(loom_vector_log1pf_facts, log1p)
LOOM_VECTOR_FLOAT_BINARY_FACTS(loom_vector_powf_facts, pow)
LOOM_VECTOR_FLOAT_UNARY_FACTS(loom_vector_sqrtf_facts, sqrt)
LOOM_VECTOR_FLOAT_UNARY_FACTS(loom_vector_rsqrtf_facts, loom_vector_rsqrt_f64)
LOOM_VECTOR_FLOAT_UNARY_FACTS(loom_vector_cbrtf_facts, cbrt)
LOOM_VECTOR_FLOAT_UNARY_FACTS(loom_vector_sinf_facts, sin)
LOOM_VECTOR_FLOAT_UNARY_FACTS(loom_vector_cosf_facts, cos)
LOOM_VECTOR_FLOAT_UNARY_FACTS(loom_vector_sinturnsf_facts,
                              loom_vector_sinturns_f64)
LOOM_VECTOR_FLOAT_UNARY_FACTS(loom_vector_costurnsf_facts,
                              loom_vector_costurns_f64)
LOOM_VECTOR_FLOAT_UNARY_FACTS(loom_vector_tanf_facts, tan)
LOOM_VECTOR_FLOAT_UNARY_FACTS(loom_vector_asinf_facts, asin)
LOOM_VECTOR_FLOAT_UNARY_FACTS(loom_vector_acosf_facts, acos)
LOOM_VECTOR_FLOAT_UNARY_FACTS(loom_vector_atanf_facts, atan)
LOOM_VECTOR_FLOAT_BINARY_FACTS(loom_vector_atan2f_facts, atan2)
LOOM_VECTOR_FLOAT_UNARY_FACTS(loom_vector_sinhf_facts, sinh)
LOOM_VECTOR_FLOAT_UNARY_FACTS(loom_vector_coshf_facts, cosh)
LOOM_VECTOR_FLOAT_UNARY_FACTS(loom_vector_tanhf_facts, tanh)
LOOM_VECTOR_FLOAT_UNARY_FACTS(loom_vector_asinhf_facts, asinh)
LOOM_VECTOR_FLOAT_UNARY_FACTS(loom_vector_acoshf_facts, acosh)
LOOM_VECTOR_FLOAT_UNARY_FACTS(loom_vector_atanhf_facts, atanh)
LOOM_VECTOR_FLOAT_UNARY_FACTS(loom_vector_erff_facts, erf)
LOOM_VECTOR_FLOAT_UNARY_FACTS(loom_vector_erfcf_facts, erfc)
LOOM_VECTOR_FLOAT_UNARY_FACTS(loom_vector_logisticf_facts,
                              loom_vector_logistic_f64)
LOOM_VECTOR_FLOAT_UNARY_FACTS(loom_vector_siluf_facts, loom_vector_silu_f64)
LOOM_VECTOR_FLOAT_UNARY_FACTS(loom_vector_softplusf_facts,
                              loom_vector_softplus_f64)
LOOM_VECTOR_FLOAT_UNARY_FACTS(loom_vector_ceilf_facts, ceil)
LOOM_VECTOR_FLOAT_UNARY_FACTS(loom_vector_floorf_facts, floor)
LOOM_VECTOR_FLOAT_UNARY_FACTS(loom_vector_roundf_facts, round)
LOOM_VECTOR_FLOAT_UNARY_FACTS(loom_vector_roundevenf_facts,
                              loom_vector_roundeven_f64)
LOOM_VECTOR_FLOAT_UNARY_FACTS(loom_vector_truncf_facts, trunc)
LOOM_VECTOR_UNARY_FACTS(loom_vector_isnanf_facts, loom_vector_isnanf_transfer)
LOOM_VECTOR_UNARY_FACTS(loom_vector_isinff_facts, loom_vector_isinff_transfer)
LOOM_VECTOR_UNARY_FACTS(loom_vector_isfinitef_facts,
                        loom_vector_isfinitef_transfer)
LOOM_VECTOR_UNARY_FACTS(loom_vector_signf_facts, loom_vector_signf_transfer)
LOOM_VECTOR_UNARY_FACTS(loom_vector_signi_facts, loom_vector_signi_transfer)
LOOM_VECTOR_UNARY_FACTS(loom_vector_extf_facts,
                        loom_vector_passthrough_transfer)
LOOM_VECTOR_UNARY_FACTS(loom_vector_extsi_facts,
                        loom_vector_passthrough_transfer)
LOOM_VECTOR_UNARY_FACTS(loom_vector_sitofp_facts, loom_vector_sitofp_transfer)

typedef struct loom_vector_geluf_transfer_t {
  loom_vector_geluf_variant_t variant;
  double scale;
} loom_vector_geluf_transfer_t;

static double loom_vector_geluf_transfer(double input, const void* user_data) {
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
  return loom_vector_float_unary_data_summary_facts(
      context, operand_facts, result_facts, loom_vector_geluf_transfer,
      &transfer);
}

iree_status_t loom_vector_fmai_facts(loom_fact_context_t* context,
                                     const loom_module_t* module,
                                     const loom_op_t* op,
                                     const loom_value_facts_t* operand_facts,
                                     loom_value_facts_t* result_facts) {
  return loom_vector_ternary_summary_facts(context, operand_facts, result_facts,
                                           loom_vector_fmai_transfer);
}

iree_status_t loom_vector_fmaf_facts(loom_fact_context_t* context,
                                     const loom_module_t* module,
                                     const loom_op_t* op,
                                     const loom_value_facts_t* operand_facts,
                                     loom_value_facts_t* result_facts) {
  bool fragment_handled = false;
  IREE_RETURN_IF_ERROR(loom_vector_try_preserve_accumulator_fragment_facts(
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
    double a_value = 0.0;
    double b_value = 0.0;
    double c_value = 0.0;
    loom_value_facts_t element = loom_value_facts_unknown();
    if (loom_vector_facts_query_exact_f64(a, &a_value) &&
        loom_vector_facts_query_exact_f64(b, &b_value) &&
        loom_vector_facts_query_exact_f64(c, &c_value)) {
      element = loom_value_facts_exact_f64(fma(a_value, b_value, c_value));
    }
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
    double a_value = 0.0;
    double b_value = 0.0;
    double c_value = 0.0;
    lanes[i] = loom_value_facts_unknown();
    if (loom_vector_facts_query_exact_f64(a, &a_value) &&
        loom_vector_facts_query_exact_f64(b, &b_value) &&
        loom_vector_facts_query_exact_f64(c, &c_value)) {
      lanes[i] = loom_value_facts_exact_f64(fma(a_value, b_value, c_value));
    }
  }
  loom_value_fact_small_static_lanes_t lane_slice = {
      .lanes = lanes,
      .count = lane_count,
  };
  return loom_value_facts_make_small_static_lanes(context, lane_slice,
                                                  &result_facts[0]);
}

iree_status_t loom_vector_select_facts(loom_fact_context_t* context,
                                       const loom_module_t* module,
                                       const loom_op_t* op,
                                       const loom_value_facts_t* operand_facts,
                                       loom_value_facts_t* result_facts) {
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
    if (loom_vector_facts_query_exact_f64(lhs, &lhs_value) &&
        loom_vector_facts_query_exact_f64(rhs, &rhs_value) &&
        loom_scalar_cmpf_exact_result(predicate, lhs_value, rhs_value,
                                      &result)) {
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
    if (loom_vector_facts_query_exact_f64(lhs, &lhs_value) &&
        loom_vector_facts_query_exact_f64(rhs, &rhs_value) &&
        loom_scalar_cmpf_exact_result(predicate, lhs_value, rhs_value,
                                      &result)) {
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
  uint64_t field_mask = loom_mask_to_bitwidth_u64(UINT64_MAX, (int32_t)width);
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
        loom_mask_to_bitwidth_u64(lane_bits >> source_shift, piece_bits);
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
    const loom_fact_context_t* context, loom_value_facts_t input,
    loom_value_facts_t threshold_facts, iree_host_size_t threshold_count,
    uint8_t nan_policy, uint8_t tie_policy, loom_value_facts_t* out_element) {
  double input_value = 0.0;
  if (!loom_vector_facts_query_exact_f64(input, &input_value)) {
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
        !loom_vector_facts_query_exact_f64(threshold, &threshold_value)) {
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
            context, uniform_input, operand_facts[1], threshold_count,
            nan_policy, tie_policy, &element)) {
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
        !loom_vector_table_quantize_exact_lane(context, input, operand_facts[1],
                                               threshold_count, nan_policy,
                                               tie_policy, &lanes[lane])) {
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

static bool loom_vector_reduce_apply_float(loom_combining_kind_t kind,
                                           double accumulator, double element,
                                           double* out) {
  switch (kind) {
    case LOOM_COMBINING_KIND_ADDF:
      *out = accumulator + element;
      return true;
    case LOOM_COMBINING_KIND_MULF:
      *out = accumulator * element;
      return true;
    case LOOM_COMBINING_KIND_MINIMUMF:
    case LOOM_COMBINING_KIND_MINNUMF:
      *out = loom_vector_minimum_f64(accumulator, element);
      return true;
    case LOOM_COMBINING_KIND_MAXIMUMF:
    case LOOM_COMBINING_KIND_MAXNUMF:
      *out = loom_vector_maximum_f64(accumulator, element);
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

static bool loom_vector_reduce_static_uniform(loom_combining_kind_t kind,
                                              uint64_t element_count,
                                              loom_value_facts_t element,
                                              loom_value_facts_t init,
                                              loom_value_facts_t* out) {
  if (element_count == 0) {
    *out = init;
    return true;
  }
  if (element_count > LOOM_VECTOR_FACT_STATIC_LOOP_LIMIT) return false;

  double float_accumulator = 0.0;
  double float_element = 0.0;
  bool float_reduce =
      loom_vector_facts_query_exact_f64(init, &float_accumulator) &&
      loom_vector_facts_query_exact_f64(element, &float_element);
  if (float_reduce) {
    for (uint64_t i = 0; i < element_count; ++i) {
      if (!loom_vector_reduce_apply_float(kind, float_accumulator,
                                          float_element, &float_accumulator)) {
        return false;
      }
    }
    *out = loom_value_facts_exact_f64(float_accumulator);
    return true;
  }

  loom_value_facts_t accumulator = init;
  for (uint64_t i = 0; i < element_count; ++i) {
    loom_value_facts_t next = loom_value_facts_unknown();
    if (!loom_vector_reduce_apply_integer(kind, &accumulator, &element,
                                          &next)) {
      return false;
    }
    accumulator = next;
  }
  *out = accumulator;
  return true;
}

static bool loom_vector_reduce_small_static_lanes(
    loom_combining_kind_t kind, loom_value_fact_small_static_lanes_t lanes,
    loom_value_facts_t init, loom_value_facts_t* out) {
  if (lanes.count == 0) {
    *out = init;
    return true;
  }

  double float_accumulator = 0.0;
  if (loom_vector_facts_query_exact_f64(init, &float_accumulator)) {
    for (iree_host_size_t i = 0; i < lanes.count; ++i) {
      double element = 0.0;
      if (!loom_vector_facts_query_exact_f64(lanes.lanes[i], &element) ||
          !loom_vector_reduce_apply_float(kind, float_accumulator, element,
                                          &float_accumulator)) {
        return false;
      }
    }
    *out = loom_value_facts_exact_f64(float_accumulator);
    return true;
  }

  loom_value_facts_t accumulator = init;
  for (iree_host_size_t i = 0; i < lanes.count; ++i) {
    loom_value_facts_t next = loom_value_facts_unknown();
    if (!loom_vector_reduce_apply_integer(kind, &accumulator, &lanes.lanes[i],
                                          &next)) {
      return false;
    }
    accumulator = next;
  }
  *out = accumulator;
  return true;
}

static bool loom_vector_reduce_apply_facts(loom_combining_kind_t kind,
                                           loom_value_facts_t accumulator,
                                           loom_value_facts_t element,
                                           loom_value_facts_t* out) {
  double float_accumulator = 0.0;
  double float_element = 0.0;
  if (loom_vector_facts_query_exact_f64(accumulator, &float_accumulator) &&
      loom_vector_facts_query_exact_f64(element, &float_element)) {
    if (!loom_vector_reduce_apply_float(kind, float_accumulator, float_element,
                                        &float_accumulator)) {
      return false;
    }
    *out = loom_value_facts_exact_f64(float_accumulator);
    return true;
  }
  return loom_vector_reduce_apply_integer(kind, &accumulator, &element, out);
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
    if (loom_vector_reduce_small_static_lanes(kind, lanes, init_facts,
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
    if (loom_vector_reduce_static_uniform(kind, element_count, element,
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
      if (!loom_vector_reduce_apply_facts(kind, accumulator, element, &next)) {
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
      if (loom_vector_reduce_static_uniform(kind, reduced_element_count,
                                            input_element, init_element,
                                            &reduced_element)) {
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

iree_status_t loom_vector_dotf_facts(loom_fact_context_t* context,
                                     const loom_module_t* module,
                                     const loom_op_t* op,
                                     const loom_value_facts_t* operand_facts,
                                     loom_value_facts_t* result_facts) {
  uint64_t element_count = 0;
  loom_type_t lhs_type =
      loom_module_value_type(module, loom_vector_dotf_lhs(op));
  if (loom_type_static_element_count(lhs_type, &element_count) &&
      element_count == 0) {
    result_facts[0] = operand_facts[2];
    return iree_ok_status();
  }

  iree_host_size_t lane_count = 0;
  if (loom_vector_facts_query_binary_lane_count(
          context, operand_facts[0], operand_facts[1], &lane_count)) {
    double accumulator = 0.0;
    if (!loom_vector_facts_query_exact_f64(operand_facts[2], &accumulator)) {
      result_facts[0] = loom_value_facts_unknown();
      return iree_ok_status();
    }
    for (iree_host_size_t i = 0; i < lane_count; ++i) {
      loom_value_facts_t lhs = {0};
      loom_value_facts_t rhs = {0};
      double lhs_value = 0.0;
      double rhs_value = 0.0;
      if (!loom_vector_facts_query_lane(context, operand_facts[0], i, &lhs) ||
          !loom_vector_facts_query_lane(context, operand_facts[1], i, &rhs) ||
          !loom_vector_facts_query_exact_f64(lhs, &lhs_value) ||
          !loom_vector_facts_query_exact_f64(rhs, &rhs_value)) {
        result_facts[0] = loom_value_facts_unknown();
        return iree_ok_status();
      }
      accumulator = fma(lhs_value, rhs_value, accumulator);
    }
    result_facts[0] = loom_value_facts_exact_f64(accumulator);
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

  double lhs_value = 0.0;
  double rhs_value = 0.0;
  double accumulator = 0.0;
  if (!loom_vector_facts_query_exact_f64(lhs, &lhs_value) ||
      !loom_vector_facts_query_exact_f64(rhs, &rhs_value) ||
      !loom_vector_facts_query_exact_f64(operand_facts[2], &accumulator)) {
    result_facts[0] = loom_value_facts_unknown();
    return iree_ok_status();
  }
  for (uint64_t i = 0; i < element_count; ++i) {
    accumulator = fma(lhs_value, rhs_value, accumulator);
  }
  result_facts[0] = loom_value_facts_exact_f64(accumulator);
  return iree_ok_status();
}

iree_status_t loom_vector_dot2f_facts(loom_fact_context_t* context,
                                      const loom_module_t* module,
                                      const loom_op_t* op,
                                      const loom_value_facts_t* operand_facts,
                                      loom_value_facts_t* result_facts) {
  loom_value_facts_t lhs_element = {0};
  loom_value_facts_t rhs_element = {0};
  loom_value_facts_t acc_element = {0};
  if (loom_vector_facts_query_uniform_element(context, operand_facts[0],
                                              &lhs_element) &&
      loom_vector_facts_query_uniform_element(context, operand_facts[1],
                                              &rhs_element) &&
      loom_vector_facts_query_uniform_element(context, operand_facts[2],
                                              &acc_element)) {
    double lhs_value = 0.0;
    double rhs_value = 0.0;
    double accumulator = 0.0;
    if (!loom_vector_facts_query_exact_f64(lhs_element, &lhs_value) ||
        !loom_vector_facts_query_exact_f64(rhs_element, &rhs_value) ||
        !loom_vector_facts_query_exact_f64(acc_element, &accumulator)) {
      return loom_vector_make_unknown_facts(result_facts);
    }
    for (uint8_t group_lane = 0; group_lane < 2; ++group_lane) {
      accumulator = fma(lhs_value, rhs_value, accumulator);
    }
    return loom_value_facts_make_uniform_element(
        context, loom_value_facts_exact_f64(accumulator), &result_facts[0]);
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
    double accumulator = 0.0;
    if (!loom_vector_facts_query_lane(context, operand_facts[2], result_lane,
                                      &acc) ||
        !loom_vector_facts_query_exact_f64(acc, &accumulator)) {
      return loom_vector_make_unknown_facts(result_facts);
    }
    for (uint8_t group_lane = 0; group_lane < 2; ++group_lane) {
      iree_host_size_t source_lane = 0;
      loom_value_facts_t lhs = {0};
      loom_value_facts_t rhs = {0};
      double lhs_value = 0.0;
      double rhs_value = 0.0;
      if (!loom_vector_grouped_dot_source_lane(shape, result_lane, 2,
                                               group_lane, &source_lane) ||
          !loom_vector_facts_query_lane(context, operand_facts[0], source_lane,
                                        &lhs) ||
          !loom_vector_facts_query_lane(context, operand_facts[1], source_lane,
                                        &rhs) ||
          !loom_vector_facts_query_exact_f64(lhs, &lhs_value) ||
          !loom_vector_facts_query_exact_f64(rhs, &rhs_value)) {
        return loom_vector_make_unknown_facts(result_facts);
      }
      accumulator = fma(lhs_value, rhs_value, accumulator);
    }
    lanes[result_lane] = loom_value_facts_exact_f64(accumulator);
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
    double accumulator = 0.0;
    if (!loom_vector_facts_query_exact_u32_bits(lhs_element, &lhs_bits) ||
        !loom_vector_facts_query_exact_u32_bits(rhs_element, &rhs_bits) ||
        !loom_vector_facts_query_exact_f64(acc_element, &accumulator) ||
        !loom_vector_dot4f8_apply(kind, lhs_bits, rhs_bits, &accumulator)) {
      return loom_vector_make_unknown_facts(result_facts);
    }
    return loom_value_facts_make_uniform_element(
        context, loom_value_facts_exact_f64(accumulator), &result_facts[0]);
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
    double accumulator = 0.0;
    if (!loom_vector_facts_query_lane(context, operand_facts[0], result_lane,
                                      &lhs) ||
        !loom_vector_facts_query_lane(context, operand_facts[1], result_lane,
                                      &rhs) ||
        !loom_vector_facts_query_lane(context, operand_facts[2], result_lane,
                                      &acc) ||
        !loom_vector_facts_query_exact_u32_bits(lhs, &lhs_bits) ||
        !loom_vector_facts_query_exact_u32_bits(rhs, &rhs_bits) ||
        !loom_vector_facts_query_exact_f64(acc, &accumulator) ||
        !loom_vector_dot4f8_apply(kind, lhs_bits, rhs_bits, &accumulator)) {
      return loom_vector_make_unknown_facts(result_facts);
    }
    lanes[result_lane] = loom_value_facts_exact_f64(accumulator);
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

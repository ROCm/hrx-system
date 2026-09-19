// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string.h>

#include "loom/analysis/scc.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"
#include "loom/util/fact_cfg.h"
#include "loom/util/fact_table.h"

//===----------------------------------------------------------------------===//
// Argument and type facts
//===----------------------------------------------------------------------===//

static int64_t loom_value_fact_static_element_byte_count(loom_type_t type) {
  int32_t bit_count = loom_scalar_type_bitwidth(loom_type_element_type(type));
  if (bit_count <= 0 || (bit_count % 8) != 0) {
    return -1;
  }
  return bit_count / 8;
}

static iree_status_t loom_value_fact_table_seed_buffer_arg(
    loom_value_fact_table_t* table, loom_value_id_t value_id,
    loom_value_fact_memory_space_t memory_space,
    loom_value_fact_reference_origin_t origin) {
  loom_value_fact_buffer_reference_t reference = {
      .maximum_byte_extent = loom_value_facts_make(0, INT64_MAX, 1),
      .minimum_alignment = 1,
      .memory_space = memory_space,
      .root_value_id = value_id,
      .alias_scope_id = LOOM_VALUE_FACT_ALIAS_SCOPE_ID_NONE,
      .nullability = LOOM_VALUE_FACT_REFERENCE_NULLABILITY_UNKNOWN,
      .origin = origin,
  };
  loom_value_facts_t facts = loom_value_fact_table_lookup(table, value_id);
  if (loom_value_facts_query_buffer_reference(&table->context, facts,
                                              &reference) &&
      loom_value_fact_reference_origin_equal(reference.origin, origin)) {
    return iree_ok_status();
  }
  if (origin.kind != LOOM_VALUE_FACT_REFERENCE_ORIGIN_UNKNOWN) {
    reference.origin = origin;
  }
  loom_value_facts_t reference_facts = loom_value_facts_unknown();
  IREE_RETURN_IF_ERROR(loom_value_facts_make_buffer_reference(
      &table->context, reference, &reference_facts));
  facts.extension_id = reference_facts.extension_id;
  return loom_value_fact_table_define(table, value_id, facts);
}

static iree_status_t loom_value_fact_table_seed_view_arg(
    loom_value_fact_table_t* table, loom_value_id_t value_id, loom_type_t type,
    loom_value_fact_reference_origin_t origin) {
  loom_value_fact_view_reference_t reference = {
      .base_byte_offset = loom_value_facts_exact_i64(0),
      .footprint_byte_length = loom_value_facts_make(0, INT64_MAX, 1),
      .minimum_alignment = 1,
      .root_minimum_alignment = 1,
      .static_element_byte_count =
          loom_value_fact_static_element_byte_count(type),
      .memory_space = LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN,
      .root_value_id = value_id,
      .buffer_value_id = LOOM_VALUE_ID_INVALID,
      .alias_scope_id = LOOM_VALUE_FACT_ALIAS_SCOPE_ID_NONE,
      .nullability = LOOM_VALUE_FACT_REFERENCE_NULLABILITY_UNKNOWN,
      .origin = origin,
  };
  loom_value_facts_t facts = loom_value_fact_table_lookup(table, value_id);
  if (loom_value_facts_query_view_reference(&table->context, facts,
                                            &reference) &&
      loom_value_fact_reference_origin_equal(reference.origin, origin)) {
    return iree_ok_status();
  }
  if (origin.kind != LOOM_VALUE_FACT_REFERENCE_ORIGIN_UNKNOWN) {
    reference.origin = origin;
  }
  loom_value_facts_t reference_facts = loom_value_facts_unknown();
  IREE_RETURN_IF_ERROR(loom_value_facts_make_view_reference(
      &table->context, reference, &reference_facts));
  facts.extension_id = reference_facts.extension_id;
  return loom_value_fact_table_define(table, value_id, facts);
}

static iree_status_t loom_value_fact_table_define_if_changed(
    loom_value_fact_table_t* table, loom_value_id_t value_id,
    loom_value_facts_t facts, bool* out_changed) {
  loom_value_facts_t old_facts = loom_value_fact_table_lookup(table, value_id);
  if (out_changed && !loom_value_facts_equal(old_facts, facts)) {
    *out_changed = true;
  }
  return loom_value_fact_table_define(table, value_id, facts);
}

static bool loom_value_fact_table_find_result(const loom_value_id_t* results,
                                              uint16_t result_count,
                                              loom_value_id_t value_id,
                                              uint16_t* out_result_index) {
  if (!results) {
    return false;
  }
  for (uint16_t i = 0; i < result_count; ++i) {
    if (results[i] != value_id) {
      continue;
    }
    *out_result_index = i;
    return true;
  }
  return false;
}

static bool loom_value_fact_table_dynamic_extent_type_supported(
    loom_type_t type) {
  if (!loom_type_is_scalar(type)) {
    return false;
  }
  loom_scalar_type_t scalar_type = loom_type_element_type(type);
  return scalar_type == LOOM_SCALAR_TYPE_INDEX ||
         scalar_type == LOOM_SCALAR_TYPE_OFFSET ||
         loom_scalar_type_is_integer(scalar_type);
}

static loom_value_facts_t loom_value_fact_table_clamp_scalar_type_domain(
    const loom_module_t* module, loom_value_id_t value_id,
    loom_value_facts_t facts) {
  loom_type_t type = loom_module_value_type(module, value_id);
  if (!loom_type_is_scalar(type)) {
    return facts;
  }
  int64_t lo = 0;
  int64_t hi = 0;
  if (!loom_value_facts_scalar_type_domain(loom_type_element_type(type), &lo,
                                           &hi)) {
    return facts;
  }
  return loom_value_facts_clamp_domain(facts, lo, hi);
}

static iree_status_t loom_value_fact_table_seed_dynamic_extent(
    loom_value_fact_table_t* table, const loom_module_t* module,
    loom_value_id_t value_id, const loom_value_id_t* result_ids,
    uint16_t result_count, loom_value_facts_t* result_facts,
    bool* out_changed) {
  if (value_id == LOOM_VALUE_ID_INVALID || value_id >= module->values.count) {
    return iree_ok_status();
  }
  if (!loom_value_fact_table_dynamic_extent_type_supported(
          loom_module_value_type(module, value_id))) {
    return iree_ok_status();
  }

  uint16_t result_index = 0;
  if (result_facts && loom_value_fact_table_find_result(
                          result_ids, result_count, value_id, &result_index)) {
    result_facts[result_index] = loom_value_fact_table_clamp_scalar_type_domain(
        module, value_id,
        loom_value_facts_non_negative_extent(result_facts[result_index]));
    return iree_ok_status();
  }

  loom_value_facts_t current = loom_value_fact_table_lookup(table, value_id);
  loom_value_facts_t extent = loom_value_fact_table_clamp_scalar_type_domain(
      module, value_id, loom_value_facts_non_negative_extent(current));
  return loom_value_fact_table_define_if_changed(table, value_id, extent,
                                                 out_changed);
}

static iree_status_t loom_value_fact_table_seed_scalar_arg(
    loom_value_fact_table_t* table, const loom_module_t* module,
    loom_value_id_t value_id) {
  if (!loom_value_facts_is_unknown(
          loom_value_fact_table_lookup(table, value_id))) {
    return iree_ok_status();
  }
  loom_type_t type = loom_module_value_type(module, value_id);
  if (!loom_type_is_scalar(type)) {
    return iree_ok_status();
  }
  return loom_value_fact_table_define(
      table, value_id,
      loom_value_fact_table_clamp_scalar_type_domain(
          module, value_id, loom_value_facts_unknown()));
}

static loom_value_facts_t loom_value_fact_table_unknown_for_value(
    const loom_module_t* module, loom_value_id_t value_id) {
  loom_value_facts_t facts = loom_value_facts_unknown();
  if (value_id >= module->values.count) {
    return facts;
  }
  loom_type_t type = loom_module_value_type(module, value_id);
  if (loom_type_is_scalar(type)) {
    facts =
        loom_value_fact_table_clamp_scalar_type_domain(module, value_id, facts);
  }
  return facts;
}

static iree_status_t loom_value_fact_table_seed_type_extent_facts(
    loom_value_fact_table_t* table, const loom_module_t* module,
    loom_type_t type, const loom_value_id_t* result_ids, uint16_t result_count,
    loom_value_facts_t* result_facts, bool* out_changed) {
  if (!loom_type_is_shaped(type)) {
    return iree_ok_status();
  }
  uint8_t rank = loom_type_rank(type);
  for (uint8_t i = 0; i < rank; ++i) {
    if (!loom_type_dim_is_dynamic_at(type, i)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_value_fact_table_seed_dynamic_extent(
        table, module, loom_type_dim_value_id_at(type, i), result_ids,
        result_count, result_facts, out_changed));
  }
  return iree_ok_status();
}

static void loom_value_fact_table_apply_operand_distribution(
    const loom_module_t* module, const loom_op_t* op,
    const loom_value_facts_t* operand_facts, loom_value_id_t result_id,
    loom_value_facts_t* result_facts) {
  if (result_id == LOOM_VALUE_ID_INVALID || result_id >= module->values.count) {
    return;
  }
  if (loom_value_facts_is_exact(*result_facts)) {
    loom_value_facts_mark_cluster_uniform(result_facts);
    return;
  }
  if (iree_any_bit_set(
          result_facts->flags,
          LOOM_VALUE_FACT_DISTRIBUTION_MASK | LOOM_VALUE_FACT_LANE_PREDICATE)) {
    return;
  }
  if (op->operand_count == 0) {
    return;
  }

  bool any_operand_lane_varying = false;
  loom_value_fact_uniform_scope_t uniform_scope =
      LOOM_VALUE_FACT_UNIFORM_SCOPE_CLUSTER;
  for (uint16_t i = 0; i < op->operand_count; ++i) {
    const loom_value_facts_t facts = operand_facts[i];
    if (loom_value_facts_is_lane_varying(facts) ||
        loom_value_facts_is_lane_predicate(facts)) {
      any_operand_lane_varying = true;
    }
    uniform_scope =
        iree_min(uniform_scope, loom_value_facts_uniform_scope(facts));
  }

  if (any_operand_lane_varying) {
    loom_value_facts_mark_lane_distribution_for_type(
        loom_module_value_type(module, result_id), result_facts);
  } else {
    loom_value_facts_mark_uniform_at_scope(result_facts, uniform_scope);
  }
}

static int64_t loom_value_fact_loop_iv_base_divisor(loom_value_facts_t facts) {
  if (!loom_value_facts_is_exact(facts) || facts.range_lo == INT64_MIN) {
    return facts.known_divisor;
  }
  // Preserve gcd(0, step) so a zero lower bound keeps step divisibility.
  return facts.range_lo >= 0 ? facts.range_lo : -facts.range_lo;
}

static bool loom_value_fact_counted_loop_trip_count(int64_t lower,
                                                    int64_t upper, int64_t step,
                                                    uint64_t* out_count) {
  if (step <= 0) {
    return false;
  }
  if (lower >= upper) {
    *out_count = 0;
    return true;
  }
  int64_t span = 0;
  if (!iree_checked_sub_i64(upper, lower, &span)) {
    return false;
  }
  *out_count = ((uint64_t)span + (uint64_t)step - 1) / (uint64_t)step;
  return true;
}

static bool loom_value_fact_counted_loop_last_reachable_iv(
    loom_value_facts_t lower_bound, loom_value_facts_t upper_bound,
    loom_value_facts_t step, int64_t* out_hi) {
  if (!loom_value_facts_is_exact(step) || step.range_lo <= 0 ||
      lower_bound.range_lo >= upper_bound.range_hi) {
    return false;
  }

  if (loom_value_facts_is_exact(upper_bound)) {
    uint64_t lower_min_trip_count = 0;
    uint64_t lower_max_trip_count = 0;
    if (loom_value_fact_counted_loop_trip_count(
            lower_bound.range_lo, upper_bound.range_lo, step.range_lo,
            &lower_min_trip_count) &&
        loom_value_fact_counted_loop_trip_count(
            lower_bound.range_hi, upper_bound.range_hi, step.range_hi,
            &lower_max_trip_count) &&
        lower_min_trip_count == lower_max_trip_count &&
        lower_max_trip_count > 0 && lower_max_trip_count <= INT64_MAX) {
      int64_t stepped_offset = 0;
      if (iree_checked_mul_i64((int64_t)(lower_max_trip_count - 1),
                               step.range_lo, &stepped_offset) &&
          iree_checked_add_i64(lower_bound.range_hi, stepped_offset, out_hi)) {
        return true;
      }
    }
  }

  if (!loom_value_facts_is_exact(lower_bound)) {
    return false;
  }

  int64_t last_possible_offset = 0;
  if (!iree_checked_sub_i64(upper_bound.range_hi, lower_bound.range_lo,
                            &last_possible_offset) ||
      !iree_checked_sub_i64(last_possible_offset, 1, &last_possible_offset)) {
    return false;
  }

  int64_t stepped_offset = 0;
  const int64_t trip_index = last_possible_offset / step.range_lo;
  if (!iree_checked_mul_i64(trip_index, step.range_lo, &stepped_offset) ||
      !iree_checked_add_i64(lower_bound.range_lo, stepped_offset, out_hi)) {
    return false;
  }
  return true;
}

static loom_value_facts_t loom_value_fact_counted_loop_iv_facts(
    loom_value_facts_t lower_bound, loom_value_facts_t upper_bound,
    loom_value_facts_t step) {
  if (loom_value_facts_is_float(lower_bound) ||
      loom_value_facts_is_float(upper_bound) ||
      loom_value_facts_is_float(step) || !loom_value_facts_is_positive(step)) {
    return loom_value_facts_unknown();
  }

  int64_t lower_divisor = loom_value_fact_loop_iv_base_divisor(lower_bound);
  int64_t divisor = iree_math_gcd_i64(lower_divisor, step.known_divisor);

  int64_t hi = INT64_MAX;
  if (loom_value_fact_counted_loop_last_reachable_iv(lower_bound, upper_bound,
                                                     step, &hi)) {
    return loom_value_facts_make(lower_bound.range_lo, hi, divisor);
  }

  if (iree_checked_sub_i64(upper_bound.range_hi, 1, &hi)) {
    if (lower_bound.range_lo <= hi) {
      return loom_value_facts_make(lower_bound.range_lo, hi, divisor);
    }
  }
  return loom_value_facts_unknown();
}

static iree_status_t loom_value_fact_table_seed_loop_iv_arg(
    loom_value_fact_table_t* table, const loom_module_t* module,
    const loom_block_t* block, loom_op_t* parent_op) {
  loom_loop_like_t loop = loom_loop_like_cast(module, parent_op);
  if (!loom_loop_like_isa(loop) || !loom_loop_like_has_counted_range(loop)) {
    return iree_ok_status();
  }
  if (loop.vtable->iv_block_arg_index == LOOM_BLOCK_ARG_INDEX_NONE ||
      loop.vtable->iv_block_arg_index >= block->arg_count) {
    return iree_ok_status();
  }
  loom_region_t* body = loom_loop_like_body(loop);
  if (!body || body->block_count == 0 ||
      block != loom_region_const_entry_block(body)) {
    return iree_ok_status();
  }

  loom_value_facts_t lower_bound =
      loom_value_fact_table_lookup(table, loom_loop_like_lower_bound(loop));
  loom_value_facts_t upper_bound =
      loom_value_fact_table_lookup(table, loom_loop_like_upper_bound(loop));
  loom_value_facts_t step =
      loom_value_fact_table_lookup(table, loom_loop_like_step(loop));
  loom_value_facts_t iv_facts =
      loom_value_fact_counted_loop_iv_facts(lower_bound, upper_bound, step);
  loom_value_facts_propagate_ternary_distribution(lower_bound, upper_bound,
                                                  step, &iv_facts);
  loom_value_id_t iv_id =
      loom_block_arg_id(block, loop.vtable->iv_block_arg_index);
  return loom_value_fact_table_define(table, iv_id, iv_facts);
}

static const loom_region_descriptor_t*
loom_value_fact_table_seeded_region_descriptor(const loom_module_t* module,
                                               const loom_block_t* block,
                                               const loom_op_t* parent_op) {
  const loom_region_t* region = block->parent_region;
  if (!parent_op || !region) {
    return NULL;
  }

  const loom_op_vtable_t* vtable = loom_op_vtable(module, parent_op);
  loom_region_t* const* regions = loom_op_regions(parent_op);
  for (uint8_t i = 0; i < parent_op->region_count; ++i) {
    if (regions[i] != region) {
      continue;
    }
    return loom_op_vtable_region_descriptor(vtable, i);
  }
  return NULL;
}

static bool loom_value_fact_table_block_contains_arg(const loom_block_t* block,
                                                     loom_value_id_t value_id) {
  for (uint16_t i = 0; i < block->arg_count; ++i) {
    if (loom_block_arg_id(block, i) == value_id) {
      return true;
    }
  }
  return false;
}

static iree_status_t loom_value_fact_table_apply_func_predicates(
    loom_value_fact_table_t* table, const loom_module_t* module,
    const loom_block_t* block, loom_op_t* parent_op) {
  loom_func_like_t function = loom_func_like_cast(module, parent_op);
  if (!loom_func_like_isa(function)) {
    return iree_ok_status();
  }
  loom_region_t* body = loom_func_like_body(function);
  if (!body || block != loom_region_const_entry_block(body)) {
    return iree_ok_status();
  }

  uint16_t predicate_count = 0;
  const loom_predicate_t* predicates =
      loom_func_like_predicates(function, &predicate_count);
  for (uint16_t i = 0; i < predicate_count; ++i) {
    const loom_predicate_t* predicate = &predicates[i];
    if (predicate->arg_tags[0] != LOOM_PRED_ARG_VALUE) {
      continue;
    }
    int64_t raw_value_id = predicate->args[0];
    if (raw_value_id < 0 || raw_value_id > UINT32_MAX) {
      continue;
    }
    loom_value_id_t value_id = (loom_value_id_t)raw_value_id;
    if (!loom_value_fact_table_block_contains_arg(block, value_id)) {
      continue;
    }
    loom_value_facts_t facts = loom_value_fact_table_lookup(table, value_id);
    loom_value_facts_apply_predicate(&facts, predicate);
    IREE_RETURN_IF_ERROR(loom_value_fact_table_define(table, value_id, facts));
  }
  return iree_ok_status();
}

static iree_status_t loom_value_fact_table_seed_block_args(
    loom_value_fact_table_t* table, const loom_module_t* module,
    const loom_block_t* block, loom_op_t* parent_op) {
  loom_value_fact_reference_origin_t origin = {0};
  if (parent_op == table->context.function.op &&
      block == loom_region_const_entry_block(block->parent_region)) {
    origin = table->context.reference_origin;
  }
  const loom_region_descriptor_t* region_descriptor =
      loom_value_fact_table_seeded_region_descriptor(module, block, parent_op);
  const loom_value_fact_memory_space_t buffer_memory_space =
      region_descriptor && iree_any_bit_set(region_descriptor->flags,
                                            LOOM_REGION_GLOBAL_BUFFER_ARGS)
          ? LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL
          : LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN;
  const bool has_workgroup_uniform_args =
      region_descriptor && iree_any_bit_set(region_descriptor->flags,
                                            LOOM_REGION_WORKGROUP_UNIFORM_ARGS);
  const bool has_cluster_uniform_args =
      region_descriptor && iree_any_bit_set(region_descriptor->flags,
                                            LOOM_REGION_CLUSTER_UNIFORM_ARGS);
  for (uint16_t i = 0; i < block->arg_count; ++i) {
    loom_value_id_t value_id = loom_block_arg_id(block, i);
    if (value_id >= module->values.count) {
      continue;
    }
    loom_type_t type = loom_module_value_type(module, value_id);
    const bool has_entry = loom_value_fact_table_has_entry(table, value_id);
    if (origin.kind == LOOM_VALUE_FACT_REFERENCE_ORIGIN_ENTRY) {
      origin.entry_value_id = value_id;
    }
    if (!has_entry || origin.kind == LOOM_VALUE_FACT_REFERENCE_ORIGIN_ENTRY) {
      if (loom_type_is_buffer(type)) {
        IREE_RETURN_IF_ERROR(loom_value_fact_table_seed_buffer_arg(
            table, value_id, buffer_memory_space, origin));
      } else if (loom_type_is_view(type)) {
        IREE_RETURN_IF_ERROR(
            loom_value_fact_table_seed_view_arg(table, value_id, type, origin));
      } else if (!has_entry) {
        IREE_RETURN_IF_ERROR(
            loom_value_fact_table_seed_scalar_arg(table, module, value_id));
      }
    }
    if (has_workgroup_uniform_args || has_cluster_uniform_args) {
      loom_value_facts_t facts = loom_value_fact_table_lookup(table, value_id);
      if (has_cluster_uniform_args) {
        loom_value_facts_mark_cluster_uniform(&facts);
      } else {
        loom_value_facts_mark_workgroup_uniform(&facts);
      }
      IREE_RETURN_IF_ERROR(
          loom_value_fact_table_define(table, value_id, facts));
    }
    IREE_RETURN_IF_ERROR(loom_value_fact_table_seed_type_extent_facts(
        table, module, type, /*result_ids=*/NULL, /*result_count=*/0,
        /*result_facts=*/NULL, /*out_changed=*/NULL));
  }
  IREE_RETURN_IF_ERROR(loom_value_fact_table_apply_func_predicates(
      table, module, block, parent_op));
  return loom_value_fact_table_seed_loop_iv_arg(table, module, block,
                                                parent_op);
}

//===----------------------------------------------------------------------===//
// CFG block argument summaries
//===----------------------------------------------------------------------===//

static iree_status_t loom_value_fact_table_define_block_arg_facts(
    loom_value_fact_table_t* table, const loom_module_t* module,
    loom_value_id_t arg_id, loom_value_facts_t facts, bool* out_changed) {
  if (arg_id >= module->values.count) {
    return iree_ok_status();
  }
  loom_type_t type = loom_module_value_type(module, arg_id);
  if (out_changed &&
      (!loom_value_fact_table_has_entry(table, arg_id) ||
       !loom_value_fact_table_facts_equal_for_type(
           module, type, table, loom_value_fact_table_lookup(table, arg_id),
           table, facts))) {
    *out_changed = true;
  }
  IREE_RETURN_IF_ERROR(loom_value_fact_table_define(table, arg_id, facts));
  return loom_value_fact_table_seed_type_extent_facts(
      table, module, type, /*result_ids=*/NULL, /*result_count=*/0,
      /*result_facts=*/NULL, out_changed);
}

static iree_status_t loom_value_fact_table_join_cfg_block_arg_incoming(
    loom_value_fact_table_t* table, const loom_module_t* module,
    loom_type_t type, loom_value_id_t source_value, bool* inout_has_facts,
    loom_value_facts_t* inout_facts) {
  // A forward CFG solve can see backedge operands before the producing block
  // has run in the current iteration. Undefined entries are skipped until the
  // producer contributes facts; values that are defined-but-unknown still have
  // entries and participate in the meet below.
  if (!loom_value_fact_table_has_entry(table, source_value)) {
    return iree_ok_status();
  }
  loom_value_facts_t source_facts =
      loom_value_fact_table_lookup(table, source_value);
  if (!*inout_has_facts) {
    *inout_facts = source_facts;
    *inout_has_facts = true;
    return iree_ok_status();
  }
  loom_value_facts_t joined = loom_value_facts_unknown();
  IREE_RETURN_IF_ERROR(loom_value_fact_table_meet_for_type(
      table, module, type, table, *inout_facts, table, source_facts, &joined));
  *inout_facts = joined;
  return iree_ok_status();
}

static iree_status_t loom_value_fact_table_compute_cfg_block_arg(
    loom_value_fact_table_t* table, const loom_module_t* module,
    const loom_value_fact_cfg_region_t* region, const loom_scc_t* component,
    uint16_t block_index, uint16_t arg_index, bool widen, uint32_t iteration,
    loom_value_fact_cfg_changed_fn_t on_changed, void* user_data,
    bool* out_changed) {
  const loom_cfg_graph_t* graph = &region->graph;
  const loom_block_t* block = graph->blocks[block_index].block;
  if (!block || arg_index >= block->arg_count) {
    return iree_ok_status();
  }
  loom_value_id_t arg_id = loom_block_arg_id(block, arg_index);
  if (arg_id >= module->values.count) {
    return iree_ok_status();
  }
  loom_type_t type = loom_module_value_type(module, arg_id);

  const uint16_t header_loop =
      loom_cfg_loop_nest_innermost(&region->loops, block_index);
  const bool exits_at_header =
      header_loop != LOOM_CFG_LOOP_NEST_NONE &&
      region->loops.loops[header_loop].header_index == block_index &&
      region->inductions[header_loop].exits_at_header;

  bool has_facts = false;
  loom_value_facts_t control_facts = loom_value_facts_unknown();
  loom_value_facts_mark_cluster_uniform(&control_facts);
  bool all_source_values_match = true;
  loom_value_id_t first_source_value = LOOM_VALUE_ID_INVALID;
  loom_value_facts_t incoming_facts = loom_value_facts_unknown();
  // A forwarding component has one least fixed-point join over its external
  // inputs. Internal edges only circulate that same set and add no facts.
  iree_host_size_t member_count = component ? component->node_count : 1;
  iree_host_size_t component_index =
      component ? region->argument_components[component->nodes[0]]
                : IREE_HOST_SIZE_MAX;
  for (iree_host_size_t member = 0; member < member_count; ++member) {
    if (component) {
      const loom_value_fact_cfg_argument_t* argument =
          &region->arguments[component->nodes[member]];
      block_index = argument->block_index;
      arg_index = argument->argument_index;
      block = graph->blocks[block_index].block;
    }
    loom_cfg_edge_index_span_t predecessor_edges =
        loom_cfg_graph_predecessor_edges(graph, block_index);
    for (iree_host_size_t i = 0; i < predecessor_edges.count; ++i) {
      const loom_cfg_edge_info_t* predecessor_edge =
          loom_cfg_graph_edge(graph, predecessor_edges.values[i]);
      if (predecessor_edge == NULL) {
        continue;
      }
      uint16_t predecessor_index = predecessor_edge->source_block_index;
      if (!loom_cfg_graph_block_is_reachable(graph, predecessor_index)) {
        continue;
      }
      if (exits_at_header &&
          predecessor_edges.values[i] ==
              region->loops.loops[header_loop].backedges.unique_index) {
        continue;
      }
      const loom_block_t* predecessor_block =
          graph->blocks[predecessor_index].block;
      const loom_value_id_t* edge_args = NULL;
      uint16_t edge_arg_count = 0;
      if (!predecessor_block ||
          !loom_cfg_terminator_payload_for_successor(
              predecessor_block->last_op, block, &edge_args, &edge_arg_count) ||
          arg_index >= edge_arg_count) {
        continue;
      }
      // Internal forwarding edges still select dynamic observations, even
      // when their numeric inputs add nothing to the component's value join.
      loom_value_facts_propagate_binary_distribution(
          control_facts,
          loom_value_fact_control_execution(region->control, predecessor_index),
          &control_facts);
      const loom_value_id_t source_value = edge_args[arg_index];
      if (component) {
        iree_host_size_t source =
            loom_value_fact_cfg_region_argument_index(region, source_value);
        if (source != IREE_HOST_SIZE_MAX &&
            region->argument_components[source] == component_index) {
          continue;
        }
      }
      if (first_source_value == LOOM_VALUE_ID_INVALID) {
        first_source_value = source_value;
      } else if (source_value != first_source_value) {
        all_source_values_match = false;
      }
      IREE_RETURN_IF_ERROR(loom_value_fact_table_join_cfg_block_arg_incoming(
          table, module, type, source_value, &has_facts, &incoming_facts));
    }
  }
  if (!has_facts) {
    return iree_ok_status();
  }

  // Control participates in the incoming equation before widening. Comparing
  // an already constrained value against unconstrained inputs would otherwise
  // make an unchanged numeric range appear unstable on every iteration.
  if (!all_source_values_match) {
    loom_value_facts_propagate_binary_distribution(
        incoming_facts, control_facts, &incoming_facts);
    if (loom_value_facts_is_lane_varying(incoming_facts)) {
      loom_value_facts_mark_lane_distribution_for_type(type, &incoming_facts);
    }
  }
  loom_value_facts_t facts = incoming_facts;
  if (widen && !exits_at_header &&
      loom_value_fact_table_has_entry(table, arg_id)) {
    loom_value_facts_t current_facts =
        loom_value_fact_table_lookup(table, arg_id);
    IREE_RETURN_IF_ERROR(loom_value_fact_table_widen_for_type(
        table, module, type, table, current_facts, table, incoming_facts,
        iteration, &facts));
  }
  if (header_loop != LOOM_CFG_LOOP_NEST_NONE &&
      region->inductions[header_loop].value == arg_id) {
    const loom_loop_recurrence_facts_t recurrence =
        loom_value_fact_cfg_induction_facts(table, module,
                                            &region->inductions[header_loop]);
    facts = loom_value_facts_clamp_domain(facts, recurrence.values.range_lo,
                                          recurrence.values.range_hi);
  }
  for (iree_host_size_t member = 0; member < member_count; ++member) {
    loom_value_id_t value_id =
        component ? region->arguments[component->nodes[member]].value_id
                  : arg_id;
    bool changed = false;
    IREE_RETURN_IF_ERROR(loom_value_fact_table_define_block_arg_facts(
        table, module, value_id, facts, &changed));
    if (changed) {
      *out_changed = true;
      if (on_changed) {
        IREE_RETURN_IF_ERROR(on_changed(user_data, value_id));
      }
    }
  }
  return iree_ok_status();
}

// Every direct-forwarding component consumes already joined source components.
// Arithmetic producers remain external inputs and advance in the outer solve.
static iree_status_t loom_value_fact_table_compute_cfg_forwarding(
    loom_value_fact_table_t* table, const loom_module_t* module,
    const loom_value_fact_cfg_region_t* region,
    iree_host_size_t control_flow_component, uint32_t iteration,
    bool* out_changed) {
  IREE_RETURN_IF_ERROR(loom_value_fact_cfg_update_forwarding(
      region, control_flow_component, table->transient_arena));
  const loom_scc_t* blocks =
      &region->control_flow.components.values[control_flow_component];
  for (iree_host_size_t i = 0; i < blocks->node_count; ++i) {
    loom_value_fact_cfg_update_induction(table, module, region,
                                         (uint16_t)blocks->nodes[i]);
  }
  const loom_value_fact_cfg_forwarding_t* partition =
      &region->control_flow.forwarding[control_flow_component];
  for (iree_host_size_t i = 0; i < partition->component_count; ++i) {
    const loom_scc_t* component =
        &region->components[partition->argument_offset + i];
    const loom_value_fact_cfg_argument_t* argument =
        &region->arguments[component->nodes[0]];
    IREE_RETURN_IF_ERROR(loom_value_fact_table_compute_cfg_block_arg(
        table, module, region, component->is_cycle ? component : NULL,
        argument->block_index, argument->argument_index, true, iteration, NULL,
        NULL, out_changed));
  }
  return iree_ok_status();
}

static iree_status_t loom_value_fact_table_compute_cfg_block_args(
    loom_value_fact_table_t* table, const loom_module_t* module,
    const loom_value_fact_cfg_region_t* region, uint16_t block_index,
    bool* out_changed) {
  loom_value_fact_cfg_update_induction(table, module, region, block_index);
  if (block_index == 0) {
    return iree_ok_status();
  }
  const loom_block_t* block = region->graph.blocks[block_index].block;
  for (uint16_t i = 0; i < block->arg_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_value_fact_table_compute_cfg_block_arg(
        table, module, region, NULL, block_index, i, false, 0, NULL, NULL,
        out_changed));
  }
  return iree_ok_status();
}

iree_status_t loom_value_fact_table_update_cfg_block_args(
    loom_value_fact_table_t* table, const loom_module_t* module,
    const loom_value_fact_cfg_region_t* region, uint16_t block_index,
    loom_value_fact_cfg_changed_fn_t on_changed, void* user_data) {
  if (block_index == 0 ||
      !loom_cfg_graph_block_is_reachable(&region->graph, block_index)) {
    return iree_ok_status();
  }
  const loom_block_t* block = region->graph.blocks[block_index].block;
  bool changed = false;
  for (uint16_t i = 0; i < block->arg_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_value_fact_table_compute_cfg_block_arg(
        table, module, region, NULL, block_index, i, false, 0, on_changed,
        user_data, &changed));
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Structured region summaries
//===----------------------------------------------------------------------===//

#define LOOM_VALUE_FACT_CFG_MAX_ITERATIONS 16
#define LOOM_VALUE_FACT_LOOP_MAX_ITERATIONS 8

static loom_op_t* loom_value_fact_region_terminator(loom_region_t* region) {
  if (!region || region->block_count == 0) {
    return NULL;
  }
  loom_block_t* block = loom_region_entry_block(region);
  return block && block->op_count > 0 ? block->last_op : NULL;
}

static iree_status_t loom_value_fact_table_compute_region_tree(
    loom_value_fact_table_t* table, const loom_module_t* module,
    loom_region_t* region, loom_op_t* parent_op);

static bool loom_value_fact_op_summarizes_nested_regions(
    const loom_op_vtable_t* vtable) {
  return vtable && (vtable->loop_like || vtable->region_branch);
}

static iree_status_t loom_value_fact_table_compute_cfg_block_tree(
    loom_value_fact_table_t* table, const loom_module_t* module,
    const loom_block_t* block, bool* out_changed) {
  loom_op_t* op = NULL;
  loom_block_for_each_op((loom_block_t*)block, op) {
    bool op_changed = false;
    IREE_RETURN_IF_ERROR(loom_value_fact_table_compute_op_and_report(
        table, module, op, &op_changed));
    *out_changed = *out_changed || op_changed;
    const loom_op_vtable_t* vtable = loom_op_vtable(module, op);
    if (loom_value_fact_op_summarizes_nested_regions(vtable)) {
      // Structured fact summaries visit their own nested regions.
      continue;
    }
    loom_region_t** regions = loom_op_regions(op);
    for (uint8_t i = 0; i < op->region_count; ++i) {
      IREE_RETURN_IF_ERROR(loom_value_fact_table_compute_region_tree(
          table, module, regions[i], op));
    }
  }
  const loom_value_fact_cfg_region_t* structure =
      loom_value_fact_table_lookup_cfg_region(table, block->parent_region);
  const bool control_changed =
      loom_value_fact_cfg_update_control(table, structure, block->region_index);
  *out_changed = *out_changed || control_changed;
  return iree_ok_status();
}

// Saved facts distinguish changed results from values merely revisited while
// solving a cycle. Extension IDs continue to name payloads in the same table.
typedef struct loom_value_fact_cfg_saved_value_t {
  // Value whose equation is being restarted.
  loom_value_id_t value_id;
  // Facts before this update.
  loom_value_facts_t facts;
} loom_value_fact_cfg_saved_value_t;

typedef struct loom_value_fact_cfg_saved_values_t {
  // Compact records for the component's definitions and nested definitions.
  loom_value_fact_cfg_saved_value_t* values;
  // Number of saved definitions.
  iree_host_size_t count;
  // Allocated record capacity.
  iree_host_size_t capacity;
} loom_value_fact_cfg_saved_values_t;

static iree_status_t loom_value_fact_table_save_cfg_value(
    loom_value_fact_table_t* table, loom_value_id_t value_id,
    iree_arena_allocator_t* arena, loom_value_fact_cfg_saved_values_t* saved) {
  if (value_id == LOOM_VALUE_ID_INVALID) {
    return iree_ok_status();
  }
  if (saved->count == saved->capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        arena, saved->count, saved->count + 1, sizeof(*saved->values),
        &saved->capacity, (void**)&saved->values));
  }
  saved->values[saved->count++] = (loom_value_fact_cfg_saved_value_t){
      .value_id = value_id,
      .facts = loom_value_fact_table_lookup(table, value_id),
  };
  loom_value_fact_table_undefine(table, value_id);
  return iree_ok_status();
}

static iree_status_t loom_value_fact_table_save_cfg_block(
    loom_value_fact_table_t* table, const loom_block_t* block,
    bool include_args, iree_arena_allocator_t* arena,
    loom_value_fact_cfg_saved_values_t* saved) {
  if (include_args) {
    for (uint16_t i = 0; i < block->arg_count; ++i) {
      IREE_RETURN_IF_ERROR(loom_value_fact_table_save_cfg_value(
          table, loom_block_arg_id(block, i), arena, saved));
    }
  }
  loom_op_t* op = NULL;
  loom_block_for_each_op((loom_block_t*)block, op) {
    for (uint16_t i = 0; i < op->result_count; ++i) {
      IREE_RETURN_IF_ERROR(loom_value_fact_table_save_cfg_value(
          table, loom_op_const_results(op)[i], arena, saved));
    }
    loom_region_t** regions = loom_op_regions(op);
    for (uint8_t i = 0; i < op->region_count; ++i) {
      if (!regions[i]) {
        continue;
      }
      loom_block_t* child_block = NULL;
      loom_region_for_each_block(regions[i], child_block) {
        IREE_RETURN_IF_ERROR(loom_value_fact_table_save_cfg_block(
            table, child_block, true, arena, saved));
      }
    }
  }
  return iree_ok_status();
}

// A bounded solve may stop before a changed input reaches every dependent
// value. Reset the complete solve scope, including provisional op results,
// before evaluating it once with unconstrained block arguments. Captures and
// function entry arguments retain the facts established outside this scope.
static iree_status_t loom_value_fact_table_reset_cfg_values(
    loom_value_fact_table_t* table, const loom_module_t* module,
    const loom_value_fact_cfg_saved_values_t* saved) {
  for (iree_host_size_t i = 0; i < saved->count; ++i) {
    loom_value_id_t value_id = saved->values[i].value_id;
    IREE_RETURN_IF_ERROR(loom_value_fact_table_define(
        table, value_id,
        loom_value_fact_table_unknown_for_value(module, value_id)));
  }
  return iree_ok_status();
}

iree_status_t loom_value_fact_table_recompute_cfg_component(
    loom_value_fact_table_t* table, const loom_module_t* module,
    const loom_value_fact_cfg_region_t* region, const loom_scc_t* component,
    iree_arena_allocator_t* scratch_arena,
    loom_value_fact_cfg_changed_fn_t on_changed, void* user_data) {
  iree_host_size_t component_index =
      component - region->control_flow.components.values;
  IREE_RETURN_IF_ERROR(loom_value_fact_cfg_update_forwarding(
      region, component_index, scratch_arena));
  const iree_host_size_t* blocks = component->nodes;
  loom_value_fact_cfg_saved_values_t saved = {0};
  for (iree_host_size_t i = 0; i < component->node_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_value_fact_table_save_cfg_block(
        table, region->graph.blocks[blocks[i]].block, blocks[i] != 0,
        scratch_arena, &saved));
  }
  loom_value_fact_cfg_seed_control(table, region, component);
  bool converged = false;
  for (uint32_t iteration = 0; iteration < LOOM_VALUE_FACT_CFG_MAX_ITERATIONS;
       ++iteration) {
    bool changed = false;
    // Seed producers in execution order before joining the whole forwarding
    // partition. An inner loop's initializer can live inside this component.
    if (iteration != 0) {
      IREE_RETURN_IF_ERROR(loom_value_fact_table_compute_cfg_forwarding(
          table, module, region, component_index, iteration, &changed));
    }
    for (iree_host_size_t i = 0; i < component->node_count; ++i) {
      uint16_t block_index = blocks[i];
      if (iteration == 0) {
        IREE_RETURN_IF_ERROR(loom_value_fact_table_compute_cfg_block_args(
            table, module, region, block_index, &changed));
      }
      IREE_RETURN_IF_ERROR(loom_value_fact_table_compute_cfg_block_tree(
          table, module, region->graph.blocks[block_index].block, &changed));
    }
    if (!changed) {
      converged = true;
      break;
    }
  }
  if (!converged) {
    IREE_RETURN_IF_ERROR(
        loom_value_fact_table_reset_cfg_values(table, module, &saved));
    loom_value_fact_cfg_seed_control(table, region, component);
    for (iree_host_size_t i = 0; i < component->node_count; ++i) {
      bool changed = false;
      IREE_RETURN_IF_ERROR(loom_value_fact_table_compute_cfg_block_tree(
          table, module, region->graph.blocks[blocks[i]].block, &changed));
    }
  }
  for (iree_host_size_t i = 0; i < saved.count; ++i) {
    loom_value_id_t value_id = saved.values[i].value_id;
    if (!loom_value_fact_table_facts_equal_for_type(
            module, loom_module_value_type(module, value_id), table,
            saved.values[i].facts, table,
            loom_value_fact_table_lookup(table, value_id))) {
      IREE_RETURN_IF_ERROR(on_changed(user_data, value_id));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_value_fact_table_compute_cfg_region_tree(
    loom_value_fact_table_t* table, const loom_module_t* module,
    loom_region_t* region, loom_op_t* parent_op) {
  const loom_value_fact_cfg_region_t* structure = NULL;
  IREE_RETURN_IF_ERROR(loom_value_fact_table_get_or_build_cfg_region(
      table, module, region, &structure));
  const loom_cfg_graph_t* graph = &structure->graph;
  uint32_t* visited_components = NULL;
  if (structure->control_flow.components.count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        table->transient_arena, structure->control_flow.components.count,
        sizeof(*visited_components), (void**)&visited_components));
    memset(
        visited_components, 0,
        structure->control_flow.components.count * sizeof(*visited_components));
  }

  if (region->block_count == 0) {
    return iree_ok_status();
  }
  loom_block_t* entry_block = loom_region_entry_block(region);
  IREE_RETURN_IF_ERROR(loom_value_fact_table_seed_block_args(
      table, module, entry_block, parent_op));

  bool converged = false;
  for (uint32_t iteration = 0; iteration < LOOM_VALUE_FACT_CFG_MAX_ITERATIONS;
       ++iteration) {
    bool changed = false;
    for (iree_host_size_t i = 0; i < graph->reverse_postorder.count; ++i) {
      const uint16_t block_index = graph->reverse_postorder.values[i];
      const loom_cfg_block_info_t* block_info = &graph->blocks[block_index];
      const loom_block_t* block = block_info->block;
      // The initial sweep must reach in-cycle producers before their users.
      // Otherwise an unseeded nested loop publishes unknown feedback before
      // its initializer has contributed facts, permanently losing precision.
      if (block_info->component_is_cyclic && iteration != 0) {
        const iree_host_size_t component_index = block_info->component;
        if (visited_components[component_index] != iteration + 1) {
          visited_components[component_index] = iteration + 1;
          IREE_RETURN_IF_ERROR(loom_value_fact_table_compute_cfg_forwarding(
              table, module, structure, component_index, iteration, &changed));
        }
      } else {
        IREE_RETURN_IF_ERROR(loom_value_fact_table_compute_cfg_block_args(
            table, module, structure, block_index, &changed));
      }
      IREE_RETURN_IF_ERROR(loom_value_fact_table_compute_cfg_block_tree(
          table, module, block, &changed));
    }
    if (!changed) {
      converged = true;
      break;
    }
  }
  if (!converged) {
    loom_value_fact_cfg_saved_values_t saved = {0};
    for (iree_host_size_t i = 0; i < graph->reverse_postorder.count; ++i) {
      uint16_t block_index = graph->reverse_postorder.values[i];
      IREE_RETURN_IF_ERROR(loom_value_fact_table_save_cfg_block(
          table, graph->blocks[block_index].block, block_index != 0,
          table->transient_arena, &saved));
    }
    IREE_RETURN_IF_ERROR(
        loom_value_fact_table_reset_cfg_values(table, module, &saved));
    loom_value_fact_cfg_seed_control(table, structure, NULL);
    for (iree_host_size_t i = 0; i < graph->reverse_postorder.count; ++i) {
      uint16_t block_index = graph->reverse_postorder.values[i];
      bool changed = false;
      IREE_RETURN_IF_ERROR(loom_value_fact_table_compute_cfg_block_tree(
          table, module, graph->blocks[block_index].block, &changed));
    }
  }
  uint16_t changed_block = 0;
  while (loom_value_fact_control_take_changed_block(structure->control,
                                                    &changed_block)) {
  }
  return iree_ok_status();
}

static iree_status_t loom_value_fact_table_compute_region_tree(
    loom_value_fact_table_t* table, const loom_module_t* module,
    loom_region_t* region, loom_op_t* parent_op) {
  if (!region) {
    return iree_ok_status();
  }
  loom_value_facts_t temporal_scope = loom_value_facts_unknown();
  if (!parent_op || parent_op == table->context.function.op) {
    loom_value_facts_mark_cluster_uniform(&temporal_scope);
  } else {
    temporal_scope = loom_value_fact_table_block_temporal_scope(
        table, parent_op->parent_block);
  }
  IREE_RETURN_IF_ERROR(loom_value_fact_table_set_region_temporal_scope(
      table, region, temporal_scope));
  if (iree_any_bit_set(region->flags, LOOM_REGION_INSTANCE_FLAG_CFG)) {
    return loom_value_fact_table_compute_cfg_region_tree(table, module, region,
                                                         parent_op);
  }
  loom_block_t* block = NULL;
  loom_region_for_each_block(region, block) {
    IREE_RETURN_IF_ERROR(
        loom_value_fact_table_seed_block_args(table, module, block, parent_op));
    loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      IREE_RETURN_IF_ERROR(loom_value_fact_table_compute_op(table, module, op));
      const loom_op_vtable_t* vtable = loom_op_vtable(module, op);
      if (loom_value_fact_op_summarizes_nested_regions(vtable)) {
        // Structured fact summaries visit their own nested regions.
        continue;
      }
      loom_region_t** regions = loom_op_regions(op);
      for (uint8_t i = 0; i < op->region_count; ++i) {
        IREE_RETURN_IF_ERROR(loom_value_fact_table_compute_region_tree(
            table, module, regions[i], op));
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_value_fact_table_seed_projected_func_args(
    loom_value_fact_table_t* table, const loom_module_t* module,
    loom_func_like_t function, loom_region_t* region, loom_op_t* parent_op) {
  if (!loom_func_like_isa(function) || parent_op != function.op || !region ||
      region->block_count == 0) {
    return iree_ok_status();
  }

  uint8_t region_index = LOOM_REGION_INDEX_NONE;
  for (uint8_t i = 0; i < loom_func_like_region_count(function); ++i) {
    if (loom_func_like_region(function, i) == region) {
      region_index = i;
      break;
    }
  }
  if (region_index == LOOM_REGION_INDEX_NONE ||
      !loom_func_like_region_projects_args(module, function, region_index)) {
    return iree_ok_status();
  }

  uint16_t argument_count = 0;
  const loom_value_id_t* arguments =
      loom_func_like_arg_ids(function, &argument_count);
  loom_block_t* entry_block = loom_region_entry_block(region);
  if (!arguments || !entry_block) {
    return iree_ok_status();
  }

  uint16_t projected_count = iree_min(argument_count, entry_block->arg_count);
  for (uint16_t i = 0; i < projected_count; ++i) {
    loom_value_facts_t facts =
        loom_value_fact_table_lookup(table, arguments[i]);
    if (loom_value_facts_is_unknown(facts)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_value_fact_table_define(
        table, loom_block_arg_id(entry_block, i), facts));
  }
  return iree_ok_status();
}

static bool loom_value_fact_counted_loop_proven_zero_trip(
    loom_value_facts_t lower_bound, loom_value_facts_t upper_bound,
    loom_value_facts_t step) {
  if (loom_value_facts_is_float(lower_bound) ||
      loom_value_facts_is_float(upper_bound) ||
      loom_value_facts_is_float(step) || !loom_value_facts_is_positive(step)) {
    return false;
  }
  return lower_bound.range_lo >= upper_bound.range_hi;
}

static bool loom_value_fact_counted_loop_proven_at_least_one_trip(
    loom_value_facts_t lower_bound, loom_value_facts_t upper_bound,
    loom_value_facts_t step) {
  if (loom_value_facts_is_float(lower_bound) ||
      loom_value_facts_is_float(upper_bound) ||
      loom_value_facts_is_float(step) || !loom_value_facts_is_positive(step)) {
    return false;
  }
  return lower_bound.range_hi < upper_bound.range_lo;
}

static uint16_t loom_value_fact_loop_carried_arg_offset(loom_loop_like_t loop) {
  return loop.vtable->iv_block_arg_index == LOOM_BLOCK_ARG_INDEX_NONE
             ? 0
             : (uint16_t)loop.vtable->iv_block_arg_index + 1;
}

static uint16_t loom_value_fact_loop_state_count(loom_loop_like_t loop) {
  loom_value_slice_t iter_args = loom_loop_like_iter_args(loop);
  uint16_t count = iter_args.count;
  if (count > loop.op->result_count) {
    count = loop.op->result_count;
  }
  return count;
}

static iree_status_t loom_value_fact_table_allocate_fact_array(
    loom_value_fact_table_t* table, iree_host_size_t count,
    loom_value_facts_t** out_facts) {
  *out_facts = NULL;
  if (count == 0) {
    return iree_ok_status();
  }
  return iree_arena_allocate_array(table->transient_arena, count,
                                   sizeof(loom_value_facts_t),
                                   (void**)out_facts);
}

static iree_status_t loom_value_fact_table_initialize_loop_state(
    loom_value_fact_table_t* table, const loom_module_t* module,
    loom_loop_like_t loop, loom_value_facts_t** out_init_facts,
    loom_value_facts_t** out_current_facts, loom_type_t** out_types) {
  uint16_t count = loom_value_fact_loop_state_count(loop);
  IREE_RETURN_IF_ERROR(
      loom_value_fact_table_allocate_fact_array(table, count, out_init_facts));
  IREE_RETURN_IF_ERROR(loom_value_fact_table_allocate_fact_array(
      table, count, out_current_facts));
  *out_types = NULL;
  if (count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        table->transient_arena, count, sizeof(loom_type_t), (void**)out_types));
  }

  loom_value_slice_t iter_args = loom_loop_like_iter_args(loop);
  for (uint16_t i = 0; i < count; ++i) {
    loom_value_id_t value_id = iter_args.values[i];
    (*out_init_facts)[i] = loom_value_fact_table_lookup(table, value_id);
    (*out_current_facts)[i] = (*out_init_facts)[i];
    (*out_types)[i] = value_id < module->values.count
                          ? loom_module_value_type(module, value_id)
                          : loom_type_none();
  }
  return iree_ok_status();
}

// Lack of change in one slot is not convergence: a long carried queue can
// delay another slot's change past the solve budget. Unknown state bounds all
// iterations and lets the final body evaluation publish sound derived facts.
static void loom_value_fact_loop_forget_state(uint16_t count,
                                              loom_value_facts_t* facts) {
  for (uint16_t i = 0; i < count; ++i) {
    facts[i] = loom_value_facts_unknown();
  }
}

// Direct state forwarding is solved in dependency order. Closed rotations
// contain only their initial values; computed producers remain outer feedback.
typedef struct loom_value_fact_loop_forwarding_t {
  // State-slot order, or NULL for a trivial or not-yet-built loop body.
  uint16_t* order;
  // Earlier solved source slot, or UINT16_MAX to consume the evaluated yield.
  uint16_t* sources;
} loom_value_fact_loop_forwarding_t;

static uint16_t loom_value_fact_loop_forwarded_argument(
    const loom_module_t* module, loom_value_id_t value_id,
    const loom_block_t* block, uint16_t argument_offset, uint16_t count) {
  const loom_value_t* value = loom_module_value(module, value_id);
  if (!loom_value_is_block_arg(value) || loom_value_def_block(value) != block) {
    return UINT16_MAX;
  }
  const uint16_t index = loom_value_def_index(value);
  return index >= argument_offset && index - argument_offset < count
             ? index - argument_offset
             : UINT16_MAX;
}

static iree_status_t loom_value_fact_loop_visit_forwarded_argument(
    void* user_data, iree_host_size_t node,
    loom_scc_successor_callback_t successor) {
  const uint16_t* sources = user_data;
  return sources[node] != UINT16_MAX
             ? successor.fn(successor.user_data, sources[node])
             : iree_ok_status();
}

static iree_status_t loom_value_fact_table_initialize_loop_forwarding(
    loom_value_fact_table_t* table, const loom_module_t* module,
    loom_loop_like_t loop, const loom_type_t* types,
    loom_value_facts_t* current_facts,
    loom_value_fact_loop_forwarding_t* out_forwarding) {
  *out_forwarding = (loom_value_fact_loop_forwarding_t){0};
  const uint16_t count = loom_value_fact_loop_state_count(loop);
  if (count < 2) {
    return iree_ok_status();
  }
  loom_region_t* body = loom_loop_like_body(loop);
  const loom_op_t* yield = loom_value_fact_region_terminator(body);
  if (!yield || yield->operand_count < count) {
    return iree_ok_status();
  }
  loom_region_t* condition_region = loom_loop_like_condition_region(loop);
  const loom_op_t* condition =
      loom_value_fact_region_terminator(condition_region);
  if (condition_region &&
      (!condition || condition->operand_count < count + 1)) {
    return iree_ok_status();
  }
  const loom_block_t* body_block = loom_region_const_entry_block(body);
  const uint16_t argument_offset =
      loom_value_fact_loop_carried_arg_offset(loop);
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      table->transient_arena, count, 2 * sizeof(uint16_t),
      (void**)&out_forwarding->order));
  out_forwarding->sources = out_forwarding->order + count;
  for (uint16_t i = 0; i < count; ++i) {
    uint16_t source = loom_value_fact_loop_forwarded_argument(
        module, loom_op_const_operands(yield)[i], body_block, argument_offset,
        count);
    if (source != UINT16_MAX && condition) {
      source = loom_value_fact_loop_forwarded_argument(
          module, loom_op_const_operands(condition)[1 + source],
          loom_region_const_entry_block(condition_region), 0, count);
    }
    out_forwarding->sources[i] = source;
  }
  const loom_scc_graph_t graph = {
      .node_count = count,
      .visit_successors = loom_scc_visit_successors_callback_make(
          loom_value_fact_loop_visit_forwarded_argument,
          out_forwarding->sources),
  };
  loom_scc_list_t components = {0};
  IREE_RETURN_IF_ERROR(
      loom_scc_compute(&graph, NULL, table->transient_arena, &components));
  uint16_t ordinal = 0;
  for (iree_host_size_t i = 0; i < components.count; ++i) {
    const loom_scc_t* component = &components.values[i];
    for (iree_host_size_t j = 0; j < component->node_count; ++j) {
      out_forwarding->order[ordinal++] = component->nodes[j];
    }
    if (!component->is_cycle) {
      continue;
    }
    const loom_type_t type = types[component->nodes[0]];
    loom_value_facts_t facts = current_facts[component->nodes[0]];
    for (iree_host_size_t j = 1; j < component->node_count; ++j) {
      IREE_RETURN_IF_ERROR(loom_value_fact_table_meet_for_type(
          table, module, type, table, facts, table,
          current_facts[component->nodes[j]], &facts));
    }
    for (iree_host_size_t j = 0; j < component->node_count; ++j) {
      const iree_host_size_t slot = component->nodes[j];
      current_facts[slot] = facts;
      // Every cycle member has the same complete invariant. Its yielded
      // facts consume that invariant rather than a partially visited peer.
      out_forwarding->sources[slot] = UINT16_MAX;
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_value_fact_table_define_loop_entry_args(
    loom_value_fact_table_t* table, const loom_module_t* module,
    loom_region_t* region, uint16_t arg_offset, const loom_value_facts_t* facts,
    uint16_t count) {
  if (!region || region->block_count == 0) {
    return iree_ok_status();
  }
  loom_block_t* block = loom_region_entry_block(region);
  for (uint16_t i = 0; i < count; ++i) {
    uint16_t arg_index = arg_offset + i;
    if (arg_index >= block->arg_count) {
      break;
    }
    loom_value_id_t arg_id = loom_block_arg_id(block, arg_index);
    if (arg_id >= module->values.count) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_value_fact_table_define(table, arg_id, facts[i]));
  }
  return iree_ok_status();
}

static void loom_value_fact_table_collect_terminator_operands(
    loom_value_fact_table_t* table, const loom_op_t* terminator,
    uint16_t operand_offset, loom_value_facts_t* facts, uint16_t count) {
  const loom_value_id_t* operands =
      terminator ? loom_op_const_operands(terminator) : NULL;
  for (uint16_t i = 0; i < count; ++i) {
    uint16_t operand_index = operand_offset + i;
    facts[i] =
        (terminator && operand_index < terminator->operand_count)
            ? loom_value_fact_table_lookup(table, operands[operand_index])
            : loom_value_facts_unknown();
  }
}

static iree_status_t loom_value_fact_table_join_loop_backedge(
    loom_value_fact_table_t* table, const loom_module_t* module,
    const loom_type_t* types, const loom_value_facts_t* init_facts,
    const loom_value_facts_t* yielded_facts, const loom_value_facts_t* current,
    uint16_t count, const loom_value_fact_loop_forwarding_t* forwarding,
    uint32_t iteration, loom_value_facts_t* next, bool* out_changed) {
  *out_changed = false;
  for (uint16_t ordinal = 0; ordinal < count; ++ordinal) {
    const uint16_t i = forwarding->order ? forwarding->order[ordinal] : ordinal;
    const uint16_t source =
        forwarding->sources ? forwarding->sources[i] : UINT16_MAX;
    const loom_value_facts_t yielded =
        source != UINT16_MAX ? next[source] : yielded_facts[i];
    loom_value_facts_t joined = loom_value_facts_unknown();
    IREE_RETURN_IF_ERROR(loom_value_fact_table_meet_for_type(
        table, module, types[i], table, init_facts[i], table, yielded,
        &joined));
    IREE_RETURN_IF_ERROR(loom_value_fact_table_widen_for_type(
        table, module, types[i], table, current[i], table, joined, iteration,
        &next[i]));
    if (!loom_value_fact_table_facts_equal_for_type(
            module, types[i], table, current[i], table, next[i])) {
      *out_changed = true;
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_value_fact_table_define_region_results(
    loom_value_fact_table_t* table, const loom_module_t* module, loom_op_t* op,
    loom_value_facts_t* result_facts, uint16_t result_count,
    bool* out_changed) {
  const loom_value_id_t* results = loom_op_const_results(op);
  if (result_count < op->result_count) {
    loom_value_facts_t* full_result_facts = NULL;
    IREE_RETURN_IF_ERROR(loom_value_fact_table_allocate_fact_array(
        table, op->result_count, &full_result_facts));
    for (uint16_t i = 0; i < op->result_count; ++i) {
      full_result_facts[i] =
          i < result_count ? result_facts[i] : loom_value_facts_unknown();
    }
    result_facts = full_result_facts;
    result_count = op->result_count;
  }
  for (uint16_t i = 0; i < op->result_count; ++i) {
    loom_value_id_t result = results[i];
    if (result == LOOM_VALUE_ID_INVALID || result >= module->values.count) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_value_fact_table_seed_type_extent_facts(
        table, module, loom_module_value_type(module, result), results,
        op->result_count, result_facts, out_changed));
  }
  for (uint16_t i = 0; i < op->result_count; ++i) {
    loom_value_id_t result = results[i];
    if (result == LOOM_VALUE_ID_INVALID || result >= module->values.count) {
      continue;
    }
    loom_value_facts_t facts =
        i < result_count ? result_facts[i] : loom_value_facts_unknown();
    loom_type_t type = loom_module_value_type(module, result);
    if (out_changed &&
        (!loom_value_fact_table_has_entry(table, result) ||
         !loom_value_fact_table_facts_equal_for_type(
             module, type, table, loom_value_fact_table_lookup(table, result),
             table, facts))) {
      *out_changed = true;
    }
    IREE_RETURN_IF_ERROR(loom_value_fact_table_define(table, result, facts));
  }
  return iree_ok_status();
}

typedef struct loom_value_fact_region_branch_result_state_t {
  // Meet of facts yielded for this result by every visited branch region.
  loom_value_facts_t facts;
  // Value yielded by the first branch region.
  loom_value_id_t first_source_value;
  // Whether every branch yields the same SSA value as the first branch.
  bool all_source_values_match;
} loom_value_fact_region_branch_result_state_t;

static iree_status_t loom_value_fact_table_compute_region_branch_summary(
    loom_value_fact_table_t* table, const loom_module_t* module, loom_op_t* op,
    bool* out_changed) {
  loom_region_branch_t branch = loom_region_branch_cast(module, op);
  IREE_ASSERT(loom_region_branch_isa(branch));

  loom_value_fact_region_branch_result_state_t* result_states = NULL;
  if (op->result_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        table->transient_arena, op->result_count,
        sizeof(loom_value_fact_region_branch_result_state_t),
        (void**)&result_states));
  }

  uint8_t visited_region_count = 0;
  const loom_value_id_t* results = loom_op_const_results(op);
  for (uint8_t region_index = 0; region_index < op->region_count;
       ++region_index) {
    loom_region_t* region =
        loom_region_branch_region(module, branch, region_index);
    if (!region) {
      continue;
    }
    IREE_RETURN_IF_ERROR(
        loom_value_fact_table_compute_region_tree(table, module, region, op));
    loom_op_t* terminator =
        loom_region_branch_region_terminator(module, branch, region_index);
    IREE_ASSERT(terminator);
    if (op->result_count == 0) {
      ++visited_region_count;
      continue;
    }

    IREE_ASSERT_EQ(terminator->operand_count, op->result_count);
    const loom_value_id_t* yielded_values = loom_op_const_operands(terminator);
    for (uint16_t result_index = 0; result_index < op->result_count;
         ++result_index) {
      const loom_value_id_t yielded_value = yielded_values[result_index];
      loom_value_fact_region_branch_result_state_t* state =
          &result_states[result_index];
      const loom_value_facts_t yielded_facts =
          loom_value_fact_table_lookup(table, yielded_value);
      if (visited_region_count == 0) {
        state->facts = yielded_facts;
        state->first_source_value = yielded_value;
        state->all_source_values_match = true;
        continue;
      }

      if (yielded_value != state->first_source_value) {
        state->all_source_values_match = false;
      }
      loom_value_facts_t joined_facts = loom_value_facts_unknown();
      const loom_type_t result_type =
          loom_module_value_type(module, results[result_index]);
      IREE_RETURN_IF_ERROR(loom_value_fact_table_meet_for_type(
          table, module, result_type, table, state->facts, table, yielded_facts,
          &joined_facts));
      state->facts = joined_facts;
    }
    ++visited_region_count;
  }

  if (op->result_count == 0) {
    return iree_ok_status();
  }
  IREE_ASSERT_GT(visited_region_count, 0);

  loom_value_facts_t* result_facts = NULL;
  IREE_RETURN_IF_ERROR(loom_value_fact_table_facts_scratch(
      table, op->result_count, &result_facts));
  const loom_value_facts_t selector_facts =
      loom_value_fact_table_lookup(table, loom_region_branch_selector(branch));
  for (uint16_t result_index = 0; result_index < op->result_count;
       ++result_index) {
    loom_value_facts_t facts = result_states[result_index].facts;
    if (!result_states[result_index].all_source_values_match) {
      loom_value_facts_propagate_binary_distribution(facts, selector_facts,
                                                     &facts);
      if (loom_value_facts_is_lane_varying(facts)) {
        loom_value_facts_mark_lane_distribution_for_type(
            loom_module_value_type(module, results[result_index]), &facts);
      }
    }
    result_facts[result_index] = facts;
  }
  return loom_value_fact_table_define_region_results(
      table, module, op, result_facts, op->result_count, out_changed);
}

static iree_status_t loom_value_fact_table_compute_counted_loop_summary(
    loom_value_fact_table_t* table, const loom_module_t* module,
    loom_loop_like_t loop, bool* out_changed) {
  loom_region_t* body = loom_loop_like_body(loop);
  if (!body || body->block_count == 0) {
    return iree_ok_status();
  }
  uint16_t count = loom_value_fact_loop_state_count(loop);
  if (count == 0) {
    return loom_value_fact_table_compute_region_tree(table, module, body,
                                                     loop.op);
  }

  loom_value_facts_t* init_facts = NULL;
  loom_value_facts_t* current_facts = NULL;
  loom_type_t* types = NULL;
  IREE_RETURN_IF_ERROR(loom_value_fact_table_initialize_loop_state(
      table, module, loop, &init_facts, &current_facts, &types));
  loom_value_fact_loop_forwarding_t forwarding;
  IREE_RETURN_IF_ERROR(loom_value_fact_table_initialize_loop_forwarding(
      table, module, loop, types, current_facts, &forwarding));

  loom_value_facts_t* yielded_facts = NULL;
  loom_value_facts_t* next_facts = NULL;
  IREE_RETURN_IF_ERROR(
      loom_value_fact_table_allocate_fact_array(table, count, &yielded_facts));
  IREE_RETURN_IF_ERROR(
      loom_value_fact_table_allocate_fact_array(table, count, &next_facts));

  uint16_t carried_arg_offset = loom_value_fact_loop_carried_arg_offset(loop);
  bool converged = false;
  for (uint32_t iteration = 0; iteration < LOOM_VALUE_FACT_LOOP_MAX_ITERATIONS;
       ++iteration) {
    IREE_RETURN_IF_ERROR(loom_value_fact_table_define_loop_entry_args(
        table, module, body, carried_arg_offset, current_facts, count));
    IREE_RETURN_IF_ERROR(loom_value_fact_table_compute_region_tree(
        table, module, body, loop.op));
    loom_op_t* yield = loom_value_fact_region_terminator(body);
    loom_value_fact_table_collect_terminator_operands(
        table, yield, /*operand_offset=*/0, yielded_facts, count);

    bool changed = false;
    IREE_RETURN_IF_ERROR(loom_value_fact_table_join_loop_backedge(
        table, module, types, init_facts, yielded_facts, current_facts, count,
        &forwarding, iteration, next_facts, &changed));
    memcpy(current_facts, next_facts, count * sizeof(loom_value_facts_t));
    if (!changed) {
      converged = true;
      break;
    }
  }
  if (!converged) {
    loom_value_fact_loop_forget_state(count, current_facts);
    IREE_RETURN_IF_ERROR(loom_value_fact_table_define_loop_entry_args(
        table, module, body, carried_arg_offset, current_facts, count));
    IREE_RETURN_IF_ERROR(loom_value_fact_table_compute_region_tree(
        table, module, body, loop.op));
    loom_op_t* yield = loom_value_fact_region_terminator(body);
    loom_value_fact_table_collect_terminator_operands(
        table, yield, /*operand_offset=*/0, yielded_facts, count);
  }

  loom_value_facts_t lower_bound =
      loom_value_fact_table_lookup(table, loom_loop_like_lower_bound(loop));
  loom_value_facts_t upper_bound =
      loom_value_fact_table_lookup(table, loom_loop_like_upper_bound(loop));
  loom_value_facts_t step =
      loom_value_fact_table_lookup(table, loom_loop_like_step(loop));
  bool zero_trip = loom_value_fact_counted_loop_proven_zero_trip(
      lower_bound, upper_bound, step);
  bool at_least_one_trip =
      loom_value_fact_counted_loop_proven_at_least_one_trip(lower_bound,
                                                            upper_bound, step);

  loom_value_facts_t* result_facts = NULL;
  IREE_RETURN_IF_ERROR(
      loom_value_fact_table_allocate_fact_array(table, count, &result_facts));
  loom_value_facts_t trip_facts = loom_value_facts_unknown();
  loom_value_facts_propagate_ternary_distribution(lower_bound, upper_bound,
                                                  step, &trip_facts);
  const loom_block_t* body_block = loom_region_const_entry_block(body);
  const loom_op_t* yield = loom_value_fact_region_terminator(body);
  // Rewriter builders finalize the new loop before moving its body. Missing
  // terminators contribute unknown facts until the region edit is published.
  const loom_value_id_t* yielded_values =
      yield ? loom_op_const_operands(yield) : NULL;
  const loom_value_slice_t initial_values = loom_loop_like_iter_args(loop);
  for (uint16_t i = 0; i < count; ++i) {
    if (zero_trip) {
      result_facts[i] = init_facts[i];
    } else if (at_least_one_trip) {
      result_facts[i] = yielded_facts[i];
    } else {
      IREE_RETURN_IF_ERROR(loom_value_fact_table_meet_for_type(
          table, module, types[i], table, init_facts[i], table,
          yielded_facts[i], &result_facts[i]));
    }
    // Uniform state at one iteration need not be uniform after invocations
    // exit on different iterations. Directly unchanged state is independent
    // of the trip count, as is a result proven exact by the scalar domain.
    const bool unchanged =
        yielded_values &&
        (yielded_values[i] ==
             loom_block_arg_id(body_block, carried_arg_offset + i) ||
         yielded_values[i] == initial_values.values[i]);
    if (!zero_trip && !unchanged) {
      loom_value_facts_propagate_binary_distribution(
          result_facts[i], trip_facts, &result_facts[i]);
      if (loom_value_facts_is_lane_varying(result_facts[i])) {
        loom_value_facts_mark_lane_distribution_for_type(types[i],
                                                         &result_facts[i]);
      }
    }
  }
  return loom_value_fact_table_define_region_results(
      table, module, loop.op, result_facts, count, out_changed);
}

static iree_status_t loom_value_fact_table_compute_condition_loop_summary(
    loom_value_fact_table_t* table, const loom_module_t* module,
    loom_loop_like_t loop, bool* out_changed) {
  loom_region_t* condition_region = loom_loop_like_condition_region(loop);
  loom_region_t* body = loom_loop_like_body(loop);
  if (!condition_region || !body || condition_region->block_count == 0 ||
      body->block_count == 0) {
    return iree_ok_status();
  }
  uint16_t count = loom_value_fact_loop_state_count(loop);
  if (count == 0) {
    IREE_RETURN_IF_ERROR(loom_value_fact_table_compute_region_tree(
        table, module, condition_region, loop.op));
    return loom_value_fact_table_compute_region_tree(table, module, body,
                                                     loop.op);
  }

  loom_value_facts_t* init_facts = NULL;
  loom_value_facts_t* current_facts = NULL;
  loom_type_t* types = NULL;
  IREE_RETURN_IF_ERROR(loom_value_fact_table_initialize_loop_state(
      table, module, loop, &init_facts, &current_facts, &types));

  loom_value_fact_loop_forwarding_t forwarding;
  IREE_RETURN_IF_ERROR(loom_value_fact_table_initialize_loop_forwarding(
      table, module, loop, types, current_facts, &forwarding));

  loom_value_facts_t* forwarded_facts = NULL;
  loom_value_facts_t* yielded_facts = NULL;
  loom_value_facts_t* next_facts = NULL;
  IREE_RETURN_IF_ERROR(loom_value_fact_table_allocate_fact_array(
      table, count, &forwarded_facts));
  IREE_RETURN_IF_ERROR(
      loom_value_fact_table_allocate_fact_array(table, count, &yielded_facts));
  IREE_RETURN_IF_ERROR(
      loom_value_fact_table_allocate_fact_array(table, count, &next_facts));

  bool converged = false;
  for (uint32_t iteration = 0; iteration < LOOM_VALUE_FACT_LOOP_MAX_ITERATIONS;
       ++iteration) {
    IREE_RETURN_IF_ERROR(loom_value_fact_table_define_loop_entry_args(
        table, module, condition_region, /*arg_offset=*/0, current_facts,
        count));
    IREE_RETURN_IF_ERROR(loom_value_fact_table_compute_region_tree(
        table, module, condition_region, loop.op));
    loom_op_t* condition = loom_value_fact_region_terminator(condition_region);
    loom_value_fact_table_collect_terminator_operands(
        table, condition, /*operand_offset=*/1, forwarded_facts, count);

    IREE_RETURN_IF_ERROR(loom_value_fact_table_define_loop_entry_args(
        table, module, body, /*arg_offset=*/0, forwarded_facts, count));
    IREE_RETURN_IF_ERROR(loom_value_fact_table_compute_region_tree(
        table, module, body, loop.op));
    loom_op_t* yield = loom_value_fact_region_terminator(body);
    loom_value_fact_table_collect_terminator_operands(
        table, yield, /*operand_offset=*/0, yielded_facts, count);

    bool changed = false;
    IREE_RETURN_IF_ERROR(loom_value_fact_table_join_loop_backedge(
        table, module, types, init_facts, yielded_facts, current_facts, count,
        &forwarding, iteration, next_facts, &changed));
    memcpy(current_facts, next_facts, count * sizeof(loom_value_facts_t));
    if (!changed) {
      converged = true;
      break;
    }
  }
  if (!converged) {
    loom_value_fact_loop_forget_state(count, current_facts);
    IREE_RETURN_IF_ERROR(loom_value_fact_table_define_loop_entry_args(
        table, module, condition_region, /*arg_offset=*/0, current_facts,
        count));
    IREE_RETURN_IF_ERROR(loom_value_fact_table_compute_region_tree(
        table, module, condition_region, loop.op));
    loom_op_t* condition = loom_value_fact_region_terminator(condition_region);
    loom_value_fact_table_collect_terminator_operands(
        table, condition, /*operand_offset=*/1, forwarded_facts, count);

    IREE_RETURN_IF_ERROR(loom_value_fact_table_define_loop_entry_args(
        table, module, body, /*arg_offset=*/0, forwarded_facts, count));
    IREE_RETURN_IF_ERROR(loom_value_fact_table_compute_region_tree(
        table, module, body, loop.op));
    loom_op_t* yield = loom_value_fact_region_terminator(body);
    loom_value_fact_table_collect_terminator_operands(
        table, yield, /*operand_offset=*/0, yielded_facts, count);
  }

  const loom_op_t* condition =
      loom_value_fact_region_terminator(condition_region);
  const loom_value_id_t* condition_values =
      condition ? loom_op_const_operands(condition) : NULL;
  const loom_value_facts_t condition_facts =
      condition_values
          ? loom_value_fact_table_lookup(table, condition_values[0])
          : loom_value_facts_unknown();
  const loom_op_t* yield = loom_value_fact_region_terminator(body);
  const loom_value_id_t* yielded_values =
      yield ? loom_op_const_operands(yield) : NULL;
  const loom_block_t* condition_block =
      loom_region_const_entry_block(condition_region);
  const loom_block_t* body_block = loom_region_const_entry_block(body);
  const loom_value_slice_t initial_values = loom_loop_like_iter_args(loop);
  for (uint16_t i = 0; i < count; ++i) {
    const bool unchanged =
        condition_values && yielded_values &&
        (condition_values[1 + i] == initial_values.values[i] ||
         (condition_values[1 + i] == loom_block_arg_id(condition_block, i) &&
          (yielded_values[i] == loom_block_arg_id(body_block, i) ||
           yielded_values[i] == initial_values.values[i])));
    if (!unchanged) {
      loom_value_facts_propagate_binary_distribution(
          forwarded_facts[i], condition_facts, &forwarded_facts[i]);
      if (loom_value_facts_is_lane_varying(forwarded_facts[i])) {
        loom_value_facts_mark_lane_distribution_for_type(types[i],
                                                         &forwarded_facts[i]);
      }
    }
  }
  return loom_value_fact_table_define_region_results(
      table, module, loop.op, forwarded_facts, count, out_changed);
}

static iree_status_t loom_value_fact_table_compute_loop_like_summary(
    loom_value_fact_table_t* table, const loom_module_t* module, loom_op_t* op,
    bool* out_changed) {
  loom_loop_like_t loop = loom_loop_like_cast(module, op);
  if (!loom_loop_like_isa(loop)) {
    return iree_ok_status();
  }
  if (loom_loop_like_condition_region(loop)) {
    return loom_value_fact_table_compute_condition_loop_summary(
        table, module, loop, out_changed);
  }
  if (loom_loop_like_has_counted_range(loop)) {
    return loom_value_fact_table_compute_counted_loop_summary(
        table, module, loop, out_changed);
  }
  loom_region_t** regions = loom_op_regions(op);
  for (uint8_t i = 0; i < op->region_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_value_fact_table_compute_region_tree(
        table, module, regions[i], op));
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Forward pass: compute facts for ops
//===----------------------------------------------------------------------===//

iree_status_t loom_value_fact_table_compute_op(loom_value_fact_table_t* table,
                                               const loom_module_t* module,
                                               const loom_op_t* op) {
  return loom_value_fact_table_compute_op_and_report(table, module, op, NULL);
}

iree_status_t loom_value_fact_table_compute_op_and_report(
    loom_value_fact_table_t* table, const loom_module_t* module,
    const loom_op_t* op, bool* out_changed) {
  if (out_changed) {
    *out_changed = false;
  }
  const loom_op_vtable_t* vtable = loom_op_vtable(module, op);
  IREE_RETURN_IF_ERROR(loom_value_fact_table_propagate_select_dependencies(
      table, module, op, vtable, out_changed));
  if (vtable && vtable->loop_like) {
    return loom_value_fact_table_compute_loop_like_summary(
        table, module, (loom_op_t*)op, out_changed);
  }
  if (vtable && vtable->region_branch) {
    return loom_value_fact_table_compute_region_branch_summary(
        table, module, (loom_op_t*)op, out_changed);
  }
  if (!vtable || !vtable->infer_facts) {
    const loom_value_id_t* results = loom_op_const_results(op);
    for (uint16_t i = 0; i < op->result_count; ++i) {
      if (results[i] == LOOM_VALUE_ID_INVALID ||
          results[i] >= module->values.count) {
        continue;
      }
      // A result with no op-specific inference is still defined. Absence from
      // the table is reserved for not-yet-computed or unreachable values.
      loom_value_facts_t unknown_facts =
          loom_value_fact_table_unknown_for_value(module, results[i]);
      if (out_changed &&
          (!loom_value_fact_table_has_entry(table, results[i]) ||
           !loom_value_fact_table_facts_equal_for_type(
               module, loom_module_value_type(module, results[i]), table,
               loom_value_fact_table_lookup(table, results[i]), table,
               unknown_facts))) {
        *out_changed = true;
      }
      IREE_RETURN_IF_ERROR(
          loom_value_fact_table_define(table, results[i], unknown_facts));
      IREE_RETURN_IF_ERROR(loom_value_fact_table_seed_type_extent_facts(
          table, module, loom_module_value_type(module, results[i]),
          /*result_ids=*/NULL, /*result_count=*/0, /*result_facts=*/NULL,
          out_changed));
    }
    return loom_value_fact_table_propagate_origins(table, module, op,
                                                   out_changed);
  }

  // Get scratch for operand + result facts.
  iree_host_size_t total =
      (iree_host_size_t)op->operand_count + op->result_count;
  loom_value_facts_t* scratch = NULL;
  IREE_RETURN_IF_ERROR(
      loom_value_fact_table_facts_scratch(table, total, &scratch));
  loom_value_facts_t* operand_facts = scratch;
  loom_value_facts_t* result_facts = scratch + op->operand_count;

  // Gather operand facts from the table.
  const loom_value_id_t* operands = loom_op_const_operands(op);
  for (uint16_t i = 0; i < op->operand_count; ++i) {
    operand_facts[i] = loom_value_fact_table_lookup(table, operands[i]);
  }

  // Call the fact inference function.
  IREE_RETURN_IF_ERROR(vtable->infer_facts(&table->context, module, op,
                                           operand_facts, result_facts));

  const bool observes_execution = loom_traits_may_read(op->traits) ||
                                  loom_traits_are_convergent(op->traits) ||
                                  loom_traits_has_unique_identity(op->traits);
  loom_value_facts_t temporal_scope = loom_value_facts_unknown();
  if (observes_execution && op->result_count) {
    temporal_scope =
        loom_value_fact_table_block_temporal_scope(table, op->parent_block);
  }
  const loom_value_id_t* results = loom_op_const_results(op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    loom_value_fact_table_apply_operand_distribution(
        module, op, operand_facts, results[i], &result_facts[i]);
    if (observes_execution) {
      // A per-execution observation can escape a divergent CFG cycle with a
      // different result in each lane. Pure invariant expressions do not
      // depend on execution time; exact observations remain exact as well.
      loom_value_facts_propagate_binary_distribution(
          result_facts[i], temporal_scope, &result_facts[i]);
      if (loom_value_facts_is_lane_varying(result_facts[i])) {
        loom_value_facts_mark_lane_distribution_for_type(
            loom_module_value_type(module, results[i]), &result_facts[i]);
      }
    }
  }

  for (uint16_t i = 0; i < op->result_count; ++i) {
    if (results[i] == LOOM_VALUE_ID_INVALID ||
        results[i] >= module->values.count) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_value_fact_table_seed_type_extent_facts(
        table, module, loom_module_value_type(module, results[i]), results,
        op->result_count, result_facts, out_changed));
  }

  // Store result facts.
  for (uint16_t i = 0; i < op->result_count; ++i) {
    if (results[i] != LOOM_VALUE_ID_INVALID) {
      if (out_changed) {
        loom_value_facts_t old_facts =
            loom_value_fact_table_lookup(table, results[i]);
        loom_type_t result_type = loom_module_value_type(module, results[i]);
        if (!loom_value_fact_table_facts_equal_for_type(module, result_type,
                                                        table, old_facts, table,
                                                        result_facts[i])) {
          *out_changed = true;
        }
      }
      IREE_RETURN_IF_ERROR(
          loom_value_fact_table_define(table, results[i], result_facts[i]));
    }
  }

  return loom_value_fact_table_propagate_origins(table, module, op,
                                                 out_changed);
}

iree_status_t loom_value_fact_table_compute_region(
    loom_value_fact_table_t* table, const loom_module_t* module,
    loom_func_like_t function, loom_region_t* region, loom_op_t* parent_op) {
  table->context.function = function;
  table->context.reference_origin = (loom_value_fact_reference_origin_t){0};
  if (loom_func_like_isa(function) && parent_op == function.op) {
    for (uint8_t i = 0; i < loom_func_like_region_count(function); ++i) {
      if (loom_func_like_region(function, i) == region) {
        table->context.reference_origin = (loom_value_fact_reference_origin_t){
            .function_symbol_id = loom_func_like_callee(function).symbol_id,
            .entry_value_id = LOOM_VALUE_ID_INVALID,
            .region_index = i,
            .kind = LOOM_VALUE_FACT_REFERENCE_ORIGIN_ENTRY,
        };
        break;
      }
    }
  }
  IREE_RETURN_IF_ERROR(loom_value_fact_table_seed_projected_func_args(
      table, module, function, region, parent_op));
  return loom_value_fact_table_compute_region_tree(table, module, region,
                                                   parent_op);
}

iree_status_t loom_value_fact_table_compute(loom_value_fact_table_t* table,
                                            const loom_module_t* module,
                                            loom_func_like_t function) {
  loom_region_t* body = loom_func_like_body(function);
  if (!body) {
    return iree_ok_status();
  }
  const uint8_t body_region_index = loom_func_like_body_region_index(function);
  for (uint8_t i = 0; i < loom_func_like_region_count(function); ++i) {
    if (i == body_region_index) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_value_fact_table_compute_region(
        table, module, function, loom_func_like_region(function, i),
        function.op));
  }
  return loom_value_fact_table_compute_region(table, module, function, body,
                                              function.op);
}

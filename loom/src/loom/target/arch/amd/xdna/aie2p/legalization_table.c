// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/legalization_table.h"

#include "loom/ir/module.h"
#include "loom/ops/vector/ops.h"
#include "loom/util/fact_table.h"

enum { LOOM_AIE2P_TABLE_LOOKUP_MAX_LEVEL_COUNT = 6 };

typedef struct loom_aie2p_table_lookup_tree_t {
  // Builder positioned immediately before the source lookup.
  loom_builder_t* builder;
  // Source location retained by the complete selection tree.
  loom_location_id_t location;
  // Original SSA table, shared by all indexed-broadcast leaves.
  loom_value_id_t table;
  // Logical shape and integer width of the index vector.
  loom_type_t index_type;
  // Logical shape and payload type of every selection-tree value.
  loom_type_t result_type;
  // Per-lane bit tests, from least to most significant index bit.
  loom_value_id_t selectors[LOOM_AIE2P_TABLE_LOOKUP_MAX_LEVEL_COUNT];
  // Number of defined entries in the original table.
  uint32_t table_count;
} loom_aie2p_table_lookup_tree_t;

static iree_status_t loom_aie2p_table_lookup_build_subtree(
    const loom_aie2p_table_lookup_tree_t* tree, uint32_t first, uint32_t span,
    uint32_t level, loom_value_id_t* out_value) {
  loom_op_t* op = NULL;
  if (span == 1) {
    IREE_RETURN_IF_ERROR(
        loom_vector_constant_build(tree->builder, loom_attr_i64(first),
                                   tree->index_type, tree->location, &op));
    IREE_RETURN_IF_ERROR(loom_vector_table_lookup_build(
        tree->builder, tree->table, loom_vector_constant_result(op),
        tree->result_type, tree->location, &op));
    *out_value = loom_vector_table_lookup_result(op);
    return iree_ok_status();
  }

  loom_value_id_t low = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_aie2p_table_lookup_build_subtree(
      tree, first, span / 2, level - 1, &low));
  if (first + span / 2 >= tree->table_count) {
    // Out-of-range indices have no defined table lane. A partial final subtree
    // needs only its defined leaves, without padding the source table.
    *out_value = low;
    return iree_ok_status();
  }
  loom_value_id_t high = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_aie2p_table_lookup_build_subtree(
      tree, first + span / 2, span / 2, level - 1, &high));
  IREE_RETURN_IF_ERROR(
      loom_vector_select_build(tree->builder, tree->selectors[level - 1], high,
                               low, tree->result_type, tree->location, &op));
  *out_value = loom_vector_select_result(op);
  return iree_ok_status();
}

static iree_status_t loom_aie2p_table_lookup_build_selector(
    const loom_aie2p_table_lookup_tree_t* tree, loom_value_id_t indices,
    loom_type_t predicate_type, loom_value_id_t zero, uint32_t level,
    loom_value_id_t* out_selector) {
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_constant_build(
      tree->builder, loom_attr_i64(UINT64_C(1) << level), tree->index_type,
      tree->location, &op));
  IREE_RETURN_IF_ERROR(loom_vector_andi_build(
      tree->builder, indices, loom_vector_constant_result(op), tree->index_type,
      tree->location, &op));
  IREE_RETURN_IF_ERROR(loom_vector_cmpi_build(
      tree->builder, LOOM_VECTOR_CMPI_PREDICATE_ULT, zero,
      loom_vector_andi_result(op), tree->index_type, predicate_type,
      tree->location, &op));
  *out_selector = loom_vector_cmpi_result(op);
  return iree_ok_status();
}

static bool loom_aie2p_table_lookup_has_vector_carrier(loom_type_t type,
                                                       uint64_t count) {
  const uint32_t bit_count =
      loom_scalar_type_bitwidth(loom_type_element_type(type));
  return (bit_count == 8 || bit_count == 16 || bit_count == 32) && count > 0 &&
         count <= 512 / bit_count;
}

static bool loom_aie2p_table_lookup_has_packet_result_carriers(loom_type_t type,
                                                               uint64_t count) {
  const loom_scalar_type_t element_type = loom_type_element_type(type);
  const uint32_t bit_count = loom_scalar_type_bitwidth(element_type);
  if ((bit_count != 8 && bit_count != 16 && bit_count != 32) || count == 0 ||
      count > 1024 / bit_count) {
    return false;
  }
  // F32x32 uses the accumulator file, while each packed selection result uses
  // an ordinary X carrier. Their representation boundary needs an explicit
  // conversion instead of an ordinary vector concat.
  return element_type != LOOM_SCALAR_TYPE_F32 || count != 32;
}

iree_status_t loom_aie2p_table_lookup_rewrite(
    loom_target_legalization_context_t* context, loom_op_t* op,
    const loom_vector_packet_policy_t* packet_policy, bool* out_rewritten) {
  *out_rewritten = false;
  const loom_value_id_t table = loom_vector_table_lookup_table(op);
  const loom_value_id_t indices = loom_vector_table_lookup_indices(op);
  const loom_type_t table_type = loom_module_value_type(context->module, table);
  const loom_type_t index_type =
      loom_module_value_type(context->module, indices);
  const loom_type_t result_type = loom_module_value_type(
      context->module, loom_vector_table_lookup_result(op));
  uint64_t table_count = 0;
  uint64_t result_count = 0;
  if (!loom_type_static_element_count(table_type, &table_count) ||
      !loom_type_static_element_count(result_type, &result_count) ||
      !loom_aie2p_table_lookup_has_vector_carrier(table_type, table_count)) {
    return iree_ok_status();
  }
  const uint32_t index_bit_count =
      loom_scalar_type_bitwidth(loom_type_element_type(index_type));
  if (loom_aie2p_table_lookup_has_packet_result_carriers(result_type,
                                                         result_count) &&
      (index_bit_count == 8 || index_bit_count == 16 ||
       index_bit_count == 32)) {
    IREE_RETURN_IF_ERROR(loom_vector_packet_legalize_table_lookup(
        context, op, packet_policy, out_rewritten));
    if (*out_rewritten) {
      return iree_ok_status();
    }
  }
  if (!loom_aie2p_table_lookup_has_vector_carrier(result_type, result_count) ||
      !loom_aie2p_table_lookup_has_vector_carrier(index_type, result_count)) {
    return iree_ok_status();
  }
  const loom_value_facts_t index_facts =
      loom_value_fact_table_lookup(context->fact_table, indices);
  loom_value_fact_uniform_element_t uniform = {0};
  if (loom_value_facts_query_uniform_element(&context->fact_table->context,
                                             index_facts, &uniform)) {
    // Uniform lookups already admit one broadcast or a single scalar extract.
    return iree_ok_status();
  }
  loom_value_fact_small_static_lanes_t static_indices = {0};
  uint32_t known_index_count = 0;
  if (loom_value_facts_query_small_static_lanes(&context->fact_table->context,
                                                index_facts, &static_indices)) {
    for (iree_host_size_t lane = 0; lane < static_indices.count; ++lane) {
      known_index_count +=
          loom_value_facts_is_exact(static_indices.lanes[lane]);
    }
    if (known_index_count == static_indices.count) {
      // Static lane specialization can discard the index carrier and fold
      // selected table entries. Preserve that path without materializing
      // dynamic bit tests and the source index assembly they keep alive.
      return iree_ok_status();
    }
  }

  uint32_t levels = 0;
  uint32_t span = 1;
  while (span < table_count) {
    span *= 2;
    ++levels;
  }
  // Each index bit needs a constant, splat, bitwise AND, and comparison.
  // Halfword and word comparisons also complete their partial predicate.
  const uint32_t selector_cost = index_bit_count == 8 ? 4 : 5;
  // Each table entry needs one native broadcast, followed by T-1 selects.
  const uint32_t packed_cost =
      2 + selector_cost * levels + 2 * (uint32_t)table_count - 1;
  // Scalar lookup needs an index extract, an optional narrow-index extension,
  // a table extract, and three result insertion/control operations per lane.
  // A known index folds the first two or three operations into one immediate
  // table extract.
  const uint32_t scalar_dynamic_lane_cost = index_bit_count == 32 ? 5 : 6;
  const uint32_t scalar_known_lane_savings = index_bit_count == 32 ? 1 : 2;
  const uint32_t scalar_cost =
      scalar_dynamic_lane_cost * (uint32_t)result_count - 2 -
      scalar_known_lane_savings * known_index_count;
  if (packed_cost >= scalar_cost) {
    return iree_ok_status();
  }

  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_set_before(&rewriter->builder, op);
  const loom_value_id_t checkpoint = loom_rewriter_value_checkpoint(rewriter);
  loom_aie2p_table_lookup_tree_t tree = {
      .builder = &rewriter->builder,
      .location = op->location,
      .table = table,
      .index_type = index_type,
      .result_type = result_type,
      .table_count = (uint32_t)table_count,
  };
  loom_type_t predicate_type = index_type;
  predicate_type.header = loom_type_make_header(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I1, loom_type_rank(index_type),
      loom_type_flags(index_type));
  loom_op_t* value_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_constant_build(
      tree.builder, loom_attr_i64(0), index_type, op->location, &value_op));
  const loom_value_id_t zero = loom_vector_constant_result(value_op);
  for (uint32_t level = 0; level < levels; ++level) {
    IREE_RETURN_IF_ERROR(loom_aie2p_table_lookup_build_selector(
        &tree, indices, predicate_type, zero, level, &tree.selectors[level]));
  }
  loom_value_id_t replacement = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_aie2p_table_lookup_build_subtree(
      &tree, 0, span, levels, &replacement));
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, checkpoint));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement, 1));
  *out_rewritten = true;
  return iree_ok_status();
}

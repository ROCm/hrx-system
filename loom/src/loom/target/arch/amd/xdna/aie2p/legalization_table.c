// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/legalization_table.h"

#include "loom/ir/module.h"
#include "loom/ops/vector/ops.h"
#include "loom/util/fact_table.h"

enum {
  LOOM_AIE2P_TABLE_LOOKUP_MAX_LEVEL_COUNT = 7,
  LOOM_AIE2P_TABLE_LOOKUP_PREDICATE_PACKET_LANE_COUNT = 64,
  LOOM_AIE2P_TABLE_LOOKUP_MAX_PREDICATE_PACKET_COUNT = 2,
};

typedef struct loom_aie2p_table_lookup_tree_t {
  // Builder positioned immediately before the source lookup.
  loom_builder_t* builder;
  // Source location retained by the complete selection tree.
  loom_location_id_t location;
  // Original SSA table, shared by all indexed-broadcast leaves.
  loom_value_id_t table;
  // Original rank-one table type.
  loom_type_t table_type;
  // Byte-extended predicate-table packets used by native broadcast leaves.
  loom_value_id_t
      predicate_byte_tables[LOOM_AIE2P_TABLE_LOOKUP_MAX_PREDICATE_PACKET_COUNT];
  // Logical shape and integer width of the index vector.
  loom_type_t index_type;
  // Logical shape and payload type of every selection-tree value.
  loom_type_t result_type;
  // Per-lane bit tests, from least to most significant index bit.
  loom_value_id_t selectors[LOOM_AIE2P_TABLE_LOOKUP_MAX_LEVEL_COUNT];
  // Number of defined entries in the original table.
  uint32_t table_count;
} loom_aie2p_table_lookup_tree_t;

static iree_status_t loom_aie2p_table_lookup_prepare_predicate_table(
    loom_aie2p_table_lookup_tree_t* tree) {
  if (loom_type_element_type(tree->table_type) != LOOM_SCALAR_TYPE_I1) {
    return iree_ok_status();
  }
  const uint32_t packet_lane_count =
      LOOM_AIE2P_TABLE_LOOKUP_PREDICATE_PACKET_LANE_COUNT;
  const uint32_t packet_count =
      (tree->table_count - 1u) / packet_lane_count + 1u;
  for (uint32_t packet_index = 0; packet_index < packet_count; ++packet_index) {
    const uint32_t lane_offset = packet_index * packet_lane_count;
    const uint32_t lane_count =
        iree_min(tree->table_count - lane_offset, packet_lane_count);
    loom_type_t predicate_packet_type = tree->table_type;
    predicate_packet_type.dims[0] = loom_dim_pack_static(lane_count);
    loom_value_id_t predicate_packet = tree->table;
    if (lane_offset != 0 || lane_count != tree->table_count) {
      const int64_t static_offset = lane_offset;
      loom_op_t* slice_op = NULL;
      IREE_RETURN_IF_ERROR(loom_vector_slice_build(
          tree->builder, tree->table, /*offsets=*/NULL, /*offsets_count=*/0,
          &static_offset, /*static_offsets_count=*/1, predicate_packet_type,
          tree->location, &slice_op));
      predicate_packet = loom_vector_slice_result(slice_op);
    }
    loom_type_t byte_packet_type = predicate_packet_type;
    byte_packet_type.header =
        loom_type_make_header(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I8,
                              loom_type_rank(predicate_packet_type),
                              loom_type_flags(predicate_packet_type));
    loom_op_t* extend_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_extui_build(
        tree->builder, predicate_packet, predicate_packet_type,
        byte_packet_type, tree->location, &extend_op));
    tree->predicate_byte_tables[packet_index] =
        loom_vector_extui_result(extend_op);
  }
  return iree_ok_status();
}

static iree_status_t loom_aie2p_table_lookup_build_leaf(
    const loom_aie2p_table_lookup_tree_t* tree, uint32_t table_index,
    loom_value_id_t* out_value) {
  loom_op_t* op = NULL;
  if (loom_type_element_type(tree->table_type) != LOOM_SCALAR_TYPE_I1) {
    IREE_RETURN_IF_ERROR(
        loom_vector_constant_build(tree->builder, loom_attr_i64(table_index),
                                   tree->index_type, tree->location, &op));
    IREE_RETURN_IF_ERROR(loom_vector_table_lookup_build(
        tree->builder, tree->table, loom_vector_constant_result(op),
        tree->result_type, tree->location, &op));
    *out_value = loom_vector_table_lookup_result(op);
    return iree_ok_status();
  }

  const uint32_t packet_lane_count =
      LOOM_AIE2P_TABLE_LOOKUP_PREDICATE_PACKET_LANE_COUNT;
  const uint32_t packet_index = table_index / packet_lane_count;
  IREE_RETURN_IF_ERROR(loom_vector_constant_build(
      tree->builder, loom_attr_i64(table_index % packet_lane_count),
      tree->index_type, tree->location, &op));
  IREE_RETURN_IF_ERROR(loom_vector_table_lookup_build(
      tree->builder, tree->predicate_byte_tables[packet_index],
      loom_vector_constant_result(op), tree->result_type, tree->location, &op));
  *out_value = loom_vector_table_lookup_result(op);
  return iree_ok_status();
}

static iree_status_t loom_aie2p_table_lookup_build_subtree(
    const loom_aie2p_table_lookup_tree_t* tree, uint32_t first, uint32_t span,
    uint32_t level, loom_value_id_t* out_value) {
  loom_op_t* op = NULL;
  if (span == 1) {
    return loom_aie2p_table_lookup_build_leaf(tree, first, out_value);
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
  if (bit_count == 1) {
    return count > 0 &&
           count <= LOOM_AIE2P_TABLE_LOOKUP_PREDICATE_PACKET_LANE_COUNT *
                        LOOM_AIE2P_TABLE_LOOKUP_MAX_PREDICATE_PACKET_COUNT;
  }
  return (bit_count == 8 || bit_count == 16 || bit_count == 32 ||
          bit_count == 64) &&
         count > 0 && count <= 512 / bit_count;
}

static bool loom_aie2p_table_lookup_has_packet_result_carriers(loom_type_t type,
                                                               uint64_t count) {
  const loom_scalar_type_t element_type = loom_type_element_type(type);
  const uint32_t bit_count = loom_scalar_type_bitwidth(element_type);
  if (count == 0) {
    return false;
  }
  if (bit_count == 1) {
    return count <= LOOM_AIE2P_TABLE_LOOKUP_PREDICATE_PACKET_LANE_COUNT *
                        LOOM_AIE2P_TABLE_LOOKUP_MAX_PREDICATE_PACKET_COUNT;
  }
  if (bit_count != 8 && bit_count != 16 && bit_count != 32 && bit_count != 64) {
    return false;
  }
  if (count <= 1024 / bit_count) {
    return true;
  }
  // The exact 2048-bit accumulator types retain four 512-bit selection
  // packets. Other vectors above 1024 bits have no source type mapping.
  return loom_type_rank(type) == 1 &&
         ((count == 64 && (element_type == LOOM_SCALAR_TYPE_I32 ||
                           element_type == LOOM_SCALAR_TYPE_F32)) ||
          (count == 32 && element_type == LOOM_SCALAR_TYPE_I64));
}

iree_status_t loom_aie2p_table_lookup_rewrite(
    loom_target_legalization_context_t* context, loom_op_t* op,
    const loom_target_vector_packet_policy_t* packet_policy,
    bool* out_rewritten) {
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
      (index_bit_count == 8 || index_bit_count == 16 || index_bit_count == 32 ||
       index_bit_count == 64)) {
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
  int64_t uniform_index = 0;
  const bool has_uniform_index =
      loom_value_facts_query_uniform_element(&context->fact_table->context,
                                             index_facts, &uniform) &&
      loom_value_facts_as_exact_i64(uniform.element, &uniform_index);
  if (has_uniform_index &&
      loom_type_element_type(table_type) != LOOM_SCALAR_TYPE_I1) {
    // One exact value shared by every lane maps directly to one native indexed
    // broadcast. A non-exact uniform element is only a common lane envelope
    // and does not prove equal runtime indices.
    return iree_ok_status();
  }
  if (has_uniform_index &&
      (uniform_index < 0 || (uint64_t)uniform_index >= table_count)) {
    return iree_ok_status();
  }
  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_set_before(&rewriter->builder, op);
  const loom_value_id_t checkpoint = loom_rewriter_value_checkpoint(rewriter);
  loom_value_id_t selection_indices = indices;
  loom_type_t selection_index_type = index_type;
  if (!has_uniform_index && index_bit_count == 64) {
    // Every defined lookup index is below the at-most-128-lane table extent.
    // Narrowing to the native word comparison width therefore preserves all
    // defined executions and avoids expanding each selector into a double-word
    // comparison.
    selection_index_type.header = loom_type_make_header(
        LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I32, loom_type_rank(index_type),
        loom_type_flags(index_type));
    loom_op_t* narrow_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_trunci_build(
        &rewriter->builder, indices, index_type, selection_index_type,
        op->location, &narrow_op));
    selection_indices = loom_vector_trunci_result(narrow_op);
  }
  loom_aie2p_table_lookup_tree_t tree = {
      .builder = &rewriter->builder,
      .location = op->location,
      .table = table,
      .table_type = table_type,
      .predicate_byte_tables = {LOOM_VALUE_ID_INVALID, LOOM_VALUE_ID_INVALID},
      .index_type = selection_index_type,
      .result_type = result_type,
      .table_count = (uint32_t)table_count,
  };
  if (loom_type_element_type(table_type) == LOOM_SCALAR_TYPE_I1) {
    tree.result_type.header = loom_type_make_header(
        LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I8, loom_type_rank(result_type),
        loom_type_flags(result_type));
  }
  IREE_RETURN_IF_ERROR(loom_aie2p_table_lookup_prepare_predicate_table(&tree));
  loom_value_id_t replacement = LOOM_VALUE_ID_INVALID;
  if (has_uniform_index) {
    IREE_RETURN_IF_ERROR(loom_aie2p_table_lookup_build_leaf(
        &tree, (uint32_t)uniform_index, &replacement));
  } else {
    uint32_t levels = 0;
    uint32_t span = 1;
    while (span < table_count) {
      span *= 2;
      ++levels;
    }
    loom_type_t predicate_type = selection_index_type;
    predicate_type.header =
        loom_type_make_header(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I1,
                              loom_type_rank(selection_index_type),
                              loom_type_flags(selection_index_type));
    loom_op_t* value_op = NULL;
    IREE_RETURN_IF_ERROR(
        loom_vector_constant_build(tree.builder, loom_attr_i64(0),
                                   tree.index_type, op->location, &value_op));
    const loom_value_id_t zero = loom_vector_constant_result(value_op);
    for (uint32_t level = 0; level < levels; ++level) {
      IREE_RETURN_IF_ERROR(loom_aie2p_table_lookup_build_selector(
          &tree, selection_indices, predicate_type, zero, level,
          &tree.selectors[level]));
    }
    IREE_RETURN_IF_ERROR(loom_aie2p_table_lookup_build_subtree(
        &tree, 0, span, levels, &replacement));
  }
  if (loom_type_element_type(table_type) == LOOM_SCALAR_TYPE_I1) {
    loom_op_t* value_op = NULL;
    IREE_RETURN_IF_ERROR(
        loom_vector_constant_build(tree.builder, loom_attr_i64(0),
                                   tree.result_type, op->location, &value_op));
    const loom_value_id_t zero = loom_vector_constant_result(value_op);
    IREE_RETURN_IF_ERROR(loom_vector_cmpi_build(
        tree.builder, LOOM_VECTOR_CMPI_PREDICATE_ULT, zero, replacement,
        tree.result_type, result_type, op->location, &value_op));
    replacement = loom_vector_cmpi_result(value_op);
  }
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, checkpoint));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement, 1));
  *out_rewritten = true;
  return iree_ok_status();
}

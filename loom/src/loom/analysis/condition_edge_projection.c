// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/condition_edge_projection.h"

#include "loom/util/adaptive_sort.h"

static bool loom_condition_edge_mapping_less(
    const loom_condition_edge_mapping_t* left,
    const loom_condition_edge_mapping_t* right) {
  return left->source < right->source ||
         (left->source == right->source && left->target < right->target);
}

LOOM_DEFINE_ADAPTIVE_SORT(loom_condition_sort_edge_mappings,
                          loom_condition_edge_mapping_t,
                          loom_condition_edge_mapping_less)

void loom_condition_edge_projection_initialize(
    iree_arena_allocator_t* arena,
    loom_condition_edge_projection_t* out_projection) {
  *out_projection = (loom_condition_edge_projection_t){0};
  loom_condition_derivation_initialize(arena,
                                       &out_projection->source_derivation);
}

void loom_condition_edge_projection_reset(
    loom_condition_edge_projection_t* projection) {
  loom_condition_derivation_reset(&projection->source_derivation);
  projection->module = NULL;
  projection->source_region = NULL;
  projection->target_block = NULL;
  projection->mapping_count = 0;
  projection->visible_integer_relation_count = 0;
  projection->visible_boolean_fact_count = 0;
}

static bool loom_condition_edge_projection_value_is_source_local(
    const loom_condition_edge_projection_t* projection,
    loom_value_id_t value_id) {
  const loom_value_t* value = loom_module_value(projection->module, value_id);
  if (!value) {
    return false;
  }
  if (loom_value_is_block_arg(value)) {
    const loom_block_t* block = loom_value_def_block(value);
    return block && block->parent_region == projection->source_region;
  }
  const loom_op_t* defining_op = loom_value_def_op(value);
  return defining_op && defining_op->parent_block &&
         defining_op->parent_block->parent_region == projection->source_region;
}

static const loom_condition_edge_mapping_t*
loom_condition_edge_projection_find_source(
    const loom_condition_edge_projection_t* projection,
    loom_value_id_t source) {
  iree_host_size_t begin = 0;
  iree_host_size_t end = projection->mapping_count;
  while (begin < end) {
    const iree_host_size_t middle = begin + (end - begin) / 2;
    if (projection->mappings[middle].source < source) {
      begin = middle + 1;
    } else {
      end = middle;
    }
  }
  return begin < projection->mapping_count &&
                 projection->mappings[begin].source == source
             ? &projection->mappings[begin]
             : NULL;
}

loom_condition_integer_operand_t loom_condition_edge_projection_source_operand(
    const loom_condition_edge_projection_t* projection,
    loom_condition_integer_operand_t operand) {
  if (operand.kind != LOOM_CONDITION_INTEGER_OPERAND_VALUE ||
      !projection->target_block) {
    return operand;
  }
  const loom_value_t* value =
      loom_module_value(projection->module, operand.value_id);
  if (!value || !loom_value_is_block_arg(value) ||
      loom_value_def_block(value) != projection->target_block) {
    return operand;
  }
  const uint16_t argument_index = loom_value_def_index(value);
  if (argument_index < projection->mapping_count) {
    operand.value_id = projection->target_sources[argument_index];
  }
  return operand;
}

bool loom_condition_edge_projection_target_operand(
    const loom_condition_edge_projection_t* projection,
    loom_condition_integer_operand_t operand,
    loom_condition_integer_operand_t* out_operand) {
  *out_operand = operand;
  if (operand.kind == LOOM_CONDITION_INTEGER_OPERAND_CONSTANT ||
      !loom_condition_edge_projection_value_is_source_local(projection,
                                                            operand.value_id)) {
    return true;
  }
  const loom_condition_edge_mapping_t* mapping =
      loom_condition_edge_projection_find_source(projection, operand.value_id);
  if (!mapping) {
    return false;
  }
  out_operand->value_id = mapping->target;
  return true;
}

iree_status_t loom_condition_edge_projection_update_mapping(
    loom_condition_edge_projection_t* projection, const loom_module_t* module,
    const loom_region_t* source_region, const loom_block_t* target_block,
    const loom_value_id_t* sources, iree_host_size_t source_count) {
  IREE_ASSERT_EQ(source_count, target_block->arg_count,
                 "verified edge payload must match successor arguments");
  if (source_count > projection->target_source_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        projection->source_derivation.arena, projection->mapping_count,
        source_count, sizeof(*projection->target_sources),
        &projection->target_source_capacity,
        (void**)&projection->target_sources));
  }
  if (source_count > projection->mapping_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        projection->source_derivation.arena, projection->mapping_count,
        source_count, sizeof(*projection->mappings),
        &projection->mapping_capacity, (void**)&projection->mappings));
  }
  projection->module = module;
  projection->source_region = source_region;
  projection->target_block = target_block;
  projection->mapping_count = source_count;
  for (uint16_t i = 0; i < source_count; ++i) {
    projection->target_sources[i] = sources[i];
    projection->mappings[i] = (loom_condition_edge_mapping_t){
        .source = sources[i],
        .target = loom_block_arg_id(target_block, i),
    };
  }
  loom_condition_sort_edge_mappings(projection->mappings, source_count);

  projection->visible_integer_relation_count = 0;
  const loom_condition_fact_set_t* facts =
      &projection->source_derivation.integer_facts;
  for (iree_host_size_t i = 0; i < facts->integer_relation_count; ++i) {
    loom_condition_integer_operand_t projected = {0};
    const loom_condition_integer_relation_t* relation =
        &facts->integer_relations[i];
    if (loom_condition_edge_projection_target_operand(
            projection, relation->left, &projected) ||
        loom_condition_edge_projection_target_operand(
            projection, relation->right, &projected)) {
      ++projection->visible_integer_relation_count;
    }
  }

  projection->visible_boolean_fact_count = 0;
  for (iree_host_size_t i = 0;
       i < projection->source_derivation.boolean_fact_count; ++i) {
    const loom_condition_integer_operand_t source = {
        .kind = LOOM_CONDITION_INTEGER_OPERAND_VALUE,
        .value_id = projection->source_derivation.boolean_facts[i].value_id,
    };
    loom_condition_integer_operand_t projected = {0};
    if (loom_condition_edge_projection_target_operand(projection, source,
                                                      &projected)) {
      ++projection->visible_boolean_fact_count;
    }
  }
  return iree_ok_status();
}

bool loom_condition_edge_projection_is_empty(
    const loom_condition_edge_projection_t* projection) {
  return projection->visible_integer_relation_count == 0 &&
         projection->visible_boolean_fact_count == 0;
}

bool loom_condition_edge_projection_query_boolean(
    const loom_condition_edge_projection_t* projection,
    loom_value_id_t value_id, bool* out_value) {
  const loom_condition_integer_operand_t target = {
      .kind = LOOM_CONDITION_INTEGER_OPERAND_VALUE,
      .value_id = value_id,
  };
  value_id = loom_condition_edge_projection_source_operand(projection, target)
                 .value_id;
  const loom_condition_derivation_t* derivation =
      &projection->source_derivation;
  iree_host_size_t begin = 0;
  iree_host_size_t end = derivation->boolean_fact_count;
  while (begin < end) {
    const iree_host_size_t middle = begin + (end - begin) / 2;
    if (derivation->boolean_facts[middle].value_id < value_id) {
      begin = middle + 1;
    } else {
      end = middle;
    }
  }
  bool known[2] = {false, false};
  while (begin < derivation->boolean_fact_count &&
         derivation->boolean_facts[begin].value_id == value_id) {
    known[derivation->boolean_facts[begin].value ? 1 : 0] = true;
    ++begin;
  }
  if (known[0] == known[1]) {
    return false;
  }
  *out_value = known[1];
  return true;
}

bool loom_condition_edge_projection_apply_to_value_facts(
    const loom_condition_edge_projection_t* projection,
    const loom_value_fact_table_t* fact_table, loom_value_id_t value_id,
    loom_value_facts_t* inout_facts) {
  const loom_condition_integer_operand_t target = {
      .kind = LOOM_CONDITION_INTEGER_OPERAND_VALUE,
      .value_id = value_id,
  };
  const loom_condition_integer_operand_t source =
      loom_condition_edge_projection_source_operand(projection, target);
  return loom_condition_fact_set_apply_to_value_facts(
      &projection->source_derivation.integer_facts, fact_table, source.value_id,
      inout_facts);
}

loom_integer_order_outcome_bits_t
loom_condition_edge_projection_query_integer_order(
    const loom_condition_edge_projection_t* projection,
    const loom_value_fact_table_t* fact_table,
    loom_condition_integer_operand_t left,
    loom_condition_integer_operand_t right) {
  return loom_condition_fact_set_query_integer_order(
      &projection->source_derivation.integer_facts, fact_table,
      loom_condition_edge_projection_source_operand(projection, left),
      loom_condition_edge_projection_source_operand(projection, right));
}

bool loom_condition_edge_projection_proves_integer_relation(
    const loom_condition_edge_projection_t* projection,
    const loom_value_fact_table_t* fact_table,
    const loom_condition_integer_relation_t* queried, bool* out_result) {
  const loom_integer_order_outcome_bits_t outcomes =
      loom_condition_edge_projection_query_integer_order(
          projection, fact_table, queried->left, queried->right);
  return loom_symbolic_integer_relation_resolve(outcomes, queried->relation,
                                                out_result);
}

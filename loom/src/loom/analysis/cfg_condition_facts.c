// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/cfg_condition_facts.h"

#include <string.h>

#include "loom/ops/cfg/ops.h"

static loom_condition_integer_operand_t loom_cfg_condition_value_operand(
    loom_value_id_t value_id) {
  return (loom_condition_integer_operand_t){
      .kind = LOOM_CONDITION_INTEGER_OPERAND_VALUE,
      .value_id = value_id,
      .constant = 0,
  };
}

static bool loom_cfg_condition_edge_implies_bool(const loom_block_t* target,
                                                 const loom_op_t* terminator,
                                                 loom_value_id_t* out_condition,
                                                 bool* out_value) {
  if (!terminator || !loom_cfg_cond_br_isa(terminator)) {
    return false;
  }

  bool is_true_edge = loom_cfg_cond_br_true_dest(terminator) == target;
  bool is_false_edge = loom_cfg_cond_br_false_dest(terminator) == target;
  if (is_true_edge == is_false_edge) return false;
  *out_condition = loom_cfg_cond_br_condition(terminator);
  *out_value = is_true_edge;
  return true;
}

static void loom_cfg_condition_append_relation_unique(
    loom_condition_integer_relation_t* relations,
    iree_host_size_t relation_capacity, iree_host_size_t* inout_relation_count,
    loom_condition_integer_relation_t relation) {
  for (iree_host_size_t i = 0; i < *inout_relation_count; ++i) {
    if (loom_condition_integer_relations_equivalent(&relations[i], &relation)) {
      return;
    }
  }
  if (*inout_relation_count >= relation_capacity) {
    return;
  }
  relations[(*inout_relation_count)++] = relation;
}

static iree_host_size_t loom_cfg_condition_intersect_relations(
    loom_condition_integer_relation_t* inout_relations,
    iree_host_size_t inout_relation_count,
    const loom_condition_integer_relation_t* edge_relations,
    iree_host_size_t edge_relation_count) {
  iree_host_size_t new_relation_count = 0;
  for (iree_host_size_t existing_index = 0;
       existing_index < inout_relation_count; ++existing_index) {
    loom_condition_integer_relation_t common_relation = {0};
    bool found_common_relation = false;
    for (iree_host_size_t edge_index = 0; edge_index < edge_relation_count;
         ++edge_index) {
      if (!loom_condition_integer_relation_meet(
              &inout_relations[existing_index], &edge_relations[edge_index],
              &common_relation)) {
        continue;
      }
      found_common_relation = true;
      break;
    }
    if (!found_common_relation) continue;
    inout_relations[new_relation_count++] = common_relation;
  }
  return new_relation_count;
}

static bool loom_cfg_condition_try_map_branch_arg_to_block_arg(
    const loom_block_t* block, loom_value_slice_t branch_args,
    loom_value_id_t value_id, loom_value_id_t* out_block_arg,
    bool* out_ambiguous) {
  *out_block_arg = LOOM_VALUE_ID_INVALID;
  *out_ambiguous = false;
  if (!block || branch_args.count != block->arg_count) return false;

  bool found_mapping = false;
  for (uint16_t i = 0; i < branch_args.count; ++i) {
    if (branch_args.values[i] != value_id) continue;
    if (found_mapping) {
      *out_ambiguous = true;
      *out_block_arg = LOOM_VALUE_ID_INVALID;
      return false;
    }
    *out_block_arg = loom_block_arg_id(block, i);
    found_mapping = true;
  }
  return found_mapping;
}

static bool loom_cfg_condition_value_available_at_block_entry(
    const loom_module_t* module, const loom_dominance_info_t* dominance,
    loom_value_id_t value_id, const loom_block_t* block) {
  if (value_id == LOOM_VALUE_ID_INVALID || value_id >= module->values.count ||
      !block || !block->first_op) {
    return false;
  }
  return loom_value_is_available_before_op(dominance, value_id,
                                           block->first_op) &&
         loom_value_type_is_available_before_op(dominance, value_id,
                                                block->first_op);
}

static bool loom_cfg_condition_remap_operand_to_block_entry(
    const loom_module_t* module, const loom_dominance_info_t* dominance,
    const loom_block_t* block, const loom_op_t* predecessor_terminator,
    loom_condition_integer_operand_t operand,
    loom_condition_integer_operand_t* out_operand) {
  *out_operand = operand;
  if (operand.kind != LOOM_CONDITION_INTEGER_OPERAND_VALUE) return true;

  if (loom_cfg_br_isa(predecessor_terminator) &&
      loom_cfg_br_dest(predecessor_terminator) == block) {
    loom_value_id_t block_arg = LOOM_VALUE_ID_INVALID;
    bool ambiguous_mapping = false;
    bool found_mapping = loom_cfg_condition_try_map_branch_arg_to_block_arg(
        block, loom_cfg_br_args(predecessor_terminator), operand.value_id,
        &block_arg, &ambiguous_mapping);
    if (ambiguous_mapping) return false;
    if (found_mapping) {
      *out_operand = loom_cfg_condition_value_operand(block_arg);
      return true;
    }
  }

  return loom_cfg_condition_value_available_at_block_entry(
      module, dominance, operand.value_id, block);
}

static bool loom_cfg_condition_remap_value_to_block_entry(
    const loom_module_t* module, const loom_dominance_info_t* dominance,
    const loom_block_t* block, const loom_op_t* predecessor_terminator,
    loom_value_id_t value_id, loom_value_id_t* out_value_id) {
  *out_value_id = value_id;
  if (loom_cfg_br_isa(predecessor_terminator) &&
      loom_cfg_br_dest(predecessor_terminator) == block) {
    loom_value_id_t block_arg = LOOM_VALUE_ID_INVALID;
    bool ambiguous_mapping = false;
    bool found_mapping = loom_cfg_condition_try_map_branch_arg_to_block_arg(
        block, loom_cfg_br_args(predecessor_terminator), value_id, &block_arg,
        &ambiguous_mapping);
    if (ambiguous_mapping) return false;
    if (found_mapping) {
      *out_value_id = block_arg;
      return true;
    }
  }
  return loom_cfg_condition_value_available_at_block_entry(module, dominance,
                                                           value_id, block);
}

static bool loom_cfg_condition_remap_relation_to_block_entry(
    const loom_module_t* module, const loom_dominance_info_t* dominance,
    const loom_block_t* block, const loom_op_t* predecessor_terminator,
    const loom_condition_integer_relation_t* relation,
    loom_condition_integer_relation_t* out_relation) {
  *out_relation = *relation;
  if (!loom_cfg_condition_remap_operand_to_block_entry(
          module, dominance, block, predecessor_terminator, relation->left,
          &out_relation->left)) {
    return false;
  }
  if (!loom_cfg_condition_remap_operand_to_block_entry(
          module, dominance, block, predecessor_terminator, relation->right,
          &out_relation->right)) {
    return false;
  }
  return true;
}

static void loom_cfg_condition_append_remapped_relations(
    const loom_module_t* module, const loom_dominance_info_t* dominance,
    const loom_block_t* block, const loom_op_t* predecessor_terminator,
    const loom_condition_integer_relation_t* source_relations,
    iree_host_size_t source_relation_count,
    loom_condition_integer_relation_t* relations,
    iree_host_size_t relation_capacity,
    iree_host_size_t* inout_relation_count) {
  for (iree_host_size_t i = 0; i < source_relation_count; ++i) {
    loom_condition_integer_relation_t remapped_relation = {0};
    if (!loom_cfg_condition_remap_relation_to_block_entry(
            module, dominance, block, predecessor_terminator,
            &source_relations[i], &remapped_relation)) {
      continue;
    }
    loom_cfg_condition_append_relation_unique(
        relations, relation_capacity, inout_relation_count, remapped_relation);
  }
}

static bool loom_cfg_condition_set_edge_bool_fact(
    loom_value_id_t condition, bool value, loom_value_id_t* inout_condition,
    bool* inout_value, bool* inout_known) {
  if (!*inout_known) {
    *inout_condition = condition;
    *inout_value = value;
    *inout_known = true;
    return true;
  }
  return *inout_condition == condition && *inout_value == value;
}

void loom_cfg_condition_facts_compute_predecessor_edge(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_dominance_info_t* dominance, const loom_block_t* block,
    const loom_op_t* predecessor_terminator, uint16_t predecessor_index,
    const loom_cfg_block_entry_condition_facts_t* current_facts,
    loom_condition_integer_relation_t* relation_storage,
    iree_host_size_t relation_capacity,
    loom_cfg_block_entry_condition_facts_t* out_fact) {
  *out_fact = (loom_cfg_block_entry_condition_facts_t){
      .condition = LOOM_VALUE_ID_INVALID,
      .integer_relations = relation_storage,
  };

  loom_value_id_t edge_condition = LOOM_VALUE_ID_INVALID;
  bool edge_value = false;
  bool edge_has_condition = loom_cfg_condition_edge_implies_bool(
      block, predecessor_terminator, &edge_condition, &edge_value);
  if (edge_has_condition) {
    out_fact->condition = edge_condition;
    out_fact->condition_value = edge_value;
    out_fact->condition_known = true;
  } else if (current_facts &&
             current_facts[predecessor_index].condition_known) {
    loom_value_id_t remapped_condition = LOOM_VALUE_ID_INVALID;
    if (loom_cfg_condition_remap_value_to_block_entry(
            module, dominance, block, predecessor_terminator,
            current_facts[predecessor_index].condition, &remapped_condition)) {
      out_fact->condition = remapped_condition;
      out_fact->condition_value =
          current_facts[predecessor_index].condition_value;
      out_fact->condition_known = true;
    }
  }

  loom_condition_fact_set_t edge_facts;
  loom_condition_fact_set_initialize(relation_storage, relation_capacity,
                                     &edge_facts);
  if (edge_has_condition) {
    loom_condition_facts_query(module, fact_table, edge_condition, edge_value,
                               &edge_facts);
  }
  if (current_facts) {
    const loom_cfg_block_entry_condition_facts_t* predecessor_facts =
        &current_facts[predecessor_index];
    loom_cfg_condition_append_remapped_relations(
        module, dominance, block, predecessor_terminator,
        predecessor_facts->integer_relations,
        predecessor_facts->integer_relation_count, relation_storage,
        relation_capacity, &edge_facts.integer_relation_count);
  }

  out_fact->integer_relation_count = edge_facts.integer_relation_count;
}

static bool loom_cfg_condition_block_entry_facts_equal(
    const loom_cfg_block_entry_condition_facts_t* left,
    const loom_cfg_block_entry_condition_facts_t* right) {
  if (left->condition != right->condition ||
      left->condition_value != right->condition_value ||
      left->condition_known != right->condition_known ||
      left->integer_relation_count != right->integer_relation_count) {
    return false;
  }
  for (iree_host_size_t i = 0; i < left->integer_relation_count; ++i) {
    if (!loom_condition_integer_relations_equivalent(
            &left->integer_relations[i], &right->integer_relations[i])) {
      return false;
    }
  }
  return true;
}

static void loom_cfg_condition_compute_block_entry_facts(
    const loom_module_t* module, const loom_cfg_graph_t* graph,
    const loom_value_fact_table_t* fact_table,
    const loom_dominance_info_t* dominance,
    const loom_cfg_block_entry_condition_facts_t* current_facts,
    uint16_t block_index, loom_condition_integer_relation_t* relation_storage,
    loom_cfg_block_entry_condition_facts_t* out_fact) {
  *out_fact = (loom_cfg_block_entry_condition_facts_t){
      .condition = LOOM_VALUE_ID_INVALID,
  };
  if (block_index == 0 ||
      !loom_cfg_graph_block_is_reachable(graph, block_index)) {
    return;
  }

  const loom_block_t* block = graph->blocks[block_index].block;
  if (!block) return;

  bool saw_reachable_predecessor = false;
  bool condition_candidate_initialized = false;
  bool condition_candidate_valid = true;
  iree_host_size_t relation_count = 0;
  bool relation_meet_initialized = false;
  loom_cfg_block_index_span_t predecessors =
      loom_cfg_graph_predecessors(graph, block_index);
  for (iree_host_size_t i = 0; i < predecessors.count; ++i) {
    uint16_t predecessor_index = predecessors.values[i];
    if (!loom_cfg_graph_block_is_reachable(graph, predecessor_index)) continue;
    const loom_block_t* predecessor = graph->blocks[predecessor_index].block;
    if (!predecessor) return;
    saw_reachable_predecessor = true;

    loom_condition_integer_relation_t
        edge_relation_storage[LOOM_CFG_CONDITION_FACT_RELATION_CAPACITY];
    loom_cfg_block_entry_condition_facts_t edge_fact = {0};
    loom_cfg_condition_facts_compute_predecessor_edge(
        module, fact_table, dominance, block, predecessor->last_op,
        predecessor_index, current_facts, edge_relation_storage,
        IREE_ARRAYSIZE(edge_relation_storage), &edge_fact);
    if (!edge_fact.condition_known ||
        !loom_cfg_condition_set_edge_bool_fact(
            edge_fact.condition, edge_fact.condition_value,
            &out_fact->condition, &out_fact->condition_value,
            &condition_candidate_initialized)) {
      condition_candidate_valid = false;
    }

    if (!relation_meet_initialized) {
      relation_count = edge_fact.integer_relation_count;
      memcpy(relation_storage, edge_fact.integer_relations,
             relation_count * sizeof(relation_storage[0]));
      relation_meet_initialized = true;
      continue;
    }

    relation_count = loom_cfg_condition_intersect_relations(
        relation_storage, relation_count, edge_fact.integer_relations,
        edge_fact.integer_relation_count);
  }

  if (!saw_reachable_predecessor) return;
  out_fact->condition_known =
      condition_candidate_valid && condition_candidate_initialized;
  if (!out_fact->condition_known) {
    out_fact->condition = LOOM_VALUE_ID_INVALID;
  }
  out_fact->integer_relations = relation_storage;
  out_fact->integer_relation_count = relation_count;
}

static iree_status_t loom_cfg_condition_copy_relations(
    iree_arena_allocator_t* arena,
    const loom_condition_integer_relation_t* relations,
    iree_host_size_t relation_count,
    loom_cfg_block_entry_condition_facts_t* inout_fact) {
  if (relation_count == 0) return iree_ok_status();
  loom_condition_integer_relation_t* copied_relations = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, relation_count,
                                                 sizeof(*copied_relations),
                                                 (void**)&copied_relations));
  memcpy(copied_relations, relations, relation_count * sizeof(*relations));
  inout_fact->integer_relations = copied_relations;
  inout_fact->integer_relation_count = relation_count;
  return iree_ok_status();
}

iree_status_t loom_cfg_condition_fact_table_compute(
    const loom_module_t* module, const loom_cfg_graph_t* graph,
    const loom_value_fact_table_t* fact_table,
    const loom_dominance_info_t* dominance, iree_arena_allocator_t* arena,
    loom_cfg_condition_fact_table_t* out_table) {
  *out_table = (loom_cfg_condition_fact_table_t){0};
  if (!module || !graph || !dominance || !arena || graph->malformed ||
      graph->block_count == 0) {
    return iree_ok_status();
  }

  loom_cfg_block_entry_condition_facts_t* facts = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, graph->block_count, sizeof(*facts), (void**)&facts));
  memset(facts, 0, graph->block_count * sizeof(*facts));

  iree_host_size_t max_iterations = (iree_host_size_t)graph->block_count + 1;
  for (iree_host_size_t iteration = 0; iteration < max_iterations;
       ++iteration) {
    bool changed = false;
    for (uint16_t block_index = 0; block_index < graph->block_count;
         ++block_index) {
      loom_condition_integer_relation_t
          relation_storage[LOOM_CFG_CONDITION_FACT_RELATION_CAPACITY];
      loom_cfg_block_entry_condition_facts_t computed_facts = {0};
      loom_cfg_condition_compute_block_entry_facts(
          module, graph, fact_table, dominance, facts, block_index,
          relation_storage, &computed_facts);
      if (loom_cfg_condition_block_entry_facts_equal(&facts[block_index],
                                                     &computed_facts)) {
        continue;
      }
      loom_cfg_block_entry_condition_facts_t copied_facts = computed_facts;
      copied_facts.integer_relations = NULL;
      copied_facts.integer_relation_count = 0;
      IREE_RETURN_IF_ERROR(loom_cfg_condition_copy_relations(
          arena, computed_facts.integer_relations,
          computed_facts.integer_relation_count, &copied_facts));
      facts[block_index] = copied_facts;
      changed = true;
    }
    if (!changed) {
      out_table->block_facts = facts;
      out_table->block_count = graph->block_count;
      return iree_ok_status();
    }
  }

  // The lattice should converge before the region-size bound. If it does not,
  // fail closed by leaving the table empty; consumers use missing facts as
  // unknown instead of proving transformations or memory safety from a partial
  // fixed point.
  return iree_ok_status();
}

const loom_cfg_block_entry_condition_facts_t*
loom_cfg_condition_fact_table_block(
    const loom_cfg_condition_fact_table_t* table, uint16_t block_index) {
  if (!table || block_index >= table->block_count) return NULL;
  return &table->block_facts[block_index];
}

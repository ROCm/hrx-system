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
#include "loom/util/fact_loop.h"
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
  const loom_scalar_type_t scalar_type = loom_type_element_type(type);
  if (loom_scalar_type_is_float(scalar_type)) {
    // Predicate transfer must distinguish integer intervals from the compact
    // floating-point payload carried in the same fields.
    facts.flags |= LOOM_VALUE_FACT_FLOAT;
    return facts;
  }
  int64_t lo = 0;
  int64_t hi = 0;
  if (!loom_value_facts_scalar_type_domain(scalar_type, &lo, &hi)) {
    return facts;
  }
  return loom_value_facts_clamp_domain(facts, lo, hi);
}

// Declaration arguments have no executing definition.
static const loom_block_t* loom_value_fact_table_definition_block(
    const loom_value_t* value) {
  if (loom_value_is_block_arg(value)) {
    return loom_value_def_block(value);
  }
  const loom_op_t* op = loom_value_def_op(value);
  return op ? op->parent_block : NULL;
}

// A shaped type references scalar values only through its dimensions; its
// optional layout reference has encoding type. Consume the maintained reverse
// type-use index instead of finding extents by walking other definitions.
static bool loom_value_fact_table_has_extent_domain(const loom_module_t* module,
                                                    loom_value_id_t value_id) {
  if (!loom_value_fact_table_dynamic_extent_type_supported(
          loom_module_value_type(module, value_id)) ||
      !loom_module_value_has_type_uses(module, value_id)) {
    return false;
  }
  const loom_block_t* block = loom_value_fact_table_definition_block(
      loom_module_value(module, value_id));
  if (!block) {
    return false;
  }
  loom_type_use_iterator_t users;
  loom_module_value_type_users(module, value_id, &users);
  for (loom_value_id_t user_id = loom_type_users_next(&users);
       user_id != LOOM_VALUE_ID_INVALID;
       user_id = loom_type_users_next(&users)) {
    if (!loom_type_is_shaped(loom_module_value_type(module, user_id))) {
      continue;
    }
    // A conditional view cannot constrain captures on paths that skip it.
    if (loom_value_fact_table_definition_block(
            loom_module_value(module, user_id)) == block) {
      return true;
    }
  }
  return false;
}

static loom_value_facts_t loom_value_fact_table_clamp_extent_domain(
    const loom_module_t* module, loom_value_id_t value_id,
    loom_value_facts_t facts) {
  return loom_value_fact_table_clamp_scalar_type_domain(
      module, value_id, loom_value_facts_non_negative_extent(facts));
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
    if (loom_value_fact_table_has_extent_domain(module, value_id)) {
      IREE_RETURN_IF_ERROR(loom_value_fact_table_define(
          table, value_id,
          loom_value_fact_table_clamp_extent_domain(
              module, value_id,
              loom_value_fact_table_lookup(table, value_id))));
    }
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
  return loom_value_fact_table_define(table, arg_id, facts);
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

// Numeric loop constraints are independent of widening and lane distribution.
// Every member of a forwarding component uses its own header observation.
static loom_value_facts_t loom_value_fact_table_refine_cfg_recurrence(
    const loom_value_fact_table_t* table, const loom_module_t* module,
    const loom_value_fact_cfg_region_t* region, loom_value_id_t argument,
    loom_value_facts_t facts) {
  const loom_value_t* value = loom_module_value(module, argument);
  const uint16_t block_index = loom_value_def_block(value)->region_index;
  const uint16_t loop_index =
      loom_cfg_loop_nest_innermost(&region->loops, block_index);
  if (loop_index == LOOM_CFG_LOOP_NEST_NONE ||
      region->loops.loops[loop_index].header_index != block_index) {
    return facts;
  }
  const loom_value_fact_induction_t* induction =
      &region->inductions[loop_index];
  const loom_loop_recurrence_facts_t counter =
      loom_value_fact_induction_facts(table, module, induction);
  if (induction->value == argument) {
    facts = loom_value_fact_recurrence_refine(facts, counter.values);
  }
  const loom_value_fact_recurrence_t* equation =
      loom_value_fact_recurrence_set_lookup(&region->recurrences[loop_index],
                                            loom_value_def_index(value));
  loom_loop_recurrence_facts_t recurrence;
  if (counter.trip_count_known && equation &&
      loom_value_fact_recurrence_evaluate(table, module, equation,
                                          counter.trip_count, &recurrence)) {
    facts = loom_value_fact_recurrence_refine(facts, recurrence.values);
  }
  return facts;
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
  // Forwarding members share numeric inputs, but their type constraints are
  // local to each definition's execution block. Settle each complete equation
  // before widening or comparing it with the previous iteration.
  for (iree_host_size_t member = 0; member < member_count; ++member) {
    loom_value_id_t value_id =
        component ? region->arguments[component->nodes[member]].value_id
                  : arg_id;
    loom_type_t member_type = loom_module_value_type(module, value_id);
    const bool is_extent =
        loom_value_fact_table_has_extent_domain(module, value_id);
    loom_value_facts_t facts = incoming_facts;
    if (is_extent) {
      facts =
          loom_value_fact_table_clamp_extent_domain(module, value_id, facts);
    }
    if (widen && !exits_at_header &&
        loom_value_fact_table_has_entry(table, value_id)) {
      loom_value_facts_t current_facts =
          loom_value_fact_table_lookup(table, value_id);
      IREE_RETURN_IF_ERROR(loom_value_fact_table_widen_for_type(
          table, module, member_type, table, current_facts, table, facts,
          iteration, &facts));
      if (is_extent) {
        facts =
            loom_value_fact_table_clamp_extent_domain(module, value_id, facts);
      }
    }
    facts = loom_value_fact_table_refine_cfg_recurrence(table, module, region,
                                                        value_id, facts);
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
    iree_host_size_t control_flow_component, uint32_t iteration, bool widen,
    iree_arena_allocator_t* recurrence_arena, bool* out_changed) {
  IREE_RETURN_IF_ERROR(loom_value_fact_cfg_update_forwarding(
      region, control_flow_component, table->transient_arena));
  const loom_scc_t* blocks =
      &region->control_flow.components.values[control_flow_component];
  if (widen && iteration == 1) {
    for (iree_host_size_t i = 0; i < blocks->node_count; ++i) {
      IREE_RETURN_IF_ERROR(loom_value_fact_cfg_build_recurrences(
          table, module, region, (uint16_t)blocks->nodes[i], recurrence_arena));
    }
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
        argument->block_index, argument->argument_index, widen, iteration, NULL,
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

static loom_op_t* loom_value_fact_region_terminator(loom_region_t* region) {
  if (!region || region->block_count == 0) {
    return NULL;
  }
  loom_block_t* block = loom_region_entry_block(region);
  return block && block->op_count > 0 ? block->last_op : NULL;
}

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

#define LOOM_VALUE_FACT_CFG_SAVED_CHUNK_CAPACITY 128u

static_assert(LOOM_VALUE_FACT_CFG_SAVED_CHUNK_CAPACITY <= UINT8_MAX,
              "saved fact chunk count must fit in one byte");

// Saved facts distinguish changed results from values merely revisited while
// solving a cycle. Extension IDs continue to name payloads in the same table.
typedef struct loom_value_fact_cfg_saved_chunk_t {
  // Next chunk in definition order, or NULL at the end of the stream.
  struct loom_value_fact_cfg_saved_chunk_t* next;
  // Values whose equations are being restarted.
  loom_value_id_t value_ids[LOOM_VALUE_FACT_CFG_SAVED_CHUNK_CAPACITY];
  // Facts before this update, indexed by the same row as value_ids.
  loom_value_facts_t facts[LOOM_VALUE_FACT_CFG_SAVED_CHUNK_CAPACITY];
} loom_value_fact_cfg_saved_chunk_t;

typedef struct loom_value_fact_cfg_saved_values_t {
  // First saved chunk for the component and its nested definitions.
  loom_value_fact_cfg_saved_chunk_t* first;
  // Append cursor. Every preceding chunk has all rows initialized.
  struct {
    // Final chunk, or NULL when no definitions have been saved.
    loom_value_fact_cfg_saved_chunk_t* chunk;
    // Initialized rows in the final chunk's value-ID and fact columns.
    uint8_t count;
  } tail;
} loom_value_fact_cfg_saved_values_t;

static iree_status_t loom_value_fact_table_save_cfg_value(
    loom_value_fact_table_t* table, loom_value_id_t value_id,
    iree_arena_allocator_t* arena, loom_value_fact_cfg_saved_values_t* saved) {
  if (value_id == LOOM_VALUE_ID_INVALID) {
    return iree_ok_status();
  }
  loom_value_fact_cfg_saved_chunk_t* chunk = saved->tail.chunk;
  if (!chunk || saved->tail.count == LOOM_VALUE_FACT_CFG_SAVED_CHUNK_CAPACITY) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate(arena, sizeof(*chunk), (void**)&chunk));
    chunk->next = NULL;
    if (saved->tail.chunk) {
      saved->tail.chunk->next = chunk;
    } else {
      saved->first = chunk;
    }
    saved->tail.chunk = chunk;
    saved->tail.count = 0;
  }
  chunk->value_ids[saved->tail.count] = value_id;
  chunk->facts[saved->tail.count] =
      loom_value_fact_table_lookup(table, value_id);
  ++saved->tail.count;
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
  for (const loom_value_fact_cfg_saved_chunk_t* chunk = saved->first; chunk;
       chunk = chunk->next) {
    const uint32_t value_count = chunk->next
                                     ? LOOM_VALUE_FACT_CFG_SAVED_CHUNK_CAPACITY
                                     : saved->tail.count;
    for (uint32_t i = 0; i < value_count; ++i) {
      loom_value_id_t value_id = chunk->value_ids[i];
      IREE_RETURN_IF_ERROR(loom_value_fact_table_define(
          table, value_id,
          loom_value_fact_table_unknown_for_value(module, value_id)));
    }
  }
  return iree_ok_status();
}

// A reset discards provisional feedback, not the loop's declared equations.
// Evaluate those equations against the reset table: an inner loop whose count
// depends on forgotten outer state cannot reuse its earlier numeric proof.
static iree_status_t loom_value_fact_table_seed_cfg_recurrences(
    loom_value_fact_table_t* table, const loom_module_t* module,
    const loom_value_fact_cfg_region_t* region, uint16_t block_index) {
  const uint16_t loop_index =
      loom_cfg_loop_nest_innermost(&region->loops, block_index);
  if (loop_index == LOOM_CFG_LOOP_NEST_NONE ||
      region->loops.loops[loop_index].header_index != block_index) {
    return iree_ok_status();
  }
  const loom_block_t* block = region->graph.blocks[block_index].block;
  for (uint16_t i = 0; i < block->arg_count; ++i) {
    const loom_value_id_t value_id = loom_block_arg_id(block, i);
    const loom_value_facts_t facts =
        loom_value_fact_table_refine_cfg_recurrence(
            table, module, region, value_id,
            loom_value_fact_table_lookup(table, value_id));
    IREE_RETURN_IF_ERROR(loom_value_fact_table_define(table, value_id, facts));
  }
  return iree_ok_status();
}

// A component's inputs are stable before its solve. Only its feedback needs
// iteration; definitions in predecessor and successor components do not.
static iree_status_t loom_value_fact_table_solve_cfg_component(
    loom_value_fact_table_t* table, const loom_module_t* module,
    const loom_value_fact_cfg_region_t* region, const loom_scc_t* component,
    iree_arena_allocator_t* scratch_arena,
    const loom_value_fact_cfg_saved_values_t* restart_values) {
  iree_host_size_t component_index =
      component - region->control_flow.components.values;
  const iree_host_size_t* blocks = component->nodes;
  loom_value_fact_cfg_seed_control(table, region, component);
  bool converged = false;
  for (uint32_t iteration = 0; iteration < LOOM_VALUE_FACT_CFG_MAX_ITERATIONS;
       ++iteration) {
    bool changed = false;
    // Seed producers in execution order before joining the whole forwarding
    // partition. An inner loop's initializer can live inside this component.
    if (iteration != 0) {
      IREE_RETURN_IF_ERROR(loom_value_fact_table_compute_cfg_forwarding(
          table, module, region, component_index, iteration, /*widen=*/true,
          scratch_arena, &changed));
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
  if (converged) {
    // The widening solve has established a post-fixed point. A descending
    // transfer sweep can recover bounded inputs that reached a carried value
    // after widening, without another convergence solve or new analysis.
    bool changed = false;
    IREE_RETURN_IF_ERROR(loom_value_fact_table_compute_cfg_forwarding(
        table, module, region, component_index, /*iteration=*/0,
        /*widen=*/false, scratch_arena, &changed));
    if (changed) {
      for (iree_host_size_t i = 0; i < component->node_count; ++i) {
        IREE_RETURN_IF_ERROR(loom_value_fact_table_compute_cfg_block_tree(
            table, module, region->graph.blocks[blocks[i]].block, &changed));
      }
    }
  } else {
    loom_value_fact_cfg_saved_values_t initial_values = {0};
    if (!restart_values) {
      for (iree_host_size_t i = 0; i < component->node_count; ++i) {
        IREE_RETURN_IF_ERROR(loom_value_fact_table_save_cfg_block(
            table, region->graph.blocks[blocks[i]].block, blocks[i] != 0,
            scratch_arena, &initial_values));
      }
      restart_values = &initial_values;
    }
    IREE_RETURN_IF_ERROR(
        loom_value_fact_table_reset_cfg_values(table, module, restart_values));
    loom_value_fact_cfg_seed_control(table, region, component);
    for (iree_host_size_t i = 0; i < component->node_count; ++i) {
      bool changed = false;
      IREE_RETURN_IF_ERROR(loom_value_fact_table_seed_cfg_recurrences(
          table, module, region, blocks[i]));
      IREE_RETURN_IF_ERROR(loom_value_fact_table_compute_cfg_block_tree(
          table, module, region->graph.blocks[blocks[i]].block, &changed));
    }
  }
  return iree_ok_status();
}

iree_status_t loom_value_fact_table_recompute_cfg_component(
    loom_value_fact_table_t* table, const loom_module_t* module,
    const loom_value_fact_cfg_region_t* region, const loom_scc_t* component,
    iree_arena_allocator_t* scratch_arena,
    loom_value_fact_cfg_changed_fn_t on_changed, void* user_data) {
  loom_value_fact_cfg_saved_values_t saved = {0};
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < component->node_count && iree_status_is_ok(status); ++i) {
    const iree_host_size_t block_index = component->nodes[i];
    status = loom_value_fact_table_save_cfg_block(
        table, region->graph.blocks[block_index].block, block_index != 0,
        scratch_arena, &saved);
  }
  if (iree_status_is_ok(status)) {
    status = loom_value_fact_table_solve_cfg_component(
        table, module, region, component, scratch_arena, &saved);
  }
  for (const loom_value_fact_cfg_saved_chunk_t* chunk = saved.first;
       chunk && iree_status_is_ok(status); chunk = chunk->next) {
    const uint32_t value_count = chunk->next
                                     ? LOOM_VALUE_FACT_CFG_SAVED_CHUNK_CAPACITY
                                     : saved.tail.count;
    for (uint32_t i = 0; i < value_count && iree_status_is_ok(status); ++i) {
      loom_value_id_t value_id = chunk->value_ids[i];
      if (!loom_value_fact_table_facts_equal_for_type(
              module, loom_module_value_type(module, value_id), table,
              chunk->facts[i], table,
              loom_value_fact_table_lookup(table, value_id))) {
        status = on_changed(user_data, value_id);
      }
    }
  }
  for (iree_host_size_t i = 0; i < component->node_count; ++i) {
    const uint16_t block_index = component->nodes[i];
    const uint16_t loop_index =
        loom_cfg_loop_nest_innermost(&region->loops, block_index);
    if (loop_index != LOOM_CFG_LOOP_NEST_NONE &&
        region->loops.loops[loop_index].header_index == block_index) {
      region->recurrences[loop_index] = (loom_value_fact_recurrence_set_t){0};
    }
  }
  return status;
}

static iree_status_t loom_value_fact_table_solve_cfg_region_tree(
    loom_value_fact_table_t* table, const loom_module_t* module,
    loom_region_t* region, loom_op_t* parent_op,
    const loom_value_fact_cfg_region_t* structure,
    iree_arena_allocator_t* scratch_arena) {
  if (region->block_count == 0) {
    return iree_ok_status();
  }
  loom_block_t* entry_block = loom_region_entry_block(region);
  IREE_RETURN_IF_ERROR(loom_value_fact_table_seed_block_args(
      table, module, entry_block, parent_op));

  const loom_cfg_graph_t* graph = &structure->graph;
  const loom_scc_list_t* components = &structure->control_flow.components;
  const iree_host_size_t count =
      graph->has_cycles ? components->count : graph->reverse_postorder.count;
  for (iree_host_size_t i = 0; i < count; ++i) {
    // Retained components have successor-before-predecessor ordinals. Solve
    // predecessors first so a cycle publishes its fixed point before any
    // downstream block consumes it. A DAG already has topological RPO.
    const loom_scc_t* component =
        graph->has_cycles ? &components->values[count - 1 - i] : NULL;
    if (component && component->is_cycle) {
      IREE_RETURN_IF_ERROR(loom_value_fact_table_solve_cfg_component(
          table, module, structure, component, scratch_arena, NULL));
    } else {
      const uint16_t block_index =
          component ? component->nodes[0] : graph->reverse_postorder.values[i];
      bool changed = false;
      IREE_RETURN_IF_ERROR(loom_value_fact_table_compute_cfg_block_args(
          table, module, structure, block_index, &changed));
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

static iree_status_t loom_value_fact_table_compute_cfg_region_tree(
    loom_value_fact_table_t* table, const loom_module_t* module,
    loom_region_t* region, loom_op_t* parent_op) {
  const loom_value_fact_cfg_region_t* structure = NULL;
  IREE_RETURN_IF_ERROR(loom_value_fact_table_get_or_build_cfg_region(
      table, module, region, &structure));
  iree_arena_allocator_t recurrence_arena;
  iree_arena_initialize(table->transient_arena->block_pool, &recurrence_arena);
  iree_status_t status = loom_value_fact_table_solve_cfg_region_tree(
      table, module, region, parent_op, structure, &recurrence_arena);
  if (structure->loops.loop_count != 0) {
    memset(structure->recurrences, 0,
           structure->loops.loop_count * sizeof(*structure->recurrences));
  }
  iree_arena_deinitialize(&recurrence_arena);
  return status;
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

iree_status_t loom_value_fact_table_define_results(
    loom_value_fact_table_t* table, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* result_facts,
    uint16_t result_count, bool* out_changed) {
  const loom_value_id_t* results = loom_op_const_results(op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    loom_value_id_t result = results[i];
    if (result == LOOM_VALUE_ID_INVALID || result >= module->values.count) {
      continue;
    }
    loom_value_facts_t facts =
        i < result_count ? result_facts[i] : loom_value_facts_unknown();
    if (loom_value_fact_table_has_extent_domain(module, result)) {
      facts = loom_value_facts_non_negative_extent(facts);
    }
    // Inference may be less precise than the verified scalar result type.
    facts =
        loom_value_fact_table_clamp_scalar_type_domain(module, result, facts);
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

// Branch regions define disjoint SSA values. Join their yields after all
// regions have been evaluated, without retaining a native call frame during
// descent.
static iree_status_t loom_value_fact_table_compute_region_branch_results(
    loom_value_fact_table_t* table, const loom_module_t* module, loom_op_t* op,
    bool* out_changed) {
  if (op->result_count == 0) {
    return iree_ok_status();
  }
  loom_region_branch_t branch = loom_region_branch_cast(module, op);
  loom_value_fact_region_branch_result_state_t* result_states = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      table->transient_arena, op->result_count,
      sizeof(loom_value_fact_region_branch_result_state_t),
      (void**)&result_states));

  uint8_t visited_region_count = 0;
  const loom_value_id_t* results = loom_op_const_results(op);
  for (uint8_t region_index = 0; region_index < op->region_count;
       ++region_index) {
    loom_region_t* region =
        loom_region_branch_region(module, branch, region_index);
    if (!region) {
      continue;
    }
    loom_op_t* terminator =
        loom_region_branch_region_terminator(module, branch, region_index);
    IREE_ASSERT(terminator);

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
  return loom_value_fact_table_define_results(table, module, op, result_facts,
                                              op->result_count, out_changed);
}

//===----------------------------------------------------------------------===//
// Forward pass: compute facts for ops
//===----------------------------------------------------------------------===//

iree_status_t loom_value_fact_table_compute_op(loom_value_fact_table_t* table,
                                               const loom_module_t* module,
                                               const loom_op_t* op) {
  return loom_value_fact_table_compute_op_and_report(table, module, op, NULL);
}

static iree_status_t loom_value_fact_table_compute_scoped_op(
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
    return loom_value_fact_table_compute_region_branch_results(
        table, module, (loom_op_t*)op, out_changed);
  }
  if (!vtable || !vtable->infer_facts) {
    // Opaque operations still define complete results. Apply sibling type
    // constraints before publishing, so a later result cannot overwrite a
    // range established by an earlier shaped result.
    loom_value_facts_t* result_facts = NULL;
    IREE_RETURN_IF_ERROR(loom_value_fact_table_facts_scratch(
        table, op->result_count, &result_facts));
    const loom_value_id_t* results = loom_op_const_results(op);
    for (uint16_t i = 0; i < op->result_count; ++i) {
      const loom_value_id_t result = results[i];
      if (result / 64 < table->seeds.word_count &&
          (table->seeds.selected_bits[result / 64] &
           (UINT64_C(1) << (result % 64)))) {
        IREE_RETURN_IF_ERROR(loom_value_fact_table_clone_fact_for_type(
            table, table->seeds.table, module,
            loom_module_value_type(module, result),
            loom_value_fact_table_lookup(table->seeds.table, result),
            &result_facts[i]));
      } else {
        result_facts[i] =
            loom_value_fact_table_unknown_for_value(module, result);
      }
    }
    IREE_RETURN_IF_ERROR(loom_value_fact_table_define_results(
        table, module, op, result_facts, op->result_count, out_changed));
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
  if (table->context.refine_operands.fn) {
    table->context.refine_operands.fn(table->context.refine_operands.user_data,
                                      table, op, operand_facts);
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

  IREE_RETURN_IF_ERROR(loom_value_fact_table_define_results(
      table, module, op, result_facts, op->result_count, out_changed));

  return loom_value_fact_table_propagate_origins(table, module, op,
                                                 out_changed);
}

static iree_status_t loom_value_fact_table_compute_op_in_target_scope(
    loom_value_fact_table_t* table, const loom_module_t* module,
    const loom_op_t* op, bool* out_changed) {
  if (!table->regions.has_independent_targets) {
    return loom_value_fact_table_compute_scoped_op(table, module, op,
                                                   out_changed);
  }
  const loom_target_facts_t* parent_target = table->context.target_facts;
  table->context.target_facts =
      loom_value_fact_table_block_target_facts(table, op->parent_block);
  iree_status_t status =
      loom_value_fact_table_compute_scoped_op(table, module, op, out_changed);
  table->context.target_facts = parent_target;
  return status;
}

static iree_status_t loom_value_fact_table_seed_branch_region(
    loom_value_fact_table_t* table, loom_region_branch_t branch,
    loom_region_t* region, uint8_t region_index) {
  const loom_region_branch_truth_t truth =
      loom_region_branch_region_truth(branch, region_index);
  if (truth == LOOM_REGION_BRANCH_TRUTH_UNKNOWN) {
    return iree_ok_status();
  }
  return loom_value_fact_table_set_region_branch_truth(table, region, truth);
}

iree_status_t loom_value_fact_table_compute_op_and_report(
    loom_value_fact_table_t* table, const loom_module_t* module,
    const loom_op_t* op, bool* out_changed) {
  IREE_RETURN_IF_ERROR(
      loom_value_fact_table_seed_nested_target_scopes(table, module, op));
  const loom_region_branch_t branch =
      op->region_count ? loom_region_branch_cast(module, (loom_op_t*)op)
                       : (loom_region_branch_t){0};
  if (loom_region_branch_isa(branch)) {
    for (uint8_t i = 0; i < op->region_count; ++i) {
      loom_region_t* region = loom_region_branch_region(module, branch, i);
      if (!region) {
        continue;
      }
      IREE_RETURN_IF_ERROR(
          loom_value_fact_table_seed_branch_region(table, branch, region, i));
      IREE_RETURN_IF_ERROR(loom_value_fact_table_compute_region_tree(
          table, module, region, (loom_op_t*)op));
    }
  }
  return loom_value_fact_table_compute_op_in_target_scope(table, module, op,
                                                          out_changed);
}

// Continuations for acyclic lexical descent. Loop and CFG solvers retain their
// own fixed-point iteration; each body evaluation uses this same traversal.
typedef struct loom_value_fact_region_frame_t {
  // Region whose blocks are being evaluated.
  loom_region_t* region;
  // Owner supplying block-argument and execution-scope facts.
  loom_op_t* parent_op;
  // Next block to seed after the current block's operations finish.
  uint16_t next_block_index;
  // Next operation in the current block, or NULL at the block's end.
  loom_op_t* next_op;
  // Suspended operation while its nested regions are evaluated.
  struct {
    // Owner of the regions, or NULL when no descent is pending.
    loom_op_t* op;
    // Branch contract requiring a result join, or NULL for ordinary regions.
    const loom_region_branch_vtable_t* branch;
    // Next region of op to evaluate.
    uint8_t next_region_index;
  } children;
} loom_value_fact_region_frame_t;

static iree_status_t loom_value_fact_table_begin_region(
    loom_value_fact_table_t* table, const loom_module_t* module,
    loom_region_t* region, loom_op_t* parent_op,
    loom_value_fact_region_frame_t* out_frame) {
  *out_frame = (loom_value_fact_region_frame_t){
      .region = region,
      .parent_op = parent_op,
  };
  loom_value_facts_t temporal_scope = loom_value_facts_unknown();
  bool may_repeat = false;
  if (!parent_op || parent_op == table->context.function.op) {
    loom_value_facts_mark_cluster_uniform(&temporal_scope);
  } else {
    temporal_scope = loom_value_fact_table_block_temporal_scope(
        table, parent_op->parent_block);
    may_repeat =
        loom_loop_like_isa(loom_loop_like_cast(module, parent_op)) ||
        loom_value_fact_table_block_may_repeat(table, parent_op->parent_block);
  }
  IREE_RETURN_IF_ERROR(loom_value_fact_table_set_region_temporal_scope(
      table, region, temporal_scope, may_repeat));
  if (iree_any_bit_set(region->flags, LOOM_REGION_INSTANCE_FLAG_CFG)) {
    out_frame->next_block_index = region->block_count;
    return loom_value_fact_table_compute_cfg_region_tree(table, module, region,
                                                         parent_op);
  }
  return iree_ok_status();
}

iree_status_t loom_value_fact_table_compute_region_tree(
    loom_value_fact_table_t* table, const loom_module_t* module,
    loom_region_t* region, loom_op_t* parent_op) {
  if (!region) {
    return iree_ok_status();
  }
  // Flat regions need no traversal allocation. Nested regions grow the stack
  // in the existing transient arena instead of consuming native stack space.
  loom_value_fact_region_frame_t root_frame;
  IREE_RETURN_IF_ERROR(loom_value_fact_table_begin_region(
      table, module, region, parent_op, &root_frame));
  loom_value_fact_region_frame_t* stack = &root_frame;
  iree_host_size_t stack_capacity = 1;
  iree_host_size_t stack_count = 1;
  while (stack_count > 0) {
    loom_value_fact_region_frame_t* frame = &stack[stack_count - 1];
    if (frame->children.op) {
      loom_op_t* op = frame->children.op;
      if (frame->children.next_region_index < op->region_count) {
        const uint8_t region_index = frame->children.next_region_index++;
        loom_region_t* child = loom_op_regions(op)[region_index];
        if (!child) {
          continue;
        }
        if (frame->children.branch) {
          const loom_region_branch_t branch = {
              .op = op, .vtable = frame->children.branch};
          IREE_RETURN_IF_ERROR(loom_value_fact_table_seed_branch_region(
              table, branch, child, region_index));
        }
        if (stack_count == stack_capacity) {
          IREE_RETURN_IF_ERROR(iree_arena_grow_array(
              table->transient_arena, stack_count, stack_count + 1,
              sizeof(*stack), &stack_capacity, (void**)&stack));
        }
        IREE_RETURN_IF_ERROR(loom_value_fact_table_begin_region(
            table, module, child, op, &stack[stack_count]));
        ++stack_count;
        continue;
      }
      if (frame->children.branch) {
        IREE_RETURN_IF_ERROR(loom_value_fact_table_compute_op_in_target_scope(
            table, module, op, NULL));
      }
      frame->children.op = NULL;
      continue;
    }
    if (!frame->next_op) {
      if (frame->next_block_index == frame->region->block_count) {
        --stack_count;
        continue;
      }
      loom_block_t* block = frame->region->blocks[frame->next_block_index++];
      if (!block) {
        continue;
      }
      IREE_RETURN_IF_ERROR(loom_value_fact_table_seed_block_args(
          table, module, block, frame->parent_op));
      frame->next_op = block->first_op;
      continue;
    }
    loom_op_t* op = frame->next_op;
    frame->next_op = op->next_op;
    IREE_RETURN_IF_ERROR(
        loom_value_fact_table_seed_nested_target_scopes(table, module, op));
    const loom_op_vtable_t* vtable = loom_op_vtable(module, op);
    const loom_region_branch_vtable_t* branch =
        vtable ? vtable->region_branch : NULL;
    if (!branch) {
      IREE_RETURN_IF_ERROR(loom_value_fact_table_compute_op_in_target_scope(
          table, module, op, NULL));
      if (!op->region_count || (vtable && vtable->loop_like)) {
        continue;
      }
    }
    frame->children.op = op;
    frame->children.branch = branch;
    frame->children.next_region_index = 0;
  }
  return iree_ok_status();
}

iree_status_t loom_value_fact_table_compute_region(
    loom_value_fact_table_t* table, const loom_module_t* module,
    loom_func_like_t function, loom_region_t* region, loom_op_t* parent_op) {
  if (region) {
    IREE_RETURN_IF_ERROR(loom_value_fact_table_seed_root_target_scope(
        table, module, region, parent_op));
  }
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

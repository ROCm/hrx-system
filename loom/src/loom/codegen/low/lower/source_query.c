// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/lower/source_query.h"

#include <string.h>

#include "loom/codegen/low/lower/context.h"
#include "loom/codegen/low/lower/contract_query.h"
#include "loom/codegen/low/lower/rule_match.h"
#include "loom/ir/module.h"
#include "loom/target/low_descriptor_registry.h"
#include "loom/target/registers.h"
#include "loom/util/walk.h"

typedef struct loom_low_lower_contract_query_state_t {
  // Mutable lowering context backing the read-only contract query.
  loom_low_lower_context_t* context;
  // Query environment provided by the target-low legality caller.
  const loom_target_contract_query_environment_t* environment;
} loom_low_lower_contract_query_state_t;

// Source-graph ownership established once for one immutable query scope.
// Result ordinals provide the dense identity for nearly every semantic op;
// result-less roots use the sparse fallback index.
typedef struct loom_low_lower_source_query_graph_t {
  // Root operation whose contract selected the source graph.
  const loom_op_t* root_op;
  // Contract result retained for the root's later worklist query.
  loom_target_contract_query_result_t root_result;
} loom_low_lower_source_query_graph_t;

typedef struct loom_low_lower_source_query_zero_result_entry_t {
  // Result-less source operation, or NULL for an empty slot.
  const loom_op_t* op;
  // Selected graph or one of the root-state marker addresses below.
  const loom_low_lower_source_query_graph_t* graph;
} loom_low_lower_source_query_zero_result_entry_t;

typedef struct loom_low_lower_source_query_coverage_t {
  // Source-value-ordinal indexed graph for each operation's first result.
  const loom_low_lower_source_query_graph_t** value_graphs;
  // Number of entries in value_graphs.
  loom_value_ordinal_t value_graph_count;
  // Sparse identity index for result-less operations.
  loom_low_lower_source_query_zero_result_entry_t* zero_result_entries;
  // Number of populated zero_result_entries.
  iree_host_size_t zero_result_count;
  // Power-of-two capacity of zero_result_entries.
  iree_host_size_t zero_result_capacity;
  // True after all source graphs in the function have been selected.
  bool prepared;
  // True while the one-time source walk is selecting graphs.
  bool preparing;
} loom_low_lower_source_query_coverage_t;

typedef struct loom_low_lower_source_query_coverage_build_state_t {
  // Lowering context that owns the query scope and coverage storage.
  loom_low_lower_context_t* context;
  // Contract environment fixed for the immutable query scope.
  const loom_target_contract_query_environment_t* environment;
  // Base contract matcher options without the graph ownership filter.
  const loom_low_lower_contract_query_options_t* options;
  // Coverage being populated or queried.
  loom_low_lower_source_query_coverage_t* coverage;
} loom_low_lower_source_query_coverage_build_state_t;

static const uint8_t loom_low_lower_source_query_coverage_state_key = 0;
static const loom_low_lower_source_query_graph_t
    loom_low_lower_source_query_legal_root_marker = {0};
static const loom_low_lower_source_query_graph_t
    loom_low_lower_source_query_unavailable_root_marker = {0};

static iree_host_size_t loom_low_lower_source_query_op_hash(
    const loom_op_t* op) {
  uint64_t value = (uint64_t)(uintptr_t)op >> 4;
  value ^= value >> 16;
  return (iree_host_size_t)(value * UINT64_C(0x9e3779b97f4a7c15));
}

static bool loom_low_lower_source_query_op_value_ordinal(
    const loom_low_lower_context_t* context, const loom_op_t* op,
    loom_value_ordinal_t* out_ordinal) {
  *out_ordinal = LOOM_VALUE_ORDINAL_INVALID;
  if (op->result_count == 0 ||
      !loom_local_value_domain_is_acquired(&context->lowering.value_domain)) {
    return false;
  }
  const loom_value_ordinal_t ordinal = loom_local_value_domain_try_ordinal(
      &context->lowering.value_domain, loom_op_const_results(op)[0]);
  if (ordinal == LOOM_VALUE_ORDINAL_INVALID) {
    return false;
  }
  *out_ordinal = ordinal;
  return true;
}

static const loom_low_lower_source_query_graph_t*
loom_low_lower_source_query_coverage_lookup(
    const loom_low_lower_context_t* context,
    const loom_low_lower_source_query_coverage_t* coverage,
    const loom_op_t* op) {
  loom_value_ordinal_t ordinal = LOOM_VALUE_ORDINAL_INVALID;
  if (loom_low_lower_source_query_op_value_ordinal(context, op, &ordinal)) {
    return coverage->value_graphs != NULL &&
                   ordinal < coverage->value_graph_count
               ? coverage->value_graphs[ordinal]
               : NULL;
  }
  if (coverage->zero_result_capacity == 0) {
    return NULL;
  }
  const iree_host_size_t mask = coverage->zero_result_capacity - 1;
  iree_host_size_t slot = loom_low_lower_source_query_op_hash(op) & mask;
  while (coverage->zero_result_entries[slot].op != NULL) {
    if (coverage->zero_result_entries[slot].op == op) {
      return coverage->zero_result_entries[slot].graph;
    }
    slot = (slot + 1) & mask;
  }
  return NULL;
}

static void loom_low_lower_source_query_coverage_insert_zero_result(
    loom_low_lower_source_query_zero_result_entry_t* entries,
    iree_host_size_t capacity, const loom_op_t* op,
    const loom_low_lower_source_query_graph_t* graph) {
  const iree_host_size_t mask = capacity - 1;
  iree_host_size_t slot = loom_low_lower_source_query_op_hash(op) & mask;
  while (entries[slot].op != NULL && entries[slot].op != op) {
    slot = (slot + 1) & mask;
  }
  entries[slot] = (loom_low_lower_source_query_zero_result_entry_t){
      .op = op,
      .graph = graph,
  };
}

static iree_status_t loom_low_lower_source_query_coverage_set(
    loom_low_lower_context_t* context,
    loom_low_lower_source_query_coverage_t* coverage, const loom_op_t* op,
    const loom_low_lower_source_query_graph_t* graph) {
  loom_value_ordinal_t ordinal = LOOM_VALUE_ORDINAL_INVALID;
  if (loom_low_lower_source_query_op_value_ordinal(context, op, &ordinal)) {
    const loom_value_ordinal_t required_count =
        context->lowering.value_domain.value_count;
    if (coverage->value_graphs == NULL ||
        coverage->value_graph_count < required_count) {
      const loom_value_ordinal_t previous_count =
          coverage->value_graphs != NULL ? coverage->value_graph_count : 0;
      const loom_low_lower_source_query_graph_t** new_value_graphs = NULL;
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          &context->function_arena, required_count, sizeof(*new_value_graphs),
          (void**)&new_value_graphs));
      if (previous_count != 0) {
        memcpy(new_value_graphs, coverage->value_graphs,
               previous_count * sizeof(*new_value_graphs));
      }
      memset(new_value_graphs + previous_count, 0,
             (required_count - previous_count) * sizeof(*new_value_graphs));
      coverage->value_graphs = new_value_graphs;
      coverage->value_graph_count = required_count;
    }
    IREE_ASSERT_LT(ordinal, coverage->value_graph_count);
    coverage->value_graphs[ordinal] = graph;
    return iree_ok_status();
  }

  const bool already_present = loom_low_lower_source_query_coverage_lookup(
                                   context, coverage, op) != NULL;
  if (!already_present &&
      (coverage->zero_result_capacity == 0 ||
       coverage->zero_result_count >= coverage->zero_result_capacity / 2)) {
    const iree_host_size_t new_capacity =
        coverage->zero_result_capacity == 0
            ? 8
            : coverage->zero_result_capacity * 2;
    if (new_capacity < coverage->zero_result_capacity) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "source query coverage capacity overflow");
    }
    loom_low_lower_source_query_zero_result_entry_t* new_entries = NULL;
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(&context->function_arena, new_capacity,
                                  sizeof(*new_entries), (void**)&new_entries));
    memset(new_entries, 0, new_capacity * sizeof(*new_entries));
    for (iree_host_size_t i = 0; i < coverage->zero_result_capacity; ++i) {
      const loom_low_lower_source_query_zero_result_entry_t entry =
          coverage->zero_result_entries[i];
      if (entry.op != NULL) {
        loom_low_lower_source_query_coverage_insert_zero_result(
            new_entries, new_capacity, entry.op, entry.graph);
      }
    }
    coverage->zero_result_entries = new_entries;
    coverage->zero_result_capacity = new_capacity;
  }

  loom_low_lower_source_query_coverage_insert_zero_result(
      coverage->zero_result_entries, coverage->zero_result_capacity, op, graph);
  if (!already_present) {
    ++coverage->zero_result_count;
  }
  return iree_ok_status();
}

static iree_status_t loom_low_lower_source_query_coverage_record_graph(
    loom_low_lower_context_t* context,
    loom_low_lower_source_query_coverage_t* coverage,
    const loom_op_t* const* source_nodes, uint8_t source_node_count,
    const loom_target_contract_query_result_t* root_result) {
  loom_low_lower_source_query_graph_t* graph = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(&context->function_arena,
                                           sizeof(*graph), (void**)&graph));
  *graph = (loom_low_lower_source_query_graph_t){
      .root_op = source_nodes[0],
      .root_result = *root_result,
  };
  for (uint8_t i = 0; i < source_node_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_lower_source_query_coverage_set(
        context, coverage, source_nodes[i], graph));
  }
  return iree_ok_status();
}

static bool loom_low_lower_source_query_case_has_related_nodes(
    const loom_low_lower_contract_query_options_t* options,
    const loom_target_contract_case_t* contract_case) {
  const loom_target_contract_binding_t* binding =
      &options->contract_index->bindings[contract_case->binding_index];
  if (!loom_target_contract_fragment_queries_target(binding->fragment)) {
    return false;
  }
  uint16_t rule_index = UINT16_MAX;
  if (!loom_low_lower_contract_case_lower_rule_index(
          options->contract_index, contract_case, &rule_index)) {
    return false;
  }
  const loom_low_lower_rule_set_t* rule_set =
      options->rule_sets.values[binding->rule_set_index];
  return loom_low_lower_rule_source_node_count(&rule_set->rules[rule_index]) !=
         0;
}

static bool loom_low_lower_source_query_has_related_rules(
    const loom_low_lower_contract_query_options_t* options) {
  if (options->contract_index == NULL) {
    return false;
  }
  for (uint8_t i = 0; i < options->contract_index->binding_count; ++i) {
    const loom_target_contract_binding_t* binding =
        &options->contract_index->bindings[i];
    if (binding->rule_set_index < options->rule_sets.count &&
        loom_target_contract_fragment_queries_target(binding->fragment) &&
        options->rule_sets.values[binding->rule_set_index]->source_node_count !=
            0) {
      return true;
    }
  }
  return false;
}

static bool loom_low_lower_source_query_op_has_related_rule(
    const loom_low_lower_contract_query_options_t* options,
    const loom_op_t* op) {
  const loom_target_contract_op_entry_t op_entry =
      loom_target_contract_index_lookup_kind(options->contract_index, op->kind);
  for (uint16_t i = 0; i < op_entry.case_count; ++i) {
    const loom_target_contract_case_t* contract_case =
        &options->contract_index->cases[op_entry.case_start + i];
    if (loom_low_lower_source_query_case_has_related_nodes(options,
                                                           contract_case)) {
      return true;
    }
  }
  return false;
}

static iree_status_t loom_low_lower_source_query_one(
    loom_low_lower_context_t* context,
    const loom_target_contract_query_environment_t* environment,
    const loom_low_lower_contract_query_options_t* options,
    loom_low_lower_contract_accept_rule_callback_t accept_rule,
    const loom_op_t* source_op, loom_target_contract_query_result_t* out_result,
    loom_low_lower_rule_selection_t* out_selection) {
  *out_result = loom_target_contract_query_result_empty();
  if (out_selection != NULL) {
    *out_selection = (loom_low_lower_rule_selection_t){
        .rule_index = UINT16_MAX,
    };
  }
  const loom_target_contract_query_callback_t query_op_contract =
      context->policy->query_op_contract;
  if (query_op_contract.fn != NULL) {
    IREE_RETURN_IF_ERROR(query_op_contract.fn(
        query_op_contract.user_data, environment, source_op, out_result));
  }
  if (out_result->outcome != LOOM_TARGET_CONTRACT_QUERY_UNHANDLED) {
    return iree_ok_status();
  }
  loom_low_lower_contract_query_options_t filtered_options = *options;
  filtered_options.accept_rule = accept_rule;
  return loom_low_lower_query_target_contract_with_selection(
      environment, &filtered_options, source_op, out_result, out_selection);
}

static iree_status_t loom_low_lower_source_query_accept_graph(
    void* user_data, const loom_low_lower_rule_selection_t* selection,
    bool* out_accepted) {
  *out_accepted = false;
  loom_low_lower_source_query_coverage_build_state_t* state =
      (loom_low_lower_source_query_coverage_build_state_t*)user_data;
  const loom_op_t* root_op = selection->source_nodes[0];
  for (uint8_t i = 1; i < selection->source_node_count; ++i) {
    const loom_op_t* source_node = selection->source_nodes[i];
    const loom_low_lower_source_query_graph_t* graph =
        loom_low_lower_source_query_coverage_lookup(
            state->context, state->coverage, source_node);
    if (graph != NULL && graph->root_op == root_op) {
      continue;
    }
    if (graph == &loom_low_lower_source_query_legal_root_marker) {
      continue;
    }
    if (graph != NULL) {
      return iree_ok_status();
    }
    IREE_ASSERT_EQ(source_node->parent_block, root_op->parent_block);
    if (source_node->block_ordinal > root_op->block_ordinal) {
      continue;
    }

    loom_target_contract_query_result_t related_result =
        loom_target_contract_query_result_empty();
    loom_low_lower_rule_selection_t related_selection;
    IREE_RETURN_IF_ERROR(loom_low_lower_source_query_one(
        state->context, state->environment, state->options,
        (loom_low_lower_contract_accept_rule_callback_t){
            .fn = loom_low_lower_source_query_accept_graph,
            .user_data = state,
        },
        source_node, &related_result, &related_selection));
    const bool related_is_legal =
        related_result.outcome == LOOM_TARGET_CONTRACT_QUERY_LEGAL &&
        related_selection.source_node_count <= 1;
    IREE_RETURN_IF_ERROR(loom_low_lower_source_query_coverage_set(
        state->context, state->coverage, source_node,
        related_is_legal
            ? &loom_low_lower_source_query_legal_root_marker
            : &loom_low_lower_source_query_unavailable_root_marker));
    if (!related_is_legal) {
      return iree_ok_status();
    }
  }
  *out_accepted = true;
  return iree_ok_status();
}

static iree_status_t loom_low_lower_source_query_coverage_visit(
    void* user_data, loom_op_t* op, const loom_walk_context_t* walk_context,
    loom_walk_result_t* out_result) {
  (void)walk_context;
  *out_result = LOOM_WALK_CONTINUE;
  loom_low_lower_source_query_coverage_build_state_t* state =
      (loom_low_lower_source_query_coverage_build_state_t*)user_data;
  const loom_low_lower_source_query_graph_t* graph =
      loom_low_lower_source_query_coverage_lookup(state->context,
                                                  state->coverage, op);
  if ((graph != NULL && graph->root_op != NULL) ||
      !loom_low_lower_source_query_op_has_related_rule(state->options, op)) {
    return iree_ok_status();
  }

  loom_target_contract_query_result_t result =
      loom_target_contract_query_result_empty();
  loom_low_lower_rule_selection_t selection;
  IREE_RETURN_IF_ERROR(loom_low_lower_source_query_one(
      state->context, state->environment, state->options,
      (loom_low_lower_contract_accept_rule_callback_t){
          .fn = loom_low_lower_source_query_accept_graph,
          .user_data = state,
      },
      op, &result, &selection));
  if (result.outcome != LOOM_TARGET_CONTRACT_QUERY_LEGAL ||
      selection.source_node_count <= 1) {
    return loom_low_lower_source_query_coverage_set(
        state->context, state->coverage, op,
        result.outcome == LOOM_TARGET_CONTRACT_QUERY_LEGAL
            ? &loom_low_lower_source_query_legal_root_marker
            : &loom_low_lower_source_query_unavailable_root_marker);
  }

  return loom_low_lower_source_query_coverage_record_graph(
      state->context, state->coverage, selection.source_nodes,
      selection.source_node_count, &result);
}

static iree_status_t loom_low_lower_source_query_prepare_coverage(
    loom_low_lower_context_t* context,
    const loom_target_contract_query_environment_t* environment,
    const loom_low_lower_contract_query_options_t* options,
    loom_low_lower_source_query_coverage_t** out_coverage) {
  *out_coverage = NULL;
  loom_low_lower_source_query_coverage_t* coverage = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_get_or_allocate_target_state(
      context, &loom_low_lower_source_query_coverage_state_key,
      sizeof(*coverage), (void**)&coverage));
  *out_coverage = coverage;
  if (coverage->prepared || coverage->preparing) {
    return iree_ok_status();
  }
  if (!loom_low_lower_source_query_has_related_rules(options)) {
    coverage->prepared = true;
    return iree_ok_status();
  }

  const loom_local_value_domain_t* value_domain =
      &context->lowering.value_domain;
  coverage->value_graph_count = value_domain->value_count;

  coverage->preparing = true;
  loom_low_lower_source_query_coverage_build_state_t state = {
      .context = context,
      .environment = environment,
      .options = options,
      .coverage = coverage,
  };
  loom_walk_result_t walk_result = LOOM_WALK_CONTINUE;
  iree_status_t status = loom_walk_function(
      context->module, context->source_function, LOOM_WALK_PRE_ORDER,
      (loom_walk_callback_t){
          .fn = loom_low_lower_source_query_coverage_visit,
          .user_data = &state,
      },
      &walk_result);
  coverage->preparing = false;
  coverage->prepared = iree_status_is_ok(status);
  return status;
}

static iree_status_t loom_low_lower_source_query_allocate_target_state(
    void* user_data, const void* key, iree_host_size_t data_length,
    void** out_data) {
  return loom_low_lower_get_or_allocate_target_state(
      (loom_low_lower_context_t*)user_data, key, data_length, out_data);
}

iree_status_t loom_low_lower_source_query_environment_initialize(
    loom_low_lower_context_t* context,
    const loom_low_descriptor_set_t* descriptor_set,
    loom_target_contract_query_environment_t* out_environment) {
  const loom_view_region_table_t* view_regions = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_context_view_regions(context, &view_regions));
  *out_environment = (loom_target_contract_query_environment_t){
      .module = context->module,
      .function = context->source_function,
      .target_facts = context->options->target_facts,
      .descriptor_set = descriptor_set,
      .fact_table = context->lowering.fact_table,
      .value_domain = &context->lowering.value_domain,
      .view_regions = view_regions,
      .arena = &context->function_arena,
      .target_state_allocator =
          {
              .fn = loom_low_lower_source_query_allocate_target_state,
              .user_data = context,
          },
  };
  return iree_ok_status();
}

static iree_status_t loom_low_lower_source_query_map_value(
    void* user_data, const loom_low_lower_rule_match_context_t* match_context,
    const loom_op_t* source_op, loom_value_id_t source_value_id,
    loom_low_lower_rule_mapped_value_t* out_mapped_value) {
  *out_mapped_value = loom_low_lower_rule_mapped_value_none();
  loom_low_lower_contract_query_state_t* state =
      (loom_low_lower_contract_query_state_t*)user_data;
  const loom_type_t actual_type =
      loom_module_value_type(state->context->module, source_value_id);
  const loom_type_t query_type =
      loom_low_lower_rule_match_value_type(match_context, source_value_id);
  if (!loom_type_equal(actual_type, query_type)) {
    loom_type_t low_type = loom_type_none();
    IREE_RETURN_IF_ERROR(state->context->policy->map_type.fn(
        state->context->policy->map_type.user_data, state->context, source_op,
        query_type, &low_type));
    if (loom_type_kind(low_type) != LOOM_TYPE_NONE) {
      *out_mapped_value = loom_low_lower_rule_mapped_value_register(
          loom_low_register_type_class_id(low_type),
          loom_low_register_type_unit_count(low_type));
    }
    return iree_ok_status();
  }
  const loom_low_lower_map_contract_value_callback_t map_contract_value =
      state->context->policy->map_contract_value;
  if (map_contract_value.fn != NULL) {
    IREE_RETURN_IF_ERROR(
        map_contract_value.fn(map_contract_value.user_data, state->environment,
                              source_op, source_value_id, out_mapped_value));
    if (out_mapped_value->is_register) {
      return iree_ok_status();
    }
  }
  loom_type_t low_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_lower_query_value(state->context, source_op,
                                                  source_value_id, &low_type));
  if (loom_type_kind(low_type) != LOOM_TYPE_NONE) {
    *out_mapped_value = loom_low_lower_rule_mapped_value_register(
        loom_low_register_type_class_id(low_type),
        loom_low_register_type_unit_count(low_type));
  }
  return iree_ok_status();
}

static iree_status_t loom_low_lower_source_query_can_materialize(
    void* user_data, const loom_low_lower_rule_match_context_t* match_context,
    const loom_low_lower_rule_set_t* rule_set, const loom_op_t* source_op,
    uint16_t value_ref_index, loom_value_id_t source_value_id,
    bool* out_can_materialize) {
  (void)match_context;
  loom_low_lower_contract_query_state_t* state =
      (loom_low_lower_contract_query_state_t*)user_data;
  const loom_low_lower_value_ref_t* value_ref =
      &rule_set->value_refs[value_ref_index];
  const uint16_t materializer_index =
      (uint16_t)(value_ref->materializer_index - 1);
  const loom_low_lower_value_materializer_t* materializer =
      &rule_set->materializers[materializer_index];
  return materializer->can_materialize(state->context, source_op,
                                       source_value_id, out_can_materialize);
}

static iree_status_t loom_low_lower_source_query_contract(
    void* user_data,
    const loom_target_contract_query_environment_t* environment,
    const loom_op_t* source_op,
    loom_target_contract_query_result_t* out_result) {
  loom_low_lower_context_t* context = (loom_low_lower_context_t*)user_data;
  const loom_low_descriptor_set_t* saved_descriptor_set =
      context->descriptor_set;
  loom_value_fact_table_t* saved_fact_table = context->lowering.fact_table;
  const bool fact_table_changed =
      saved_fact_table != (loom_value_fact_table_t*)environment->fact_table;
  loom_cfg_value_identity_table_t saved_identities;
  context->descriptor_set = environment->descriptor_set;
  context->lowering.fact_table =
      (loom_value_fact_table_t*)environment->fact_table;
  if (fact_table_changed) {
    saved_identities = context->lowering.function_analysis.value_identities;
    context->lowering.function_analysis =
        (loom_low_lower_function_analysis_t){0};
  }

  iree_status_t status = iree_ok_status();
  loom_target_contract_query_environment_t query_environment = *environment;
  if (query_environment.value_domain == NULL) {
    query_environment.value_domain =
        loom_low_lower_context_value_domain(context);
  }
  query_environment.arena = &context->function_arena;
  query_environment.target_state_allocator =
      (loom_target_contract_query_state_allocator_t){
          .fn = loom_low_lower_source_query_allocate_target_state,
          .user_data = context,
      };
  if (query_environment.view_regions == NULL) {
    const loom_view_region_table_t* view_regions = NULL;
    status = loom_low_lower_context_view_regions(context, &view_regions);
    query_environment.view_regions = view_regions;
  }

  loom_low_lower_contract_query_state_t state = {
      .context = context,
      .environment = &query_environment,
  };
  const loom_low_lower_contract_query_options_t query_options = {
      .contract_index = context->policy->contract.index,
      .rule_sets = context->policy->contract.rule_sets,
      .map_value =
          {
              .fn = loom_low_lower_source_query_map_value,
              .user_data = &state,
          },
      .can_materialize =
          {
              .fn = loom_low_lower_source_query_can_materialize,
              .user_data = &state,
          },
      .descriptor_ref =
          {
              .fn = loom_low_lower_rule_match_descriptor_ref_from_lowering,
              .user_data = context,
          },
      .descriptor_matrix = context->policy->descriptor_matrix,
  };
  loom_low_lower_source_query_coverage_t* coverage = NULL;
  if (iree_status_is_ok(status) && !fact_table_changed &&
      query_environment.vector_lane_projection.source_lane_count == 0 &&
      loom_local_value_domain_is_acquired(&context->lowering.value_domain)) {
    status = loom_low_lower_source_query_prepare_coverage(
        context, &query_environment, &query_options, &coverage);
  }
  if (iree_status_is_ok(status) && coverage != NULL && coverage->prepared) {
    const loom_low_lower_source_query_graph_t* graph =
        loom_low_lower_source_query_coverage_lookup(context, coverage,
                                                    source_op);
    if (graph != NULL && graph->root_op != NULL &&
        graph->root_op != source_op) {
      *out_result = loom_target_contract_query_result_empty();
      out_result->outcome = LOOM_TARGET_CONTRACT_QUERY_LEGAL;
    } else if (graph != NULL && graph->root_op == source_op) {
      *out_result = graph->root_result;
    } else {
      loom_low_lower_source_query_coverage_build_state_t coverage_state = {
          .context = context,
          .environment = &query_environment,
          .options = &query_options,
          .coverage = coverage,
      };
      status = loom_low_lower_source_query_one(
          context, &query_environment, &query_options,
          (loom_low_lower_contract_accept_rule_callback_t){
              .fn = loom_low_lower_source_query_accept_graph,
              .user_data = &coverage_state,
          },
          source_op, out_result, /*out_selection=*/NULL);
    }
  } else if (iree_status_is_ok(status)) {
    status = loom_low_lower_source_query_one(
        context, &query_environment, &query_options,
        (loom_low_lower_contract_accept_rule_callback_t){0}, source_op,
        out_result, /*out_selection=*/NULL);
  }

  context->descriptor_set = saved_descriptor_set;
  context->lowering.fact_table = saved_fact_table;
  if (fact_table_changed) {
    context->lowering.function_analysis = (loom_low_lower_function_analysis_t){
        .value_identities = saved_identities,
    };
  }
  return status;
}

loom_target_contract_query_callback_t loom_low_lower_source_query_callback(
    loom_low_lower_context_t* context) {
  return (loom_target_contract_query_callback_t){
      .fn = loom_low_lower_source_query_contract,
      .user_data = context,
  };
}

struct loom_low_lower_source_query_scope_t {
  // Read-only lowering context backing source-to-Low contract queries.
  loom_low_lower_context_t context;
  // Diagnostic and result scratch required by the lowering context.
  loom_low_lower_result_t result;
  // True after context.lowering.value_domain acquires module storage.
  bool value_domain_initialized;
};

iree_status_t loom_low_lower_source_query_scope_create(
    loom_module_t* module, loom_func_like_t source_function,
    const loom_low_lower_options_t* options, iree_arena_allocator_t* arena,
    loom_low_lower_source_query_scope_t** out_scope) {
  *out_scope = NULL;
  loom_low_lower_source_query_scope_t* scope = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, sizeof(*scope), (void**)&scope));
  memset(scope, 0, sizeof(*scope));
  scope->result = (loom_low_lower_result_t){
      .low_func_ref = loom_symbol_ref_null(),
  };
  if (!iree_allocator_is_null(options->report_allocator)) {
    scope->result.report_allocator = options->report_allocator;
    scope->result.memory_report_row_allocator = module->allocator;
  }
  scope->context = (loom_low_lower_context_t){
      .module = module,
      .source_function = source_function,
      .options = options,
      .policy = options->policy,
      .result = &scope->result,
  };
  scope->context.lowering.fact_table = options->fact_table;
  const loom_region_descriptor_t* source_body_descriptor =
      loom_func_like_body_region_descriptor(module, source_function);
  if (source_body_descriptor != NULL) {
    scope->context.lowering.source_callable_exit_kind =
        source_body_descriptor->terminator;
  }
  iree_arena_initialize(module->arena.block_pool,
                        &scope->context.function_arena);

  iree_status_t status =
      loom_target_low_descriptor_set_select_for_source_lowering(
          options->descriptor_registry,
          loom_target_facts_bundle(options->target_facts),
          &scope->context.descriptor_set);
  loom_region_t* source_body = loom_func_like_body(source_function);
  if (iree_status_is_ok(status) && source_body != NULL) {
    status = loom_local_value_domain_acquire_for_region_tree(
        module, source_body, &scope->context.function_arena,
        &scope->context.lowering.value_domain);
    scope->value_domain_initialized = iree_status_is_ok(status);
  }
  if (iree_status_is_ok(status)) {
    loom_condition_query_initialize(module,
                                    scope->value_domain_initialized
                                        ? &scope->context.lowering.value_domain
                                        : NULL,
                                    &scope->context.function_arena,
                                    &scope->context.lowering.condition_query);
  }
  if (!iree_status_is_ok(status)) {
    loom_low_lower_source_query_scope_deinitialize(scope);
    return status;
  }

  *out_scope = scope;
  return iree_ok_status();
}

void loom_low_lower_source_query_scope_deinitialize(
    loom_low_lower_source_query_scope_t* scope) {
  if (scope == NULL) {
    return;
  }
  if (scope->value_domain_initialized) {
    loom_local_value_domain_release(&scope->context.lowering.value_domain);
    scope->value_domain_initialized = false;
  }
  loom_low_lower_result_deinitialize(&scope->result);
  iree_arena_deinitialize(&scope->context.function_arena);
  memset(scope, 0, sizeof(*scope));
}

loom_target_contract_query_callback_t
loom_low_lower_source_query_scope_callback(
    loom_low_lower_source_query_scope_t* scope) {
  return loom_low_lower_source_query_callback(&scope->context);
}

iree_status_t loom_low_lower_source_query_scope_environment_initialize(
    loom_low_lower_source_query_scope_t* scope,
    loom_target_contract_query_environment_t* out_environment) {
  return loom_low_lower_source_query_environment_initialize(
      &scope->context, scope->context.descriptor_set, out_environment);
}

loom_local_value_domain_t* loom_low_lower_source_query_scope_value_domain(
    loom_low_lower_source_query_scope_t* scope) {
  return scope->value_domain_initialized ? &scope->context.lowering.value_domain
                                         : NULL;
}

iree_status_t loom_low_lower_source_query_scope_view_regions(
    loom_low_lower_source_query_scope_t* scope,
    const loom_view_region_table_t** out_view_regions) {
  return loom_low_lower_context_view_regions(&scope->context, out_view_regions);
}

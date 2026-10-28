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
#include "loom/codegen/low/lower/source_graph_nodes.h"
#include "loom/codegen/low/lower/source_graph_placement.h"
#include "loom/codegen/low/lower/source_graph_plan.h"
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

typedef struct loom_low_lower_source_query_alternative_t
    loom_low_lower_source_query_alternative_t;

struct loom_low_lower_source_query_alternative_t {
  // Exact legal contract result retained from candidate matching.
  loom_target_contract_query_result_t result;
  // Resolver availability and target-ownership state.
  loom_low_lower_source_graph_alternative_state_t state;
};

typedef struct loom_low_lower_source_query_coverage_t {
  // Sparse source operations participating in multi-node graphs.
  loom_low_lower_source_graph_node_index_t nodes;
  // Shared candidate ownership resolver.
  loom_low_lower_source_graph_plan_t* graph_plan;
  // True after all source graphs in the function have been selected.
  bool prepared;
  // True while the one-time source walk is collecting graphs.
  bool preparing;
} loom_low_lower_source_query_coverage_t;

typedef struct loom_low_lower_source_query_coverage_build_state_t {
  // Lowering context that owns the query scope and coverage storage.
  loom_low_lower_context_t* context;
  // Contract environment fixed for the immutable query scope.
  const loom_target_contract_query_environment_t* environment;
  // Base contract matcher options without an ownership filter.
  const loom_low_lower_contract_query_options_t* options;
  // Coverage being populated or queried.
  loom_low_lower_source_query_coverage_t* coverage;
  // Root operation whose contract candidates are being visited.
  const loom_op_t* root_op;
} loom_low_lower_source_query_coverage_build_state_t;

static iree_status_t loom_low_lower_source_query_coverage_add_graph(
    loom_low_lower_source_query_coverage_build_state_t* state,
    const loom_op_t* const* source_nodes, uint8_t source_node_count,
    const loom_low_lower_rule_set_t* rule_set,
    const loom_low_lower_rule_t* rule, uint8_t priority,
    const loom_target_contract_query_result_t* result,
    loom_low_lower_source_graph_alternative_state_t alternative_state) {
  uint32_t node_indices[LOOM_LOW_LOWER_MAX_SOURCE_NODES];
  for (uint8_t i = 0; i < source_node_count; ++i) {
    loom_low_lower_source_graph_node_t* node = NULL;
    IREE_RETURN_IF_ERROR(loom_low_lower_source_graph_node_index_get_or_add(
        &state->coverage->nodes, source_nodes[i], &node));
    node_indices[i] = node->index;
  }
  loom_low_lower_source_graph_edge_t edges[LOOM_LOW_LOWER_MAX_SOURCE_NODES - 1];
  const loom_low_lower_source_graph_edge_t* graph_edges = NULL;
  if (source_node_count > 1) {
    IREE_ASSERT(rule_set != NULL);
    IREE_ASSERT(rule != NULL);
    loom_low_lower_source_graph_rule_edges(rule_set, rule, edges);
    graph_edges = edges;
  }
  loom_low_lower_source_query_alternative_t* alternative = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(&state->context->analysis_arena,
                                           sizeof(*alternative),
                                           (void**)&alternative));
  *alternative = (loom_low_lower_source_query_alternative_t){
      .result = *result,
      .state = alternative_state,
  };
  return loom_low_lower_source_graph_plan_add_alternative(
      state->coverage->graph_plan, node_indices, source_node_count, graph_edges,
      priority, alternative, /*out_alternative=*/NULL);
}

static bool loom_low_lower_source_query_case_has_related_nodes(
    const loom_low_lower_contract_query_options_t* options,
    const loom_target_contract_case_t* contract_case) {
  if (!loom_low_lower_contract_case_queries_target(options->contract_index,
                                                   contract_case)) {
    return false;
  }
  const loom_target_contract_binding_t* binding =
      &options->contract_index->bindings[contract_case->binding_index];
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

static bool loom_low_lower_source_query_op_participates_in_graphs(
    const loom_low_lower_contract_query_options_t* options,
    loom_op_kind_t op_kind) {
  if (options->contract_index == NULL) {
    return false;
  }
  for (uint8_t i = 0; i < options->contract_index->binding_count; ++i) {
    const loom_target_contract_binding_t* binding =
        &options->contract_index->bindings[i];
    if (binding->rule_set_index < options->rule_sets.count &&
        loom_target_contract_fragment_has_source_graph_op(binding->fragment,
                                                          op_kind)) {
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

static iree_status_t loom_low_lower_source_query_target_contract(
    loom_low_lower_context_t* context,
    const loom_target_contract_query_environment_t* environment,
    const loom_op_t* source_op,
    loom_target_contract_query_result_t* out_result) {
  *out_result = loom_target_contract_query_result_empty();
  const loom_low_lower_op_contract_query_t* query =
      &context->policy->query_op_contract;
  if (query->fn == NULL ||
      !loom_low_lower_op_contract_query_accepts(query, source_op->kind)) {
    return iree_ok_status();
  }
  return query->fn(query->user_data, environment, source_op, out_result);
}

static iree_status_t loom_low_lower_source_query_one(
    loom_low_lower_context_t* context,
    const loom_target_contract_query_environment_t* environment,
    const loom_low_lower_contract_query_options_t* options,
    const loom_op_t* source_op, loom_target_contract_query_result_t* out_result,
    loom_low_lower_rule_selection_t* out_selection) {
  *out_result = loom_target_contract_query_result_empty();
  if (out_selection != NULL) {
    loom_low_lower_rule_selection_initialize(out_selection);
  }
  IREE_RETURN_IF_ERROR(loom_low_lower_source_query_target_contract(
      context, environment, source_op, out_result));
  if (out_result->outcome != LOOM_TARGET_CONTRACT_QUERY_UNHANDLED) {
    return iree_ok_status();
  }
  return loom_low_lower_query_target_contract_with_selection(
      environment, options, source_op, out_result, out_selection);
}

static iree_status_t loom_low_lower_source_query_collect_contract_match(
    void* user_data, const loom_target_contract_query_result_t* result,
    const loom_low_lower_rule_set_t* rule_set,
    const loom_low_lower_rule_selection_t* selection,
    loom_low_lower_rule_primary_descriptor_status_t primary_status) {
  loom_low_lower_source_query_coverage_build_state_t* state =
      (loom_low_lower_source_query_coverage_build_state_t*)user_data;
  const uint8_t source_node_count =
      selection->rule != NULL ? selection->source_node_count : 1;
  if (selection->rule != NULL) {
    bool placement_legal = false;
    IREE_RETURN_IF_ERROR(loom_low_lower_source_graph_placement_evaluate(
        state->context, selection, &placement_legal));
    if (!placement_legal) {
      return iree_ok_status();
    }
  }
  const loom_op_t* singleton_node = state->root_op;
  const loom_op_t* const* source_nodes =
      selection->rule != NULL ? selection->source_nodes : &singleton_node;
  const uint8_t priority = selection->rule != NULL
                               ? loom_low_lower_rule_priority(selection->rule)
                               : 0;
  const loom_low_lower_source_graph_alternative_state_t alternative_state =
      primary_status == LOOM_LOW_LOWER_RULE_PRIMARY_DESCRIPTOR_AVAILABLE
          ? LOOM_LOW_LOWER_SOURCE_GRAPH_ALTERNATIVE_ENABLED
          : LOOM_LOW_LOWER_SOURCE_GRAPH_ALTERNATIVE_FALLBACK;
  return loom_low_lower_source_query_coverage_add_graph(
      state, source_nodes, source_node_count, rule_set, selection->rule,
      priority, result, alternative_state);
}

static iree_status_t loom_low_lower_source_query_coverage_visit(
    void* user_data, loom_op_t* op, const loom_walk_context_t* walk_context,
    loom_walk_result_t* out_result) {
  (void)walk_context;
  *out_result = LOOM_WALK_CONTINUE;
  loom_low_lower_source_query_coverage_build_state_t* state =
      (loom_low_lower_source_query_coverage_build_state_t*)user_data;
  if (!loom_low_lower_source_query_op_has_related_rule(state->options, op)) {
    return iree_ok_status();
  }
  loom_target_contract_query_result_t target_result =
      loom_target_contract_query_result_empty();
  IREE_RETURN_IF_ERROR(loom_low_lower_source_query_target_contract(
      state->context, state->environment, op, &target_result));
  if (target_result.outcome != LOOM_TARGET_CONTRACT_QUERY_UNHANDLED) {
    return iree_ok_status();
  }
  state->root_op = op;
  return loom_low_lower_visit_target_contract_matches(
      state->environment, state->options, op,
      LOOM_LOW_LOWER_CONTRACT_CASE_SCOPE_SOURCE_GRAPH,
      (loom_low_lower_contract_match_callback_t){
          .fn = loom_low_lower_source_query_collect_contract_match,
          .user_data = state,
      });
}

static iree_status_t loom_low_lower_source_query_collect_singletons(
    loom_low_lower_source_query_coverage_build_state_t* state) {
  for (loom_low_lower_source_graph_node_t* node = state->coverage->nodes.first;
       node != NULL; node = node->next) {
    state->root_op = node->source_op;
    loom_target_contract_query_result_t target_result =
        loom_target_contract_query_result_empty();
    IREE_RETURN_IF_ERROR(loom_low_lower_source_query_target_contract(
        state->context, state->environment, node->source_op, &target_result));
    if (target_result.outcome != LOOM_TARGET_CONTRACT_QUERY_UNHANDLED) {
      const loom_op_t* source_nodes[] = {node->source_op};
      IREE_RETURN_IF_ERROR(loom_low_lower_source_query_coverage_add_graph(
          state, source_nodes, /*source_node_count=*/1,
          /*rule_set=*/NULL, /*rule=*/NULL, /*priority=*/0, &target_result,
          LOOM_LOW_LOWER_SOURCE_GRAPH_ALTERNATIVE_FIXED));
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_low_lower_visit_target_contract_matches(
        state->environment, state->options, node->source_op,
        LOOM_LOW_LOWER_CONTRACT_CASE_SCOPE_SINGLETON,
        (loom_low_lower_contract_match_callback_t){
            .fn = loom_low_lower_source_query_collect_contract_match,
            .user_data = state,
        }));
  }
  return iree_ok_status();
}

static loom_low_lower_source_graph_alternative_state_t
loom_low_lower_source_query_alternative_state(
    void* user_data,
    const loom_low_lower_source_graph_alternative_t* alternative) {
  (void)user_data;
  const loom_low_lower_source_query_alternative_t* query_alternative =
      (const loom_low_lower_source_query_alternative_t*)
          loom_low_lower_source_graph_alternative_user_data(alternative);
  return query_alternative->state;
}

static iree_status_t loom_low_lower_source_query_accept_singleton(
    void* user_data, const loom_low_lower_rule_selection_t* selection,
    bool* out_accepted) {
  (void)user_data;
  *out_accepted = selection->source_node_count <= 1;
  return iree_ok_status();
}

static iree_status_t loom_low_lower_source_query_accept_placement(
    void* user_data, const loom_low_lower_rule_selection_t* selection,
    bool* out_accepted) {
  return loom_low_lower_source_graph_placement_evaluate(
      (loom_low_lower_context_t*)user_data, selection, out_accepted);
}

static iree_status_t loom_low_lower_source_query_prepare_coverage(
    loom_low_lower_context_t* context,
    const loom_target_contract_query_environment_t* environment,
    const loom_low_lower_contract_query_options_t* options,
    loom_low_lower_source_query_coverage_t** out_coverage) {
  *out_coverage = NULL;
  loom_low_lower_source_query_coverage_t* coverage =
      context->source_query_coverage;
  if (coverage == NULL) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate(
        &context->analysis_arena, sizeof(*coverage), (void**)&coverage));
    memset(coverage, 0, sizeof(*coverage));
    loom_low_lower_source_graph_node_index_initialize(
        &context->lowering->value_domain, &context->analysis_arena,
        &coverage->nodes);
    context->source_query_coverage = coverage;
  }
  *out_coverage = coverage;
  if (coverage->prepared || coverage->preparing) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_low_lower_source_graph_plan_allocate(
      &context->analysis_arena, &coverage->graph_plan));
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
  if (iree_status_is_ok(status)) {
    status = loom_low_lower_source_query_collect_singletons(&state);
  }
  bool complete = false;
  if (iree_status_is_ok(status)) {
    status = loom_low_lower_source_graph_plan_resolve(
        coverage->graph_plan,
        (loom_low_lower_source_graph_alternative_query_callback_t){
            .fn = loom_low_lower_source_query_alternative_state,
            .user_data = NULL,
        },
        &complete);
  }
  (void)complete;
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

static iree_status_t loom_low_lower_source_query_function_analysis(
    void* user_data, loom_local_value_domain_t** out_value_domain,
    const loom_view_region_table_t** out_view_regions) {
  loom_low_lower_context_t* context = user_data;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_context_view_regions(context, out_view_regions));
  *out_value_domain = &context->lowering->value_domain;
  return iree_ok_status();
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
      .schedule_model = context->schedule_model,
      .fact_table = context->fact_table,
      .storage_access = context->storage_access,
      .value_domain = &context->lowering->value_domain,
      .view_regions = view_regions,
      .arena = context->function_arena,
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
          LOOM_LOW_REPRESENTATION_ID_NONE,
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
        LOOM_LOW_REPRESENTATION_ID_NONE,
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
  loom_value_fact_table_t* saved_fact_table = context->fact_table;
  const bool fact_table_changed =
      saved_fact_table != (loom_value_fact_table_t*)environment->fact_table;
  loom_low_lower_function_analysis_t saved_analysis;
  context->descriptor_set = environment->descriptor_set;
  context->fact_table = (loom_value_fact_table_t*)environment->fact_table;
  if (fact_table_changed) {
    saved_analysis = context->function_analysis;
    context->function_analysis = (loom_low_lower_function_analysis_t){0};
  }

  iree_status_t status = iree_ok_status();
  loom_target_contract_query_environment_t query_environment = *environment;
  query_environment.storage_access = context->storage_access;
  if (query_environment.value_domain == NULL &&
      loom_local_value_domain_is_acquired(&context->lowering->value_domain)) {
    query_environment.value_domain =
        loom_low_lower_context_value_domain(context);
  }
  query_environment.arena = context->function_arena;
  query_environment.target_state_allocator =
      (loom_target_contract_query_state_allocator_t){
          .fn = loom_low_lower_source_query_allocate_target_state,
          .user_data = context,
      };
  query_environment.function_analysis =
      (loom_target_contract_function_analysis_callback_t){
          .fn = loom_low_lower_source_query_function_analysis,
          .user_data = context,
      };

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
      .accept_rule =
          {
              .fn = loom_low_lower_source_query_accept_placement,
              .user_data = context,
          },
  };
  loom_low_lower_source_query_coverage_t* coverage = NULL;
  if (iree_status_is_ok(status) && !fact_table_changed &&
      query_environment.vector_lane_projection.source_lane_count == 0 &&
      loom_low_lower_source_query_op_participates_in_graphs(&query_options,
                                                            source_op->kind)) {
    status = loom_target_contract_query_resolve_function_analysis(
        &query_environment);
    if (iree_status_is_ok(status)) {
      status = loom_low_lower_source_query_prepare_coverage(
          context, &query_environment, &query_options, &coverage);
    }
  }
  if (iree_status_is_ok(status) && coverage != NULL && coverage->prepared) {
    const loom_low_lower_source_graph_node_t* node =
        loom_low_lower_source_graph_node_index_lookup(&coverage->nodes,
                                                      source_op);
    const loom_low_lower_source_graph_alternative_t* alternative =
        node != NULL ? loom_low_lower_source_graph_plan_selected_alternative(
                           coverage->graph_plan, node->index)
                     : NULL;
    if (alternative != NULL) {
      const loom_low_lower_source_graph_candidate_t* candidate =
          loom_low_lower_source_graph_alternative_candidate(alternative);
      const uint32_t root_index =
          loom_low_lower_source_graph_candidate_node(candidate, 0);
      if (root_index != node->index) {
        *out_result = loom_target_contract_query_result_empty();
        out_result->outcome = LOOM_TARGET_CONTRACT_QUERY_LEGAL;
      } else {
        const loom_low_lower_source_query_alternative_t* query_alternative =
            (const loom_low_lower_source_query_alternative_t*)
                loom_low_lower_source_graph_alternative_user_data(alternative);
        *out_result = query_alternative->result;
      }
    } else {
      loom_low_lower_contract_query_options_t singleton_options = query_options;
      if (node != NULL) {
        singleton_options.accept_rule =
            (loom_low_lower_contract_accept_rule_callback_t){
                .fn = loom_low_lower_source_query_accept_singleton,
                .user_data = NULL,
            };
      }
      status = loom_low_lower_source_query_one(context, &query_environment,
                                               &singleton_options, source_op,
                                               out_result,
                                               /*out_selection=*/NULL);
    }
  } else if (iree_status_is_ok(status)) {
    status = loom_low_lower_source_query_one(
        context, &query_environment, &query_options, source_op, out_result,
        /*out_selection=*/NULL);
  }

  context->descriptor_set = saved_descriptor_set;
  context->fact_table = saved_fact_table;
  if (fact_table_changed) {
    context->function_analysis = saved_analysis;
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
  // Function state shared by contract queries in this scope.
  loom_low_lowering_frame_t frame;
  // Reference/access graph owned by this immutable-source query scope.
  loom_storage_access_scope_t storage_access;
  // Diagnostic and result scratch required by the lowering context.
  loom_low_lower_result_t result;
};

static void loom_low_lower_source_query_scope_initialize(
    loom_low_lower_source_query_scope_t* scope, loom_module_t* module,
    loom_func_like_t source_function, const loom_low_lower_options_t* options,
    loom_target_low_source_binding_t binding) {
  memset(scope, 0, sizeof(*scope));
  scope->result = (loom_low_lower_result_t){
      .low_func_ref = loom_symbol_ref_null(),
  };
  scope->context = (loom_low_lower_context_t){
      // Queries retain no emission plan: target payloads and analyses share
      // the immutable-source scope lifetime and one arena.
      .function_arena = &scope->context.analysis_arena,
      .lowering = &scope->frame,
      .module = module,
      .source_function = source_function,
      .options = options,
      .policy = options->policy,
      .descriptor_set = binding.descriptor_set,
      .schedule_model = binding.schedule_model,
      .result = &scope->result,
      .storage_access = &scope->storage_access,
  };
  scope->context.fact_table = options->fact_table;
  const loom_region_descriptor_t* source_body_descriptor =
      loom_func_like_body_region_descriptor(module, source_function);
  if (source_body_descriptor != NULL) {
    scope->context.lowering->source_callable_exit_kind =
        source_body_descriptor->terminator;
  }
  iree_arena_initialize(module->arena.block_pool,
                        &scope->context.analysis_arena);
  loom_storage_access_scope_initialize(module, &scope->context.analysis_arena,
                                       &scope->storage_access);
}

iree_status_t loom_low_lower_source_query_scope_allocate(
    loom_module_t* module, loom_func_like_t source_function,
    const loom_low_lower_options_t* options,
    loom_target_low_source_binding_t binding, iree_arena_allocator_t* arena,
    loom_low_lower_source_query_scope_t** out_scope) {
  *out_scope = NULL;
  loom_low_lower_source_query_scope_t* scope = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, sizeof(*scope), (void**)&scope));
  loom_low_lower_source_query_scope_initialize(scope, module, source_function,
                                               options, binding);
  *out_scope = scope;
  return iree_ok_status();
}

void loom_low_lower_source_query_scope_deinitialize(
    loom_low_lower_source_query_scope_t* scope) {
  if (scope == NULL) {
    return;
  }
  if (loom_local_value_domain_is_acquired(
          &scope->context.lowering->value_domain)) {
    loom_local_value_domain_release(&scope->context.lowering->value_domain);
  }
  loom_low_lower_result_deinitialize(&scope->result);
  iree_arena_deinitialize(&scope->context.analysis_arena);
  memset(scope, 0, sizeof(*scope));
}

void loom_low_lower_source_query_scope_reset(
    loom_low_lower_source_query_scope_t* scope) {
  loom_module_t* module = scope->context.module;
  const loom_func_like_t source_function = scope->context.source_function;
  const loom_low_lower_options_t* options = scope->context.options;
  const loom_low_descriptor_set_t* descriptor_set =
      scope->context.descriptor_set;
  const loom_low_schedule_model_t* schedule_model =
      scope->context.schedule_model;
  loom_low_lower_source_query_scope_deinitialize(scope);
  loom_low_lower_source_query_scope_initialize(
      scope, module, source_function, options,
      (loom_target_low_source_binding_t){
          .descriptor_set = descriptor_set,
          .schedule_model = schedule_model,
      });
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
  return loom_local_value_domain_is_acquired(
             &scope->context.lowering->value_domain)
             ? &scope->context.lowering->value_domain
             : NULL;
}

iree_status_t loom_low_lower_source_query_scope_view_regions(
    loom_low_lower_source_query_scope_t* scope,
    const loom_view_region_table_t** out_view_regions) {
  return loom_low_lower_context_view_regions(&scope->context, out_view_regions);
}

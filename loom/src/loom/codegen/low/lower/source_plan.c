// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/lower/source_plan.h"

#include <string.h>

#include "loom/analysis/consumption.h"
#include "loom/analysis/contract_vector.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/codegen/low/lower/context.h"
#include "loom/codegen/low/lower/contract_query.h"
#include "loom/codegen/low/lower/contract_selection.h"
#include "loom/codegen/low/lower/function_boundary.h"
#include "loom/codegen/low/lower/rule_match.h"
#include "loom/codegen/low/lower/rule_plan.h"
#include "loom/codegen/low/lower/rule_source_memory.h"
#include "loom/codegen/low/lower/rule_source_memory_plan.h"
#include "loom/codegen/low/lower/rule_value.h"
#include "loom/codegen/low/lower/source_call.h"
#include "loom/codegen/low/lower/source_graph_nodes.h"
#include "loom/codegen/low/lower/source_graph_placement.h"
#include "loom/codegen/low/lower/source_graph_plan.h"
#include "loom/codegen/low/lower/source_query.h"
#include "loom/codegen/low/lower/source_value_storage.h"
#include "loom/codegen/low/source_memory_plan.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/context.h"
#include "loom/ir/facts.h"
#include "loom/ir/module.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/cfg/ops.h"
#include "loom/ops/encoding/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/ops/view/ops.h"

typedef enum loom_low_lower_source_candidate_kind_e {
  LOOM_LOW_LOWER_SOURCE_CANDIDATE_RULE = 0,
  LOOM_LOW_LOWER_SOURCE_CANDIDATE_DESCRIPTOR_MATRIX = 1,
  LOOM_LOW_LOWER_SOURCE_CANDIDATE_PLAN = 2,
  LOOM_LOW_LOWER_SOURCE_CANDIDATE_CALLBACK = 3,
} loom_low_lower_source_candidate_kind_t;

typedef struct loom_low_lower_source_candidate_descriptor_matrix_t {
  // Generated descriptor-matrix row selected by source adaptation.
  const loom_target_contract_descriptor_matrix_rule_t* matrix_rule;
  // Target-independent adapted request.
  loom_contract_request_t contract_request;
  // Target projection retained without materialized immediate attributes.
  loom_target_contract_query_result_t query_result;
} loom_low_lower_source_candidate_descriptor_matrix_t;

typedef struct loom_low_lower_source_candidate_callback_t {
  // Target callback evaluated only when this fallback is selected.
  loom_low_lower_select_op_callback_t callback;
  // Selected plan populated after successful evaluation.
  loom_low_lower_selected_plan_t selected_plan;
  // True after the callback has been evaluated.
  bool evaluated;
} loom_low_lower_source_candidate_callback_t;

typedef struct loom_low_lower_source_candidate_t {
  // Candidate payload representation.
  loom_low_lower_source_candidate_kind_t kind;
  // Resolver state before storage-demand filtering.
  loom_low_lower_source_graph_alternative_state_t state;
  // True when deferred evaluation proved the candidate unavailable.
  bool disabled;
  union {
    struct {
      // Rule set owning the selected rule row.
      const loom_low_lower_rule_set_t* rule_set;
      // Canonical root memory selection, or NULL when unused.
      const loom_low_lower_selected_source_memory_t* source_memory;
      // Policy rule-set ordinal.
      uint16_t rule_set_index;
      // Rule-table ordinal.
      uint16_t rule_index;
      // Source-node ordinal where every emitted operand is available.
      uint8_t emission_source_node_index;
    } rule;
    // Retained descriptor-matrix query awaiting selected-plan materialization.
    loom_low_lower_source_candidate_descriptor_matrix_t* descriptor_matrix;
    // Complete non-rule selected plan.
    loom_low_lower_selected_plan_t* selected_plan;
    // Deferred target callback state.
    loom_low_lower_source_candidate_callback_t* callback;
  } data;
} loom_low_lower_source_candidate_t;

typedef struct loom_low_lower_source_candidate_graph_t {
  // Number of source nodes in root-first order.
  uint8_t source_node_count;
  // Source nodes shared by every exact graph alternative.
  const loom_op_t* source_nodes[];
} loom_low_lower_source_candidate_graph_t;

typedef struct loom_low_lower_source_candidate_rejection_t {
  // Best generated-rule rejection when no candidate covers this node.
  const loom_low_lower_rule_set_t* failed_rule_set;
  // Best generated-rule failure detail.
  loom_low_lower_rule_failure_t rule_failure;
  // Descriptor-matrix rejection retained until coverage is resolved.
  loom_target_contract_query_result_t matrix_rejection;
} loom_low_lower_source_candidate_rejection_t;

typedef struct loom_low_lower_source_candidate_node_t {
  // Ordinary source operation represented by this node.
  const loom_op_t* source_op;
  // Canonical source-memory record, or NULL for non-memory operations.
  const loom_low_lower_source_memory_record_t* memory_record;
  // Root-adjusted canonical memory selection shared by rule alternatives.
  const loom_low_lower_selected_source_memory_t* matched_source_memory;
  // Cold rejection detail allocated only for uncovered operations.
  loom_low_lower_source_candidate_rejection_t* rejection;
} loom_low_lower_source_candidate_node_t;

typedef struct loom_low_lower_source_candidate_root_t {
  // Next possible source-graph root in source traversal order.
  struct loom_low_lower_source_candidate_root_t* next;
  // Possible source-graph root operation.
  const loom_op_t* source_op;
  // Canonical source-memory record at the root operation.
  const loom_low_lower_source_memory_record_t* memory_record;
} loom_low_lower_source_candidate_root_t;

typedef struct loom_low_lower_source_candidate_plan_t {
  // Mutable lowering context owning candidate facts and retained plan storage.
  loom_low_lower_context_t* context;
  // Scratch storage retained only until source-graph selection completes.
  iree_arena_allocator_t arena;
  // Query environment initialized on the first indexed contract operation.
  loom_target_contract_query_environment_t query_environment;
  // Composed contract tables and target descriptor-matrix projection.
  loom_low_lower_contract_query_options_t query_options;
  // Shared exact-graph resolver.
  loom_low_lower_source_graph_plan_t* graph_plan;
  // Possible graph roots recorded by the owning source traversal.
  loom_low_lower_source_candidate_root_t* first_root;
  // Last possible graph root in source traversal order.
  loom_low_lower_source_candidate_root_t* last_root;
  // Sparse source operations participating in multi-operation graphs.
  loom_low_lower_source_graph_node_index_t node_index;
  // Candidate payloads indexed by dense source graph node identity.
  loom_low_lower_source_candidate_node_t** nodes;
  // Number of entries in nodes after discovery completes.
  uint32_t node_count;
} loom_low_lower_source_candidate_plan_t;

bool loom_low_lower_source_plan_uses_structured_control_flow(
    const loom_low_lower_context_t* context) {
  return context->options->control_flow_lowering ==
         LOOM_LOW_CONTROL_FLOW_LOWERING_STRUCTURED_LOW;
}

static bool loom_low_lower_supported_structured_source_op(
    const loom_low_lower_context_t* context, const loom_op_t* source_op) {
  if (!loom_low_lower_source_plan_uses_structured_control_flow(context)) {
    return false;
  }
  switch (source_op->kind) {
    case LOOM_OP_SCF_IF:
    case LOOM_OP_SCF_FOR:
    case LOOM_OP_SCF_WHILE:
      return true;
    default:
      return false;
  }
}

static bool loom_low_lower_op_is_structural(
    const loom_low_lower_context_t* context, const loom_op_t* op,
    loom_trait_flags_t traits, bool is_callable_exit) {
  if (loom_low_lower_source_value_storage_op_is_identity(context, op, traits)) {
    return true;
  }
  if (is_callable_exit) {
    return true;
  }
  if (iree_any_bit_set(traits, LOOM_TRAIT_CALLABLE_BOUNDARY) &&
      loom_low_lower_source_call_is_structural(context->module, op)) {
    return true;
  }
  switch (op->kind) {
    case LOOM_OP_BUFFER_ASSUME_SAME_ROOT:
    case LOOM_OP_CFG_BR:
    case LOOM_OP_CFG_COND_BR:
    case LOOM_OP_SCF_FOR:
    case LOOM_OP_SCF_IF:
    case LOOM_OP_SCF_WHILE:
    case LOOM_OP_SCF_CONDITION:
    case LOOM_OP_SCF_SCHEDULE_FENCE:
    case LOOM_OP_SCF_YIELD:
      return true;
    default:
      return false;
  }
}

bool loom_low_lower_source_plan_op_is_metadata(loom_op_kind_t kind) {
  switch (kind) {
    case LOOM_OP_ENCODING_ASSUME_SPEC:
    case LOOM_OP_ENCODING_DEFINE:
    case LOOM_OP_ENCODING_LAYOUT_ASSUME_DENSE:
    case LOOM_OP_ENCODING_LAYOUT_ASSUME_STRIDED:
    case LOOM_OP_ENCODING_LAYOUT_DENSE:
    case LOOM_OP_ENCODING_LAYOUT_STRIDED:
      return true;
    default:
      return false;
  }
}

static bool loom_low_lower_op_is_discardable_hint(const loom_module_t* module,
                                                  const loom_op_t* op) {
  if (op->result_count != 0 || op->region_count != 0 ||
      op->tied_result_count != 0) {
    return false;
  }
  const loom_trait_flags_t traits = loom_op_effective_traits(module, op);
  return iree_any_bit_set(traits, LOOM_TRAIT_HINT);
}

static bool loom_low_lower_source_op_is_graph_structural_root(
    const loom_low_lower_context_t* context, const loom_op_t* source_op) {
  const loom_target_contract_index_t* index = context->policy->contract.index;
  return loom_cfg_cond_br_isa(source_op) && index != NULL &&
         loom_target_contract_index_has_source_graph_root_op_kind(
             index, source_op->kind);
}

static bool loom_low_lower_source_op_has_plan_node(
    const loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_trait_flags_t traits, bool is_callable_exit) {
  return source_op->region_count == 0 &&
         (!loom_low_lower_op_is_structural(context, source_op, traits,
                                           is_callable_exit) ||
          loom_low_lower_source_op_is_graph_structural_root(context,
                                                            source_op)) &&
         !loom_low_lower_source_plan_op_is_metadata(source_op->kind);
}

static bool loom_low_lower_source_plan_has_related_rules(
    const loom_low_lower_context_t* context) {
  const loom_target_contract_index_t* index = context->policy->contract.index;
  return index != NULL && index->source_graph_root_op_kind_count != 0;
}

static iree_status_t loom_low_lower_source_candidate_plan_initialize(
    loom_low_lower_context_t* context,
    loom_low_lower_source_candidate_plan_t* out_plan) {
  *out_plan = (loom_low_lower_source_candidate_plan_t){0};
  out_plan->context = context;
  iree_arena_initialize(context->module->arena.block_pool, &out_plan->arena);
  IREE_RETURN_IF_ERROR(loom_low_lower_source_graph_plan_allocate(
      &out_plan->arena, &out_plan->graph_plan));
  loom_low_lower_source_graph_node_index_initialize(
      &context->lowering->value_domain, &out_plan->arena,
      &out_plan->node_index);
  return iree_ok_status();
}

static iree_status_t
loom_low_lower_source_candidate_plan_prepare_query_environment(
    loom_low_lower_source_candidate_plan_t* candidate_plan) {
  if (candidate_plan->query_environment.module != NULL) {
    return iree_ok_status();
  }
  loom_low_lower_context_t* context = candidate_plan->context;
  const loom_target_contract_index_t* index = context->policy->contract.index;
  IREE_ASSERT(index != NULL);
  IREE_RETURN_IF_ERROR(loom_low_lower_source_query_environment_initialize(
      context, context->descriptor_set, &candidate_plan->query_environment));
  candidate_plan->query_options = (loom_low_lower_contract_query_options_t){
      .contract_index = index,
      .rule_sets = context->policy->contract.rule_sets,
      .descriptor_matrix = context->policy->descriptor_matrix,
  };
  return iree_ok_status();
}

static iree_status_t loom_low_lower_source_candidate_plan_get_or_add_node(
    loom_low_lower_source_candidate_plan_t* plan, const loom_op_t* source_op,
    loom_low_lower_source_candidate_node_t** out_candidate_node,
    uint32_t* out_node_index) {
  *out_candidate_node = NULL;
  *out_node_index = UINT32_MAX;
  loom_low_lower_source_graph_node_t* graph_node = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_source_graph_node_index_get_or_add(
      &plan->node_index, source_op, &graph_node));
  if (graph_node->user_data == NULL) {
    loom_low_lower_source_candidate_node_t* candidate_node = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate(
        &plan->arena, sizeof(*candidate_node), (void**)&candidate_node));
    *candidate_node = (loom_low_lower_source_candidate_node_t){
        .source_op = source_op,
    };
    graph_node->user_data = candidate_node;
  }
  *out_candidate_node =
      (loom_low_lower_source_candidate_node_t*)graph_node->user_data;
  *out_node_index = graph_node->index;
  return iree_ok_status();
}

static loom_low_lower_source_candidate_node_t*
loom_low_lower_source_candidate_plan_lookup_node(
    const loom_low_lower_source_candidate_plan_t* plan,
    const loom_op_t* source_op, uint32_t* out_node_index) {
  loom_low_lower_source_graph_node_t* graph_node =
      loom_low_lower_source_graph_node_index_lookup(&plan->node_index,
                                                    source_op);
  if (graph_node == NULL) {
    *out_node_index = UINT32_MAX;
    return NULL;
  }
  *out_node_index = graph_node->index;
  return (loom_low_lower_source_candidate_node_t*)graph_node->user_data;
}

static iree_status_t loom_low_lower_source_candidate_plan_finalize_nodes(
    loom_low_lower_source_candidate_plan_t* plan) {
  plan->node_count = plan->node_index.count;
  if (plan->node_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(&plan->arena, plan->node_count,
                                                 sizeof(*plan->nodes),
                                                 (void**)&plan->nodes));
  for (loom_low_lower_source_graph_node_t* graph_node = plan->node_index.first;
       graph_node != NULL; graph_node = graph_node->next) {
    IREE_ASSERT_LT(graph_node->index, plan->node_count);
    IREE_ASSERT(graph_node->user_data != NULL);
    plan->nodes[graph_node->index] =
        (loom_low_lower_source_candidate_node_t*)graph_node->user_data;
  }
  return iree_ok_status();
}

static void loom_low_lower_planning_scope_begin(
    loom_low_lower_context_t* context) {
  context->planning_arena_active = true;
}

static void loom_low_lower_planning_scope_end(
    loom_low_lower_context_t* context) {
  context->planning_arena_active = false;
  iree_arena_reset(&context->planning_arena);
}

static void loom_low_lower_source_consumption_query_initialize(
    loom_low_lower_context_t* context, const loom_region_t* source_region,
    loom_consumption_region_query_t* out_query) {
  if (!iree_any_bit_set(source_region->flags, LOOM_REGION_INSTANCE_FLAG_CFG)) {
    loom_consumption_region_query_initialize(
        context->module, source_region, &context->planning_arena, out_query);
    return;
  }
  const loom_value_fact_cfg_region_t* fact_region =
      loom_value_fact_table_lookup_cfg_region(context->fact_table,
                                              source_region);
  IREE_ASSERT(fact_region != NULL,
              "source planning requires retained region structure");
  loom_consumption_region_query_initialize_with_cfg_graph(
      context->module, source_region, &fact_region->graph,
      /*liveness=*/NULL, /*value_domain=*/NULL, &context->planning_arena,
      out_query);
}

static iree_status_t loom_low_lower_source_candidate_plan_record_root(
    loom_low_lower_source_candidate_plan_t* candidate_plan,
    const loom_op_t* source_op,
    const loom_low_lower_source_memory_record_t* memory_record) {
  loom_low_lower_source_candidate_root_t* root = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(&candidate_plan->arena,
                                           sizeof(*root), (void**)&root));
  *root = (loom_low_lower_source_candidate_root_t){
      .source_op = source_op,
      .memory_record = memory_record,
  };
  if (candidate_plan->last_root != NULL) {
    candidate_plan->last_root->next = root;
  } else {
    candidate_plan->first_root = root;
  }
  candidate_plan->last_root = root;
  return iree_ok_status();
}

static iree_status_t loom_low_lower_visit_region_plan_ops(
    loom_low_lower_context_t* context, loom_region_t* source_region,
    const loom_low_lower_source_plan_observer_t* observer, void* observer_state,
    loom_low_lower_visibility_builder_t* visibility,
    loom_low_lower_source_memory_builder_t* memory_builder,
    loom_low_lower_source_candidate_plan_t* candidate_plan,
    iree_host_size_t* inout_plan_capacity) {
  const uint16_t* block_order =
      source_region == loom_func_like_body(context->source_function)
          ? context->lowering->source_plan.block_order
          : NULL;
  for (uint16_t position = 0; position < source_region->block_count;
       ++position) {
    const uint16_t block_index = block_order ? block_order[position] : position;
    loom_block_t* block = loom_region_block(source_region, block_index);
    IREE_RETURN_IF_ERROR(
        loom_low_lower_source_memory_enter_block(memory_builder, block));
    loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      const loom_trait_flags_t traits =
          loom_op_effective_traits(context->module, op);
      IREE_RETURN_IF_ERROR(
          loom_low_lower_source_memory_observe(memory_builder, context, op));
      IREE_RETURN_IF_ERROR(
          loom_low_lower_visibility_observe(context, visibility, op));
      const bool is_callable_exit =
          loom_low_lower_source_op_is_callable_exit(context, op);
      const bool is_structural = loom_low_lower_op_is_structural(
          context, op, traits, is_callable_exit);
      if (loom_cfg_cond_br_isa(op)) {
        IREE_RETURN_IF_ERROR(
            loom_low_lower_structural_plan_branch(context, op));
      }
      const bool has_plan_node = loom_low_lower_source_op_has_plan_node(
          context, op, traits, is_callable_exit);
      if (is_structural && !has_plan_node) {
        loom_low_lower_source_value_storage_mark_structural_demands(context, op,
                                                                    traits);
      }
      if (is_callable_exit) {
        IREE_RETURN_IF_ERROR(
            loom_low_lower_function_boundary_observe_exit(context, op));
      }
      if (observer != NULL) {
        observer->observe(observer_state, context, op);
      }
      if (candidate_plan != NULL && has_plan_node &&
          loom_target_contract_index_has_source_graph_root_op_kind(
              context->policy->contract.index, op->kind)) {
        // Retain roots during the owning source traversal, but defer contract
        // matching until observation, memory preparation and function-boundary
        // analysis have established every fact consumed by rule guards.
        IREE_RETURN_IF_ERROR(loom_low_lower_source_candidate_plan_record_root(
            candidate_plan, op, context->lowering->source_plan.memory.current));
      }
      if (loom_low_lower_supported_structured_source_op(context, op)) {
        loom_region_t* const* regions = loom_op_regions(op);
        for (uint8_t i = 0; i < op->region_count; ++i) {
          if (regions[i] != NULL) {
            IREE_RETURN_IF_ERROR(loom_low_lower_visit_region_plan_ops(
                context, regions[i], observer, observer_state, visibility,
                memory_builder, candidate_plan, inout_plan_capacity));
          }
        }
        continue;
      }
      if (op->region_count != 0) {
        continue;
      }
      if (has_plan_node) {
        ++(*inout_plan_capacity);
      }
    }
  }
  loom_low_lower_source_memory_leave_region(memory_builder, source_region);
  return iree_ok_status();
}

static iree_status_t loom_low_lower_prepare_plan(
    loom_low_lower_context_t* context, loom_region_t* source_body,
    loom_low_lower_source_candidate_plan_t* candidate_plan) {
  const loom_low_lower_source_plan_observer_t* observer =
      context->policy->source_plan_observer;
  void* observer_state = NULL;
  if (observer != NULL) {
    IREE_RETURN_IF_ERROR(
        observer->begin(observer->user_data, context, &observer_state));
  }
  iree_host_size_t plan_capacity = 0;
  loom_low_lower_visibility_builder_t visibility = {.body = source_body};
  if (context->policy->visibility_model) {
    visibility.model = context->policy->visibility_model(context);
  }
  loom_low_lower_source_memory_builder_t* memory_builder = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_source_memory_builder_allocate(context, &memory_builder));
  IREE_RETURN_IF_ERROR(loom_low_lower_visit_region_plan_ops(
      context, source_body, observer, observer_state, &visibility,
      memory_builder, candidate_plan, &plan_capacity));
  context->lowering->source_plan.memory.current = NULL;
  context->lowering->source_plan.memory.cursor =
      context->lowering->source_plan.memory.first;
  IREE_RETURN_IF_ERROR(loom_low_lower_visibility_select(
      context, &visibility,
      &context->lowering->source_plan.read_visibility_scope));
  if (observer != NULL) {
    IREE_RETURN_IF_ERROR(observer->end(observer_state, context));
  }
  if (!loom_low_lower_context_diagnostic_limit_reached(context)) {
    IREE_RETURN_IF_ERROR(loom_low_lower_source_memory_prepare(context));
  }
  if (context->result->error_count == 0) {
    IREE_RETURN_IF_ERROR(loom_low_lower_function_boundary_finalize(context));
  }
  context->lowering->source_plan.selected_plan_capacity = plan_capacity;
  context->lowering->source_plan.selected_plan_count = 0;
  context->lowering->source_plan.selected_plan_emit_index = 0;
  if (plan_capacity == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_function_array(
      context, plan_capacity,
      sizeof(*context->lowering->source_plan.selected_plans),
      (void**)&context->lowering->source_plan.selected_plans));
  return iree_ok_status();
}

static void loom_low_lower_record_selected_plan(
    loom_low_lower_context_t* context,
    loom_low_lower_selected_plan_t selected_plan) {
  loom_low_lower_source_plan_t* source_plan = &context->lowering->source_plan;
  IREE_ASSERT_LT(source_plan->selected_plan_count,
                 source_plan->selected_plan_capacity);
  source_plan->selected_plans[source_plan->selected_plan_count++] =
      selected_plan;
}

static void loom_low_lower_record_elided_hint_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op) {
  loom_low_lower_record_selected_plan(
      context, (loom_low_lower_selected_plan_t){
                   .source_op = source_op,
                   .kind = LOOM_LOW_LOWER_SELECTED_PLAN_CALLBACK,
                   .flags = LOOM_LOW_LOWER_SELECTED_PLAN_ELIDED,
                   .rule_set_index = UINT16_MAX,
                   .rule_index = UINT16_MAX,
                   .rule_set = NULL,
                   .rule = NULL,
                   .resolved_emits = NULL,
                   .data.target_plan = loom_low_lower_plan_empty(),
               });
}

static bool loom_low_lower_selected_plan_preserves_volatile_memory(
    const loom_module_t* module,
    const loom_low_lower_selected_plan_t* selected_plan) {
  if (!iree_any_bit_set(loom_memory_access_flags(loom_memory_access_cast(
                            module, selected_plan->source_op)),
                        LOOM_MEMORY_ACCESS_FLAG_VOLATILE)) {
    return true;
  }
  if (selected_plan->kind == LOOM_LOW_LOWER_SELECTED_PLAN_CALLBACK) {
    return iree_any_bit_set(selected_plan->data.target_plan.access_flags,
                            LOOM_MEMORY_ACCESS_FLAG_VOLATILE);
  }
  if (selected_plan->kind != LOOM_LOW_LOWER_SELECTED_PLAN_RULE ||
      selected_plan->rule == NULL || selected_plan->resolved_emits == NULL) {
    return false;
  }
  for (uint16_t i = 0; i < selected_plan->rule->emit_count; ++i) {
    if (iree_any_bit_set(selected_plan->resolved_emits[i].access_flags,
                         LOOM_MEMORY_ACCESS_FLAG_VOLATILE)) {
      return true;
    }
  }
  return false;
}

static iree_status_t loom_low_lower_finalize_selected_plans(
    loom_low_lower_context_t* context) {
  const loom_low_lower_source_plan_t* source_plan =
      &context->lowering->source_plan;
  iree_arena_allocator_t construction_arena;
  iree_arena_initialize(context->module->arena.block_pool, &construction_arena);
  loom_low_lower_rule_source_memory_plan_builder_t source_memory_builder = {
      .arena = &construction_arena,
  };
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < source_plan->selected_plan_count && iree_status_is_ok(status) &&
       context->result->error_count == 0;
       ++i) {
    loom_low_lower_selected_plan_t* selected_plan =
        &source_plan->selected_plans[i];
    if (selected_plan->rule != NULL) {
      loom_low_lower_planning_scope_begin(context);
      status = loom_low_lower_rule_plan_finalize(
          context, &source_memory_builder, selected_plan);
      loom_low_lower_planning_scope_end(context);
    } else if (selected_plan->kind ==
                   LOOM_LOW_LOWER_SELECTED_PLAN_DESCRIPTOR_MATRIX &&
               !iree_any_bit_set(selected_plan->flags,
                                 LOOM_LOW_LOWER_SELECTED_PLAN_ELIDED)) {
      const loom_value_id_t result =
          loom_op_const_results(selected_plan->source_op)[0];
      loom_type_t result_type = loom_type_none();
      status = loom_low_lower_map_value(context, selected_plan->source_op,
                                        result, &result_type);
      if (iree_status_is_ok(status) && context->result->error_count == 0) {
        status = loom_low_lower_plan_value_type(context, result, result_type);
      }
    } else if (selected_plan->kind == LOOM_LOW_LOWER_SELECTED_PLAN_CALLBACK &&
               !iree_any_bit_set(selected_plan->flags,
                                 LOOM_LOW_LOWER_SELECTED_PLAN_ELIDED |
                                     LOOM_LOW_LOWER_SELECTED_PLAN_CLAIMED) &&
               context->policy->finalize_plan.fn != NULL) {
      status = context->policy->finalize_plan.fn(
          context->policy->finalize_plan.user_data, context,
          selected_plan->source_op, selected_plan->data.target_plan);
    } else if (selected_plan->kind == LOOM_LOW_LOWER_SELECTED_PLAN_INVOKE) {
      status = loom_low_lower_source_invoke_finalize(
          context, selected_plan->source_op, selected_plan->data.invoke);
    }
    if (iree_status_is_ok(status) && context->result->error_count == 0 &&
        !loom_low_lower_selected_plan_preserves_volatile_memory(
            context->module, selected_plan)) {
      status = loom_low_lower_emit_target_context_error(
          context, selected_plan->source_op, LOOM_ERR_TARGET_081,
          /*extra_params=*/NULL, /*extra_param_count=*/0);
    }
  }
  iree_arena_deinitialize(&construction_arena);
  return status;
}

static bool loom_low_lower_rule_selection_can_record_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_rule_set_t* rule_set,
    const loom_low_lower_rule_selection_t* rule_selection) {
  if (!loom_low_lower_source_value_storage_rule_preserves_fact_demands(
          context, rule_set, source_op, rule_selection->source_nodes,
          rule_selection->source_node_count, rule_selection->rule)) {
    return false;
  }
  return true;
}

static iree_status_t loom_low_lower_record_selected_rule_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    uint16_t rule_set_index, const loom_low_lower_rule_set_t* rule_set,
    const loom_low_lower_rule_selection_t* rule_selection,
    const loom_low_lower_rule_source_memory_state_t* source_memory_state,
    bool* out_recorded) {
  *out_recorded = false;
  IREE_ASSERT_EQ(rule_selection->source_node_count, 1);
  IREE_ASSERT_EQ(loom_low_lower_rule_source_node_count(rule_selection->rule),
                 0);
  if (!loom_low_lower_rule_selection_can_record_plan(
          context, source_op, rule_set, rule_selection)) {
    return iree_ok_status();
  }
  const loom_low_lower_selected_source_memory_t* source_memory = NULL;
  if (rule_selection->uses_source_memory_access) {
    IREE_ASSERT(source_memory_state != NULL);
    IREE_ASSERT_EQ(source_memory_state->source_op, source_op);
    IREE_ASSERT(source_memory_state->plan_available);
    const loom_low_lower_source_memory_record_t* memory_record =
        context->lowering->source_plan.memory.current;
    IREE_ASSERT(memory_record != NULL && memory_record->source_op == source_op);
    loom_low_lower_selected_source_memory_t* retained_source_memory = NULL;
    IREE_RETURN_IF_ERROR(loom_low_lower_allocate_plan_data(
        context, sizeof(*retained_source_memory),
        (void**)&retained_source_memory));
    *retained_source_memory = (loom_low_lower_selected_source_memory_t){
        .access = *source_memory_state->access_plan,
        .origin = memory_record->origin,
        .dynamic_byte_offset_identity =
            memory_record->dynamic_byte_offset_identity,
    };
    source_memory = retained_source_memory;
  }
  loom_low_lower_record_selected_plan(
      context, (loom_low_lower_selected_plan_t){
                   .source_op = source_op,
                   .kind = LOOM_LOW_LOWER_SELECTED_PLAN_RULE,
                   .source_node_count = rule_selection->source_node_count,
                   .rule_set_index = rule_set_index,
                   .rule_index = rule_selection->rule_index,
                   .rule_set = rule_set,
                   .rule = rule_selection->rule,
                   .source_memory = source_memory,
                   .data.source_nodes = NULL,
               });
  *out_recorded = true;
  return iree_ok_status();
}

static iree_status_t loom_low_lower_source_candidate_allocate(
    loom_low_lower_source_candidate_plan_t* candidate_plan,
    loom_low_lower_source_candidate_kind_t kind,
    loom_low_lower_source_graph_alternative_state_t state,
    loom_low_lower_source_candidate_t** out_candidate) {
  *out_candidate = NULL;
  loom_low_lower_source_candidate_t* candidate = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(
      &candidate_plan->arena, sizeof(*candidate), (void**)&candidate));
  *candidate = (loom_low_lower_source_candidate_t){
      .kind = kind,
      .state = state,
  };
  *out_candidate = candidate;
  return iree_ok_status();
}

static iree_status_t loom_low_lower_source_candidate_add(
    loom_low_lower_source_candidate_plan_t* candidate_plan,
    const loom_op_t* const* source_nodes, uint8_t source_node_count,
    const loom_low_lower_source_graph_edge_t* edges, uint8_t priority,
    loom_low_lower_source_candidate_t* source_candidate) {
  uint32_t node_indices[LOOM_LOW_LOWER_MAX_SOURCE_NODES];
  for (uint8_t i = 0; i < source_node_count; ++i) {
    loom_low_lower_source_candidate_node_t* candidate_node = NULL;
    IREE_RETURN_IF_ERROR(loom_low_lower_source_candidate_plan_get_or_add_node(
        candidate_plan, source_nodes[i], &candidate_node, &node_indices[i]));
    IREE_ASSERT_EQ(candidate_node->source_op, source_nodes[i]);
  }
  loom_low_lower_source_graph_alternative_t* alternative = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_source_graph_plan_add_alternative(
      candidate_plan->graph_plan, node_indices, source_node_count, edges,
      priority, source_candidate, &alternative));
  if (source_node_count > 1) {
    const loom_low_lower_source_graph_candidate_t* graph_candidate =
        loom_low_lower_source_graph_alternative_candidate(alternative);
    if (loom_low_lower_source_graph_candidate_user_data(graph_candidate) ==
        NULL) {
      const iree_host_size_t allocation_size =
          sizeof(loom_low_lower_source_candidate_graph_t) +
          (iree_host_size_t)source_node_count * sizeof(*source_nodes);
      loom_low_lower_source_candidate_graph_t* graph = NULL;
      IREE_RETURN_IF_ERROR(iree_arena_allocate(
          &candidate_plan->arena, allocation_size, (void**)&graph));
      graph->source_node_count = source_node_count;
      for (uint8_t i = 0; i < source_node_count; ++i) {
        graph->source_nodes[i] = source_nodes[i];
      }
      loom_low_lower_source_graph_alternative_set_candidate_user_data(
          alternative, graph);
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_lower_source_candidate_add_rule(
    loom_low_lower_context_t* context,
    loom_low_lower_source_candidate_plan_t* candidate_plan,
    loom_low_lower_source_candidate_node_t* candidate_node,
    uint16_t rule_set_index, const loom_low_lower_rule_set_t* rule_set,
    const loom_low_lower_rule_selection_t* selection,
    loom_low_lower_rule_primary_descriptor_status_t primary_status,
    const loom_low_lower_rule_source_memory_state_t* source_memory_state) {
  bool placement_legal = false;
  IREE_RETURN_IF_ERROR(loom_low_lower_source_graph_placement_evaluate(
      context, selection, &placement_legal));
  if (!placement_legal) {
    return iree_ok_status();
  }
  if (selection->source_node_count > 1) {
    uint32_t root_node_index = UINT32_MAX;
    loom_low_lower_source_candidate_node_t* indexed_candidate_node = NULL;
    IREE_RETURN_IF_ERROR(loom_low_lower_source_candidate_plan_get_or_add_node(
        candidate_plan, selection->source_nodes[0], &indexed_candidate_node,
        &root_node_index));
    (void)root_node_index;
    if (candidate_node != NULL) {
      IREE_ASSERT_EQ(candidate_node->source_op,
                     indexed_candidate_node->source_op);
      IREE_ASSERT(indexed_candidate_node->memory_record == NULL ||
                  indexed_candidate_node->memory_record ==
                      candidate_node->memory_record);
      if (indexed_candidate_node->memory_record == NULL) {
        indexed_candidate_node->memory_record = candidate_node->memory_record;
      }
    } else {
      indexed_candidate_node->memory_record =
          context->lowering->source_plan.memory.current;
    }
    candidate_node = indexed_candidate_node;
  } else if (candidate_node == NULL) {
    uint32_t root_node_index = UINT32_MAX;
    IREE_RETURN_IF_ERROR(loom_low_lower_source_candidate_plan_get_or_add_node(
        candidate_plan, selection->source_nodes[0], &candidate_node,
        &root_node_index));
    (void)root_node_index;
    candidate_node->memory_record =
        context->lowering->source_plan.memory.current;
  }
  loom_low_lower_source_candidate_t* source_candidate = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_source_candidate_allocate(
      candidate_plan, LOOM_LOW_LOWER_SOURCE_CANDIDATE_RULE,
      primary_status == LOOM_LOW_LOWER_RULE_PRIMARY_DESCRIPTOR_AVAILABLE
          ? LOOM_LOW_LOWER_SOURCE_GRAPH_ALTERNATIVE_ENABLED
          : LOOM_LOW_LOWER_SOURCE_GRAPH_ALTERNATIVE_FALLBACK,
      &source_candidate));
  source_candidate->data.rule.rule_set = rule_set;
  source_candidate->data.rule.rule_set_index = rule_set_index;
  source_candidate->data.rule.rule_index = selection->rule_index;
  source_candidate->data.rule.emission_source_node_index =
      selection->emission_source_node_index;
  if (selection->uses_source_memory_access) {
    IREE_ASSERT(source_memory_state->plan_available);
    if (candidate_node->matched_source_memory == NULL) {
      const loom_low_lower_source_memory_record_t* memory_record =
          candidate_node->memory_record;
      IREE_ASSERT(memory_record != NULL && memory_record->available);
      loom_low_lower_selected_source_memory_t* retained_source_memory = NULL;
      IREE_RETURN_IF_ERROR(iree_arena_allocate(
          &candidate_plan->arena, sizeof(*retained_source_memory),
          (void**)&retained_source_memory));
      *retained_source_memory = (loom_low_lower_selected_source_memory_t){
          .access = *source_memory_state->access_plan,
          .origin = memory_record->origin,
          .dynamic_byte_offset_identity =
              memory_record->dynamic_byte_offset_identity,
      };
      candidate_node->matched_source_memory = retained_source_memory;
    }
    source_candidate->data.rule.source_memory =
        candidate_node->matched_source_memory;
  }
  loom_low_lower_source_graph_edge_t edges[LOOM_LOW_LOWER_MAX_SOURCE_NODES - 1];
  const loom_low_lower_source_graph_edge_t* graph_edges = NULL;
  if (selection->source_node_count > 1) {
    loom_low_lower_source_graph_rule_edges(rule_set, selection->rule, edges);
    graph_edges = edges;
  }
  return loom_low_lower_source_candidate_add(
      candidate_plan, selection->source_nodes, selection->source_node_count,
      graph_edges, loom_low_lower_rule_priority(selection->rule),
      source_candidate);
}

static iree_status_t loom_low_lower_source_candidate_add_descriptor_matrix(
    loom_low_lower_context_t* context,
    loom_low_lower_source_candidate_plan_t* candidate_plan,
    const loom_op_t* source_op,
    const loom_target_contract_descriptor_matrix_rule_t* matrix_rule,
    const loom_contract_request_t* contract_request,
    const loom_target_contract_query_result_t* query_result) {
  loom_low_lower_source_candidate_t* source_candidate = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_source_candidate_allocate(
      candidate_plan, LOOM_LOW_LOWER_SOURCE_CANDIDATE_DESCRIPTOR_MATRIX,
      LOOM_LOW_LOWER_SOURCE_GRAPH_ALTERNATIVE_ENABLED, &source_candidate));
  loom_low_lower_source_candidate_descriptor_matrix_t* descriptor_matrix = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(&candidate_plan->arena,
                                           sizeof(*descriptor_matrix),
                                           (void**)&descriptor_matrix));
  *descriptor_matrix = (loom_low_lower_source_candidate_descriptor_matrix_t){
      .matrix_rule = matrix_rule,
      .contract_request = *contract_request,
      .query_result = *query_result,
  };
  source_candidate->data.descriptor_matrix = descriptor_matrix;
  const loom_op_t* source_nodes[] = {source_op};
  return loom_low_lower_source_candidate_add(
      candidate_plan, source_nodes, /*source_node_count=*/1,
      /*edges=*/NULL, /*priority=*/0, source_candidate);
}

static iree_status_t loom_low_lower_source_candidate_add_selected_plan(
    loom_low_lower_context_t* context,
    loom_low_lower_source_candidate_plan_t* candidate_plan,
    const loom_low_lower_selected_plan_t* selected_plan,
    loom_low_lower_source_graph_alternative_state_t state) {
  loom_low_lower_source_candidate_t* source_candidate = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_source_candidate_allocate(
      candidate_plan, LOOM_LOW_LOWER_SOURCE_CANDIDATE_PLAN, state,
      &source_candidate));
  loom_low_lower_selected_plan_t* retained_plan = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(
      &candidate_plan->arena, sizeof(*retained_plan), (void**)&retained_plan));
  *retained_plan = *selected_plan;
  source_candidate->data.selected_plan = retained_plan;
  const loom_op_t* source_nodes[] = {selected_plan->source_op};
  return loom_low_lower_source_candidate_add(
      candidate_plan, source_nodes, /*source_node_count=*/1,
      /*edges=*/NULL, /*priority=*/0, source_candidate);
}

static iree_status_t loom_low_lower_select_fixed_plan(
    loom_low_lower_context_t* context,
    loom_low_lower_source_candidate_plan_t* candidate_plan,
    const loom_low_lower_selected_plan_t* selected_plan,
    loom_low_lower_source_graph_alternative_state_t state) {
  if (candidate_plan != NULL) {
    return loom_low_lower_source_candidate_add_selected_plan(
        context, candidate_plan, selected_plan, state);
  }
  loom_low_lower_record_selected_plan(context, *selected_plan);
  return iree_ok_status();
}

static iree_status_t loom_low_lower_try_select_op_callback(
    loom_low_lower_context_t* context,
    loom_low_lower_source_candidate_plan_t* candidate_plan,
    loom_low_lower_select_op_callback_t callback, const loom_op_t* source_op,
    bool* out_selected) {
  *out_selected = false;
  if (callback.fn == NULL) {
    return iree_ok_status();
  }

  loom_low_lower_plan_t plan = loom_low_lower_plan_empty();
  IREE_RETURN_IF_ERROR(
      callback.fn(callback.user_data, context, source_op, &plan));
  if (loom_low_lower_plan_is_empty(plan)) {
    return iree_ok_status();
  }
  const loom_low_lower_selected_plan_t selected_plan = {
      .source_op = source_op,
      .kind = LOOM_LOW_LOWER_SELECTED_PLAN_CALLBACK,
      .rule_set_index = UINT16_MAX,
      .rule_index = UINT16_MAX,
      .data.target_plan = plan,
  };
  if (candidate_plan != NULL) {
    IREE_RETURN_IF_ERROR(loom_low_lower_source_candidate_add_selected_plan(
        context, candidate_plan, &selected_plan,
        LOOM_LOW_LOWER_SOURCE_GRAPH_ALTERNATIVE_ENABLED));
  } else {
    loom_low_lower_record_selected_plan(context, selected_plan);
  }
  *out_selected = true;
  return iree_ok_status();
}

static iree_status_t loom_low_lower_source_candidate_add_callback(
    loom_low_lower_context_t* context,
    loom_low_lower_source_candidate_plan_t* candidate_plan,
    const loom_op_t* source_op, loom_low_lower_select_op_callback_t callback,
    loom_low_lower_source_graph_alternative_state_t state) {
  if (callback.fn == NULL) {
    return iree_ok_status();
  }
  loom_low_lower_source_candidate_t* source_candidate = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_source_candidate_allocate(
      candidate_plan, LOOM_LOW_LOWER_SOURCE_CANDIDATE_CALLBACK, state,
      &source_candidate));
  loom_low_lower_source_candidate_callback_t* callback_state = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(&candidate_plan->arena,
                                           sizeof(*callback_state),
                                           (void**)&callback_state));
  *callback_state = (loom_low_lower_source_candidate_callback_t){
      .callback = callback,
  };
  source_candidate->data.callback = callback_state;
  const loom_op_t* source_nodes[] = {source_op};
  return loom_low_lower_source_candidate_add(
      candidate_plan, source_nodes, /*source_node_count=*/1,
      /*edges=*/NULL, /*priority=*/0, source_candidate);
}

static iree_status_t loom_low_lower_emit_contract_query_rejection(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_target_contract_query_result_t* result) {
  if (result->rejection != NULL) {
    return loom_low_lower_emit_error_ref(
        context, source_op, result->rejection->error_ref,
        result->rejection->params, result->rejection->param_count);
  }
  return loom_low_lower_emit_no_target_contract(context, source_op);
}

static iree_status_t loom_low_lower_create_descriptor_matrix_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_target_contract_descriptor_matrix_rule_t* matrix_rule,
    const loom_contract_request_t* contract_request,
    const loom_target_contract_query_result_t* query_result,
    loom_low_lower_selected_plan_t* out_selected_plan) {
  loom_low_lower_descriptor_matrix_plan_t* plan_data = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_plan_data(
      context, sizeof(*plan_data), (void**)&plan_data));
  plan_data->source = matrix_rule->source;
  plan_data->transform_flags =
      query_result->selected_descriptor_matrix_transform_flags;
  if (query_result->selected_descriptor == NULL) {
    IREE_ASSERT_UNREACHABLE("descriptor-matrix legal query has no descriptor");
    IREE_BUILTIN_UNREACHABLE();
  }
  plan_data->descriptor.descriptor = query_result->selected_descriptor;
  plan_data->contract_request = *contract_request;
  if (iree_any_bit_set(
          plan_data->transform_flags,
          LOOM_TARGET_CONTRACT_DESCRIPTOR_MATRIX_TRANSFORM_TRANSPOSE_MN)) {
    loom_contract_request_transpose_mn(&plan_data->contract_request);
  }
  plan_data->attrs = loom_named_attr_slice_empty();
  plan_data->native_contraction_facts =
      query_result->selected_native_contraction_facts;
  if (loom_low_descriptor_layout(context->descriptor_set,
                                 plan_data->descriptor.descriptor)
          ->immediate_count != 0) {
    if (context->policy->descriptor_matrix.attrs == NULL) {
      IREE_ASSERT_UNREACHABLE("descriptor-matrix policy has no attrs callback");
      IREE_BUILTIN_UNREACHABLE();
    }
    IREE_RETURN_IF_ERROR(context->policy->descriptor_matrix.attrs(
        context->policy->descriptor_matrix.user_data, context, matrix_rule,
        &plan_data->contract_request, plan_data->descriptor.descriptor,
        &plan_data->attrs));
  }
  *out_selected_plan = (loom_low_lower_selected_plan_t){
      .source_op = source_op,
      .kind = LOOM_LOW_LOWER_SELECTED_PLAN_DESCRIPTOR_MATRIX,
      .rule_set_index = UINT16_MAX,
      .rule_index = query_result->rule_index,
      .rule_set = NULL,
      .rule = NULL,
      .resolved_emits = NULL,
      .data.target_plan = loom_low_lower_plan_make(source_op->kind, plan_data),
  };
  return iree_ok_status();
}

static iree_status_t loom_low_lower_record_descriptor_matrix_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_target_contract_descriptor_matrix_rule_t* matrix_rule,
    const loom_contract_request_t* contract_request,
    const loom_target_contract_query_result_t* query_result) {
  loom_low_lower_selected_plan_t selected_plan;
  IREE_RETURN_IF_ERROR(loom_low_lower_create_descriptor_matrix_plan(
      context, source_op, matrix_rule, contract_request, query_result,
      &selected_plan));
  loom_low_lower_record_selected_plan(context, selected_plan);
  return iree_ok_status();
}

static bool loom_low_lower_source_plan_requires_primary_descriptor_query(
    const loom_low_lower_context_t* context, uint16_t rule_set_index) {
  const uint64_t* query_rule_set_bits =
      context->lowering->source_plan.primary_descriptor_query_rule_set_bits;
  return query_rule_set_bits != NULL &&
         iree_all_bits_set(query_rule_set_bits[rule_set_index / 64u],
                           UINT64_C(1) << (rule_set_index % 64u));
}

static iree_status_t loom_low_lower_plan_primary_descriptor_available(
    const loom_low_lower_context_t* context,
    const loom_low_lower_rule_match_context_t* match_context,
    uint16_t rule_set_index, const loom_low_lower_rule_set_t* rule_set,
    const loom_low_lower_rule_selection_t* selection, bool* out_available) {
  if (!loom_low_lower_source_plan_requires_primary_descriptor_query(
          context, rule_set_index)) {
    *out_available = true;
    return iree_ok_status();
  }
  loom_low_lower_rule_primary_descriptor_status_t status =
      LOOM_LOW_LOWER_RULE_PRIMARY_DESCRIPTOR_AVAILABLE;
  IREE_RETURN_IF_ERROR(loom_low_lower_rule_query_primary_descriptor(
      match_context, rule_set, selection->rule, &status));
  IREE_ASSERT_NE(status, LOOM_LOW_LOWER_RULE_PRIMARY_DESCRIPTOR_MISSING);
  *out_available = status == LOOM_LOW_LOWER_RULE_PRIMARY_DESCRIPTOR_AVAILABLE;
  return iree_ok_status();
}

static void loom_low_lower_source_contract_retain_rule_failure(
    loom_low_lower_source_candidate_rejection_t* rejection,
    const loom_low_lower_contract_case_match_t* match) {
  if (loom_low_lower_rule_failure_is_better(match->selection.failure,
                                            rejection->rule_failure)) {
    rejection->failed_rule_set = match->rule_set;
    rejection->rule_failure = match->selection.failure;
  }
}

typedef struct loom_low_lower_source_candidate_contract_state_t {
  // Mutable lowering context owning match state and retained plans.
  loom_low_lower_context_t* context;
  // Function candidate plan receiving alternatives.
  loom_low_lower_source_candidate_plan_t* candidate_plan;
  // Candidate node receiving alternatives.
  loom_low_lower_source_candidate_node_t* candidate_node;
  // Source operation being matched.
  const loom_op_t* source_op;
  // Canonical source-memory state shared by every candidate rule.
  loom_low_lower_rule_source_memory_state_t* source_memory_state;
  // Match context used for primary descriptor availability queries.
  loom_low_lower_rule_match_context_t* match_context;
  // Best generated rejection seen when no candidate matched.
  loom_low_lower_source_candidate_rejection_t rejection;
  // Current generated contract iteration mode.
  loom_low_lower_contract_case_iteration_mode_t iteration_mode;
  // True when collection found at least one legal alternative.
  bool matched;
} loom_low_lower_source_candidate_contract_state_t;

static iree_status_t loom_low_lower_source_candidate_visit_contract_case(
    void* user_data, uint16_t case_index,
    const loom_low_lower_contract_case_match_t* match, bool* out_stop) {
  *out_stop = false;
  loom_low_lower_source_candidate_contract_state_t* state =
      (loom_low_lower_source_candidate_contract_state_t*)user_data;
  const loom_target_contract_index_t* index =
      state->context->policy->contract.index;
  const loom_target_contract_case_t* contract_case = &index->cases[case_index];
  const loom_target_contract_binding_t* binding =
      &index->bindings[contract_case->binding_index];
  if (match->kind == LOOM_LOW_LOWER_CONTRACT_CASE_MATCH_DESCRIPTOR_MATRIX) {
    if (match->result.outcome == LOOM_TARGET_CONTRACT_QUERY_LEGAL) {
      const loom_target_contract_descriptor_matrix_rule_t* matrix_rule =
          &binding->fragment->descriptor_matrices[contract_case->row_index];
      IREE_RETURN_IF_ERROR(
          loom_low_lower_source_candidate_add_descriptor_matrix(
              state->context, state->candidate_plan, state->source_op,
              matrix_rule, &match->contract_request, &match->result));
      state->matched = true;
    } else if (match->result.outcome ==
                   LOOM_TARGET_CONTRACT_QUERY_UNSUPPORTED ||
               match->result.outcome == LOOM_TARGET_CONTRACT_QUERY_INVALID_IR) {
      if (state->rejection.matrix_rejection.outcome ==
          LOOM_TARGET_CONTRACT_QUERY_UNHANDLED) {
        state->rejection.matrix_rejection = match->result;
      }
    }
    return iree_ok_status();
  }
  if (match->kind != LOOM_LOW_LOWER_CONTRACT_CASE_MATCH_LOWER_RULE) {
    return iree_ok_status();
  }
  if (match->selection.rule == NULL) {
    loom_low_lower_source_contract_retain_rule_failure(&state->rejection,
                                                       match);
    return iree_ok_status();
  }
  IREE_ASSERT_EQ(state->iteration_mode,
                 LOOM_LOW_LOWER_CONTRACT_CASE_ITERATION_CANDIDATES,
                 "generated contract candidate index omitted a matching rule");
  bool primary_descriptor_available = false;
  IREE_RETURN_IF_ERROR(loom_low_lower_plan_primary_descriptor_available(
      state->context, state->match_context, binding->rule_set_index,
      match->rule_set, &match->selection, &primary_descriptor_available));
  // A graph-structural root has a semantic fallback owned by the source plan.
  // An unavailable lowering-only fusion must not displace that legal cover
  // merely to report its missing feature.
  if (!primary_descriptor_available &&
      contract_case->system ==
          LOOM_TARGET_CONTRACT_SYSTEM_DESCRIPTOR_RULE_LOWERING_ONLY &&
      loom_low_lower_source_op_is_graph_structural_root(state->context,
                                                        state->source_op)) {
    return iree_ok_status();
  }
  state->matched = true;
  return loom_low_lower_source_candidate_add_rule(
      state->context, state->candidate_plan, state->candidate_node,
      binding->rule_set_index, match->rule_set, &match->selection,
      primary_descriptor_available
          ? LOOM_LOW_LOWER_RULE_PRIMARY_DESCRIPTOR_AVAILABLE
          : LOOM_LOW_LOWER_RULE_PRIMARY_DESCRIPTOR_FEATURES_UNAVAILABLE,
      state->source_memory_state);
}

typedef struct loom_low_lower_source_stream_contract_state_t {
  // Mutable lowering context owning match state and retained plans.
  loom_low_lower_context_t* context;
  // Source operation being matched.
  const loom_op_t* source_op;
  // Canonical source-memory state shared by every candidate rule.
  loom_low_lower_rule_source_memory_state_t* source_memory_state;
  // Match context used for primary descriptor availability queries.
  loom_low_lower_rule_match_context_t* match_context;
  // Best generated rejection seen when no candidate matched.
  loom_low_lower_source_candidate_rejection_t rejection;
  // First feature-unavailable rule retained as a same-graph fallback.
  loom_low_lower_rule_selection_t fallback_selection;
  // Rule set owning |fallback_selection|.
  const loom_low_lower_rule_set_t* fallback_rule_set;
  // Policy rule-set ordinal owning |fallback_selection|.
  uint16_t fallback_rule_set_index;
  // Current generated contract iteration mode.
  loom_low_lower_contract_case_iteration_mode_t iteration_mode;
  // True after a plan or terminal contract rejection has been recorded.
  bool selected;
} loom_low_lower_source_stream_contract_state_t;

static iree_status_t loom_low_lower_source_stream_visit_contract_case(
    void* user_data, uint16_t case_index,
    const loom_low_lower_contract_case_match_t* match, bool* out_stop) {
  *out_stop = false;
  loom_low_lower_source_stream_contract_state_t* state =
      (loom_low_lower_source_stream_contract_state_t*)user_data;
  const loom_target_contract_index_t* index =
      state->context->policy->contract.index;
  const loom_target_contract_case_t* contract_case = &index->cases[case_index];
  const loom_target_contract_binding_t* binding =
      &index->bindings[contract_case->binding_index];
  if (match->kind == LOOM_LOW_LOWER_CONTRACT_CASE_MATCH_DESCRIPTOR_MATRIX) {
    if (match->result.outcome == LOOM_TARGET_CONTRACT_QUERY_LEGAL) {
      const loom_target_contract_descriptor_matrix_rule_t* matrix_rule =
          &binding->fragment->descriptor_matrices[contract_case->row_index];
      IREE_RETURN_IF_ERROR(loom_low_lower_record_descriptor_matrix_plan(
          state->context, state->source_op, matrix_rule,
          &match->contract_request, &match->result));
      state->selected = true;
      *out_stop = true;
    } else if (match->result.outcome ==
                   LOOM_TARGET_CONTRACT_QUERY_UNSUPPORTED ||
               match->result.outcome == LOOM_TARGET_CONTRACT_QUERY_INVALID_IR) {
      IREE_RETURN_IF_ERROR(loom_low_lower_emit_contract_query_rejection(
          state->context, state->source_op, &match->result));
      state->selected = true;
      *out_stop = true;
    }
    return iree_ok_status();
  }
  if (match->kind != LOOM_LOW_LOWER_CONTRACT_CASE_MATCH_LOWER_RULE) {
    return iree_ok_status();
  }
  if (match->selection.rule == NULL) {
    loom_low_lower_source_contract_retain_rule_failure(&state->rejection,
                                                       match);
    return iree_ok_status();
  }
  if (state->iteration_mode == LOOM_LOW_LOWER_CONTRACT_CASE_ITERATION_ALL) {
    return iree_ok_status();
  }
  if (state->fallback_rule_set != NULL &&
      !loom_low_lower_rule_selections_have_same_source_graph(
          &state->fallback_selection, &match->selection)) {
    return iree_ok_status();
  }

  bool primary_descriptor_available = false;
  IREE_RETURN_IF_ERROR(loom_low_lower_plan_primary_descriptor_available(
      state->context, state->match_context, binding->rule_set_index,
      match->rule_set, &match->selection, &primary_descriptor_available));
  if (!primary_descriptor_available) {
    if (state->fallback_rule_set == NULL &&
        loom_low_lower_rule_selection_can_record_plan(
            state->context, state->source_op, match->rule_set,
            &match->selection)) {
      state->fallback_selection = match->selection;
      state->fallback_rule_set = match->rule_set;
      state->fallback_rule_set_index = binding->rule_set_index;
    }
    return iree_ok_status();
  }

  bool recorded = false;
  IREE_RETURN_IF_ERROR(loom_low_lower_record_selected_rule_plan(
      state->context, state->source_op, binding->rule_set_index,
      match->rule_set, &match->selection, state->source_memory_state,
      &recorded));
  if (recorded) {
    state->selected = true;
    *out_stop = true;
  }
  return iree_ok_status();
}

static iree_status_t loom_low_lower_plan_op_from_contract_index(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_consumption_region_query_t* consumption_query,
    const loom_low_lower_rule_set_t** inout_failed_rule_set,
    loom_low_lower_rule_failure_t* inout_rule_failure,
    loom_low_lower_rule_source_memory_state_t* source_memory_state,
    bool* out_selected) {
  *out_selected = false;
  const loom_target_contract_index_t* index = context->policy->contract.index;
  const loom_target_contract_op_entry_t op_entry =
      loom_target_contract_index_lookup_kind(index, source_op->kind);
  if (loom_target_contract_op_entry_is_empty(op_entry)) {
    return iree_ok_status();
  }

  loom_target_contract_query_environment_t environment = {0};
  IREE_RETURN_IF_ERROR(loom_low_lower_source_query_environment_initialize(
      context, context->descriptor_set, &environment));
  const loom_low_lower_contract_query_options_t query_options = {
      .contract_index = index,
      .rule_sets = context->policy->contract.rule_sets,
      .descriptor_matrix = context->policy->descriptor_matrix,
  };
  loom_low_lower_rule_match_context_t match_context;
  loom_low_lower_rule_match_context_initialize_from_lowering(
      context, environment.view_regions, source_memory_state, &match_context);
  // The lowering policy owns this rule-set list and its paired descriptor set.
  // Generated primary feature classes can therefore bypass descriptor lookup.
  match_context.flags |=
      LOOM_LOW_LOWER_RULE_MATCH_FLAG_PRIMARY_DESCRIPTORS_BOUND;
  match_context.consumption_query = consumption_query;

  const loom_low_lower_rule_set_t* initial_failed_rule_set =
      *inout_failed_rule_set;
  const loom_low_lower_rule_failure_t initial_rule_failure =
      *inout_rule_failure;
  loom_low_lower_source_stream_contract_state_t state = {
      .context = context,
      .source_op = source_op,
      .source_memory_state = source_memory_state,
      .match_context = &match_context,
      .rejection =
          {
              .failed_rule_set = initial_failed_rule_set,
              .rule_failure = initial_rule_failure,
          },
      .fallback_selection = {.rule_index = UINT16_MAX},
      .fallback_rule_set_index = UINT16_MAX,
      .iteration_mode = LOOM_LOW_LOWER_CONTRACT_CASE_ITERATION_CANDIDATES,
  };
  const loom_low_lower_contract_case_callback_t callback = {
      .fn = loom_low_lower_source_stream_visit_contract_case,
      .user_data = &state,
  };
  bool used_candidate_index = false;
  IREE_RETURN_IF_ERROR(loom_low_lower_visit_lowering_contract_cases(
      &environment, &query_options, source_op, state.iteration_mode,
      LOOM_LOW_LOWER_CONTRACT_CASE_SCOPE_SINGLETON, &match_context, callback,
      &used_candidate_index));
  if (!state.selected && state.fallback_rule_set != NULL) {
    bool recorded = false;
    IREE_RETURN_IF_ERROR(loom_low_lower_record_selected_rule_plan(
        context, source_op, state.fallback_rule_set_index,
        state.fallback_rule_set, &state.fallback_selection, source_memory_state,
        &recorded));
    IREE_ASSERT(recorded);
    state.selected = true;
  }
  if (!state.selected && used_candidate_index) {
    state.rejection.failed_rule_set = initial_failed_rule_set;
    state.rejection.rule_failure = initial_rule_failure;
    state.iteration_mode = LOOM_LOW_LOWER_CONTRACT_CASE_ITERATION_ALL;
    IREE_RETURN_IF_ERROR(loom_low_lower_visit_lowering_contract_cases(
        &environment, &query_options, source_op, state.iteration_mode,
        LOOM_LOW_LOWER_CONTRACT_CASE_SCOPE_SINGLETON, &match_context, callback,
        /*out_used_candidate_index=*/NULL));
  }
  *inout_failed_rule_set = state.rejection.failed_rule_set;
  *inout_rule_failure = state.rejection.rule_failure;
  *out_selected = state.selected;
  return iree_ok_status();
}

static iree_status_t loom_low_lower_source_candidate_retain_rejection(
    loom_low_lower_source_candidate_plan_t* candidate_plan,
    loom_low_lower_source_candidate_node_t* candidate_node,
    loom_low_lower_source_candidate_rejection_t** out_rejection) {
  if (candidate_node->rejection == NULL) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate(
        &candidate_plan->arena, sizeof(*candidate_node->rejection),
        (void**)&candidate_node->rejection));
    memset(candidate_node->rejection, 0, sizeof(*candidate_node->rejection));
  }
  *out_rejection = candidate_node->rejection;
  return iree_ok_status();
}

static iree_status_t loom_low_lower_source_candidate_collect_contract(
    loom_low_lower_context_t* context,
    loom_low_lower_source_candidate_plan_t* candidate_plan,
    loom_low_lower_source_candidate_node_t* candidate_node,
    loom_consumption_region_query_t* consumption_query) {
  const loom_op_t* source_op = candidate_node->source_op;
  const loom_target_contract_index_t* index = context->policy->contract.index;
  if (index == NULL) {
    return iree_ok_status();
  }
  const loom_target_contract_op_entry_t op_entry =
      loom_target_contract_index_lookup_kind(index, source_op->kind);
  if (loom_target_contract_op_entry_is_empty(op_entry)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_low_lower_source_candidate_plan_prepare_query_environment(
          candidate_plan));

  loom_low_source_memory_access_plan_t source_memory_access;
  loom_low_lower_rule_source_memory_state_t source_memory_state;
  loom_low_lower_rule_source_memory_state_initialize(
      source_op, &source_memory_access, &source_memory_state);
  if (candidate_node->memory_record != NULL) {
    if (candidate_node->memory_record->available) {
      source_memory_state.retained_access =
          &candidate_node->memory_record->access;
    } else {
      source_memory_state.plan_attempted = true;
      source_memory_state.diagnostic =
          candidate_node->memory_record->diagnostic;
    }
  }
  loom_low_lower_rule_match_context_t match_context;
  loom_low_lower_rule_match_context_initialize_from_lowering(
      context, candidate_plan->query_environment.view_regions,
      &source_memory_state, &match_context);
  match_context.flags |=
      LOOM_LOW_LOWER_RULE_MATCH_FLAG_PRIMARY_DESCRIPTORS_BOUND;
  match_context.consumption_query = consumption_query;
  loom_low_lower_source_candidate_contract_state_t collect_state = {
      .context = context,
      .candidate_plan = candidate_plan,
      .candidate_node = candidate_node,
      .source_op = source_op,
      .source_memory_state = &source_memory_state,
      .match_context = &match_context,
      .iteration_mode = LOOM_LOW_LOWER_CONTRACT_CASE_ITERATION_CANDIDATES,
  };
  const loom_low_lower_contract_case_callback_t callback = {
      .fn = loom_low_lower_source_candidate_visit_contract_case,
      .user_data = &collect_state,
  };
  bool used_candidate_index = false;
  IREE_RETURN_IF_ERROR(loom_low_lower_visit_lowering_contract_cases(
      &candidate_plan->query_environment, &candidate_plan->query_options,
      source_op, LOOM_LOW_LOWER_CONTRACT_CASE_ITERATION_CANDIDATES,
      LOOM_LOW_LOWER_CONTRACT_CASE_SCOPE_SINGLETON, &match_context, callback,
      &used_candidate_index));
  if (collect_state.matched || !used_candidate_index) {
    return iree_ok_status();
  }

  collect_state.iteration_mode = LOOM_LOW_LOWER_CONTRACT_CASE_ITERATION_ALL;
  IREE_RETURN_IF_ERROR(loom_low_lower_visit_lowering_contract_cases(
      &candidate_plan->query_environment, &candidate_plan->query_options,
      source_op, LOOM_LOW_LOWER_CONTRACT_CASE_ITERATION_ALL,
      LOOM_LOW_LOWER_CONTRACT_CASE_SCOPE_SINGLETON, &match_context, callback,
      /*out_used_candidate_index=*/NULL));
  if (collect_state.rejection.failed_rule_set != NULL ||
      collect_state.rejection.matrix_rejection.outcome ==
          LOOM_TARGET_CONTRACT_QUERY_UNSUPPORTED ||
      collect_state.rejection.matrix_rejection.outcome ==
          LOOM_TARGET_CONTRACT_QUERY_INVALID_IR) {
    loom_low_lower_source_candidate_rejection_t* rejection = NULL;
    IREE_RETURN_IF_ERROR(loom_low_lower_source_candidate_retain_rejection(
        candidate_plan, candidate_node, &rejection));
    *rejection = collect_state.rejection;
  }
  return iree_ok_status();
}

static iree_status_t loom_low_lower_source_graph_discover_root(
    loom_low_lower_context_t* context,
    loom_low_lower_source_candidate_plan_t* candidate_plan,
    const loom_low_lower_source_candidate_root_t* root,
    loom_consumption_region_query_t* consumption_query) {
  const loom_op_t* source_op = root->source_op;
  const loom_target_contract_index_t* index = context->policy->contract.index;
  IREE_ASSERT(loom_target_contract_index_has_source_graph_root_op_kind(
      index, source_op->kind));
  IREE_RETURN_IF_ERROR(
      loom_low_lower_source_candidate_plan_prepare_query_environment(
          candidate_plan));

  loom_low_source_memory_access_plan_t source_memory_access;
  loom_low_lower_rule_source_memory_state_t source_memory_state;
  loom_low_lower_rule_source_memory_state_initialize(
      source_op, &source_memory_access, &source_memory_state);
  const loom_low_lower_source_memory_record_t* memory_record =
      root->memory_record;
  if (memory_record != NULL) {
    if (memory_record->available) {
      source_memory_state.retained_access = &memory_record->access;
    } else {
      source_memory_state.plan_attempted = true;
      source_memory_state.diagnostic = memory_record->diagnostic;
    }
  }
  loom_low_lower_rule_match_context_t match_context;
  loom_low_lower_rule_match_context_initialize_from_lowering(
      context, candidate_plan->query_environment.view_regions,
      &source_memory_state, &match_context);
  match_context.flags |=
      LOOM_LOW_LOWER_RULE_MATCH_FLAG_PRIMARY_DESCRIPTORS_BOUND;
  match_context.consumption_query = consumption_query;
  loom_low_lower_source_candidate_node_t candidate_node = {
      .source_op = source_op,
      .memory_record = memory_record,
  };
  loom_low_lower_source_candidate_contract_state_t collect_state = {
      .context = context,
      .candidate_plan = candidate_plan,
      .candidate_node = &candidate_node,
      .source_op = source_op,
      .source_memory_state = &source_memory_state,
      .match_context = &match_context,
      .iteration_mode = LOOM_LOW_LOWER_CONTRACT_CASE_ITERATION_CANDIDATES,
  };
  return loom_low_lower_visit_lowering_contract_cases(
      &candidate_plan->query_environment, &candidate_plan->query_options,
      source_op, LOOM_LOW_LOWER_CONTRACT_CASE_ITERATION_CANDIDATES,
      LOOM_LOW_LOWER_CONTRACT_CASE_SCOPE_SOURCE_GRAPH, &match_context,
      (loom_low_lower_contract_case_callback_t){
          .fn = loom_low_lower_source_candidate_visit_contract_case,
          .user_data = &collect_state,
      },
      /*out_used_candidate_index=*/NULL);
}

static iree_status_t loom_low_lower_source_graph_discover(
    loom_low_lower_context_t* context,
    loom_low_lower_source_candidate_plan_t* candidate_plan) {
  // Possible roots remain separate from the sparse participant index. Only a
  // legal multi-operation match admits its nodes to global graph selection;
  // unmatched roots retain the direct streaming planner.
  for (loom_low_lower_source_candidate_root_t* root =
           candidate_plan->first_root;
       root != NULL; root = root->next) {
    loom_low_lower_planning_scope_begin(context);
    loom_consumption_region_query_t consumption_query;
    loom_low_lower_source_consumption_query_initialize(
        context, root->source_op->parent_block->parent_region,
        &consumption_query);
    iree_status_t status = loom_low_lower_source_graph_discover_root(
        context, candidate_plan, root, &consumption_query);
    loom_low_lower_planning_scope_end(context);
    IREE_RETURN_IF_ERROR(status);
  }
  return iree_ok_status();
}

static iree_status_t loom_low_lower_plan_op(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_consumption_region_query_t* consumption_query, bool is_callable_exit,
    loom_low_lower_source_candidate_plan_t* candidate_plan) {
  if (source_op->tied_result_count != 0) {
    loom_low_lower_plan_tied_result_types(context, source_op);
  }
  if (source_op->region_count != 0) {
    if (loom_low_lower_supported_structured_source_op(context, source_op)) {
      return loom_low_lower_structural_plan_op(context, source_op);
    }
    const loom_diagnostic_param_t params[] = {
        loom_param_u32(source_op->region_count),
    };
    return loom_low_lower_emit_target_context_error(context, source_op,
                                                    LOOM_ERR_TARGET_030, params,
                                                    IREE_ARRAYSIZE(params));
  }
  const loom_trait_flags_t traits =
      loom_op_effective_traits(context->module, source_op);
  const bool is_structural = loom_low_lower_op_is_structural(
      context, source_op, traits, is_callable_exit);
  const bool is_graph_structural_root =
      loom_low_lower_source_op_is_graph_structural_root(context, source_op);
  if (is_structural && !is_graph_structural_root) {
    return loom_low_lower_structural_plan_op(context, source_op);
  }
  if (is_graph_structural_root && candidate_plan == NULL) {
    loom_low_lower_source_value_storage_mark_structural_demands(
        context, source_op, traits);
    return loom_low_lower_structural_plan_op(context, source_op);
  }
  if (loom_low_lower_source_plan_op_is_metadata(source_op->kind)) {
    return iree_ok_status();
  }

  loom_low_lower_source_candidate_node_t* candidate_node = NULL;
  if (candidate_plan != NULL) {
    uint32_t node_index = UINT32_MAX;
    candidate_node = loom_low_lower_source_candidate_plan_lookup_node(
        candidate_plan, source_op, &node_index);
    IREE_ASSERT(candidate_node != NULL);
    IREE_ASSERT_LT(node_index, candidate_plan->node_count);
    IREE_ASSERT_EQ(candidate_node->source_op, source_op);
    IREE_ASSERT_EQ(candidate_node->memory_record,
                   context->lowering->source_plan.memory.current);
  }

  if (is_graph_structural_root) {
    IREE_ASSERT(candidate_plan != NULL);
    IREE_RETURN_IF_ERROR(loom_low_lower_structural_plan_op(context, source_op));
    const bool exact_branch = loom_low_lower_structural_branch_exact_bool(
        context, source_op, /*out_condition=*/NULL);
    const loom_low_lower_selected_plan_t selected_plan = {
        .source_op = source_op,
        .kind = LOOM_LOW_LOWER_SELECTED_PLAN_STRUCTURAL,
        .rule_set_index = UINT16_MAX,
        .rule_index = UINT16_MAX,
    };
    IREE_RETURN_IF_ERROR(loom_low_lower_select_fixed_plan(
        context, candidate_plan, &selected_plan,
        exact_branch ? LOOM_LOW_LOWER_SOURCE_GRAPH_ALTERNATIVE_FIXED
                     : LOOM_LOW_LOWER_SOURCE_GRAPH_ALTERNATIVE_FALLBACK));
    if (exact_branch) {
      return iree_ok_status();
    }
    return loom_low_lower_source_candidate_collect_contract(
        context, candidate_plan, candidate_node, consumption_query);
  }

  if (loom_low_invoke_isa(source_op)) {
    const loom_low_lower_source_invoke_plan_t* plan = NULL;
    IREE_RETURN_IF_ERROR(
        loom_low_lower_source_invoke_plan(context, source_op, &plan));
    if (plan == NULL && candidate_plan == NULL) {
      return iree_ok_status();
    }
    const loom_low_lower_selected_plan_t selected_plan = {
        .source_op = source_op,
        .kind = plan != NULL ? LOOM_LOW_LOWER_SELECTED_PLAN_INVOKE
                             : LOOM_LOW_LOWER_SELECTED_PLAN_CALLBACK,
        .flags = plan != NULL ? 0 : LOOM_LOW_LOWER_SELECTED_PLAN_ELIDED,
        .rule_set_index = UINT16_MAX,
        .rule_index = UINT16_MAX,
        .data.invoke = plan,
    };
    return loom_low_lower_select_fixed_plan(
        context, candidate_plan, &selected_plan,
        LOOM_LOW_LOWER_SOURCE_GRAPH_ALTERNATIVE_FIXED);
  }

  if (loom_buffer_alloca_isa(source_op)) {
    bool selected = false;
    const loom_low_lower_function_storage_plan_t* plan = NULL;
    IREE_RETURN_IF_ERROR(loom_low_lower_function_storage_select(
        context, source_op, &selected, &plan));
    if (selected) {
      const loom_low_lower_selected_plan_t selected_plan = {
          .source_op = source_op,
          .kind = LOOM_LOW_LOWER_SELECTED_PLAN_FUNCTION_STORAGE,
          .rule_set_index = UINT16_MAX,
          .rule_index = UINT16_MAX,
          .data.function_storage = plan,
      };
      return loom_low_lower_select_fixed_plan(
          context, candidate_plan, &selected_plan,
          LOOM_LOW_LOWER_SOURCE_GRAPH_ALTERNATIVE_FIXED);
    }
  }

  const loom_low_lower_source_memory_record_t* prepared_memory =
      candidate_node != NULL ? candidate_node->memory_record
                             : context->lowering->source_plan.memory.current;
  if (prepared_memory &&
      !loom_low_lower_plan_is_empty(prepared_memory->prepared_plan)) {
    const loom_low_lower_selected_plan_t selected_plan = {
        .source_op = source_op,
        .kind = LOOM_LOW_LOWER_SELECTED_PLAN_CALLBACK,
        .rule_set_index = UINT16_MAX,
        .rule_index = UINT16_MAX,
        .data.target_plan = prepared_memory->prepared_plan,
    };
    return loom_low_lower_select_fixed_plan(
        context, candidate_plan, &selected_plan,
        LOOM_LOW_LOWER_SOURCE_GRAPH_ALTERNATIVE_FIXED);
  }
  bool selected_callback = false;
  IREE_RETURN_IF_ERROR(loom_low_lower_try_select_op_callback(
      context, candidate_plan, context->policy->preselect_op, source_op,
      &selected_callback));
  if (selected_callback) {
    return iree_ok_status();
  }

  if (candidate_plan != NULL) {
    IREE_RETURN_IF_ERROR(loom_low_lower_source_candidate_collect_contract(
        context, candidate_plan, candidate_node, consumption_query));
    IREE_RETURN_IF_ERROR(loom_low_lower_source_candidate_add_callback(
        context, candidate_plan, source_op, context->policy->select_op,
        LOOM_LOW_LOWER_SOURCE_GRAPH_ALTERNATIVE_FALLBACK));
    if (loom_low_lower_op_is_discardable_hint(context->module, source_op)) {
      const loom_low_lower_selected_plan_t selected_plan = {
          .source_op = source_op,
          .kind = LOOM_LOW_LOWER_SELECTED_PLAN_CALLBACK,
          .flags = LOOM_LOW_LOWER_SELECTED_PLAN_ELIDED,
          .rule_set_index = UINT16_MAX,
          .rule_index = UINT16_MAX,
          .data.target_plan = {0},
      };
      IREE_RETURN_IF_ERROR(loom_low_lower_select_fixed_plan(
          context, candidate_plan, &selected_plan,
          LOOM_LOW_LOWER_SOURCE_GRAPH_ALTERNATIVE_FALLBACK));
    }
    return iree_ok_status();
  }

  const loom_low_lower_rule_set_t* failed_rule_set = NULL;
  loom_low_lower_rule_failure_t rule_failure = {0};
  loom_low_source_memory_access_plan_t source_memory_access;
  loom_low_lower_rule_source_memory_state_t source_memory_state;
  loom_low_lower_rule_source_memory_state_initialize(
      source_op, &source_memory_access, &source_memory_state);
  const loom_low_lower_source_memory_record_t* memory_record =
      context->lowering->source_plan.memory.current;
  if (memory_record != NULL) {
    if (memory_record->available) {
      source_memory_state.retained_access = &memory_record->access;
    } else {
      source_memory_state.plan_attempted = true;
      source_memory_state.diagnostic = memory_record->diagnostic;
    }
  }
  bool selected_rule = false;
  if (context->policy->contract.index != NULL) {
    IREE_RETURN_IF_ERROR(loom_low_lower_plan_op_from_contract_index(
        context, source_op, consumption_query, &failed_rule_set, &rule_failure,
        &source_memory_state, &selected_rule));
    if (selected_rule) {
      return iree_ok_status();
    }
    if (failed_rule_set != NULL) {
      return loom_low_lower_rule_set_emit_selection_failure(
          context, failed_rule_set, source_op, rule_failure,
          &source_memory_state);
    }
  }

  IREE_RETURN_IF_ERROR(loom_low_lower_try_select_op_callback(
      context, /*candidate_plan=*/NULL, context->policy->select_op, source_op,
      &selected_callback));
  if (selected_callback) {
    return iree_ok_status();
  }

  if (loom_low_lower_op_is_discardable_hint(context->module, source_op)) {
    loom_low_lower_record_elided_hint_plan(context, source_op);
    return iree_ok_status();
  }
  return loom_low_lower_emit_no_target_contract(context, source_op);
}

static iree_status_t loom_low_lower_source_graph_collect_region(
    loom_low_lower_context_t* context, loom_region_t* source_region,
    loom_low_lower_source_candidate_plan_t* candidate_plan) {
  const uint16_t* block_order =
      source_region == loom_func_like_body(context->source_function)
          ? context->lowering->source_plan.block_order
          : NULL;
  for (uint16_t position = 0; position < source_region->block_count;
       ++position) {
    const uint16_t block_index = block_order ? block_order[position] : position;
    loom_block_t* block = loom_region_block(source_region, block_index);
    loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      loom_low_lower_source_memory_enter_op(context, op);
      const bool is_callable_exit =
          loom_low_lower_source_op_is_callable_exit(context, op);
      uint32_t node_index = UINT32_MAX;
      loom_low_lower_source_candidate_node_t* candidate_node =
          loom_low_lower_source_candidate_plan_lookup_node(candidate_plan, op,
                                                           &node_index);
      if (candidate_node != NULL) {
        IREE_ASSERT_LT(node_index, candidate_plan->node_count);
        candidate_node->memory_record =
            context->lowering->source_plan.memory.current;
        loom_low_lower_planning_scope_begin(context);
        loom_consumption_region_query_t consumption_query;
        loom_low_lower_source_consumption_query_initialize(
            context, source_region, &consumption_query);
        iree_status_t status = loom_low_lower_plan_op(
            context, op, &consumption_query, is_callable_exit, candidate_plan);
        loom_low_lower_planning_scope_end(context);
        IREE_RETURN_IF_ERROR(status);
        if (loom_low_lower_context_diagnostic_limit_reached(context)) {
          return iree_ok_status();
        }
      }
      if (!loom_low_lower_supported_structured_source_op(context, op)) {
        continue;
      }
      loom_region_t* const* regions = loom_op_regions(op);
      for (uint8_t i = 0; i < op->region_count; ++i) {
        if (regions[i] == NULL) {
          continue;
        }
        IREE_RETURN_IF_ERROR(loom_low_lower_source_graph_collect_region(
            context, regions[i], candidate_plan));
        if (loom_low_lower_context_diagnostic_limit_reached(context)) {
          return iree_ok_status();
        }
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_lower_source_graph_collect(
    loom_low_lower_context_t* context, loom_region_t* source_body,
    loom_low_lower_source_candidate_plan_t* candidate_plan) {
  IREE_RETURN_IF_ERROR(loom_low_lower_source_graph_collect_region(
      context, source_body, candidate_plan));
  context->lowering->source_plan.memory.cursor =
      context->lowering->source_plan.memory.first;
  context->lowering->source_plan.memory.current = NULL;
  return iree_ok_status();
}

static loom_low_lower_source_graph_alternative_state_t
loom_low_lower_source_candidate_query(
    void* user_data,
    const loom_low_lower_source_graph_alternative_t* alternative) {
  const loom_low_lower_source_candidate_plan_t* candidate_plan =
      (const loom_low_lower_source_candidate_plan_t*)user_data;
  const loom_low_lower_source_candidate_t* source_candidate =
      (const loom_low_lower_source_candidate_t*)
          loom_low_lower_source_graph_alternative_user_data(alternative);
  if (source_candidate->disabled) {
    return LOOM_LOW_LOWER_SOURCE_GRAPH_ALTERNATIVE_DISABLED;
  }
  if (source_candidate->kind != LOOM_LOW_LOWER_SOURCE_CANDIDATE_RULE) {
    return source_candidate->state;
  }
  const loom_low_lower_source_graph_candidate_t* graph_candidate =
      loom_low_lower_source_graph_alternative_candidate(alternative);
  const uint32_t root_index =
      loom_low_lower_source_graph_candidate_node(graph_candidate, 0);
  const loom_low_lower_source_candidate_node_t* root_node =
      candidate_plan->nodes[root_index];
  const loom_low_lower_source_candidate_graph_t* graph =
      (const loom_low_lower_source_candidate_graph_t*)
          loom_low_lower_source_graph_candidate_user_data(graph_candidate);
  const loom_op_t* singleton_node = root_node->source_op;
  const loom_op_t* const* source_nodes =
      graph != NULL ? graph->source_nodes : &singleton_node;
  const uint8_t source_node_count =
      graph != NULL ? graph->source_node_count : 1;
  const loom_low_lower_rule_t* rule =
      &source_candidate->data.rule.rule_set
           ->rules[source_candidate->data.rule.rule_index];
  if (!loom_low_lower_source_value_storage_rule_preserves_fact_demands(
          candidate_plan->context, source_candidate->data.rule.rule_set,
          root_node->source_op, source_nodes, source_node_count, rule)) {
    return LOOM_LOW_LOWER_SOURCE_GRAPH_ALTERNATIVE_DISABLED;
  }
  return source_candidate->state;
}

static iree_status_t loom_low_lower_source_candidate_resolve(
    loom_low_lower_source_candidate_plan_t* candidate_plan,
    bool* out_complete) {
  loom_low_lower_context_t* context = candidate_plan->context;
  while (true) {
    IREE_RETURN_IF_ERROR(loom_low_lower_source_graph_plan_resolve(
        candidate_plan->graph_plan,
        (loom_low_lower_source_graph_alternative_query_callback_t){
            .fn = loom_low_lower_source_candidate_query,
            .user_data = candidate_plan,
        },
        out_complete));
    bool retry = false;
    for (uint32_t node_index = 0; node_index < candidate_plan->node_count;
         ++node_index) {
      const loom_low_lower_source_graph_alternative_t* alternative =
          loom_low_lower_source_graph_plan_selected_alternative(
              candidate_plan->graph_plan, node_index);
      if (alternative == NULL) {
        continue;
      }
      const loom_low_lower_source_graph_candidate_t* graph_candidate =
          loom_low_lower_source_graph_alternative_candidate(alternative);
      if (loom_low_lower_source_graph_candidate_node(graph_candidate, 0) !=
          node_index) {
        continue;
      }
      loom_low_lower_source_candidate_t* source_candidate =
          (loom_low_lower_source_candidate_t*)
              loom_low_lower_source_graph_alternative_user_data(alternative);
      if (source_candidate->kind != LOOM_LOW_LOWER_SOURCE_CANDIDATE_CALLBACK) {
        continue;
      }
      loom_low_lower_source_candidate_callback_t* callback_state =
          source_candidate->data.callback;
      IREE_ASSERT(!callback_state->evaluated);
      callback_state->evaluated = true;
      loom_low_lower_plan_t target_plan = loom_low_lower_plan_empty();
      context->lowering->source_plan.memory.current =
          candidate_plan->nodes[node_index]->memory_record;
      loom_low_lower_planning_scope_begin(context);
      iree_status_t status = callback_state->callback.fn(
          callback_state->callback.user_data, context,
          candidate_plan->nodes[node_index]->source_op, &target_plan);
      loom_low_lower_planning_scope_end(context);
      context->lowering->source_plan.memory.current = NULL;
      IREE_RETURN_IF_ERROR(status);
      if (loom_low_lower_plan_is_empty(target_plan)) {
        source_candidate->disabled = true;
        retry = true;
        continue;
      }
      callback_state->selected_plan = (loom_low_lower_selected_plan_t){
          .source_op = candidate_plan->nodes[node_index]->source_op,
          .kind = LOOM_LOW_LOWER_SELECTED_PLAN_CALLBACK,
          .rule_set_index = UINT16_MAX,
          .rule_index = UINT16_MAX,
          .data.target_plan = target_plan,
      };
      source_candidate->kind = LOOM_LOW_LOWER_SOURCE_CANDIDATE_PLAN;
      source_candidate->data.selected_plan = &callback_state->selected_plan;
    }
    if (!retry || loom_low_lower_context_diagnostic_limit_reached(context)) {
      return iree_ok_status();
    }
  }
}

static iree_status_t loom_low_lower_source_candidate_emit_rejection(
    loom_low_lower_context_t* context,
    const loom_low_lower_source_candidate_node_t* candidate_node) {
  const loom_low_lower_source_candidate_rejection_t* rejection =
      candidate_node->rejection;
  if (rejection != NULL && (rejection->matrix_rejection.outcome ==
                                LOOM_TARGET_CONTRACT_QUERY_UNSUPPORTED ||
                            rejection->matrix_rejection.outcome ==
                                LOOM_TARGET_CONTRACT_QUERY_INVALID_IR)) {
    return loom_low_lower_emit_contract_query_rejection(
        context, candidate_node->source_op, &rejection->matrix_rejection);
  }
  if (rejection != NULL && rejection->failed_rule_set != NULL) {
    loom_low_source_memory_access_plan_t source_memory_access;
    loom_low_lower_rule_source_memory_state_t source_memory_state;
    loom_low_lower_rule_source_memory_state_initialize(
        candidate_node->source_op, &source_memory_access, &source_memory_state);
    if (candidate_node->memory_record != NULL) {
      if (candidate_node->memory_record->available) {
        source_memory_state.retained_access =
            &candidate_node->memory_record->access;
      } else {
        source_memory_state.plan_attempted = true;
        source_memory_state.diagnostic =
            candidate_node->memory_record->diagnostic;
      }
    }
    return loom_low_lower_rule_set_emit_selection_failure(
        context, rejection->failed_rule_set, candidate_node->source_op,
        rejection->rule_failure, &source_memory_state);
  }
  return loom_low_lower_emit_no_target_contract(context,
                                                candidate_node->source_op);
}

static iree_status_t loom_low_lower_source_candidate_materialize_node(
    loom_low_lower_source_candidate_plan_t* candidate_plan,
    uint32_t node_index) {
  loom_low_lower_context_t* context = candidate_plan->context;
  IREE_ASSERT_LT(node_index, candidate_plan->node_count);
  const loom_low_lower_source_candidate_node_t* candidate_node =
      candidate_plan->nodes[node_index];
  const loom_low_lower_source_graph_alternative_t* alternative =
      loom_low_lower_source_graph_plan_selected_alternative(
          candidate_plan->graph_plan, node_index);
  if (alternative == NULL) {
    return loom_low_lower_source_candidate_emit_rejection(context,
                                                          candidate_node);
  }
  const loom_low_lower_source_graph_candidate_t* graph_candidate =
      loom_low_lower_source_graph_alternative_candidate(alternative);
  const uint32_t root_index =
      loom_low_lower_source_graph_candidate_node(graph_candidate, 0);
  loom_low_lower_source_candidate_t* source_candidate =
      (loom_low_lower_source_candidate_t*)
          loom_low_lower_source_graph_alternative_user_data(alternative);
  const uint8_t emission_source_node_index =
      source_candidate->kind == LOOM_LOW_LOWER_SOURCE_CANDIDATE_RULE
          ? source_candidate->data.rule.emission_source_node_index
          : 0;
  const uint32_t emission_index = loom_low_lower_source_graph_candidate_node(
      graph_candidate, emission_source_node_index);
  if (emission_index != node_index) {
    loom_low_lower_record_selected_plan(
        context, (loom_low_lower_selected_plan_t){
                     .source_op = candidate_node->source_op,
                     .kind = LOOM_LOW_LOWER_SELECTED_PLAN_RULE,
                     .flags = LOOM_LOW_LOWER_SELECTED_PLAN_CLAIMED |
                              LOOM_LOW_LOWER_SELECTED_PLAN_ELIDED,
                     .rule_set_index = UINT16_MAX,
                     .rule_index = UINT16_MAX,
                 });
    return iree_ok_status();
  }

  if (source_candidate->kind ==
      LOOM_LOW_LOWER_SOURCE_CANDIDATE_DESCRIPTOR_MATRIX) {
    const loom_low_lower_source_candidate_descriptor_matrix_t*
        descriptor_matrix = source_candidate->data.descriptor_matrix;
    loom_low_lower_selected_plan_t selected_plan;
    IREE_RETURN_IF_ERROR(loom_low_lower_create_descriptor_matrix_plan(
        context, candidate_plan->nodes[root_index]->source_op,
        descriptor_matrix->matrix_rule, &descriptor_matrix->contract_request,
        &descriptor_matrix->query_result, &selected_plan));
    loom_low_lower_record_selected_plan(context, selected_plan);
    return iree_ok_status();
  }
  if (source_candidate->kind == LOOM_LOW_LOWER_SOURCE_CANDIDATE_PLAN) {
    loom_low_lower_selected_plan_t selected_plan =
        *source_candidate->data.selected_plan;
    selected_plan.flags &= LOOM_LOW_LOWER_SELECTED_PLAN_ELIDED;
    loom_low_lower_record_selected_plan(context, selected_plan);
    return iree_ok_status();
  }

  IREE_ASSERT_EQ(source_candidate->kind, LOOM_LOW_LOWER_SOURCE_CANDIDATE_RULE);
  const loom_low_lower_source_candidate_graph_t* graph =
      (const loom_low_lower_source_candidate_graph_t*)
          loom_low_lower_source_graph_candidate_user_data(graph_candidate);
  const uint8_t source_node_count =
      loom_low_lower_source_graph_candidate_node_count(graph_candidate);
  IREE_ASSERT(source_node_count == 1 || graph != NULL);
  const loom_low_lower_rule_set_t* rule_set =
      source_candidate->data.rule.rule_set;
  const loom_op_t* const* source_nodes = NULL;
  if (graph != NULL) {
    const loom_op_t** retained_source_nodes = NULL;
    IREE_RETURN_IF_ERROR(loom_low_lower_allocate_plan_data(
        context,
        (iree_host_size_t)source_node_count * sizeof(*retained_source_nodes),
        (void**)&retained_source_nodes));
    memcpy(
        retained_source_nodes, graph->source_nodes,
        (iree_host_size_t)source_node_count * sizeof(*retained_source_nodes));
    source_nodes = retained_source_nodes;
  }
  const loom_low_lower_selected_source_memory_t* source_memory = NULL;
  if (source_candidate->data.rule.source_memory != NULL) {
    loom_low_lower_selected_source_memory_t* retained_source_memory = NULL;
    IREE_RETURN_IF_ERROR(loom_low_lower_allocate_plan_data(
        context, sizeof(*retained_source_memory),
        (void**)&retained_source_memory));
    *retained_source_memory = *source_candidate->data.rule.source_memory;
    source_memory = retained_source_memory;
  }
  loom_low_lower_record_selected_plan(
      context,
      (loom_low_lower_selected_plan_t){
          .source_op = candidate_plan->nodes[root_index]->source_op,
          .kind = LOOM_LOW_LOWER_SELECTED_PLAN_RULE,
          .source_node_count = source_node_count,
          .emission_source_node_index = emission_source_node_index,
          .rule_set_index = source_candidate->data.rule.rule_set_index,
          .rule_index = source_candidate->data.rule.rule_index,
          .rule_set = rule_set,
          .rule = &rule_set->rules[source_candidate->data.rule.rule_index],
          .source_memory = source_memory,
          .data.source_nodes = source_nodes,
      });
  return iree_ok_status();
}

static iree_status_t loom_low_lower_plan_region(
    loom_low_lower_context_t* context, loom_region_t* source_region,
    const loom_op_t* block_arg_context_op, bool skip_entry_block_args,
    bool plan_structural,
    loom_low_lower_source_candidate_plan_t* candidate_plan) {
  const uint16_t* block_order =
      source_region == loom_func_like_body(context->source_function)
          ? context->lowering->source_plan.block_order
          : NULL;
  for (uint16_t position = 0; position < source_region->block_count;
       ++position) {
    const uint16_t block_index = block_order ? block_order[position] : position;
    loom_block_t* block = loom_region_block(source_region, block_index);
    if (plan_structural && !(skip_entry_block_args && block_index == 0)) {
      IREE_RETURN_IF_ERROR(loom_low_lower_structural_plan_block(
          context, block_arg_context_op, block));
      if (loom_low_lower_context_diagnostic_limit_reached(context)) {
        return iree_ok_status();
      }
    }
    loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      const bool is_callable_exit =
          loom_low_lower_source_op_is_callable_exit(context, op);
      loom_low_lower_source_memory_enter_op(context, op);
      loom_low_lower_planning_scope_begin(context);
      iree_status_t status = iree_ok_status();
      const loom_trait_flags_t traits =
          loom_op_effective_traits(context->module, op);
      const bool has_plan_node = loom_low_lower_source_op_has_plan_node(
          context, op, traits, is_callable_exit);
      if (candidate_plan != NULL && has_plan_node) {
        uint32_t node_index = UINT32_MAX;
        const loom_low_lower_source_candidate_node_t* candidate_node =
            loom_low_lower_source_candidate_plan_lookup_node(candidate_plan, op,
                                                             &node_index);
        if (candidate_node != NULL) {
          IREE_ASSERT_EQ(candidate_node->memory_record,
                         context->lowering->source_plan.memory.current);
          status = loom_low_lower_source_candidate_materialize_node(
              candidate_plan, node_index);
        } else {
          loom_consumption_region_query_t consumption_query;
          loom_low_lower_source_consumption_query_initialize(
              context, source_region, &consumption_query);
          status = loom_low_lower_plan_op(context, op, &consumption_query,
                                          is_callable_exit,
                                          /*candidate_plan=*/NULL);
        }
      } else if (plan_structural || has_plan_node) {
        loom_consumption_region_query_t consumption_query;
        loom_low_lower_source_consumption_query_initialize(
            context, source_region, &consumption_query);
        status = loom_low_lower_plan_op(context, op, &consumption_query,
                                        is_callable_exit,
                                        /*candidate_plan=*/NULL);
      }
      loom_low_lower_planning_scope_end(context);
      IREE_RETURN_IF_ERROR(status);
      if (loom_low_lower_context_diagnostic_limit_reached(context)) {
        return iree_ok_status();
      }
      if (plan_structural &&
          context->lowering->source_plan.representation_plan != NULL &&
          is_callable_exit) {
        IREE_RETURN_IF_ERROR(
            loom_low_lower_function_boundary_observe_exit(context, op));
      }
      if (!loom_low_lower_supported_structured_source_op(context, op)) {
        continue;
      }
      loom_region_t* const* regions = loom_op_regions(op);
      for (uint8_t i = 0; i < op->region_count; ++i) {
        if (regions[i] == NULL) {
          continue;
        }
        IREE_RETURN_IF_ERROR(loom_low_lower_plan_region(
            context, regions[i], op,
            /*skip_entry_block_args=*/false, plan_structural, candidate_plan));
        if (loom_low_lower_context_diagnostic_limit_reached(context)) {
          return iree_ok_status();
        }
      }
    }
  }
  return iree_ok_status();
}

static bool loom_low_lower_try_reuse_selected_rule_plan(
    loom_low_lower_context_t* context,
    const loom_low_lower_selected_plan_t* previous_plan) {
  IREE_ASSERT_EQ(previous_plan->kind, LOOM_LOW_LOWER_SELECTED_PLAN_RULE);
  IREE_ASSERT(previous_plan->rule != NULL);
  IREE_ASSERT_EQ(previous_plan->source_node_count, 1);
  IREE_ASSERT_EQ(loom_low_lower_rule_source_node_count(previous_plan->rule), 0);
  if (!loom_low_lower_source_value_storage_rule_preserves_fact_demands(
          context, previous_plan->rule_set, previous_plan->source_op,
          previous_plan->data.source_nodes, previous_plan->source_node_count,
          previous_plan->rule)) {
    return false;
  }

  loom_low_lower_selected_plan_t reused_plan = *previous_plan;
  reused_plan.flags = 0;
  loom_low_lower_record_selected_plan(context, reused_plan);
  return true;
}

static iree_status_t loom_low_lower_refine_plan_op(
    loom_low_lower_context_t* context, loom_region_t* source_region,
    const loom_op_t* source_op, bool is_callable_exit,
    const loom_low_lower_selected_plan_t* previous_plan) {
  IREE_ASSERT_EQ(previous_plan->source_op, source_op);
  if (previous_plan->kind != LOOM_LOW_LOWER_SELECTED_PLAN_RULE) {
    loom_low_lower_selected_plan_t reused_plan = *previous_plan;
    reused_plan.flags = 0;
    loom_low_lower_record_selected_plan(context, reused_plan);
    return iree_ok_status();
  }
  if (previous_plan->rule != NULL &&
      loom_low_lower_try_reuse_selected_rule_plan(context, previous_plan)) {
    return iree_ok_status();
  }

  loom_consumption_region_query_t consumption_query;
  loom_low_lower_source_consumption_query_initialize(context, source_region,
                                                     &consumption_query);
  return loom_low_lower_plan_op(context, source_op, &consumption_query,
                                is_callable_exit, /*candidate_plan=*/NULL);
}

static iree_status_t loom_low_lower_refine_plan_region(
    loom_low_lower_context_t* context, loom_region_t* source_region,
    iree_host_size_t previous_plan_count,
    iree_host_size_t* inout_previous_plan_index) {
  loom_low_lower_source_plan_t* source_plan = &context->lowering->source_plan;
  const uint16_t* block_order =
      source_region == loom_func_like_body(context->source_function)
          ? source_plan->block_order
          : NULL;
  for (uint16_t position = 0; position < source_region->block_count;
       ++position) {
    const uint16_t block_index = block_order ? block_order[position] : position;
    loom_block_t* block = loom_region_block(source_region, block_index);
    loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      const bool is_callable_exit =
          loom_low_lower_source_op_is_callable_exit(context, op);
      loom_low_lower_source_memory_enter_op(context, op);

      if (*inout_previous_plan_index < previous_plan_count &&
          source_plan->selected_plans[*inout_previous_plan_index].source_op ==
              op) {
        IREE_ASSERT_EQ(source_plan->selected_plan_count,
                       *inout_previous_plan_index);
        const loom_low_lower_selected_plan_t previous_plan =
            source_plan->selected_plans[*inout_previous_plan_index];
        ++(*inout_previous_plan_index);
        loom_low_lower_planning_scope_begin(context);
        iree_status_t status = loom_low_lower_refine_plan_op(
            context, source_region, op, is_callable_exit, &previous_plan);
        loom_low_lower_planning_scope_end(context);
        IREE_RETURN_IF_ERROR(status);
        if (loom_low_lower_context_diagnostic_limit_reached(context)) {
          return iree_ok_status();
        }
      }

      if (!loom_low_lower_supported_structured_source_op(context, op)) {
        continue;
      }
      loom_region_t* const* regions = loom_op_regions(op);
      for (uint8_t i = 0; i < op->region_count; ++i) {
        if (regions[i] == NULL) {
          continue;
        }
        IREE_RETURN_IF_ERROR(loom_low_lower_refine_plan_region(
            context, regions[i], previous_plan_count,
            inout_previous_plan_index));
        if (loom_low_lower_context_diagnostic_limit_reached(context)) {
          return iree_ok_status();
        }
      }
    }
  }
  return iree_ok_status();
}

static bool loom_low_lower_rule_set_requires_primary_descriptor_query(
    const loom_low_lower_rule_set_t* rule_set, uint64_t available_features) {
  if (iree_any_bit_set(
          rule_set->flags,
          LOOM_LOW_LOWER_RULE_SET_FLAG_DYNAMIC_PRIMARY_DESCRIPTOR_FEATURES)) {
    return true;
  }
  for (uint16_t i = 0; i < rule_set->primary_descriptor_feature_mask_count;
       ++i) {
    if ((rule_set->primary_descriptor_feature_masks[i] & ~available_features) !=
        0) {
      return true;
    }
  }
  return false;
}

static iree_status_t
loom_low_lower_source_plan_prepare_primary_descriptor_queries(
    loom_low_lower_context_t* context) {
  const loom_low_lower_rule_set_list_t rule_sets =
      context->policy->contract.rule_sets;
  const uint64_t available_features =
      loom_low_lower_context_bundle(context)->config->contract_feature_bits;
  uint64_t* query_rule_set_bits = NULL;
  for (uint16_t i = 0; i < rule_sets.count; ++i) {
    if (!loom_low_lower_rule_set_requires_primary_descriptor_query(
            rule_sets.values[i], available_features)) {
      continue;
    }
    if (query_rule_set_bits == NULL) {
      const iree_host_size_t word_count = (rule_sets.count + 63u) / 64u;
      IREE_RETURN_IF_ERROR(loom_low_lower_allocate_function_array(
          context, word_count, sizeof(*query_rule_set_bits),
          (void**)&query_rule_set_bits));
      memset(query_rule_set_bits, 0, word_count * sizeof(*query_rule_set_bits));
      context->lowering->source_plan.primary_descriptor_query_rule_set_bits =
          query_rule_set_bits;
    }
    query_rule_set_bits[i / 64u] |= UINT64_C(1) << (i % 64u);
  }
  return iree_ok_status();
}

iree_status_t loom_low_lower_source_plan_build(
    loom_low_lower_context_t* context, loom_region_t* source_body) {
  loom_low_lower_source_plan_t* source_plan = &context->lowering->source_plan;
  IREE_ASSERT(context->lowering->source_callable_exit_kind !=
              LOOM_OP_KIND_UNKNOWN);
  *source_plan = (loom_low_lower_source_plan_t){0};
  IREE_RETURN_IF_ERROR(
      loom_low_lower_source_plan_prepare_primary_descriptor_queries(context));
  if (source_body->block_count > 1) {
    const loom_value_fact_cfg_region_t* cfg =
        loom_low_lower_context_cfg(context);
    uint16_t* block_order = NULL;
    IREE_RETURN_IF_ERROR(loom_low_lower_allocate_function_array(
        context, source_body->block_count, sizeof(*block_order),
        (void**)&block_order));
    iree_host_size_t count = cfg->dominance.preorder.count;
    memcpy(block_order, cfg->dominance.preorder.values,
           count * sizeof(*block_order));
    if (count < source_body->block_count) {
      for (uint16_t i = 0; i < source_body->block_count; ++i) {
        if (!cfg->graph.blocks[i].reachable) {
          block_order[count++] = i;
        }
      }
    }
    source_plan->block_order = block_order;
  }
  const loom_value_ordinal_t value_count =
      context->lowering->value_domain.value_count;
  if (value_count != 0) {
    IREE_RETURN_IF_ERROR(loom_low_lower_allocate_function_array(
        context, value_count, sizeof(*source_plan->value_flags),
        (void**)&source_plan->value_flags));
    memset(source_plan->value_flags, 0,
           value_count * sizeof(*source_plan->value_flags));
  }

  uint16_t argument_count = 0;
  const loom_value_id_t* arguments =
      loom_func_like_arg_ids(context->source_function, &argument_count);
  for (uint16_t i = 0; i < argument_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_lower_plan_value_type(
        context, arguments[i],
        context->lowering->boundary.argument_map[i].abi_type));
  }

  iree_arena_initialize(context->module->arena.block_pool,
                        &context->planning_arena);
  loom_low_lower_source_candidate_plan_t candidate_plan;
  loom_low_lower_source_candidate_plan_t* active_candidate_plan = NULL;
  bool candidate_plan_initialized = false;
  iree_status_t status = iree_ok_status();
  if (loom_low_lower_source_plan_has_related_rules(context)) {
    candidate_plan_initialized = true;
    status = loom_low_lower_source_candidate_plan_initialize(context,
                                                             &candidate_plan);
    if (iree_status_is_ok(status)) {
      active_candidate_plan = &candidate_plan;
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_low_lower_prepare_plan(context, source_body,
                                         active_candidate_plan);
  }
  if (iree_status_is_ok(status) && active_candidate_plan != NULL) {
    status =
        loom_low_lower_source_graph_discover(context, active_candidate_plan);
  }
  if (iree_status_is_ok(status) && active_candidate_plan != NULL) {
    status = loom_low_lower_source_candidate_plan_finalize_nodes(
        active_candidate_plan);
    if (iree_status_is_ok(status) && active_candidate_plan->node_count == 0) {
      active_candidate_plan = NULL;
    }
  }
  if (iree_status_is_ok(status) && active_candidate_plan != NULL) {
    status = loom_low_lower_source_graph_collect(context, source_body,
                                                 active_candidate_plan);
  }
  if (iree_status_is_ok(status) && active_candidate_plan != NULL &&
      !loom_low_lower_context_diagnostic_limit_reached(context)) {
    bool complete = false;
    status = loom_low_lower_source_candidate_resolve(active_candidate_plan,
                                                     &complete);
    (void)complete;
  }
  if (iree_status_is_ok(status) &&
      !loom_low_lower_context_diagnostic_limit_reached(context)) {
    status = loom_low_lower_plan_region(
        context, source_body, context->source_function.op,
        /*skip_entry_block_args=*/true, /*plan_structural=*/true,
        active_candidate_plan);
  }
  if (iree_status_is_ok(status) && context->result->error_count == 0) {
    loom_low_lower_source_value_storage_analyze(context);
    loom_value_ordinal_t previous_fact_storage_demand_count = 0;
    while (source_plan->fact_storage_demand_count != 0 &&
           !loom_low_lower_source_value_storage_plans_preserve_fact_demands(
               context)) {
      IREE_ASSERT_GT(
          source_plan->fact_storage_demand_count,
          previous_fact_storage_demand_count,
          "source planning made no progress while preserving fact references");
      previous_fact_storage_demand_count =
          source_plan->fact_storage_demand_count;
      const iree_host_size_t previous_plan_count =
          source_plan->selected_plan_count;
      loom_low_lower_source_value_storage_prepare_refinement(context);
      iree_host_size_t previous_plan_index = 0;
      if (active_candidate_plan != NULL) {
        bool complete = false;
        status = loom_low_lower_source_candidate_resolve(active_candidate_plan,
                                                         &complete);
        if (iree_status_is_ok(status)) {
          status = loom_low_lower_plan_region(
              context, source_body, context->source_function.op,
              /*skip_entry_block_args=*/true, /*plan_structural=*/false,
              active_candidate_plan);
        }
        previous_plan_index = previous_plan_count;
        (void)complete;
      } else {
        status = loom_low_lower_refine_plan_region(
            context, source_body, previous_plan_count, &previous_plan_index);
      }
      if (!iree_status_is_ok(status) || context->result->error_count != 0) {
        break;
      }
      IREE_ASSERT_EQ(previous_plan_index, previous_plan_count);
      IREE_ASSERT_EQ(source_plan->selected_plan_count, previous_plan_count);
      loom_low_lower_source_value_storage_analyze(context);
    }
  }
  if (candidate_plan_initialized) {
    iree_arena_deinitialize(&candidate_plan.arena);
  }
  if (iree_status_is_ok(status) && context->result->error_count == 0) {
    status = loom_low_lower_finalize_selected_plans(context);
  }
  if (iree_status_is_ok(status) && context->result->error_count == 0) {
    status = loom_low_lower_function_boundary_plan(context,
                                                   &context->planning_arena);
  }
  if (iree_status_is_ok(status) && context->result->error_count == 0 &&
      context->policy->entry_setup.plan != NULL) {
    loom_low_lower_planning_scope_begin(context);
    status = context->policy->entry_setup.plan(
        context->policy->entry_setup.user_data, context);
    loom_low_lower_planning_scope_end(context);
  }
  iree_arena_deinitialize(&context->planning_arena);
  return status;
}

// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/cfg/cfg_converge.h"

#include "loom/ir/local_value_domain.h"
#include "loom/ir/value_refs.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/cfg/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/special_values.h"
#include "loom/pass/value_facts.h"
#include "loom/rewrite/rewriter.h"
#include "loom/target/function_version.h"
#include "loom/util/fact_cfg.h"

#define LOOM_CFG_CONVERGE_STATISTICS(V, statistics_type)                  \
  V(statistics_type, decisions_factored, "decisions-factored",            \
    "Number of overlapping decisions factored through a predicate join.") \
  V(statistics_type, values_carried, "values-carried",                    \
    "Number of scalar live-ins carried across predicate joins.")

LOOM_PASS_STATISTICS_DEFINE(loom_cfg_converge_statistics,
                            loom_cfg_converge_statistics_t,
                            LOOM_CFG_CONVERGE_STATISTICS)

static const loom_pass_info_t loom_cfg_converge_pass_info_storage = {
    .name = IREE_SVL("cfg-converge"),
    .description = IREE_SVL("Converge overlapping acyclic CFG decisions."),
    .kind = LOOM_PASS_FUNCTION,
    .statistic_layout = &loom_cfg_converge_statistics_layout,
};

const loom_pass_info_t* loom_cfg_converge_pass_info(void) {
  return &loom_cfg_converge_pass_info_storage;
}

typedef struct loom_cfg_converge_capture_t {
  // Original scalar live-in whose definition ceases to dominate its users.
  loom_value_id_t value;
  // Join argument replacing only affected ordinary uses.
  loom_value_id_t argument;
  // Definition's block index in the immutable pre-edit snapshot.
  uint16_t definition;
  // Successor dominance subtrees containing affected users, one bit per arm.
  uint8_t arms;
} loom_cfg_converge_capture_t;

typedef struct loom_cfg_converge_replacement_t {
  // Ordinary use recorded before block insertion changes dense indices.
  loom_use_t use;
  // Index of the capture supplying this operand's join argument.
  iree_host_size_t capture;
} loom_cfg_converge_replacement_t;

typedef struct loom_cfg_converge_remat_t {
  // A non-owning view reconstructed in each successor where it remains live.
  loom_value_id_t value;
  uint8_t arms;
} loom_cfg_converge_remat_t;

typedef struct loom_cfg_converge_remat_replacement_t {
  loom_use_t use;
  iree_host_size_t remat;
  uint8_t arm;
} loom_cfg_converge_remat_replacement_t;

typedef struct loom_cfg_converge_plan_t {
  // Module containing the selected decision.
  loom_module_t* module;
  // Immutable graph, dominance and postdominance used during preflight.
  const loom_value_fact_cfg_region_t* facts;
  // Scratch storage released after completing or declining this candidate.
  iree_arena_allocator_t* arena;
  // Function-local value ordinals for deduplicating captures.
  loom_local_value_domain_t domain;
  // Conditional terminator whose decision moves to the join.
  loom_op_t* branch;
  // Original decision block index.
  uint16_t source;
  // Original true and false successor indices.
  uint16_t destinations[2];
  // Immediate dominator of the new join, derived from shared successors.
  uint16_t parent;
  // External incoming edges redirected through constant predicate payloads.
  loom_cfg_edge_index_t* incoming;
  // Number of populated incoming edges.
  iree_host_size_t incoming_count;
  // Captures in join argument order, owned by arena.
  loom_cfg_converge_capture_t* captures;
  // Number of populated captures.
  iree_host_size_t capture_count;
  // Capture index per local value ordinal, or UINT32_MAX when absent.
  uint32_t* capture_indices;
  // Exact ordinary operands that need substitution after inserting the join.
  loom_cfg_converge_replacement_t* replacements;
  // Number of populated replacement operands.
  iree_host_size_t replacement_count;
  // Reserved replacement operand capacity.
  iree_host_size_t replacement_capacity;
  loom_cfg_converge_remat_t* remats;
  iree_host_size_t remat_count;
  uint32_t* remat_indices;
  loom_cfg_converge_remat_replacement_t* remat_replacements;
  iree_host_size_t remat_replacement_count;
  iree_host_size_t remat_replacement_capacity;
  // Successor ordinal whose subtree is being inspected by the reference walk.
  uint8_t arm;
  // False when a capture cannot travel through an ordinary scalar argument.
  bool representable;
} loom_cfg_converge_plan_t;

static bool loom_cfg_converge_is_forwarding(const loom_block_t* block,
                                            const loom_block_t* continuation) {
  return block->op_count == 1 && loom_cfg_br_isa(block->last_op) &&
         loom_cfg_br_dest(block->last_op) == continuation;
}

// A decision with an externally shared arm moves behind that arm's incoming
// alternatives. The component order proves the redirected edges cannot create
// a cycle. Transparent forwarding arms already converge without moving work.
static iree_status_t loom_cfg_converge_select(loom_cfg_converge_plan_t* plan,
                                              bool* out_selected) {
  *out_selected = false;
  const loom_cfg_graph_t* graph = &plan->facts->graph;
  const loom_cfg_dominance_t* dominance = &plan->facts->dominance;
  const loom_cfg_postdominance_t* postdominance =
      &plan->facts->control_structure.postdominance;
  uint32_t continuation =
      postdominance->nodes[plan->source].immediate_postdominator;
  if (continuation == postdominance->exit_node ||
      plan->destinations[0] == plan->destinations[1]) {
    return iree_ok_status();
  }
  iree_host_size_t capacity =
      graph->blocks[plan->destinations[0]].predecessor_count +
      graph->blocks[plan->destinations[1]].predecessor_count;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      plan->arena, capacity, sizeof(*plan->incoming), (void**)&plan->incoming));
  plan->parent = plan->source;
  for (uint8_t arm = 0; arm < 2; ++arm) {
    uint16_t destination = plan->destinations[arm];
    if (destination == continuation ||
        loom_cfg_converge_is_forwarding(graph->blocks[destination].block,
                                        graph->blocks[continuation].block)) {
      continue;
    }
    loom_cfg_edge_index_span_t incoming =
        loom_cfg_graph_predecessor_edges(graph, destination);
    for (iree_host_size_t i = 0; i < incoming.count; ++i) {
      loom_cfg_edge_index_t index = incoming.values[i];
      const loom_cfg_edge_info_t* edge = &graph->edges[index];
      const loom_cfg_block_info_t* source =
          &graph->blocks[edge->source_block_index];
      if (!source->reachable ||
          loom_cfg_dominance_block_dominates(dominance, plan->source,
                                             edge->source_block_index)) {
        continue;
      }
      if (source->component <= graph->blocks[plan->destinations[0]].component ||
          source->component <= graph->blocks[plan->destinations[1]].component) {
        return iree_ok_status();
      }
      plan->incoming[plan->incoming_count++] = index;
      // The branch directly reaches each successor, so each successor's
      // immediate dominator is an ancestor of the branch. The outermost of
      // these ancestors is the incoming edges' common dominator; no ancestry
      // search or new dominance solve is needed to locate the join's parent.
      uint16_t parent = dominance->immediate_dominators[destination];
      if ((uint16_t)dominance->intervals[parent] <
          (uint16_t)dominance->intervals[plan->parent]) {
        plan->parent = parent;
      }
    }
  }
  *out_selected = plan->incoming_count != 0 &&
                  graph->block_count + plan->incoming_count + 1 <= UINT16_MAX;
  return iree_ok_status();
}

// A view may be rebuilt in its consuming successor instead of being carried
// through a join with an invented inactive view. Restrict the recipe to pure
// address operations whose leaves are available before the selected decision.
static bool loom_cfg_converge_can_rematerialize(
    const loom_cfg_converge_plan_t* plan, loom_value_id_t value_id,
    uint8_t depth) {
  if (depth == 16) {
    return false;
  }
  const loom_value_t* value = loom_module_value(plan->module, value_id);
  const loom_op_t* defining_op =
      loom_value_is_block_arg(value) ? NULL : loom_value_def_op(value);
  const loom_block_t* definition = loom_value_is_block_arg(value)
                                       ? loom_value_def_block(value)
                                   : defining_op ? defining_op->parent_block
                                                 : NULL;
  if (!definition || definition->parent_region != plan->facts->graph.region) {
    return false;
  }
  if (loom_cfg_dominance_block_dominates(
          &plan->facts->dominance, definition->region_index, plan->parent)) {
    return true;
  }
  if (loom_value_is_block_arg(value) ||
      loom_module_value_has_type_uses(plan->module, value_id) ||
      loom_value_has_attribute_uses(value)) {
    return false;
  }
  loom_type_use_iterator_t dependencies;
  loom_module_value_type_dependencies(plan->module, value_id, &dependencies);
  if (loom_type_users_next(&dependencies) != LOOM_VALUE_ID_INVALID) {
    return false;
  }
  const loom_op_t* op = defining_op;
  if (loom_index_constant_isa(op)) {
    return true;
  }
  if (loom_index_add_isa(op)) {
    return loom_cfg_converge_can_rematerialize(plan, loom_index_add_lhs(op),
                                               depth + 1) &&
           loom_cfg_converge_can_rematerialize(plan, loom_index_add_rhs(op),
                                               depth + 1);
  }
  if (loom_buffer_view_isa(op)) {
    return loom_cfg_converge_can_rematerialize(
               plan, loom_buffer_view_buffer(op), depth + 1) &&
           loom_cfg_converge_can_rematerialize(
               plan, loom_buffer_view_byte_offset(op), depth + 1);
  }
  return false;
}

static iree_status_t loom_cfg_converge_capture(loom_value_id_t value_id,
                                               void* user_data) {
  loom_cfg_converge_plan_t* plan = user_data;
  if (!plan->representable) {
    return iree_ok_status();
  }
  const loom_value_t* value = loom_module_value(plan->module, value_id);
  const loom_block_t* definition = NULL;
  if (loom_value_is_block_arg(value)) {
    definition = loom_value_def_block(value);
  } else {
    const loom_op_t* op = loom_value_def_op(value);
    definition = op ? op->parent_block : NULL;
  }
  if (!definition || definition->parent_region != plan->facts->graph.region) {
    return iree_ok_status();
  }
  const loom_cfg_dominance_t* dominance = &plan->facts->dominance;
  uint16_t index = definition->region_index;
  if (loom_cfg_dominance_block_dominates(
          dominance, plan->destinations[plan->arm], index) ||
      loom_cfg_dominance_block_dominates(dominance, index, plan->parent)) {
    return iree_ok_status();
  }
  if (loom_type_is_view(value->type) &&
      loom_cfg_converge_can_rematerialize(plan, value_id, 0)) {
    loom_value_ordinal_t ordinal =
        loom_local_value_domain_ordinal(&plan->domain, value_id);
    uint32_t remat = plan->remat_indices[ordinal];
    if (remat == UINT32_MAX) {
      remat = (uint32_t)plan->remat_count++;
      plan->remat_indices[ordinal] = remat;
      plan->remats[remat].value = value_id;
    }
    plan->remats[remat].arms |= 1u << plan->arm;
    return iree_ok_status();
  }
  // Moving only the branch would break this reference's dominance. Scalar
  // carriers have a defined inactive value; opaque/linear carriers and type
  // or attribute identities require a different ownership-preserving rewrite.
  if (!loom_type_is_scalar(value->type) ||
      loom_module_value_has_type_uses(plan->module, value_id) ||
      loom_value_has_attribute_uses(value) ||
      plan->capture_count == UINT16_MAX - 1) {
    plan->representable = false;
    return iree_ok_status();
  }
  loom_value_ordinal_t ordinal =
      loom_local_value_domain_ordinal(&plan->domain, value_id);
  uint32_t capture = plan->capture_indices[ordinal];
  if (capture == UINT32_MAX) {
    capture = (uint32_t)plan->capture_count++;
    plan->capture_indices[ordinal] = capture;
    plan->captures[capture] = (loom_cfg_converge_capture_t){
        .value = value_id,
        .definition = index,
    };
  }
  plan->captures[capture].arms |= 1u << plan->arm;
  return iree_ok_status();
}

static uint8_t loom_cfg_converge_use_arm(const loom_cfg_converge_plan_t* plan,
                                         uint8_t arms, loom_use_t use) {
  const loom_op_t* op = loom_use_user_op(use);
  // Nested-region operands inherit the containing CFG block's availability.
  while (op && op->parent_block &&
         op->parent_block->parent_region != plan->facts->graph.region) {
    op = op->parent_op;
  }
  if (!op || !op->parent_block) {
    return 2;
  }
  for (uint8_t arm = 0; arm < 2; ++arm) {
    if ((arms & (1u << arm)) &&
        loom_cfg_dominance_block_dominates(&plan->facts->dominance,
                                           plan->destinations[arm],
                                           op->parent_block->region_index)) {
      return arm;
    }
  }
  return 2;
}

static iree_status_t loom_cfg_converge_preflight(
    loom_cfg_converge_plan_t* plan) {
  IREE_RETURN_IF_ERROR(loom_local_value_domain_acquire_for_region_tree(
      plan->module, plan->facts->graph.region, plan->arena, &plan->domain));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      plan->arena, plan->domain.value_count, sizeof(*plan->captures),
      (void**)&plan->captures));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      plan->arena, plan->domain.value_count, sizeof(*plan->capture_indices),
      (void**)&plan->capture_indices));
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(plan->arena, plan->domain.value_count,
                                sizeof(*plan->remats), (void**)&plan->remats));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      plan->arena, plan->domain.value_count, sizeof(*plan->remat_indices),
      (void**)&plan->remat_indices));
  memset(plan->remats, 0, plan->domain.value_count * sizeof(*plan->remats));
  memset(plan->capture_indices, 0xFF,
         plan->domain.value_count * sizeof(*plan->capture_indices));
  memset(plan->remat_indices, 0xFF,
         plan->domain.value_count * sizeof(*plan->remat_indices));
  plan->representable = true;
  const loom_cfg_dominance_t* dominance = &plan->facts->dominance;
  for (plan->arm = 0; plan->arm < 2 && plan->representable; ++plan->arm) {
    uint32_t interval = dominance->intervals[plan->destinations[plan->arm]];
    for (uint32_t i = (uint16_t)interval;
         i <= (interval >> 16) && plan->representable; ++i) {
      const loom_block_t* block =
          plan->facts->graph.blocks[dominance->preorder.values[i]].block;
      for (uint16_t j = 0; j < block->arg_count; ++j) {
        loom_type_use_iterator_t dependencies;
        loom_module_value_type_dependencies(
            plan->module, loom_block_arg_id(block, j), &dependencies);
        for (loom_value_id_t dependency = loom_type_users_next(&dependencies);
             dependency != LOOM_VALUE_ID_INVALID;
             dependency = loom_type_users_next(&dependencies)) {
          IREE_RETURN_IF_ERROR(loom_cfg_converge_capture(dependency, plan));
        }
      }
      const loom_op_t* op = NULL;
      loom_block_for_each_op(block, op) {
        IREE_RETURN_IF_ERROR(loom_op_walk_subtree_value_refs(
            plan->module, op, loom_cfg_converge_capture, plan));
      }
    }
  }
  for (iree_host_size_t i = 0; i < plan->capture_count && plan->representable;
       ++i) {
    const loom_cfg_converge_capture_t* capture = &plan->captures[i];
    const loom_value_t* value = loom_module_value(plan->module, capture->value);
    const loom_use_t* uses = loom_value_uses(value);
    for (uint32_t j = 0; j < value->use_count; ++j) {
      if (loom_cfg_converge_use_arm(plan, capture->arms, uses[j]) == 2) {
        continue;
      }
      IREE_RETURN_IF_ERROR(iree_arena_grow_array(
          plan->arena, plan->replacement_count, plan->replacement_count + 1,
          sizeof(*plan->replacements), &plan->replacement_capacity,
          (void**)&plan->replacements));
      plan->replacements[plan->replacement_count++] =
          (loom_cfg_converge_replacement_t){.use = uses[j], .capture = i};
    }
  }
  for (iree_host_size_t i = 0; i < plan->remat_count; ++i) {
    const loom_cfg_converge_remat_t* remat = &plan->remats[i];
    const loom_value_t* value = loom_module_value(plan->module, remat->value);
    const loom_use_t* uses = loom_value_uses(value);
    for (uint32_t j = 0; j < value->use_count; ++j) {
      uint8_t arm = loom_cfg_converge_use_arm(plan, remat->arms, uses[j]);
      if (arm == 2) {
        continue;
      }
      IREE_RETURN_IF_ERROR(iree_arena_grow_array(
          plan->arena, plan->remat_replacement_count,
          plan->remat_replacement_count + 1, sizeof(*plan->remat_replacements),
          &plan->remat_replacement_capacity,
          (void**)&plan->remat_replacements));
      plan->remat_replacements[plan->remat_replacement_count++] =
          (loom_cfg_converge_remat_replacement_t){
              .use = uses[j], .remat = i, .arm = arm};
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_cfg_converge_build_payload(
    loom_cfg_converge_plan_t* plan, loom_builder_t* builder, uint16_t source,
    loom_value_id_t* payload) {
  for (iree_host_size_t i = 0; i < plan->capture_count; ++i) {
    const loom_cfg_converge_capture_t* capture = &plan->captures[i];
    loom_value_id_t value = capture->value;
    if (!loom_cfg_dominance_block_dominates(&plan->facts->dominance,
                                            capture->definition, source)) {
      loom_type_t type = loom_module_value_type(plan->module, value);
      loom_scalar_type_t element = loom_type_element_type(type);
      if (element != LOOM_SCALAR_TYPE_INDEX &&
          element != LOOM_SCALAR_TYPE_OFFSET) {
        loom_op_t* constant = NULL;
        IREE_RETURN_IF_ERROR(loom_scalar_constant_build(
            builder,
            loom_scalar_type_is_float(element) ? loom_attr_f64(0)
                                               : loom_attr_i64(0),
            type, plan->branch->location, &constant));
        value = loom_scalar_constant_result(constant);
      } else {
        IREE_RETURN_IF_ERROR(
            loom_constant_build(builder, loom_value_facts_exact_i64(0), type,
                                plan->branch->location, &value));
      }
    }
    payload[i + 1] = value;
  }
  return iree_ok_status();
}

static iree_status_t loom_cfg_converge_rematerialize(
    loom_cfg_converge_plan_t* plan, loom_builder_t* builder,
    loom_value_id_t original, loom_value_id_t* out_value) {
  const loom_value_t* value = loom_module_value(plan->module, original);
  const loom_block_t* definition = loom_value_is_block_arg(value)
                                       ? loom_value_def_block(value)
                                       : loom_value_def_op(value)->parent_block;
  if (loom_cfg_dominance_block_dominates(
          &plan->facts->dominance, definition->region_index, plan->parent)) {
    *out_value = original;
    return iree_ok_status();
  }
  const loom_op_t* source = loom_value_def_op(value);
  loom_op_t* clone = NULL;
  if (loom_index_constant_isa(source)) {
    IREE_RETURN_IF_ERROR(
        loom_index_constant_build(builder, loom_index_constant_value(source),
                                  value->type, source->location, &clone));
    *out_value = loom_index_constant_result(clone);
  } else if (loom_index_add_isa(source)) {
    loom_value_id_t lhs = LOOM_VALUE_ID_INVALID;
    loom_value_id_t rhs = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_cfg_converge_rematerialize(
        plan, builder, loom_index_add_lhs(source), &lhs));
    IREE_RETURN_IF_ERROR(loom_cfg_converge_rematerialize(
        plan, builder, loom_index_add_rhs(source), &rhs));
    IREE_RETURN_IF_ERROR(loom_index_add_build(builder, lhs, rhs, value->type,
                                              source->location, &clone));
    *out_value = loom_index_add_result(clone);
  } else if (loom_buffer_view_isa(source)) {
    loom_value_id_t buffer = LOOM_VALUE_ID_INVALID;
    loom_value_id_t byte_offset = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_cfg_converge_rematerialize(
        plan, builder, loom_buffer_view_buffer(source), &buffer));
    IREE_RETURN_IF_ERROR(loom_cfg_converge_rematerialize(
        plan, builder, loom_buffer_view_byte_offset(source), &byte_offset));
    IREE_RETURN_IF_ERROR(loom_buffer_view_build(
        builder, buffer, byte_offset, value->type, source->location, &clone));
    *out_value = loom_buffer_view_result(clone);
  } else {
    IREE_ASSERT_UNREACHABLE("preflight planned the rematerialization recipe");
    IREE_BUILTIN_UNREACHABLE();
  }
  return iree_ok_status();
}

static iree_status_t loom_cfg_converge_apply(loom_cfg_converge_plan_t* plan,
                                             loom_rewriter_t* rewriter) {
  loom_builder_t* builder = &rewriter->builder;
  loom_region_t* body = plan->branch->parent_block->parent_region;
  loom_block_t* destinations[] = {loom_cfg_cond_br_true_dest(plan->branch),
                                  loom_cfg_cond_br_false_dest(plan->branch)};
  loom_value_id_t selector = loom_cfg_cond_br_condition(plan->branch);
  loom_location_id_t location = plan->branch->location;
  for (iree_host_size_t i = 0; i < plan->remat_count; ++i) {
    for (uint8_t arm = 0; arm < 2; ++arm) {
      if (!(plan->remats[i].arms & (1u << arm))) {
        continue;
      }
      loom_builder_set_before(builder, destinations[arm]->first_op);
      loom_value_id_t rematerialized = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_cfg_converge_rematerialize(
          plan, builder, plan->remats[i].value, &rematerialized));
      for (iree_host_size_t j = 0; j < plan->remat_replacement_count; ++j) {
        const loom_cfg_converge_remat_replacement_t* replacement =
            &plan->remat_replacements[j];
        if (replacement->remat != i || replacement->arm != arm) {
          continue;
        }
        IREE_RETURN_IF_ERROR(loom_rewriter_set_operand(
            rewriter, loom_use_user_op(replacement->use),
            loom_use_operand_index(replacement->use), rematerialized));
      }
    }
  }
  builder->ip.parent_op = plan->branch->parent_op;
  loom_block_t* join = NULL;
  IREE_RETURN_IF_ERROR(loom_region_insert_block(
      plan->module, body,
      iree_min(plan->destinations[0], plan->destinations[1]), &join));
  loom_value_id_t choice = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_builder_define_block_arg(
      builder, join, loom_type_scalar(LOOM_SCALAR_TYPE_I1), &choice));
  for (iree_host_size_t i = 0; i < plan->capture_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_builder_define_block_arg(
        builder, join,
        loom_module_value_type(plan->module, plan->captures[i].value),
        &plan->captures[i].argument));
  }
  loom_builder_set_block(builder, join);
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_cfg_cond_br_build(builder, choice, destinations[0],
                                              destinations[1], location, &op));
  uint16_t payload_count = (uint16_t)(plan->capture_count + 1);
  loom_value_id_t* payload = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      plan->arena, payload_count, sizeof(*payload), (void**)&payload));
  loom_builder_set_before(builder, plan->branch);
  payload[0] = selector;
  IREE_RETURN_IF_ERROR(
      loom_cfg_converge_build_payload(plan, builder, plan->source, payload));
  IREE_RETURN_IF_ERROR(
      loom_cfg_br_build(builder, join, payload, payload_count, location, &op));
  for (iree_host_size_t i = 0; i < plan->incoming_count; ++i) {
    const loom_cfg_edge_info_t* edge =
        &plan->facts->graph.edges[plan->incoming[i]];
    loom_block_t* forwarding = NULL;
    IREE_RETURN_IF_ERROR(loom_region_insert_block(
        plan->module, body, join->region_index, &forwarding));
    loom_builder_set_block(builder, forwarding);
    IREE_RETURN_IF_ERROR(loom_scalar_constant_build(
        builder,
        loom_attr_i64(edge->target_block_index == plan->destinations[0]),
        loom_type_scalar(LOOM_SCALAR_TYPE_I1), location, &op));
    payload[0] = loom_scalar_constant_result(op);
    IREE_RETURN_IF_ERROR(loom_cfg_converge_build_payload(
        plan, builder, edge->source_block_index, payload));
    IREE_RETURN_IF_ERROR(loom_cfg_br_build(builder, join, payload,
                                           payload_count, location, &op));
    loom_op_successors(edge->terminator)[edge->successor_index] = forwarding;
  }
  for (iree_host_size_t i = 0; i < plan->replacement_count; ++i) {
    const loom_cfg_converge_replacement_t* replacement = &plan->replacements[i];
    IREE_RETURN_IF_ERROR(loom_rewriter_set_operand(
        rewriter, loom_use_user_op(replacement->use),
        loom_use_operand_index(replacement->use),
        plan->captures[replacement->capture].argument));
  }
  IREE_RETURN_IF_ERROR(loom_rewriter_erase(rewriter, plan->branch));
  return loom_rewriter_refresh_cfg_facts(rewriter, body);
}

static iree_status_t loom_cfg_converge_once(loom_pass_t* pass,
                                            loom_module_t* module,
                                            loom_region_t* body,
                                            loom_rewriter_t* rewriter,
                                            iree_arena_allocator_t* arena,
                                            bool* out_changed) {
  *out_changed = false;
  const loom_value_fact_cfg_region_t* facts = NULL;
  IREE_RETURN_IF_ERROR(loom_value_fact_table_get_or_build_cfg_region(
      rewriter->fact_table, module, body, &facts));
  const loom_cfg_graph_t* graph = &facts->graph;
  for (iree_host_size_t i = 0; i < graph->reverse_postorder.count; ++i) {
    uint16_t source = graph->reverse_postorder.values[i];
    const loom_cfg_block_info_t* block = &graph->blocks[source];
    loom_op_t* branch = (loom_op_t*)block->block->last_op;
    if (block->component_is_cyclic || !loom_cfg_cond_br_isa(branch)) {
      continue;
    }
    // Uniform participation and a uniform selector need no lane convergence.
    // A uniform selector inside varying control still needs factoring: its
    // shared successor can also be reached by lanes in the enclosing bypass.
    if (loom_value_facts_is_uniform_at_scope(
            loom_value_fact_control_execution(facts->control, source),
            LOOM_VALUE_FACT_UNIFORM_SCOPE_SUBGROUP) &&
        loom_value_facts_is_uniform_at_scope(
            loom_value_fact_table_lookup(rewriter->fact_table,
                                         loom_cfg_cond_br_condition(branch)),
            LOOM_VALUE_FACT_UNIFORM_SCOPE_SUBGROUP)) {
      continue;
    }
    iree_arena_reset(arena);
    loom_cfg_converge_plan_t plan = {
        .module = module,
        .facts = facts,
        .arena = arena,
        .branch = branch,
        .source = source,
        .destinations = {loom_cfg_cond_br_true_dest(branch)->region_index,
                         loom_cfg_cond_br_false_dest(branch)->region_index},
    };
    bool selected = false;
    iree_status_t status = loom_cfg_converge_select(&plan, &selected);
    if (iree_status_is_ok(status) && selected) {
      status = loom_cfg_converge_preflight(&plan);
      if (iree_status_is_ok(status) && plan.representable) {
        status = loom_cfg_converge_apply(&plan, rewriter);
        *out_changed = true;
        loom_cfg_converge_statistics_t* statistics =
            loom_cfg_converge_statistics(pass);
        ++statistics->decisions_factored;
        statistics->values_carried += plan.capture_count;
      }
    }
    loom_local_value_domain_release(&plan.domain);
    if (!iree_status_is_ok(status) || *out_changed) {
      return status;
    }
  }
  return iree_ok_status();
}

iree_status_t loom_cfg_converge_run(loom_pass_t* pass, loom_module_t* module,
                                    loom_func_like_t function) {
  loom_region_t* body = loom_func_like_body(function);
  if (!body || body->block_count < 4) {
    return iree_ok_status();
  }
  loom_rewriter_t rewriter = {0};
  loom_rewriter_initialize(&rewriter, module, pass->arena);
  iree_arena_allocator_t arena;
  iree_arena_initialize(pass->arena->block_pool, &arena);
  loom_value_fact_table_t* facts = NULL;
  iree_status_t status = loom_pass_value_facts_acquire(
      pass, module,
      loom_pass_value_fact_scope_function_for_target(
          function,
          loom_target_function_version_target_facts(pass->function_version)),
      &facts);
  loom_rewriter_attach_value_facts(&rewriter, facts);
  bool changed = true;
  bool any_changed = false;
  while (iree_status_is_ok(status) && changed) {
    status =
        loom_cfg_converge_once(pass, module, body, &rewriter, &arena, &changed);
    any_changed |= changed;
  }
  if (iree_status_is_ok(status) && any_changed) {
    loom_pass_mark_changed(pass);
  }
  loom_rewriter_deinitialize(&rewriter);
  if (any_changed || !iree_status_is_ok(status)) {
    loom_pass_value_fact_owner_invalidate(pass->value_facts);
  }
  iree_arena_deinitialize(&arena);
  return status;
}

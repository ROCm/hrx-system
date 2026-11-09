// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/cleanup/fact_refinement.h"

#include <stdint.h>
#include <string.h>

#include "iree/base/internal/arena.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ir/value_replacement.h"
#include "loom/ops/op_defs.h"
#include "loom/util/adaptive_sort.h"
#include "loom/util/dominance.h"
#include "loom/util/fact_cfg.h"

#define LOOM_FACT_REFINEMENT_NO_NODE IREE_HOST_SIZE_MAX

typedef uint8_t loom_fact_refinement_node_state_t;

enum loom_fact_refinement_node_state_e {
  LOOM_FACT_REFINEMENT_NODE_UNUSED = 0,
  LOOM_FACT_REFINEMENT_NODE_REQUIRED = 1,
  LOOM_FACT_REFINEMENT_NODE_VISITING = 2,
  LOOM_FACT_REFINEMENT_NODE_AVAILABLE = 3,
};

typedef uint8_t loom_fact_refinement_observation_kind_t;

enum loom_fact_refinement_observation_kind_e {
  LOOM_FACT_REFINEMENT_OBSERVATION_EXACT_ALIAS = 0,
  LOOM_FACT_REFINEMENT_OBSERVATION_PREDICATE = 1,
};

typedef struct loom_fact_refinement_observation_t {
  // Exact identity operation establishing the predicate.
  loom_op_t* anchor;
  // Dynamic value constrained by this observation.
  loom_value_id_t source;
  // Active observation payload.
  loom_fact_refinement_observation_kind_t kind;
  // Payload selected by kind.
  union {
    // Exact identity result already representing source.
    loom_value_id_t exact_alias;
    // Predicate oriented so that source is its first argument.
    loom_predicate_t predicate;
  } payload;
} loom_fact_refinement_observation_t;

typedef struct loom_fact_refinement_node_t {
  // Dynamic identity refined by this node.
  loom_value_id_t source;
  // Exact operation after which the refinement becomes valid.
  loom_op_t* anchor;
  // Predicates established together at anchor.
  const loom_predicate_t* predicates;
  // Number of predicates in the contiguous span.
  iree_host_size_t predicate_count;
  // Nearest strict dominating node for source, or NO_NODE.
  iree_host_size_t parent;
  // Materialized same-value alias.
  loom_value_id_t alias;
  // Operation defining alias.
  loom_op_t* alias_op;
  // Dialect provider selected from source's type.
  const loom_fact_refinement_value_provider_t* provider;
} loom_fact_refinement_node_t;

typedef struct loom_fact_refinement_operand_t {
  // Operation whose complete operand tuple will be republished.
  loom_op_t* op;
  // Operand position receiving value.
  uint16_t operand_index;
  // Relation node supplying value, or NO_NODE for a materialized carrier.
  iree_host_size_t node;
  // Final replacement value.
  loom_value_id_t value;
} loom_fact_refinement_operand_t;

typedef struct loom_fact_refinement_attribute_t {
  // Operation owning the attribute occurrence.
  loom_op_t* op;
  // Attribute position containing source.
  uint8_t attribute_index;
  // Dynamic identity replaced in the attribute.
  loom_value_id_t source;
  // Dominating relation node supplying the replacement alias.
  iree_host_size_t node;
} loom_fact_refinement_attribute_t;

typedef struct loom_fact_refinement_carrier_mapping_t {
  // SSA carrier whose type references source.
  loom_value_id_t carrier;
  // Operation consuming carrier under this refinement environment.
  loom_op_t* user;
  // Operand position consuming carrier.
  uint16_t operand_index;
  // Dynamic identity embedded in carrier's type.
  loom_value_id_t source;
  // Dominating relation node supplying the embedded replacement.
  iree_host_size_t node;
} loom_fact_refinement_carrier_mapping_t;

typedef struct loom_fact_refinement_carrier_definition_mapping_t {
  // SSA carrier whose definition may publish a refined type directly.
  loom_value_id_t carrier;
  // Dynamic identity embedded in carrier's type.
  loom_value_id_t source;
  // Dominating relation node supplying the embedded replacement.
  iree_host_size_t node;
} loom_fact_refinement_carrier_definition_mapping_t;

typedef struct loom_fact_refinement_carrier_use_t {
  // SSA carrier consumed by user.
  loom_value_id_t carrier;
  // Operation consuming carrier.
  loom_op_t* user;
  // Operand position consuming carrier.
  uint16_t operand_index;
  // Canonical target type after applying all mappings for this occurrence.
  loom_type_t target_type;
  // Canonical module identity of target_type.
  loom_type_id_t target_type_id;
  // First mapping in the carrier-mapping array.
  iree_host_size_t mapping_start;
  // Number of mappings in the contiguous span.
  iree_host_size_t mapping_count;
} loom_fact_refinement_carrier_use_t;

typedef struct loom_fact_refinement_block_cache_entry_t {
  // Cached block entry position; NULL marks an empty hash slot.
  const loom_block_t* block;
  // Nearest relation node visible at block entry, or NO_NODE.
  iree_host_size_t node;
} loom_fact_refinement_block_cache_entry_t;

typedef struct loom_fact_refinement_source_t {
  // Shared plan containing the source's node span.
  struct loom_fact_refinement_plan_t* plan;
  // First node for source in plan->nodes.
  iree_host_size_t node_start;
  // Number of nodes for source.
  iree_host_size_t node_count;
  // Sparse inherited-binding cache keyed by blocks reached by references.
  loom_fact_refinement_block_cache_entry_t* block_cache;
  // Power-of-two block-cache capacity.
  iree_host_size_t block_cache_capacity;
  // Number of occupied block-cache entries.
  iree_host_size_t block_cache_count;
  // Reusable uncached ancestor path.
  const loom_block_t** block_path;
  // Number of blocks in the current path.
  iree_host_size_t block_path_count;
  // Allocated path entries.
  iree_host_size_t block_path_capacity;
} loom_fact_refinement_source_t;

typedef struct loom_fact_refinement_plan_t {
  // Rewriter applying the completed sparse plan.
  loom_rewriter_t* rewriter;
  // Compiler-composed dialect materializers.
  const loom_fact_refinement_policy_t* policy;
  // Feature-local arena returned to the shared block pool after this batch.
  iree_arena_allocator_t* arena;
  // Sorted exact identity operations consumed by this batch.
  loom_op_t* const* candidate_ops;
  // Number of operations in candidate_ops.
  iree_host_size_t candidate_count;
  // Borrowed dominance over retained fact-table CFG snapshots.
  loom_dominance_info_t dominance;
  // Normalized relation nodes sorted by source, block, and operation order.
  loom_fact_refinement_node_t* nodes;
  // Number of populated relation nodes.
  iree_host_size_t node_count;
  // Direct scalar/register operand rewrites.
  loom_fact_refinement_operand_t* operands;
  // Number of direct and materialized-carrier operand rewrites.
  iree_host_size_t operand_count;
  // Allocated operand rewrite entries.
  iree_host_size_t operand_capacity;
  // Indexed attribute owner rewrites.
  loom_fact_refinement_attribute_t* attributes;
  // Number of attribute owner mappings.
  iree_host_size_t attribute_count;
  // Allocated attribute owner mappings.
  iree_host_size_t attribute_capacity;
  // Indexed dependent-type carrier occurrence mappings.
  loom_fact_refinement_carrier_mapping_t* carrier_mappings;
  // Number of carrier occurrence mappings.
  iree_host_size_t carrier_mapping_count;
  // Allocated carrier occurrence mappings.
  iree_host_size_t carrier_mapping_capacity;
  // Indexed mappings applicable at carrier definitions.
  loom_fact_refinement_carrier_definition_mapping_t*
      carrier_definition_mappings;
  // Number of carrier-definition mappings.
  iree_host_size_t carrier_definition_mapping_count;
  // Allocated carrier-definition mappings.
  iree_host_size_t carrier_definition_mapping_capacity;
} loom_fact_refinement_plan_t;

static bool loom_fact_refinement_pointer_less(loom_op_t* const* lhs,
                                              loom_op_t* const* rhs) {
  return (uintptr_t)*lhs < (uintptr_t)*rhs;
}

LOOM_DEFINE_ADAPTIVE_SORT(loom_fact_refinement_sort_op_pointers, loom_op_t*,
                          loom_fact_refinement_pointer_less)

static bool loom_fact_refinement_is_candidate(
    const loom_fact_refinement_plan_t* plan, const loom_op_t* op) {
  const uintptr_t key = (uintptr_t)op;
  iree_host_size_t low = 0;
  iree_host_size_t high = plan->candidate_count;
  while (low < high) {
    const iree_host_size_t middle = low + (high - low) / 2;
    const uintptr_t candidate = (uintptr_t)plan->candidate_ops[middle];
    if (candidate < key) {
      low = middle + 1;
    } else {
      high = middle;
    }
  }
  return low < plan->candidate_count && plan->candidate_ops[low] == op;
}

static bool loom_fact_refinement_predicate_equal(const loom_predicate_t* lhs,
                                                 const loom_predicate_t* rhs) {
  if (lhs->kind != rhs->kind || lhs->arg_count != rhs->arg_count) {
    return false;
  }
  for (uint8_t i = 0; i < lhs->arg_count; ++i) {
    if (lhs->arg_tags[i] != rhs->arg_tags[i] || lhs->args[i] != rhs->args[i]) {
      return false;
    }
  }
  return true;
}

static bool loom_fact_refinement_observation_less(
    const loom_fact_refinement_observation_t* lhs,
    const loom_fact_refinement_observation_t* rhs) {
  if (lhs->source != rhs->source) {
    return lhs->source < rhs->source;
  }
  if (lhs->anchor != rhs->anchor) {
    return (uintptr_t)lhs->anchor < (uintptr_t)rhs->anchor;
  }
  if (lhs->kind != rhs->kind) {
    return lhs->kind < rhs->kind;
  }
  if (lhs->kind == LOOM_FACT_REFINEMENT_OBSERVATION_EXACT_ALIAS) {
    return lhs->payload.exact_alias < rhs->payload.exact_alias;
  }
  if (lhs->payload.predicate.kind != rhs->payload.predicate.kind) {
    return lhs->payload.predicate.kind < rhs->payload.predicate.kind;
  }
  if (lhs->payload.predicate.arg_count != rhs->payload.predicate.arg_count) {
    return lhs->payload.predicate.arg_count < rhs->payload.predicate.arg_count;
  }
  for (uint8_t i = 0; i < lhs->payload.predicate.arg_count; ++i) {
    if (lhs->payload.predicate.arg_tags[i] !=
        rhs->payload.predicate.arg_tags[i]) {
      return lhs->payload.predicate.arg_tags[i] <
             rhs->payload.predicate.arg_tags[i];
    }
    if (lhs->payload.predicate.args[i] != rhs->payload.predicate.args[i]) {
      return lhs->payload.predicate.args[i] < rhs->payload.predicate.args[i];
    }
  }
  return false;
}

LOOM_DEFINE_ADAPTIVE_SORT(loom_fact_refinement_sort_observations,
                          loom_fact_refinement_observation_t,
                          loom_fact_refinement_observation_less)

static bool loom_fact_refinement_node_less(
    const loom_fact_refinement_node_t* lhs,
    const loom_fact_refinement_node_t* rhs) {
  if (lhs->source != rhs->source) {
    return lhs->source < rhs->source;
  }
  if (lhs->anchor->parent_block != rhs->anchor->parent_block) {
    return (uintptr_t)lhs->anchor->parent_block <
           (uintptr_t)rhs->anchor->parent_block;
  }
  if (lhs->anchor->block_ordinal != rhs->anchor->block_ordinal) {
    return lhs->anchor->block_ordinal < rhs->anchor->block_ordinal;
  }
  return (uintptr_t)lhs->anchor < (uintptr_t)rhs->anchor;
}

LOOM_DEFINE_ADAPTIVE_SORT(loom_fact_refinement_sort_nodes,
                          loom_fact_refinement_node_t,
                          loom_fact_refinement_node_less)

static bool loom_fact_refinement_operand_less(
    const loom_fact_refinement_operand_t* lhs,
    const loom_fact_refinement_operand_t* rhs) {
  if (lhs->op != rhs->op) {
    return (uintptr_t)lhs->op < (uintptr_t)rhs->op;
  }
  return lhs->operand_index < rhs->operand_index;
}

LOOM_DEFINE_ADAPTIVE_SORT(loom_fact_refinement_sort_operands,
                          loom_fact_refinement_operand_t,
                          loom_fact_refinement_operand_less)

static bool loom_fact_refinement_attribute_less(
    const loom_fact_refinement_attribute_t* lhs,
    const loom_fact_refinement_attribute_t* rhs) {
  if (lhs->op != rhs->op) {
    return (uintptr_t)lhs->op < (uintptr_t)rhs->op;
  }
  if (lhs->attribute_index != rhs->attribute_index) {
    return lhs->attribute_index < rhs->attribute_index;
  }
  return lhs->source < rhs->source;
}

LOOM_DEFINE_ADAPTIVE_SORT(loom_fact_refinement_sort_attributes,
                          loom_fact_refinement_attribute_t,
                          loom_fact_refinement_attribute_less)

static bool loom_fact_refinement_carrier_mapping_less(
    const loom_fact_refinement_carrier_mapping_t* lhs,
    const loom_fact_refinement_carrier_mapping_t* rhs) {
  if (lhs->carrier != rhs->carrier) {
    return lhs->carrier < rhs->carrier;
  }
  if (lhs->user != rhs->user) {
    return (uintptr_t)lhs->user < (uintptr_t)rhs->user;
  }
  if (lhs->operand_index != rhs->operand_index) {
    return lhs->operand_index < rhs->operand_index;
  }
  return lhs->source < rhs->source;
}

LOOM_DEFINE_ADAPTIVE_SORT(loom_fact_refinement_sort_carrier_mappings,
                          loom_fact_refinement_carrier_mapping_t,
                          loom_fact_refinement_carrier_mapping_less)

static bool loom_fact_refinement_carrier_definition_mapping_less(
    const loom_fact_refinement_carrier_definition_mapping_t* lhs,
    const loom_fact_refinement_carrier_definition_mapping_t* rhs) {
  if (lhs->carrier != rhs->carrier) {
    return lhs->carrier < rhs->carrier;
  }
  return lhs->source < rhs->source;
}

LOOM_DEFINE_ADAPTIVE_SORT(loom_fact_refinement_sort_carrier_definition_mappings,
                          loom_fact_refinement_carrier_definition_mapping_t,
                          loom_fact_refinement_carrier_definition_mapping_less)

static bool loom_fact_refinement_carrier_use_less(
    const loom_fact_refinement_carrier_use_t* lhs,
    const loom_fact_refinement_carrier_use_t* rhs) {
  if (lhs->carrier != rhs->carrier) {
    return lhs->carrier < rhs->carrier;
  }
  if (lhs->target_type_id != rhs->target_type_id) {
    return lhs->target_type_id < rhs->target_type_id;
  }
  if (lhs->user != rhs->user) {
    return (uintptr_t)lhs->user < (uintptr_t)rhs->user;
  }
  return lhs->operand_index < rhs->operand_index;
}

LOOM_DEFINE_ADAPTIVE_SORT(loom_fact_refinement_sort_carrier_uses,
                          loom_fact_refinement_carrier_use_t,
                          loom_fact_refinement_carrier_use_less)

static const loom_fact_refinement_value_provider_t*
loom_fact_refinement_find_value_provider(
    const loom_fact_refinement_policy_t* policy, loom_type_t type) {
  for (iree_host_size_t i = 0; i < policy->value_providers.count; ++i) {
    const loom_fact_refinement_value_provider_t* provider =
        policy->value_providers.values[i];
    if (provider->supports(type)) {
      return provider;
    }
  }
  return NULL;
}

static const loom_fact_refinement_carrier_provider_t*
loom_fact_refinement_find_carrier_provider(
    const loom_fact_refinement_policy_t* policy, loom_type_t source_type,
    loom_type_t target_type) {
  for (iree_host_size_t i = 0; i < policy->carrier_providers.count; ++i) {
    const loom_fact_refinement_carrier_provider_t* provider =
        policy->carrier_providers.values[i];
    if (provider->supports(source_type, target_type)) {
      return provider;
    }
  }
  return NULL;
}

static bool loom_fact_refinement_op_is_tuple_boundary(
    const loom_module_t* module, const loom_op_t* op) {
  const loom_op_vtable_t* vtable = loom_op_vtable(module, op);
  return !vtable || iree_any_bit_set(op->traits, LOOM_TRAIT_TERMINATOR) ||
         vtable->call_like || op->region_count > 0 || op->successor_count > 0;
}

static bool loom_fact_refinement_constraint_changes_with_operand_type(
    const loom_constraint_t* constraint) {
  switch ((enum loom_constraint_property_e)constraint->property) {
    case LOOM_PROPERTY_TYPE:
    case LOOM_PROPERTY_ENCODING:
    case LOOM_PROPERTY_SHAPE:
    case LOOM_PROPERTY_REGISTER_CLASS:
    case LOOM_PROPERTY_REGISTER_UNIT_COUNT:
      break;
    default:
      return false;
  }
  switch ((enum loom_constraint_relation_e)constraint->relation) {
    case LOOM_RELATION_PAIRWISE_EQ:
    case LOOM_RELATION_ALL_SAME:
    case LOOM_RELATION_REGION_ARG_MATCH:
    case LOOM_RELATION_YIELD_MATCH:
    case LOOM_RELATION_VARIADIC_MATCH:
    case LOOM_RELATION_REGISTER_UNIT_COUNT_SUM:
      return true;
    default:
      return false;
  }
}

static bool loom_fact_refinement_constraint_mentions_operand(
    const loom_op_vtable_t* vtable, const loom_op_t* op,
    const loom_constraint_t* constraint, uint16_t operand_index) {
  const loom_operand_descriptor_t* descriptor = NULL;
  uint8_t operand_field_index = 0;
  if (!loom_op_operand_descriptor_at(vtable, op, operand_index, &descriptor,
                                     &operand_field_index, NULL)) {
    return true;
  }
  (void)descriptor;
  for (uint8_t i = 0; i < constraint->arg_count; ++i) {
    const loom_field_ref_t field_ref = constraint->args[i];
    if (LOOM_FIELD_REF_CATEGORY(field_ref) == LOOM_FIELD_OPERAND &&
        LOOM_FIELD_REF_INDEX(field_ref) == operand_field_index) {
      return true;
    }
  }
  return false;
}

static bool loom_fact_refinement_allows_identity_operand(
    const loom_module_t* module, const loom_op_t* op, uint16_t operand_index) {
  return op && operand_index < op->operand_count &&
         !loom_fact_refinement_op_is_tuple_boundary(module, op);
}

static bool loom_fact_refinement_allows_refined_carrier(
    const loom_module_t* module, const loom_op_t* op, uint16_t operand_index) {
  if (!loom_fact_refinement_allows_identity_operand(module, op,
                                                    operand_index)) {
    return false;
  }
  const loom_op_vtable_t* vtable = loom_op_vtable(module, op);
  if (vtable->constraint_count > 0 && !vtable->constraints) {
    return false;
  }
  for (uint8_t i = vtable->operand_dictionary_count;
       i < vtable->constraint_count; ++i) {
    const loom_constraint_t* constraint = &vtable->constraints[i];
    if (loom_fact_refinement_constraint_changes_with_operand_type(constraint) &&
        loom_fact_refinement_constraint_mentions_operand(vtable, op, constraint,
                                                         operand_index)) {
      return false;
    }
  }
  return true;
}

static bool loom_fact_refinement_allows_refined_carrier_definition(
    const loom_module_t* module, loom_value_id_t carrier) {
  const loom_value_t* value = loom_module_value(module, carrier);
  if (loom_value_is_block_arg(value) || loom_value_has_attribute_uses(value) ||
      loom_module_value_has_type_uses(module, carrier)) {
    return false;
  }
  const loom_op_t* defining_op = loom_value_def_op(value);
  if (!defining_op || !loom_traits_have_refinable_result_type_refs(
                          loom_op_effective_traits(module, defining_op))) {
    return false;
  }
  const loom_use_t* use = NULL;
  loom_value_for_each_use(value, use) {
    if (!loom_fact_refinement_allows_refined_carrier(
            module, loom_use_user_op(*use), loom_use_operand_index(*use))) {
      return false;
    }
  }
  return true;
}

static bool loom_fact_refinement_allows_identity_attribute(
    const loom_module_t* module, const loom_op_t* op, uint8_t attribute_index) {
  return op && attribute_index < op->attribute_count &&
         !loom_fact_refinement_op_is_tuple_boundary(module, op);
}

static bool loom_fact_refinement_orient_binary_predicate(
    uint8_t kind, uint8_t* out_swapped_kind) {
  switch ((enum loom_predicate_kind_e)kind) {
    case LOOM_PREDICATE_EQ:
    case LOOM_PREDICATE_NE:
      *out_swapped_kind = kind;
      return true;
    case LOOM_PREDICATE_LT:
      *out_swapped_kind = LOOM_PREDICATE_GT;
      return true;
    case LOOM_PREDICATE_LE:
      *out_swapped_kind = LOOM_PREDICATE_GE;
      return true;
    case LOOM_PREDICATE_GT:
      *out_swapped_kind = LOOM_PREDICATE_LT;
      return true;
    case LOOM_PREDICATE_GE:
      *out_swapped_kind = LOOM_PREDICATE_LE;
      return true;
    case LOOM_PREDICATE_ULT:
      *out_swapped_kind = LOOM_PREDICATE_UGT;
      return true;
    case LOOM_PREDICATE_ULE:
      *out_swapped_kind = LOOM_PREDICATE_UGE;
      return true;
    case LOOM_PREDICATE_UGT:
      *out_swapped_kind = LOOM_PREDICATE_ULT;
      return true;
    case LOOM_PREDICATE_UGE:
      *out_swapped_kind = LOOM_PREDICATE_ULE;
      return true;
    case LOOM_PREDICATE_MULTIPLE_OF:
    case LOOM_PREDICATE_POWER_OF_TWO:
    case LOOM_PREDICATE_RANGE:
    case LOOM_PREDICATE_NOT_NAN:
    case LOOM_PREDICATE_NOT_INF:
    case LOOM_PREDICATE_FINITE:
    case LOOM_PREDICATE_COUNT_:
      return false;
  }
  return false;
}

static bool loom_fact_refinement_value_is_dynamic(
    const loom_value_fact_table_t* facts, loom_value_id_t value_id) {
  return !loom_value_facts_is_exact(
      loom_value_fact_table_lookup(facts, value_id));
}

static bool loom_fact_refinement_anchor_argument_exact_i64(
    const loom_fact_refinement_plan_t* plan, const loom_op_t* anchor,
    loom_value_id_t argument, int64_t* out_exact_value) {
  const loom_value_fact_table_t* facts = plan->rewriter->fact_table;
  if (loom_value_facts_as_exact_i64(
          loom_value_fact_table_lookup(facts, argument), out_exact_value)) {
    return true;
  }

  const loom_value_id_t argument_identity =
      loom_value_fact_table_query_identity(facts, argument);
  const loom_value_id_t* operands = loom_op_const_operands(anchor);
  const loom_value_id_t* results = loom_op_const_results(anchor);
  const uint16_t pair_count = anchor->operand_count < anchor->result_count
                                  ? anchor->operand_count
                                  : anchor->result_count;
  for (uint16_t i = 0; i < pair_count; ++i) {
    if (loom_value_fact_table_query_identity(facts, operands[i]) !=
        argument_identity) {
      continue;
    }
    if (loom_value_facts_as_exact_i64(
            loom_value_fact_table_lookup(facts, results[i]), out_exact_value)) {
      return true;
    }
  }
  return false;
}

static void loom_fact_refinement_substitute_exact_arguments(
    const loom_fact_refinement_plan_t* plan, const loom_op_t* anchor,
    loom_predicate_t* predicate) {
  for (uint8_t i = 0; i < predicate->arg_count; ++i) {
    if (predicate->arg_tags[i] != LOOM_PRED_ARG_VALUE ||
        predicate->args[i] < 0) {
      continue;
    }
    int64_t exact_value = 0;
    if (loom_fact_refinement_anchor_argument_exact_i64(
            plan, anchor, (loom_value_id_t)predicate->args[i], &exact_value)) {
      predicate->arg_tags[i] = LOOM_PRED_ARG_CONST;
      predicate->args[i] = exact_value;
    }
  }
}

static void loom_fact_refinement_append_observation(
    loom_fact_refinement_plan_t* plan, loom_op_t* anchor,
    loom_value_id_t source, uint8_t target_argument,
    const loom_predicate_t* predicate,
    loom_fact_refinement_observation_t* observations,
    iree_host_size_t observation_capacity,
    iree_host_size_t* inout_observation_count) {
  if (source >= plan->rewriter->module->values.count) {
    return;
  }
  const loom_type_t source_type =
      loom_module_value_type(plan->rewriter->module, source);
  const loom_fact_refinement_value_provider_t* provider =
      loom_fact_refinement_find_value_provider(plan->policy, source_type);
  if (!provider ||
      !loom_predicate_kind_accepts_value_type(predicate->kind, source_type)) {
    return;
  }

  loom_predicate_t normalized = *predicate;
  loom_fact_refinement_substitute_exact_arguments(plan, anchor, &normalized);
  if (target_argument != 0) {
    if (target_argument != 1 || normalized.arg_count != 2) {
      return;
    }
    uint8_t oriented_kind = 0;
    if (!loom_fact_refinement_orient_binary_predicate(normalized.kind,
                                                      &oriented_kind)) {
      return;
    }
    normalized.kind = oriented_kind;
    const loom_predicate_arg_tag_t first_tag = normalized.arg_tags[0];
    const int64_t first_argument = normalized.args[0];
    normalized.arg_tags[0] = normalized.arg_tags[1];
    normalized.args[0] = normalized.args[1];
    normalized.arg_tags[1] = first_tag;
    normalized.args[1] = first_argument;
  }
  if (normalized.arg_tags[0] != LOOM_PRED_ARG_VALUE ||
      normalized.args[0] != (int64_t)source) {
    return;
  }
  IREE_ASSERT(*inout_observation_count < observation_capacity,
              "observation capacity must cover every predicate argument");
  const iree_host_size_t observation_index = (*inout_observation_count)++;
  observations[observation_index] = (loom_fact_refinement_observation_t){
      .anchor = anchor,
      .source = source,
      .kind = LOOM_FACT_REFINEMENT_OBSERVATION_PREDICATE,
      .payload.predicate = normalized,
  };
}

static void loom_fact_refinement_append_exact_alias_observation(
    loom_fact_refinement_plan_t* plan, loom_op_t* anchor,
    loom_value_id_t source, loom_value_id_t alias,
    loom_fact_refinement_observation_t* observations,
    iree_host_size_t observation_capacity,
    iree_host_size_t* inout_observation_count) {
  if (source >= plan->rewriter->module->values.count ||
      alias >= plan->rewriter->module->values.count ||
      !loom_fact_refinement_value_is_dynamic(plan->rewriter->fact_table,
                                             source)) {
    return;
  }
  const loom_type_t source_type =
      loom_module_value_type(plan->rewriter->module, source);
  if (!loom_fact_refinement_find_value_provider(plan->policy, source_type)) {
    return;
  }
  IREE_ASSERT(*inout_observation_count < observation_capacity,
              "observation capacity must cover every identity pair");
  const iree_host_size_t observation_index = (*inout_observation_count)++;
  observations[observation_index] = (loom_fact_refinement_observation_t){
      .anchor = anchor,
      .source = source,
      .kind = LOOM_FACT_REFINEMENT_OBSERVATION_EXACT_ALIAS,
      .payload.exact_alias = alias,
  };
}

static bool loom_fact_refinement_candidate_is_still_exact(
    const loom_value_fact_table_t* facts, const loom_op_t* op) {
  if (!op || iree_any_bit_set(op->flags, LOOM_OP_FLAG_DEAD) ||
      op->result_count == 0) {
    return false;
  }
  const loom_value_id_t* results = loom_op_const_results(op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    if (results[i] == LOOM_VALUE_ID_INVALID ||
        !loom_value_facts_is_exact(
            loom_value_fact_table_lookup(facts, results[i]))) {
      return false;
    }
  }
  return true;
}

static iree_status_t loom_fact_refinement_build_nodes(
    loom_fact_refinement_plan_t* plan, loom_op_t** candidate_ops,
    iree_host_size_t candidate_count) {
  iree_host_size_t observation_capacity = 0;
  for (iree_host_size_t i = 0; i < candidate_count; ++i) {
    const loom_op_t* op = candidate_ops[i];
    const loom_attribute_t* attributes = loom_op_const_attrs(op);
    const uint16_t pair_count = op->operand_count < op->result_count
                                    ? op->operand_count
                                    : op->result_count;
    if (pair_count > IREE_HOST_SIZE_MAX - observation_capacity) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "fact-refinement observation overflow");
    }
    observation_capacity += pair_count;
    for (uint8_t j = 0; j < op->attribute_count; ++j) {
      if (attributes[j].kind != LOOM_ATTR_PREDICATE_LIST) {
        continue;
      }
      for (uint16_t k = 0; k < attributes[j].count; ++k) {
        const uint8_t argument_count =
            attributes[j].predicate_list[k].arg_count;
        if (argument_count > IREE_HOST_SIZE_MAX - observation_capacity) {
          return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                  "fact-refinement observation overflow");
        }
        observation_capacity += argument_count;
      }
    }
  }
  if (observation_capacity == 0) {
    return iree_ok_status();
  }

  iree_arena_allocator_t observation_arena;
  iree_arena_initialize(plan->arena->block_pool, &observation_arena);
  loom_fact_refinement_observation_t* observations = NULL;
  iree_status_t status =
      iree_arena_allocate_array(&observation_arena, observation_capacity,
                                sizeof(*observations), (void**)&observations);
  if (!iree_status_is_ok(status)) {
    iree_arena_deinitialize(&observation_arena);
    return status;
  }
  iree_host_size_t observation_count = 0;
  const loom_value_fact_table_t* facts = plan->rewriter->fact_table;
  for (iree_host_size_t i = 0; i < candidate_count; ++i) {
    loom_op_t* op = candidate_ops[i];
    const loom_value_id_t* operands = loom_op_const_operands(op);
    const loom_value_id_t* results = loom_op_const_results(op);
    const uint16_t pair_count = op->operand_count < op->result_count
                                    ? op->operand_count
                                    : op->result_count;
    for (uint16_t j = 0; j < pair_count; ++j) {
      if (operands[j] == LOOM_VALUE_ID_INVALID ||
          results[j] == LOOM_VALUE_ID_INVALID ||
          !loom_value_facts_is_exact(
              loom_value_fact_table_lookup(facts, results[j]))) {
        continue;
      }
      loom_fact_refinement_append_exact_alias_observation(
          plan, op, operands[j], results[j], observations, observation_capacity,
          &observation_count);
    }
    const loom_attribute_t* attributes = loom_op_const_attrs(op);
    for (uint8_t j = 0; j < op->attribute_count; ++j) {
      if (attributes[j].kind != LOOM_ATTR_PREDICATE_LIST) {
        continue;
      }
      for (uint16_t k = 0; k < attributes[j].count; ++k) {
        const loom_predicate_t* predicate = &attributes[j].predicate_list[k];
        for (uint8_t argument_index = 0; argument_index < predicate->arg_count;
             ++argument_index) {
          if (predicate->arg_tags[argument_index] != LOOM_PRED_ARG_VALUE ||
              predicate->args[argument_index] < 0) {
            continue;
          }
          const loom_value_id_t source =
              (loom_value_id_t)predicate->args[argument_index];
          if (source >= plan->rewriter->module->values.count ||
              !loom_fact_refinement_value_is_dynamic(facts, source)) {
            continue;
          }
          loom_fact_refinement_append_observation(
              plan, op, source, argument_index, predicate, observations,
              observation_capacity, &observation_count);
        }
      }
    }
  }
  if (observation_count == 0) {
    iree_arena_deinitialize(&observation_arena);
    return iree_ok_status();
  }

  loom_fact_refinement_sort_observations(observations, observation_count);
  loom_fact_refinement_node_t* nodes = NULL;
  loom_predicate_t* predicates = NULL;
  status = iree_arena_allocate_array(plan->arena, observation_count,
                                     sizeof(*nodes), (void**)&nodes);
  if (iree_status_is_ok(status)) {
    status =
        iree_arena_allocate_array(plan->arena, observation_count,
                                  sizeof(*predicates), (void**)&predicates);
  }
  if (!iree_status_is_ok(status)) {
    iree_arena_deinitialize(&observation_arena);
    return status;
  }

  iree_host_size_t node_count = 0;
  iree_host_size_t predicate_count = 0;
  for (iree_host_size_t i = 0; i < observation_count;) {
    const loom_value_id_t source = observations[i].source;
    loom_op_t* anchor = observations[i].anchor;
    const iree_host_size_t predicate_start = predicate_count;
    loom_value_id_t exact_alias = LOOM_VALUE_ID_INVALID;
    do {
      if (observations[i].kind == LOOM_FACT_REFINEMENT_OBSERVATION_PREDICATE) {
        predicates[predicate_count++] = observations[i].payload.predicate;
      } else {
        IREE_ASSERT(exact_alias == LOOM_VALUE_ID_INVALID ||
                    exact_alias == observations[i].payload.exact_alias);
        exact_alias = observations[i].payload.exact_alias;
      }
      ++i;
    } while (i < observation_count && observations[i].source == source &&
             observations[i].anchor == anchor);
    const loom_type_t source_type =
        loom_module_value_type(plan->rewriter->module, source);
    nodes[node_count++] = (loom_fact_refinement_node_t){
        .source = source,
        .anchor = anchor,
        .predicates = predicates + predicate_start,
        .predicate_count = predicate_count - predicate_start,
        .parent = LOOM_FACT_REFINEMENT_NO_NODE,
        .alias = exact_alias,
        .alias_op = exact_alias == LOOM_VALUE_ID_INVALID ? NULL : anchor,
        .provider =
            loom_fact_refinement_find_value_provider(plan->policy, source_type),
    };
  }
  iree_arena_deinitialize(&observation_arena);
  loom_fact_refinement_sort_nodes(nodes, node_count);
  plan->nodes = nodes;
  plan->node_count = node_count;
  return iree_ok_status();
}

static iree_host_size_t loom_fact_refinement_block_hash(
    const loom_block_t* block) {
  uintptr_t value = (uintptr_t)block;
  value ^= value >> 17;
  value *= (uintptr_t)UINT64_C(0x9E3779B97F4A7C15);
  value ^= value >> 23;
  return (iree_host_size_t)value;
}

static loom_fact_refinement_block_cache_entry_t*
loom_fact_refinement_block_cache_lookup(loom_fact_refinement_source_t* source,
                                        const loom_block_t* block) {
  if (source->block_cache_capacity == 0) {
    return NULL;
  }
  const iree_host_size_t mask = source->block_cache_capacity - 1;
  iree_host_size_t index = loom_fact_refinement_block_hash(block) & mask;
  while (source->block_cache[index].block) {
    if (source->block_cache[index].block == block) {
      return &source->block_cache[index];
    }
    index = (index + 1) & mask;
  }
  return NULL;
}

static iree_status_t loom_fact_refinement_block_cache_reserve(
    loom_fact_refinement_source_t* source,
    iree_host_size_t minimum_entry_count) {
  if (source->block_cache_capacity != 0 &&
      minimum_entry_count <= source->block_cache_capacity / 2) {
    return iree_ok_status();
  }
  iree_host_size_t new_capacity = source->block_cache_capacity
                                      ? source->block_cache_capacity * 2
                                      : (iree_host_size_t)16;
  while (minimum_entry_count > new_capacity / 2) {
    if (new_capacity > IREE_HOST_SIZE_MAX / 2) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "fact-refinement block cache overflow");
    }
    new_capacity *= 2;
  }
  loom_fact_refinement_block_cache_entry_t* new_entries = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(source->plan->arena, new_capacity,
                                sizeof(*new_entries), (void**)&new_entries));
  memset(new_entries, 0, new_capacity * sizeof(*new_entries));
  if (source->block_cache) {
    const iree_host_size_t mask = new_capacity - 1;
    for (iree_host_size_t i = 0; i < source->block_cache_capacity; ++i) {
      const loom_fact_refinement_block_cache_entry_t entry =
          source->block_cache[i];
      if (!entry.block) {
        continue;
      }
      iree_host_size_t index =
          loom_fact_refinement_block_hash(entry.block) & mask;
      while (new_entries[index].block) {
        index = (index + 1) & mask;
      }
      new_entries[index] = entry;
    }
  }
  source->block_cache = new_entries;
  source->block_cache_capacity = new_capacity;
  return iree_ok_status();
}

static iree_status_t loom_fact_refinement_block_cache_insert(
    loom_fact_refinement_source_t* source, const loom_block_t* block,
    iree_host_size_t node) {
  loom_fact_refinement_block_cache_entry_t* existing =
      loom_fact_refinement_block_cache_lookup(source, block);
  if (existing) {
    existing->node = node;
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_fact_refinement_block_cache_reserve(
      source, source->block_cache_count + 1));
  const iree_host_size_t mask = source->block_cache_capacity - 1;
  iree_host_size_t index = loom_fact_refinement_block_hash(block) & mask;
  while (source->block_cache[index].block) {
    index = (index + 1) & mask;
  }
  source->block_cache[index] =
      (loom_fact_refinement_block_cache_entry_t){block, node};
  ++source->block_cache_count;
  return iree_ok_status();
}

static iree_status_t loom_fact_refinement_append_block_path(
    loom_fact_refinement_source_t* source, const loom_block_t* block) {
  if (source->block_path_count == source->block_path_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        source->plan->arena, source->block_path_count,
        source->block_path_count + 1, sizeof(*source->block_path),
        &source->block_path_capacity, (void**)&source->block_path));
  }
  source->block_path[source->block_path_count++] = block;
  return iree_ok_status();
}

static iree_host_size_t loom_fact_refinement_find_local_node(
    const loom_fact_refinement_source_t* source, const loom_block_t* block,
    uint64_t before_ordinal) {
  const loom_fact_refinement_node_t* nodes = source->plan->nodes;
  iree_host_size_t low = source->node_start;
  iree_host_size_t high = source->node_start + source->node_count;
  const uintptr_t block_key = (uintptr_t)block;
  while (low < high) {
    const iree_host_size_t middle = low + (high - low) / 2;
    if ((uintptr_t)nodes[middle].anchor->parent_block < block_key) {
      low = middle + 1;
    } else {
      high = middle;
    }
  }
  const iree_host_size_t block_start = low;
  high = source->node_start + source->node_count;
  while (low < high) {
    const iree_host_size_t middle = low + (high - low) / 2;
    const loom_fact_refinement_node_t* node = &nodes[middle];
    if (node->anchor->parent_block == block &&
        node->anchor->block_ordinal < before_ordinal) {
      low = middle + 1;
    } else if ((uintptr_t)node->anchor->parent_block < block_key) {
      low = middle + 1;
    } else {
      high = middle;
    }
  }
  return low > block_start ? low - 1 : LOOM_FACT_REFINEMENT_NO_NODE;
}

static const loom_op_t* loom_fact_refinement_region_parent_op(
    const loom_block_t* block) {
  if (!block || !block->parent_region) {
    return NULL;
  }
  if (block->first_op) {
    return block->first_op->parent_op;
  }
  const loom_block_t* sibling = NULL;
  loom_region_for_each_block(block->parent_region, sibling) {
    if (sibling->first_op) {
      return sibling->first_op->parent_op;
    }
  }
  return NULL;
}

static iree_status_t loom_fact_refinement_resolve_block_entry(
    loom_fact_refinement_source_t* source, const loom_block_t* block,
    iree_host_size_t* out_node) {
  loom_fact_refinement_block_cache_entry_t* cached =
      loom_fact_refinement_block_cache_lookup(source, block);
  if (cached) {
    *out_node = cached->node;
    return iree_ok_status();
  }

  source->block_path_count = 0;
  const loom_block_t* current = block;
  iree_host_size_t result = LOOM_FACT_REFINEMENT_NO_NODE;
  while (current) {
    cached = loom_fact_refinement_block_cache_lookup(source, current);
    if (cached) {
      result = cached->node;
      break;
    }
    IREE_RETURN_IF_ERROR(
        loom_fact_refinement_append_block_path(source, current));

    const loom_block_t* immediate_dominator =
        loom_dominance_immediate_dominator_block(&source->plan->dominance,
                                                 current);
    if (immediate_dominator) {
      result = loom_fact_refinement_find_local_node(source, immediate_dominator,
                                                    UINT64_MAX);
      if (result != LOOM_FACT_REFINEMENT_NO_NODE) {
        break;
      }
      current = immediate_dominator;
      continue;
    }

    const loom_region_t* region = current->parent_region;
    if (!region || region->block_count == 0 || region->blocks[0] != current) {
      break;
    }
    const loom_op_t* parent_op = loom_fact_refinement_region_parent_op(current);
    if (!parent_op || loom_traits_is_isolated(parent_op->traits)) {
      break;
    }
    result = loom_fact_refinement_find_local_node(
        source, parent_op->parent_block, parent_op->block_ordinal);
    if (result != LOOM_FACT_REFINEMENT_NO_NODE) {
      break;
    }
    current = parent_op->parent_block;
  }

  for (iree_host_size_t i = 0; i < source->block_path_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_fact_refinement_block_cache_insert(
        source, source->block_path[i], result));
  }
  *out_node = result;
  return iree_ok_status();
}

static iree_status_t loom_fact_refinement_resolve_before_op(
    loom_fact_refinement_source_t* source, const loom_op_t* op,
    iree_host_size_t* out_node) {
  *out_node = loom_fact_refinement_find_local_node(source, op->parent_block,
                                                   op->block_ordinal);
  if (*out_node != LOOM_FACT_REFINEMENT_NO_NODE) {
    return iree_ok_status();
  }
  return loom_fact_refinement_resolve_block_entry(source, op->parent_block,
                                                  out_node);
}

static iree_status_t loom_fact_refinement_append_operand(
    loom_fact_refinement_plan_t* plan, loom_op_t* op, uint16_t operand_index,
    iree_host_size_t node, loom_value_id_t value) {
  if (plan->operand_count == plan->operand_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        plan->arena, plan->operand_count, plan->operand_count + 1,
        sizeof(*plan->operands), &plan->operand_capacity,
        (void**)&plan->operands));
  }
  plan->operands[plan->operand_count++] =
      (loom_fact_refinement_operand_t){op, operand_index, node, value};
  return iree_ok_status();
}

static iree_status_t loom_fact_refinement_append_attribute(
    loom_fact_refinement_plan_t* plan, loom_op_t* op, uint8_t attribute_index,
    loom_value_id_t source, iree_host_size_t node) {
  if (plan->attribute_count == plan->attribute_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        plan->arena, plan->attribute_count, plan->attribute_count + 1,
        sizeof(*plan->attributes), &plan->attribute_capacity,
        (void**)&plan->attributes));
  }
  plan->attributes[plan->attribute_count++] =
      (loom_fact_refinement_attribute_t){op, attribute_index, source, node};
  return iree_ok_status();
}

static iree_status_t loom_fact_refinement_append_carrier_mapping(
    loom_fact_refinement_plan_t* plan, loom_value_id_t carrier, loom_op_t* user,
    uint16_t operand_index, loom_value_id_t source, iree_host_size_t node) {
  if (plan->carrier_mapping_count == plan->carrier_mapping_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        plan->arena, plan->carrier_mapping_count,
        plan->carrier_mapping_count + 1, sizeof(*plan->carrier_mappings),
        &plan->carrier_mapping_capacity, (void**)&plan->carrier_mappings));
  }
  plan->carrier_mappings[plan->carrier_mapping_count++] =
      (loom_fact_refinement_carrier_mapping_t){carrier, user, operand_index,
                                               source, node};
  return iree_ok_status();
}

static iree_status_t loom_fact_refinement_append_carrier_definition_mapping(
    loom_fact_refinement_plan_t* plan, loom_value_id_t carrier,
    loom_value_id_t source, iree_host_size_t node) {
  if (plan->carrier_definition_mapping_count ==
      plan->carrier_definition_mapping_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        plan->arena, plan->carrier_definition_mapping_count,
        plan->carrier_definition_mapping_count + 1,
        sizeof(*plan->carrier_definition_mappings),
        &plan->carrier_definition_mapping_capacity,
        (void**)&plan->carrier_definition_mappings));
  }
  plan->carrier_definition_mappings[plan->carrier_definition_mapping_count++] =
      (loom_fact_refinement_carrier_definition_mapping_t){carrier, source,
                                                          node};
  return iree_ok_status();
}

static iree_status_t loom_fact_refinement_plan_source_references(
    loom_fact_refinement_source_t* source) {
  loom_fact_refinement_plan_t* plan = source->plan;
  loom_module_t* module = plan->rewriter->module;
  const loom_value_id_t source_value = plan->nodes[source->node_start].source;
  const loom_value_t* value = loom_module_value(module, source_value);

  const loom_use_t* use = NULL;
  loom_value_for_each_use(value, use) {
    loom_op_t* user = loom_use_user_op(*use);
    const uint16_t operand_index = loom_use_operand_index(*use);
    if (!user || iree_any_bit_set(user->flags, LOOM_OP_FLAG_DEAD) ||
        loom_fact_refinement_is_candidate(plan, user) ||
        !loom_fact_refinement_allows_identity_operand(module, user,
                                                      operand_index)) {
      continue;
    }
    iree_host_size_t node = LOOM_FACT_REFINEMENT_NO_NODE;
    IREE_RETURN_IF_ERROR(
        loom_fact_refinement_resolve_before_op(source, user, &node));
    if (node != LOOM_FACT_REFINEMENT_NO_NODE) {
      IREE_RETURN_IF_ERROR(loom_fact_refinement_append_operand(
          plan, user, operand_index, node, LOOM_VALUE_ID_INVALID));
    }
  }

  loom_type_use_iterator_t attribute_users;
  loom_attribute_users_begin(&module->type_uses, source_value,
                             &attribute_users);
  for (loom_attribute_user_t user = loom_attribute_users_next(&attribute_users);
       user.op; user = loom_attribute_users_next(&attribute_users)) {
    if (iree_any_bit_set(user.op->flags, LOOM_OP_FLAG_DEAD) ||
        loom_fact_refinement_is_candidate(plan, user.op) ||
        !loom_fact_refinement_allows_identity_attribute(module, user.op,
                                                        user.attribute_index)) {
      continue;
    }
    iree_host_size_t node = LOOM_FACT_REFINEMENT_NO_NODE;
    IREE_RETURN_IF_ERROR(
        loom_fact_refinement_resolve_before_op(source, user.op, &node));
    if (node != LOOM_FACT_REFINEMENT_NO_NODE) {
      IREE_RETURN_IF_ERROR(loom_fact_refinement_append_attribute(
          plan, user.op, user.attribute_index, source_value, node));
    }
  }

  loom_type_use_iterator_t type_users;
  loom_type_users_begin(&module->type_uses, source_value, &type_users);
  for (loom_value_id_t carrier = loom_type_users_next(&type_users);
       carrier != LOOM_VALUE_ID_INVALID;
       carrier = loom_type_users_next(&type_users)) {
    const loom_value_t* carrier_value = loom_module_value(module, carrier);
    if (loom_fact_refinement_allows_refined_carrier_definition(module,
                                                               carrier)) {
      loom_op_t* defining_op = loom_value_def_op(carrier_value);
      iree_host_size_t definition_node = LOOM_FACT_REFINEMENT_NO_NODE;
      IREE_RETURN_IF_ERROR(loom_fact_refinement_resolve_before_op(
          source, defining_op, &definition_node));
      if (definition_node != LOOM_FACT_REFINEMENT_NO_NODE) {
        IREE_RETURN_IF_ERROR(
            loom_fact_refinement_append_carrier_definition_mapping(
                plan, carrier, source_value, definition_node));
      }
    }
    const loom_use_t* carrier_use = NULL;
    loom_value_for_each_use(carrier_value, carrier_use) {
      loom_op_t* user = loom_use_user_op(*carrier_use);
      const uint16_t operand_index = loom_use_operand_index(*carrier_use);
      if (!user || iree_any_bit_set(user->flags, LOOM_OP_FLAG_DEAD) ||
          loom_fact_refinement_is_candidate(plan, user) ||
          !loom_fact_refinement_allows_refined_carrier(module, user,
                                                       operand_index)) {
        continue;
      }
      iree_host_size_t node = LOOM_FACT_REFINEMENT_NO_NODE;
      IREE_RETURN_IF_ERROR(
          loom_fact_refinement_resolve_before_op(source, user, &node));
      if (node != LOOM_FACT_REFINEMENT_NO_NODE) {
        IREE_RETURN_IF_ERROR(loom_fact_refinement_append_carrier_mapping(
            plan, carrier, user, operand_index, source_value, node));
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_fact_refinement_plan_references(
    loom_fact_refinement_plan_t* plan) {
  for (iree_host_size_t source_start = 0; source_start < plan->node_count;) {
    iree_host_size_t source_end = source_start + 1;
    while (source_end < plan->node_count &&
           plan->nodes[source_end].source == plan->nodes[source_start].source) {
      ++source_end;
    }
    loom_fact_refinement_source_t source = {
        .plan = plan,
        .node_start = source_start,
        .node_count = source_end - source_start,
    };
    for (iree_host_size_t i = source_start; i < source_end; ++i) {
      IREE_RETURN_IF_ERROR(loom_fact_refinement_resolve_before_op(
          &source, plan->nodes[i].anchor, &plan->nodes[i].parent));
    }
    IREE_RETURN_IF_ERROR(loom_fact_refinement_plan_source_references(&source));
    source_start = source_end;
  }
  return iree_ok_status();
}

static iree_status_t loom_fact_refinement_add_cfg_dominance(
    void* user_data, const loom_cfg_graph_t* graph) {
  loom_fact_refinement_plan_t* plan = (loom_fact_refinement_plan_t*)user_data;
  const loom_value_fact_cfg_region_t* structure =
      loom_value_fact_table_lookup_cfg_region(plan->rewriter->fact_table,
                                              graph->region);
  return structure ? loom_dominance_info_add_cfg_graph(&plan->dominance, graph,
                                                       &structure->dominance)
                   : iree_ok_status();
}

static iree_status_t loom_fact_refinement_prepare_dominance(
    loom_fact_refinement_plan_t* plan) {
  plan->dominance = (loom_dominance_info_t){
      .module = plan->rewriter->module,
      .arena = plan->arena,
  };
  return loom_value_fact_table_enumerate_cfg_graphs(
      plan->rewriter->fact_table,
      (loom_value_fact_cfg_graph_callback_t){
          .user_data = plan,
          .fn = loom_fact_refinement_add_cfg_dominance,
      });
}

static iree_status_t loom_fact_refinement_materialize_node(
    loom_fact_refinement_plan_t* plan, iree_host_size_t node_index) {
  loom_fact_refinement_node_t* node = &plan->nodes[node_index];
  if (node->alias != LOOM_VALUE_ID_INVALID) {
    return iree_ok_status();
  }
  if (node->parent != LOOM_FACT_REFINEMENT_NO_NODE) {
    const loom_fact_refinement_node_t* parent = &plan->nodes[node->parent];
    bool predicates_equal = node->predicate_count == parent->predicate_count;
    for (iree_host_size_t i = 0; predicates_equal && i < node->predicate_count;
         ++i) {
      predicates_equal = loom_fact_refinement_predicate_equal(
          &node->predicates[i], &parent->predicates[i]);
    }
    if (predicates_equal) {
      node->alias = parent->alias;
      node->alias_op = parent->alias_op;
      return iree_ok_status();
    }
  }
  const loom_value_id_t input = node->parent == LOOM_FACT_REFINEMENT_NO_NODE
                                    ? node->source
                                    : plan->nodes[node->parent].alias;
  IREE_ASSERT(input != LOOM_VALUE_ID_INVALID);

  loom_predicate_t* predicates = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(plan->arena, node->predicate_count,
                                sizeof(*predicates), (void**)&predicates));
  memcpy(predicates, node->predicates,
         node->predicate_count * sizeof(*predicates));
  for (iree_host_size_t i = 0; i < node->predicate_count; ++i) {
    for (uint8_t j = 0; j < predicates[i].arg_count; ++j) {
      if (predicates[i].arg_tags[j] == LOOM_PRED_ARG_VALUE &&
          predicates[i].args[j] == (int64_t)node->source) {
        predicates[i].args[j] = input;
      }
    }
  }

  loom_rewriter_t* rewriter = plan->rewriter;
  const loom_builder_ip_t saved_ip = loom_builder_save(&rewriter->builder);
  loom_builder_set_after(&rewriter->builder, node->anchor);
  const loom_type_t result_type =
      loom_module_value_type(rewriter->module, node->source);
  iree_status_t status = node->provider->materialize(
      &rewriter->builder, input, predicates, node->predicate_count, result_type,
      node->anchor->location, &node->alias);
  loom_builder_restore(&rewriter->builder, saved_ip);
  if (!iree_status_is_ok(status)) {
    return status;
  }
  node->alias_op =
      loom_value_def_op(loom_module_value(rewriter->module, node->alias));
  IREE_RETURN_IF_ERROR(loom_rewriter_try_set_derived_value_name(
      rewriter, node->source, node->alias, IREE_SV("refined")));
  return iree_ok_status();
}

static void loom_fact_refinement_require_node(
    const loom_fact_refinement_plan_t* plan,
    loom_fact_refinement_node_state_t* states, iree_host_size_t node) {
  while (node != LOOM_FACT_REFINEMENT_NO_NODE &&
         states[node] == LOOM_FACT_REFINEMENT_NODE_UNUSED) {
    states[node] = LOOM_FACT_REFINEMENT_NODE_REQUIRED;
    node = plan->nodes[node].parent;
  }
}

static iree_status_t loom_fact_refinement_materialize_nodes(
    loom_fact_refinement_plan_t* plan) {
  loom_fact_refinement_node_state_t* states = NULL;
  iree_host_size_t* stack = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      plan->arena, plan->node_count, sizeof(*states), (void**)&states));
  memset(states, 0, plan->node_count * sizeof(*states));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      plan->arena, plan->node_count, sizeof(*stack), (void**)&stack));

  for (iree_host_size_t i = 0; i < plan->node_count; ++i) {
    if (plan->nodes[i].alias != LOOM_VALUE_ID_INVALID) {
      states[i] = LOOM_FACT_REFINEMENT_NODE_AVAILABLE;
    }
  }
  for (iree_host_size_t i = 0; i < plan->operand_count; ++i) {
    loom_fact_refinement_require_node(plan, states, plan->operands[i].node);
  }
  for (iree_host_size_t i = 0; i < plan->attribute_count; ++i) {
    loom_fact_refinement_require_node(plan, states, plan->attributes[i].node);
  }
  for (iree_host_size_t i = 0; i < plan->carrier_mapping_count; ++i) {
    loom_fact_refinement_require_node(plan, states,
                                      plan->carrier_mappings[i].node);
  }
  for (iree_host_size_t i = 0; i < plan->carrier_definition_mapping_count;
       ++i) {
    loom_fact_refinement_require_node(
        plan, states, plan->carrier_definition_mappings[i].node);
  }

  for (iree_host_size_t i = 0; i < plan->node_count; ++i) {
    if (states[i] != LOOM_FACT_REFINEMENT_NODE_REQUIRED) {
      continue;
    }
    iree_host_size_t stack_count = 0;
    iree_host_size_t current = i;
    while (current != LOOM_FACT_REFINEMENT_NO_NODE &&
           states[current] == LOOM_FACT_REFINEMENT_NODE_REQUIRED) {
      states[current] = LOOM_FACT_REFINEMENT_NODE_VISITING;
      stack[stack_count++] = current;
      current = plan->nodes[current].parent;
    }
    IREE_ASSERT(current == LOOM_FACT_REFINEMENT_NO_NODE ||
                    states[current] == LOOM_FACT_REFINEMENT_NODE_AVAILABLE,
                "fact-refinement dominance parents must be acyclic");
    while (stack_count > 0) {
      const iree_host_size_t node_index = stack[--stack_count];
      IREE_RETURN_IF_ERROR(
          loom_fact_refinement_materialize_node(plan, node_index));
      states[node_index] = LOOM_FACT_REFINEMENT_NODE_AVAILABLE;
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_fact_refinement_remap_carrier_type(
    loom_fact_refinement_plan_t* plan, loom_type_t source_type,
    const loom_fact_refinement_carrier_mapping_t* mappings,
    iree_host_size_t mapping_count, loom_type_t* out_target_type) {
  loom_type_t target_type = source_type;
  for (iree_host_size_t i = 0; i < mapping_count; ++i) {
    const loom_fact_refinement_carrier_mapping_t* mapping = &mappings[i];
    const loom_value_id_t alias = plan->nodes[mapping->node].alias;
    loom_value_replacement_t replacement;
    loom_value_replacement_initialize(plan->rewriter->module, mapping->source,
                                      alias, &replacement);
    loom_type_t remapped_type = target_type;
    bool changed = false;
    iree_status_t status = loom_value_replacement_type(
        &replacement, target_type, &remapped_type, &changed);
    loom_value_replacement_deinitialize(&replacement);
    IREE_RETURN_IF_ERROR(status);
    if (changed) {
      target_type = remapped_type;
    }
  }
  *out_target_type = target_type;
  return iree_ok_status();
}

static iree_status_t loom_fact_refinement_apply_carrier_definition_types(
    loom_fact_refinement_plan_t* plan) {
  if (plan->carrier_definition_mapping_count == 0) {
    return iree_ok_status();
  }
  loom_fact_refinement_sort_carrier_definition_mappings(
      plan->carrier_definition_mappings,
      plan->carrier_definition_mapping_count);
  for (iree_host_size_t i = 0; i < plan->carrier_definition_mapping_count;) {
    const iree_host_size_t mapping_start = i;
    const loom_value_id_t carrier =
        plan->carrier_definition_mappings[i].carrier;
    do {
      ++i;
    } while (i < plan->carrier_definition_mapping_count &&
             plan->carrier_definition_mappings[i].carrier == carrier);

    const loom_type_t source_type =
        loom_module_value_type(plan->rewriter->module, carrier);
    loom_type_t target_type = source_type;
    for (iree_host_size_t j = mapping_start; j < i; ++j) {
      const loom_fact_refinement_carrier_definition_mapping_t* mapping =
          &plan->carrier_definition_mappings[j];
      const loom_value_id_t alias = plan->nodes[mapping->node].alias;
      loom_value_replacement_t replacement;
      loom_value_replacement_initialize(plan->rewriter->module, mapping->source,
                                        alias, &replacement);
      loom_type_t remapped_type = target_type;
      bool changed = false;
      iree_status_t status = loom_value_replacement_type(
          &replacement, target_type, &remapped_type, &changed);
      loom_value_replacement_deinitialize(&replacement);
      IREE_RETURN_IF_ERROR(status);
      if (changed) {
        target_type = remapped_type;
      }
    }
    if (!loom_type_equal(source_type, target_type)) {
      IREE_RETURN_IF_ERROR(
          loom_rewriter_set_value_type(plan->rewriter, carrier, target_type));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_fact_refinement_build_carrier_uses(
    loom_fact_refinement_plan_t* plan,
    loom_fact_refinement_carrier_use_t** out_uses,
    iree_host_size_t* out_use_count) {
  *out_uses = NULL;
  *out_use_count = 0;
  if (plan->carrier_mapping_count == 0) {
    return iree_ok_status();
  }
  loom_fact_refinement_sort_carrier_mappings(plan->carrier_mappings,
                                             plan->carrier_mapping_count);
  loom_fact_refinement_carrier_use_t* uses = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      plan->arena, plan->carrier_mapping_count, sizeof(*uses), (void**)&uses));

  iree_host_size_t use_count = 0;
  for (iree_host_size_t i = 0; i < plan->carrier_mapping_count;) {
    const iree_host_size_t mapping_start = i;
    const loom_value_id_t carrier = plan->carrier_mappings[i].carrier;
    loom_op_t* user = plan->carrier_mappings[i].user;
    const uint16_t operand_index = plan->carrier_mappings[i].operand_index;
    do {
      ++i;
    } while (i < plan->carrier_mapping_count &&
             plan->carrier_mappings[i].carrier == carrier &&
             plan->carrier_mappings[i].user == user &&
             plan->carrier_mappings[i].operand_index == operand_index);

    const loom_type_t source_type =
        loom_module_value_type(plan->rewriter->module, carrier);
    loom_type_t target_type = source_type;
    IREE_RETURN_IF_ERROR(loom_fact_refinement_remap_carrier_type(
        plan, source_type, plan->carrier_mappings + mapping_start,
        i - mapping_start, &target_type));
    if (loom_type_equal(source_type, target_type)) {
      continue;
    }
    const loom_type_id_t target_type_id =
        loom_module_lookup_type_id(plan->rewriter->module, target_type);
    IREE_ASSERT(target_type_id != LOOM_TYPE_ID_INVALID,
                "replacement must publish a canonical target type");
    uses[use_count++] = (loom_fact_refinement_carrier_use_t){
        .carrier = carrier,
        .user = user,
        .operand_index = operand_index,
        .target_type = target_type,
        .target_type_id = target_type_id,
        .mapping_start = mapping_start,
        .mapping_count = i - mapping_start,
    };
  }
  loom_fact_refinement_sort_carrier_uses(uses, use_count);
  *out_uses = uses;
  *out_use_count = use_count;
  return iree_ok_status();
}

static void loom_fact_refinement_set_carrier_insertion_point(
    loom_fact_refinement_plan_t* plan, loom_value_id_t carrier,
    const loom_fact_refinement_carrier_use_t* use) {
  loom_module_t* module = plan->rewriter->module;
  const loom_value_t* carrier_value = loom_module_value(module, carrier);
  loom_op_t* insertion_anchor = NULL;
  loom_block_t* insertion_block = NULL;
  if (loom_value_is_block_arg(carrier_value)) {
    insertion_block = loom_value_def_block(carrier_value);
  } else {
    insertion_anchor = loom_value_def_op(carrier_value);
  }

  for (iree_host_size_t i = 0; i < use->mapping_count; ++i) {
    const loom_fact_refinement_carrier_mapping_t* mapping =
        &plan->carrier_mappings[use->mapping_start + i];
    loom_op_t* alias_op = plan->nodes[mapping->node].alias_op;
    if (insertion_anchor) {
      if (loom_dominates_op(&plan->dominance, insertion_anchor, alias_op)) {
        insertion_anchor = alias_op;
      } else {
        IREE_ASSERT(
            loom_dominates_op(&plan->dominance, alias_op, insertion_anchor));
      }
    } else if (loom_dominates_value(&plan->dominance, carrier, alias_op)) {
      insertion_anchor = alias_op;
      insertion_block = NULL;
    }
  }

  if (insertion_anchor) {
    loom_builder_set_after(&plan->rewriter->builder, insertion_anchor);
  } else if (insertion_block->first_op) {
    loom_builder_set_before(&plan->rewriter->builder,
                            insertion_block->first_op);
  } else {
    loom_builder_set_block(&plan->rewriter->builder, insertion_block);
  }
}

static iree_status_t loom_fact_refinement_apply_carriers(
    loom_fact_refinement_plan_t* plan) {
  loom_fact_refinement_carrier_use_t* uses = NULL;
  iree_host_size_t use_count = 0;
  // Definition and use mappings name the original embedded values. Compute
  // both target types before publishing definitions, which may replace those
  // identities with an earlier refinement than a later use requires.
  IREE_RETURN_IF_ERROR(
      loom_fact_refinement_build_carrier_uses(plan, &uses, &use_count));
  IREE_RETURN_IF_ERROR(
      loom_fact_refinement_apply_carrier_definition_types(plan));
  loom_module_t* module = plan->rewriter->module;
  for (iree_host_size_t i = 0; i < use_count;) {
    const iree_host_size_t use_start = i;
    const loom_value_id_t carrier = uses[i].carrier;
    const loom_type_id_t target_type_id = uses[i].target_type_id;
    do {
      ++i;
    } while (i < use_count && uses[i].carrier == carrier &&
             uses[i].target_type_id == target_type_id);

    const loom_type_t source_type = loom_module_value_type(module, carrier);
    const loom_type_t target_type = uses[use_start].target_type;
    if (loom_type_equal(source_type, target_type)) {
      continue;
    }
    const loom_fact_refinement_carrier_provider_t* provider =
        loom_fact_refinement_find_carrier_provider(plan->policy, source_type,
                                                   target_type);
    if (!provider) {
      continue;
    }
    const loom_builder_ip_t saved_ip =
        loom_builder_save(&plan->rewriter->builder);
    loom_fact_refinement_set_carrier_insertion_point(plan, carrier,
                                                     &uses[use_start]);
    loom_value_id_t alias = LOOM_VALUE_ID_INVALID;
    iree_status_t status =
        provider->materialize(&plan->rewriter->builder, carrier, target_type,
                              uses[use_start].user->location, &alias);
    loom_builder_restore(&plan->rewriter->builder, saved_ip);
    IREE_RETURN_IF_ERROR(status);
    IREE_RETURN_IF_ERROR(loom_rewriter_try_set_derived_value_name(
        plan->rewriter, carrier, alias, IREE_SV("refined")));
    for (iree_host_size_t j = use_start; j < i; ++j) {
      IREE_RETURN_IF_ERROR(loom_fact_refinement_append_operand(
          plan, uses[j].user, uses[j].operand_index,
          LOOM_FACT_REFINEMENT_NO_NODE, alias));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_fact_refinement_apply_attributes(
    loom_fact_refinement_plan_t* plan) {
  loom_fact_refinement_sort_attributes(plan->attributes, plan->attribute_count);
  for (iree_host_size_t i = 0; i < plan->attribute_count;) {
    loom_op_t* op = plan->attributes[i].op;
    const uint8_t attribute_index = plan->attributes[i].attribute_index;
    loom_attribute_t attribute = loom_op_const_attrs(op)[attribute_index];
    do {
      const loom_fact_refinement_attribute_t* mapping = &plan->attributes[i++];
      const loom_value_id_t alias = plan->nodes[mapping->node].alias;
      loom_value_replacement_t replacement;
      loom_value_replacement_initialize(plan->rewriter->module, mapping->source,
                                        alias, &replacement);
      loom_attribute_t remapped_attribute = attribute;
      bool changed = false;
      iree_status_t status = loom_value_replacement_attribute(
          &replacement, attribute, &remapped_attribute, &changed);
      loom_value_replacement_deinitialize(&replacement);
      IREE_RETURN_IF_ERROR(status);
      if (changed) {
        attribute = remapped_attribute;
      }
    } while (i < plan->attribute_count && plan->attributes[i].op == op &&
             plan->attributes[i].attribute_index == attribute_index);
    if (!loom_attribute_equal(&loom_op_const_attrs(op)[attribute_index],
                              &attribute)) {
      IREE_RETURN_IF_ERROR(loom_rewriter_set_attr(plan->rewriter, op,
                                                  attribute_index, attribute));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_fact_refinement_apply_operands(
    loom_fact_refinement_plan_t* plan) {
  for (iree_host_size_t i = 0; i < plan->operand_count; ++i) {
    if (plan->operands[i].node != LOOM_FACT_REFINEMENT_NO_NODE) {
      plan->operands[i].value = plan->nodes[plan->operands[i].node].alias;
    }
  }
  loom_fact_refinement_sort_operands(plan->operands, plan->operand_count);
  loom_value_id_t* operand_values = NULL;
  iree_host_size_t operand_capacity = 0;
  for (iree_host_size_t i = 0; i < plan->operand_count;) {
    loom_op_t* op = plan->operands[i].op;
    if (op->operand_count > operand_capacity) {
      IREE_RETURN_IF_ERROR(iree_arena_grow_array(
          plan->arena, /*existing_count=*/0, op->operand_count,
          sizeof(*operand_values), &operand_capacity, (void**)&operand_values));
    }
    memcpy(operand_values, loom_op_const_operands(op),
           op->operand_count * sizeof(*operand_values));
    do {
      const loom_fact_refinement_operand_t* update = &plan->operands[i++];
      operand_values[update->operand_index] = update->value;
    } while (i < plan->operand_count && plan->operands[i].op == op);
    IREE_RETURN_IF_ERROR(
        loom_rewriter_set_operands(plan->rewriter, op, operand_values));
  }
  return iree_ok_status();
}

iree_status_t loom_fact_refinement_preserve_pending(
    loom_rewriter_t* rewriter,
    const loom_fact_refinement_policy_t* refinement_policy) {
  if (!refinement_policy || !loom_value_fact_table_has_pending_exact_relations(
                                rewriter->fact_table)) {
    return iree_ok_status();
  }
  loom_op_t* const* pending_ops = NULL;
  iree_host_size_t pending_count = 0;
  loom_value_fact_table_pending_exact_relations(rewriter->fact_table,
                                                &pending_ops, &pending_count);
  if (pending_count == 0) {
    return iree_ok_status();
  }

  iree_arena_allocator_t arena;
  iree_arena_initialize(rewriter->arena->block_pool, &arena);
  loom_op_t** candidate_ops = NULL;
  iree_status_t status = iree_arena_allocate_array(
      &arena, pending_count, sizeof(*candidate_ops), (void**)&candidate_ops);
  if (iree_status_is_ok(status)) {
    memcpy(candidate_ops, pending_ops, pending_count * sizeof(*candidate_ops));
    loom_value_fact_table_clear_pending_exact_relations(rewriter->fact_table);
    loom_fact_refinement_sort_op_pointers(candidate_ops, pending_count);
    iree_host_size_t candidate_count = 0;
    for (iree_host_size_t i = 0; i < pending_count; ++i) {
      if (candidate_count == 0 ||
          candidate_ops[i] != candidate_ops[candidate_count - 1]) {
        candidate_ops[candidate_count++] = candidate_ops[i];
      }
    }

    iree_host_size_t exact_candidate_count = 0;
    for (iree_host_size_t i = 0; i < candidate_count; ++i) {
      if (loom_fact_refinement_candidate_is_still_exact(rewriter->fact_table,
                                                        candidate_ops[i])) {
        candidate_ops[exact_candidate_count++] = candidate_ops[i];
      }
    }
    candidate_count = exact_candidate_count;

    loom_fact_refinement_plan_t plan = {
        .rewriter = rewriter,
        .policy = refinement_policy,
        .arena = &arena,
        .candidate_ops = candidate_ops,
        .candidate_count = candidate_count,
    };
    status =
        loom_fact_refinement_build_nodes(&plan, candidate_ops, candidate_count);
    if (iree_status_is_ok(status) && plan.node_count > 0) {
      status = loom_fact_refinement_prepare_dominance(&plan);
    }
    if (iree_status_is_ok(status) && plan.node_count > 0) {
      status = loom_fact_refinement_plan_references(&plan);
    }
    if (iree_status_is_ok(status) && plan.node_count > 0) {
      status = loom_fact_refinement_materialize_nodes(&plan);
    }
    if (iree_status_is_ok(status) && plan.node_count > 0) {
      status = loom_fact_refinement_apply_carriers(&plan);
    }
    if (iree_status_is_ok(status) && plan.node_count > 0) {
      status = loom_fact_refinement_apply_attributes(&plan);
    }
    if (iree_status_is_ok(status) && plan.node_count > 0) {
      status = loom_fact_refinement_apply_operands(&plan);
    }
  }
  iree_arena_deinitialize(&arena);
  return status;
}

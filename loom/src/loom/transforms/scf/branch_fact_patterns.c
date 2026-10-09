// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/scf/branch_fact_patterns.h"

#include <string.h>

#include "loom/analysis/condition_facts.h"
#include "loom/analysis/symbolic_expr.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/rewrite/rewriter.h"
#include "loom/transforms/cleanup/patterns.h"
#include "loom/util/walk.h"

//===----------------------------------------------------------------------===//
// Branch-edge fact materialization
//===----------------------------------------------------------------------===//

// Opportunistic branch-edge materialization has a local rewrite budget, not an
// IR legality limit. Over-budget facts remain true but unmaterialized; later
// canonicalization/fact passes still see the original program semantics.
#define LOOM_BRANCH_FACTS_EDGE_RELATION_CAPACITY 32
#define LOOM_BRANCH_FACTS_EDGE_CANDIDATE_CAPACITY 16
#define LOOM_BRANCH_FACTS_EDGE_PREDICATE_CAPACITY 64

typedef uint8_t loom_branch_facts_edge_assume_kind_t;

enum loom_branch_facts_edge_assume_kind_e {
  LOOM_BRANCH_FACTS_EDGE_ASSUME_PREDICATES = 0,
  LOOM_BRANCH_FACTS_EDGE_ASSUME_SEMANTIC = 1,
};

typedef struct loom_branch_facts_edge_assume_candidate_t {
  // Value used before the branch whose edge-local replacement dominates the
  // region body.
  loom_value_id_t source;
  // Source type, reused for the assume result.
  loom_type_t type;
  // Refinement representation selected for this candidate.
  loom_branch_facts_edge_assume_kind_t kind;
  // Refinement details interpreted according to kind.
  union {
    // Scalar or index predicate refinement.
    struct {
      // True when the source uses index.assume instead of scalar.assume.
      bool uses_index_assume;
      // First predicate in the flat predicate storage.
      uint16_t predicate_offset;
      // Number of predicates attached to the assume.
      uint16_t predicate_count;
    } predicates;
    // Dialect-owned semantic refinement.
    struct {
      // Descriptor whose callback materializes the fact identity.
      const loom_condition_refinement_descriptor_t* descriptor;
      // Query operation that implies the refinement.
      const loom_op_t* condition_op;
      // Truth value assumed for condition_op on this edge.
      bool assumed_truth;
    } semantic;
  } refinement;
  // True when the source has an operand use or rewritable type reference inside
  // the target region.
  bool has_region_use;
  // Value consumed by the materialized fact identity. This differs from source
  // when multiple refinements for one value are chained on the same edge.
  loom_value_id_t materialized_input;
  // Region-entry assume op built for this source.
  loom_op_t* assume_op;
  // Result of assume_op that replaces region-local source uses.
  loom_value_id_t replacement;
} loom_branch_facts_edge_assume_candidate_t;

typedef struct loom_branch_facts_edge_assume_set_t {
  // Candidate sources requiring region-entry assumes.
  loom_branch_facts_edge_assume_candidate_t
      candidates[LOOM_BRANCH_FACTS_EDGE_CANDIDATE_CAPACITY];
  // Number of populated candidates.
  uint16_t candidate_count;
  // Flat predicate storage referenced by each candidate.
  loom_predicate_t predicates[LOOM_BRANCH_FACTS_EDGE_PREDICATE_CAPACITY];
  // Number of populated predicates.
  uint16_t predicate_count;
} loom_branch_facts_edge_assume_set_t;

static bool loom_branch_facts_type_uses_index_assume(loom_type_t type,
                                                     bool* out_uses_index) {
  if (!loom_type_is_scalar(type)) {
    return false;
  }
  loom_scalar_type_t scalar_type = loom_type_element_type(type);
  if (scalar_type == LOOM_SCALAR_TYPE_INDEX ||
      scalar_type == LOOM_SCALAR_TYPE_OFFSET) {
    *out_uses_index = true;
    return true;
  }
  if (loom_scalar_type_is_integer(scalar_type)) {
    *out_uses_index = false;
    return true;
  }
  return false;
}

static bool loom_branch_facts_predicate_equal(const loom_predicate_t* lhs,
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

static bool loom_branch_facts_value_exact_integer(
    const loom_value_fact_table_t* fact_table, loom_value_id_t value_id) {
  loom_value_facts_t facts = loom_value_fact_table_lookup(fact_table, value_id);
  return loom_value_facts_is_exact(facts) && !loom_value_facts_is_float(facts);
}

static iree_status_t loom_branch_facts_edge_assume_set_append_predicate(
    loom_rewriter_t* rewriter, loom_branch_facts_edge_assume_set_t* assume_set,
    loom_value_id_t source, loom_predicate_t predicate) {
  if (source == LOOM_VALUE_ID_INVALID ||
      source >= rewriter->module->values.count) {
    return iree_ok_status();
  }
  if (loom_branch_facts_value_exact_integer(rewriter->fact_table, source)) {
    return iree_ok_status();
  }

  loom_type_t source_type = loom_module_value_type(rewriter->module, source);
  bool uses_index_assume = false;
  if (!loom_branch_facts_type_uses_index_assume(source_type,
                                                &uses_index_assume)) {
    return iree_ok_status();
  }

  uint16_t candidate_index = 0;
  bool found_candidate = false;
  for (uint16_t i = 0; i < assume_set->candidate_count; ++i) {
    if (assume_set->candidates[i].source == source &&
        assume_set->candidates[i].kind ==
            LOOM_BRANCH_FACTS_EDGE_ASSUME_PREDICATES) {
      candidate_index = i;
      found_candidate = true;
      break;
    }
  }
  if (!found_candidate) {
    if (assume_set->candidate_count >=
        LOOM_BRANCH_FACTS_EDGE_CANDIDATE_CAPACITY) {
      return iree_ok_status();
    }
    candidate_index = assume_set->candidate_count++;
    assume_set->candidates[candidate_index] =
        (loom_branch_facts_edge_assume_candidate_t){
            .source = source,
            .type = source_type,
            .kind = LOOM_BRANCH_FACTS_EDGE_ASSUME_PREDICATES,
            .refinement.predicates =
                {
                    .uses_index_assume = uses_index_assume,
                    .predicate_offset = assume_set->predicate_count,
                    .predicate_count = 0,
                },
            .has_region_use = false,
            .materialized_input = LOOM_VALUE_ID_INVALID,
            .assume_op = NULL,
            .replacement = LOOM_VALUE_ID_INVALID,
        };
  }

  loom_branch_facts_edge_assume_candidate_t* candidate =
      &assume_set->candidates[candidate_index];
  for (uint16_t i = 0; i < candidate->refinement.predicates.predicate_count;
       ++i) {
    const loom_predicate_t* existing =
        &assume_set
             ->predicates[candidate->refinement.predicates.predicate_offset +
                          i];
    if (loom_branch_facts_predicate_equal(existing, &predicate)) {
      return iree_ok_status();
    }
  }
  if (assume_set->predicate_count >=
      LOOM_BRANCH_FACTS_EDGE_PREDICATE_CAPACITY) {
    return iree_ok_status();
  }

  uint16_t insert_index =
      (uint16_t)(candidate->refinement.predicates.predicate_offset +
                 candidate->refinement.predicates.predicate_count);
  if (insert_index < assume_set->predicate_count) {
    memmove(&assume_set->predicates[insert_index + 1],
            &assume_set->predicates[insert_index],
            ((iree_host_size_t)assume_set->predicate_count - insert_index) *
                sizeof(*assume_set->predicates));
    for (uint16_t i = 0; i < assume_set->candidate_count; ++i) {
      if (i != candidate_index &&
          assume_set->candidates[i].kind ==
              LOOM_BRANCH_FACTS_EDGE_ASSUME_PREDICATES &&
          assume_set->candidates[i].refinement.predicates.predicate_offset >=
              insert_index) {
        ++assume_set->candidates[i].refinement.predicates.predicate_offset;
      }
    }
  }
  assume_set->predicates[insert_index] = predicate;
  ++assume_set->predicate_count;
  ++candidate->refinement.predicates.predicate_count;
  return iree_ok_status();
}

static void loom_branch_facts_edge_assume_set_append_semantic_refinement(
    loom_branch_facts_edge_assume_set_t* assume_set,
    const loom_condition_edge_refinement_t* refinement) {
  for (uint16_t i = 0; i < assume_set->candidate_count; ++i) {
    const loom_branch_facts_edge_assume_candidate_t* existing =
        &assume_set->candidates[i];
    if (existing->kind == LOOM_BRANCH_FACTS_EDGE_ASSUME_SEMANTIC &&
        existing->refinement.semantic.condition_op ==
            refinement->condition_op &&
        existing->refinement.semantic.assumed_truth ==
            refinement->assumed_truth) {
      return;
    }
  }
  if (assume_set->candidate_count >=
      LOOM_BRANCH_FACTS_EDGE_CANDIDATE_CAPACITY) {
    return;
  }
  assume_set->candidates[assume_set->candidate_count++] =
      (loom_branch_facts_edge_assume_candidate_t){
          .source = refinement->source,
          .type = (loom_type_t){0},
          .kind = LOOM_BRANCH_FACTS_EDGE_ASSUME_SEMANTIC,
          .refinement.semantic =
              {
                  .descriptor = refinement->descriptor,
                  .condition_op = refinement->condition_op,
                  .assumed_truth = refinement->assumed_truth,
              },
          .has_region_use = false,
          .materialized_input = LOOM_VALUE_ID_INVALID,
          .assume_op = NULL,
          .replacement = LOOM_VALUE_ID_INVALID,
      };
}

static iree_status_t loom_branch_facts_edge_assume_set_append_relation(
    loom_rewriter_t* rewriter, loom_branch_facts_edge_assume_set_t* assume_set,
    const loom_condition_integer_relation_t* relation) {
  if (relation->left.kind == LOOM_CONDITION_INTEGER_OPERAND_VALUE) {
    loom_predicate_t predicate = {0};
    if (loom_condition_integer_relation_make_predicate_for_value(
            relation, rewriter->fact_table, relation->left.value_id,
            &predicate)) {
      IREE_RETURN_IF_ERROR(loom_branch_facts_edge_assume_set_append_predicate(
          rewriter, assume_set, relation->left.value_id, predicate));
    }
  }
  if (relation->right.kind == LOOM_CONDITION_INTEGER_OPERAND_VALUE) {
    loom_predicate_t predicate = {0};
    if (loom_condition_integer_relation_make_predicate_for_value(
            relation, rewriter->fact_table, relation->right.value_id,
            &predicate)) {
      IREE_RETURN_IF_ERROR(loom_branch_facts_edge_assume_set_append_predicate(
          rewriter, assume_set, relation->right.value_id, predicate));
    }
  }
  return iree_ok_status();
}

static bool loom_branch_facts_field_ref_matches_operand(
    const loom_op_vtable_t* vtable, loom_field_ref_t field_ref,
    uint16_t operand_index) {
  if (LOOM_FIELD_REF_CATEGORY(field_ref) != LOOM_FIELD_OPERAND) {
    return false;
  }
  uint8_t field_index = LOOM_FIELD_REF_INDEX(field_ref);
  if (field_index == operand_index) {
    return true;
  }
  if (!vtable || !vtable->operand_descriptors ||
      field_index < vtable->fixed_operand_count) {
    return false;
  }
  const loom_operand_descriptor_t* descriptor =
      &vtable->operand_descriptors[field_index];
  return iree_any_bit_set(descriptor->flags, LOOM_OPERAND_VARIADIC) &&
         operand_index >= field_index;
}

static bool loom_branch_facts_type_constraint_mentions_operand(
    const loom_op_vtable_t* vtable, const loom_constraint_t* constraint,
    uint16_t operand_index) {
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
      break;
    default:
      return false;
  }
  for (uint8_t i = 0; i < constraint->arg_count; ++i) {
    if (loom_branch_facts_field_ref_matches_operand(vtable, constraint->args[i],
                                                    operand_index)) {
      return true;
    }
  }
  return false;
}

static bool loom_branch_facts_op_type_constraints_mention_operand(
    const loom_op_vtable_t* vtable, uint16_t operand_index) {
  if (!vtable) {
    return true;
  }
  if (vtable->constraint_count > 0 && !vtable->constraints) {
    return true;
  }
  for (uint8_t i = vtable->operand_dictionary_count;
       i < vtable->constraint_count; ++i) {
    if (loom_branch_facts_type_constraint_mentions_operand(
            vtable, &vtable->constraints[i], operand_index)) {
      return true;
    }
  }
  return false;
}

static bool loom_branch_facts_value_has_type_sensitive_use(
    const loom_module_t* module, loom_value_id_t value_id) {
  if (value_id == LOOM_VALUE_ID_INVALID || value_id >= module->values.count) {
    return true;
  }
  const loom_value_t* value = loom_module_value(module, value_id);
  const loom_use_t* uses = loom_value_uses(value);
  for (uint32_t i = 0; i < value->use_count; ++i) {
    loom_op_t* user = loom_use_user_op(uses[i]);
    if (!user || iree_any_bit_set(user->flags, LOOM_OP_FLAG_DEAD)) {
      continue;
    }
    const loom_op_vtable_t* vtable = loom_op_vtable(module, user);
    if (!vtable) {
      return true;
    }
    if (iree_any_bit_set(vtable->traits, LOOM_TRAIT_TERMINATOR)) {
      return true;
    }
    if (loom_branch_facts_op_type_constraints_mention_operand(
            vtable, loom_use_operand_index(uses[i]))) {
      return true;
    }
  }
  return false;
}

static bool loom_branch_facts_can_rewrite_edge_type_refs_for_result(
    const loom_module_t* module, const loom_op_t* op,
    loom_value_id_t result_id) {
  if (result_id == LOOM_VALUE_ID_INVALID || result_id >= module->values.count) {
    return false;
  }
  const loom_op_vtable_t* vtable = loom_op_vtable(module, op);
  if (!vtable || !loom_traits_have_refinable_result_type_refs(vtable->traits)) {
    return false;
  }
  return !loom_branch_facts_value_has_type_sensitive_use(module, result_id);
}

static void loom_branch_facts_scan_type_for_edge_uses(
    const loom_module_t* module,
    loom_branch_facts_edge_assume_set_t* assume_set, loom_type_t type) {
  for (uint16_t candidate_index = 0;
       candidate_index < assume_set->candidate_count; ++candidate_index) {
    loom_branch_facts_edge_assume_candidate_t* candidate =
        &assume_set->candidates[candidate_index];
    if (loom_type_references_value(module, type, candidate->source)) {
      candidate->has_region_use = true;
    }
  }
}

typedef struct loom_branch_facts_region_use_scan_t {
  // Rewriter used for module, facts, and candidate construction.
  loom_rewriter_t* rewriter;
  // Module owning the walked region.
  const loom_module_t* module;
  // Edge-local condition facts that may apply to region-used cast values.
  const loom_condition_fact_set_t* condition_facts;
  // Candidate set whose has_region_use flags are being populated.
  loom_branch_facts_edge_assume_set_t* assume_set;
} loom_branch_facts_region_use_scan_t;

static bool loom_branch_facts_index_cast_preserves_value(
    const loom_module_t* module, const loom_op_t* op) {
  loom_value_id_t input = loom_index_cast_input(op);
  loom_value_id_t result = loom_index_cast_result(op);
  if (input == LOOM_VALUE_ID_INVALID || result == LOOM_VALUE_ID_INVALID ||
      input >= module->values.count || result >= module->values.count) {
    return false;
  }
  loom_type_t input_type = loom_module_value_type(module, input);
  loom_type_t result_type = loom_module_value_type(module, result);
  if (!loom_type_is_scalar(input_type) || !loom_type_is_scalar(result_type)) {
    return false;
  }
  loom_scalar_type_t input_scalar_type = loom_type_element_type(input_type);
  loom_scalar_type_t result_scalar_type = loom_type_element_type(result_type);
  if (result_scalar_type == LOOM_SCALAR_TYPE_OFFSET &&
      input_scalar_type != LOOM_SCALAR_TYPE_OFFSET) {
    return false;
  }
  int32_t input_bitwidth = loom_scalar_type_bitwidth(input_scalar_type);
  int32_t result_bitwidth = loom_scalar_type_bitwidth(result_scalar_type);
  return input_bitwidth > 0 && result_bitwidth > 0 &&
         input_bitwidth <= result_bitwidth;
}

static bool loom_branch_facts_value_preserves_condition_value(
    const loom_module_t* module, loom_value_id_t value_id,
    loom_value_id_t condition_value_id) {
  if (!module) {
    return false;
  }
  iree_host_size_t remaining_steps = module->values.count;
  loom_value_id_t current_value = value_id;
  while (remaining_steps-- > 0) {
    if (current_value == condition_value_id) {
      return true;
    }
    if (current_value >= module->values.count) {
      return false;
    }
    const loom_value_t* value = loom_module_value(module, current_value);
    if (loom_value_is_block_arg(value)) {
      return false;
    }
    const loom_op_t* defining_op = loom_value_def_op(value);
    if (!defining_op) {
      return false;
    }
    if (loom_index_cast_isa(defining_op)) {
      if (!loom_branch_facts_index_cast_preserves_value(module, defining_op)) {
        return false;
      }
      current_value = loom_index_cast_input(defining_op);
      continue;
    }
    return false;
  }
  return false;
}

static iree_status_t
loom_branch_facts_edge_assume_set_append_relation_for_value(
    loom_rewriter_t* rewriter, loom_branch_facts_edge_assume_set_t* assume_set,
    const loom_condition_integer_relation_t* relation,
    loom_value_id_t value_id) {
  if (value_id == LOOM_VALUE_ID_INVALID ||
      value_id >= rewriter->module->values.count) {
    return iree_ok_status();
  }

  loom_condition_integer_relation_t mapped_relation = *relation;
  bool matched = false;
  if (relation->left.kind == LOOM_CONDITION_INTEGER_OPERAND_VALUE &&
      value_id != relation->left.value_id &&
      loom_branch_facts_value_preserves_condition_value(
          rewriter->module, value_id, relation->left.value_id)) {
    mapped_relation.left.value_id = value_id;
    matched = true;
  }
  if (relation->right.kind == LOOM_CONDITION_INTEGER_OPERAND_VALUE &&
      value_id != relation->right.value_id &&
      loom_branch_facts_value_preserves_condition_value(
          rewriter->module, value_id, relation->right.value_id)) {
    mapped_relation.right.value_id = value_id;
    matched = true;
  }
  if (!matched) {
    return iree_ok_status();
  }

  loom_predicate_t predicate = {0};
  if (loom_condition_integer_relation_make_predicate_for_value(
          &mapped_relation, rewriter->fact_table, value_id, &predicate)) {
    IREE_RETURN_IF_ERROR(loom_branch_facts_edge_assume_set_append_predicate(
        rewriter, assume_set, value_id, predicate));
  }
  return iree_ok_status();
}

static iree_status_t loom_branch_facts_scan_edge_value_use(
    loom_branch_facts_region_use_scan_t* scan, loom_value_id_t value_id) {
  for (iree_host_size_t i = 0;
       scan->condition_facts &&
       i < scan->condition_facts->integer_relation_count;
       ++i) {
    IREE_RETURN_IF_ERROR(
        loom_branch_facts_edge_assume_set_append_relation_for_value(
            scan->rewriter, scan->assume_set,
            &scan->condition_facts->integer_relations[i], value_id));
  }
  for (uint16_t candidate_index = 0;
       candidate_index < scan->assume_set->candidate_count; ++candidate_index) {
    loom_branch_facts_edge_assume_candidate_t* candidate =
        &scan->assume_set->candidates[candidate_index];
    if (value_id == candidate->source) {
      candidate->has_region_use = true;
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_branch_facts_scan_region_uses(
    void* user_data, loom_op_t* op, const loom_walk_context_t* context,
    loom_walk_result_t* out_result) {
  (void)context;
  *out_result = LOOM_WALK_CONTINUE;
  loom_branch_facts_region_use_scan_t* scan =
      (loom_branch_facts_region_use_scan_t*)user_data;
  if (loom_index_assume_isa(op) || loom_scalar_assume_isa(op)) {
    return iree_ok_status();
  }
  const loom_value_id_t* operands = loom_op_const_operands(op);
  for (uint16_t operand_index = 0; operand_index < op->operand_count;
       ++operand_index) {
    IREE_RETURN_IF_ERROR(
        loom_branch_facts_scan_edge_value_use(scan, operands[operand_index]));
  }
  const loom_value_id_t* results = loom_op_const_results(op);
  for (uint16_t result_index = 0; result_index < op->result_count;
       ++result_index) {
    loom_value_id_t result = results[result_index];
    if (!loom_branch_facts_can_rewrite_edge_type_refs_for_result(scan->module,
                                                                 op, result)) {
      continue;
    }
    loom_type_t result_type = loom_module_value_type(scan->module, result);
    loom_branch_facts_scan_type_for_edge_uses(scan->module, scan->assume_set,
                                              result_type);
  }
  return iree_ok_status();
}

static bool loom_branch_facts_is_edge_assume_op(
    const loom_branch_facts_edge_assume_set_t* assume_set,
    const loom_op_t* op) {
  for (uint16_t i = 0; i < assume_set->candidate_count; ++i) {
    if (assume_set->candidates[i].assume_op == op) {
      return true;
    }
  }
  return false;
}

static bool loom_branch_facts_region_contains_block(
    const loom_region_t* region, const loom_block_t* target_block) {
  if (!region || !target_block) {
    return false;
  }
  const loom_block_t* block = NULL;
  loom_region_for_each_block(region, block) {
    if (block == target_block) {
      return true;
    }
    const loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      loom_region_t** regions = loom_op_regions(op);
      for (uint8_t i = 0; i < op->region_count; ++i) {
        if (loom_branch_facts_region_contains_block(regions[i], target_block)) {
          return true;
        }
      }
    }
  }
  return false;
}

static loom_op_t* loom_branch_facts_edge_assume_insertion_anchor(
    const loom_module_t* module, const loom_region_t* region,
    loom_value_id_t source) {
  if (!module || source == LOOM_VALUE_ID_INVALID ||
      source >= module->values.count) {
    return NULL;
  }
  const loom_value_t* value = loom_module_value(module, source);
  if (loom_value_is_block_arg(value)) {
    return NULL;
  }

  loom_op_t* defining_op = loom_value_def_op(value);
  if (!defining_op || !defining_op->parent_block) {
    return NULL;
  }
  return loom_branch_facts_region_contains_block(region,
                                                 defining_op->parent_block)
             ? defining_op
             : NULL;
}

typedef struct loom_branch_facts_region_replacement_t {
  // Rewriter used for operand updates.
  loom_rewriter_t* rewriter;
  // Candidate replacements to apply.
  loom_branch_facts_edge_assume_set_t* assume_set;
} loom_branch_facts_region_replacement_t;

static iree_status_t loom_branch_facts_rewrite_type_with_edge_assumes(
    loom_rewriter_t* rewriter,
    const loom_branch_facts_edge_assume_set_t* assume_set, loom_type_t type,
    loom_type_t* out_type, bool* out_changed) {
  *out_type = type;
  *out_changed = false;
  for (uint16_t candidate_index = 0;
       candidate_index < assume_set->candidate_count; ++candidate_index) {
    const loom_branch_facts_edge_assume_candidate_t* candidate =
        &assume_set->candidates[candidate_index];
    if (candidate->replacement == LOOM_VALUE_ID_INVALID) {
      continue;
    }
    loom_type_t rewritten_type = *out_type;
    bool candidate_changed = false;
    IREE_RETURN_IF_ERROR(loom_module_replace_type_value_references(
        rewriter->module, *out_type, candidate->materialized_input,
        candidate->replacement, &rewritten_type, &candidate_changed));
    if (candidate_changed) {
      *out_type = rewritten_type;
      *out_changed = true;
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_branch_facts_replace_result_type_refs(
    loom_branch_facts_region_replacement_t* replacement, loom_op_t* op) {
  const loom_value_id_t* results = loom_op_const_results(op);
  for (uint16_t result_index = 0; result_index < op->result_count;
       ++result_index) {
    loom_value_id_t result = results[result_index];
    if (!loom_branch_facts_can_rewrite_edge_type_refs_for_result(
            replacement->rewriter->module, op, result)) {
      continue;
    }
    loom_type_t old_type =
        loom_module_value_type(replacement->rewriter->module, result);
    loom_type_t new_type = old_type;
    bool changed = false;
    IREE_RETURN_IF_ERROR(loom_branch_facts_rewrite_type_with_edge_assumes(
        replacement->rewriter, replacement->assume_set, old_type, &new_type,
        &changed));
    if (changed) {
      IREE_RETURN_IF_ERROR(loom_rewriter_set_value_type(replacement->rewriter,
                                                        result, new_type));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_branch_facts_replace_region_uses(
    void* user_data, loom_op_t* op, const loom_walk_context_t* context,
    loom_walk_result_t* out_result) {
  (void)context;
  *out_result = LOOM_WALK_CONTINUE;
  loom_branch_facts_region_replacement_t* replacement =
      (loom_branch_facts_region_replacement_t*)user_data;
  if (loom_branch_facts_is_edge_assume_op(replacement->assume_set, op) ||
      loom_index_assume_isa(op) || loom_scalar_assume_isa(op)) {
    return iree_ok_status();
  }

  const loom_value_id_t* operands = loom_op_const_operands(op);
  for (uint16_t operand_index = 0; operand_index < op->operand_count;
       ++operand_index) {
    loom_value_id_t operand = operands[operand_index];
    for (uint16_t candidate_count = replacement->assume_set->candidate_count;
         candidate_count > 0; --candidate_count) {
      const loom_branch_facts_edge_assume_candidate_t* candidate =
          &replacement->assume_set->candidates[candidate_count - 1];
      if (operand == candidate->source &&
          candidate->replacement != LOOM_VALUE_ID_INVALID) {
        IREE_RETURN_IF_ERROR(loom_rewriter_set_operand(
            replacement->rewriter, op, operand_index, candidate->replacement));
        break;
      }
    }
  }
  return loom_branch_facts_replace_result_type_refs(replacement, op);
}

static bool loom_branch_facts_candidate_exact_integer(
    loom_rewriter_t* rewriter,
    const loom_branch_facts_edge_assume_candidate_t* candidate,
    int64_t* out_value) {
  return candidate->replacement != LOOM_VALUE_ID_INVALID &&
         loom_value_facts_as_exact_i64(
             loom_rewriter_value_facts(rewriter, candidate->replacement),
             out_value);
}

static bool loom_branch_facts_source_exact_integer(
    loom_rewriter_t* rewriter,
    const loom_branch_facts_edge_assume_set_t* assume_set,
    loom_value_id_t source, int64_t* out_value) {
  for (uint16_t i = assume_set->candidate_count; i > 0; --i) {
    const loom_branch_facts_edge_assume_candidate_t* candidate =
        &assume_set->candidates[i - 1];
    if (candidate->source == source &&
        loom_branch_facts_candidate_exact_integer(rewriter, candidate,
                                                  out_value)) {
      return true;
    }
  }
  return loom_value_facts_as_exact_i64(
      loom_rewriter_value_facts(rewriter, source), out_value);
}

static iree_status_t loom_branch_facts_set_predicates(
    loom_rewriter_t* rewriter,
    loom_branch_facts_edge_assume_candidate_t* candidate,
    const loom_predicate_t* predicates, uint16_t predicate_count) {
  loom_predicate_t* storage = NULL;
  IREE_RETURN_IF_ERROR(loom_builder_copy_predicate_list_attr_storage(
      &rewriter->builder, predicates, predicate_count,
      IREE_SV("branch fact predicates"), &storage));
  return loom_rewriter_set_attr(
      rewriter, candidate->assume_op, 0,
      loom_attr_predicate_list(storage, predicate_count));
}

// Branch candidates describe one simultaneous edge fact set. Rewrite references
// to candidates proven exact as literals before ordinary folding can erase the
// identities that established those path-local values.
static iree_status_t loom_branch_facts_normalize_exact_assumes(
    loom_rewriter_t* rewriter,
    loom_branch_facts_edge_assume_set_t* assume_set) {
  for (uint16_t iteration = 0; iteration <= assume_set->candidate_count;
       ++iteration) {
    bool changed = false;
    for (uint16_t candidate_index = 0;
         candidate_index < assume_set->candidate_count; ++candidate_index) {
      loom_branch_facts_edge_assume_candidate_t* candidate =
          &assume_set->candidates[candidate_index];
      if (candidate->kind != LOOM_BRANCH_FACTS_EDGE_ASSUME_PREDICATES ||
          !candidate->assume_op) {
        continue;
      }
      const loom_attribute_t source =
          loom_op_const_attrs(candidate->assume_op)[0];
      loom_predicate_t predicates[LOOM_BRANCH_FACTS_EDGE_PREDICATE_CAPACITY];
      memcpy(predicates, source.predicate_list,
             source.count * sizeof(*predicates));
      bool candidate_changed = false;
      for (uint16_t predicate_index = 0; predicate_index < source.count;
           ++predicate_index) {
        loom_predicate_t* predicate = &predicates[predicate_index];
        for (uint8_t argument_index = 0; argument_index < predicate->arg_count;
             ++argument_index) {
          if (predicate->arg_tags[argument_index] != LOOM_PRED_ARG_VALUE ||
              predicate->args[argument_index] < 0) {
            continue;
          }
          const loom_value_id_t referenced =
              (loom_value_id_t)predicate->args[argument_index];
          if (argument_index == 0) {
            predicate->args[argument_index] = candidate->materialized_input;
            candidate_changed |= referenced != candidate->materialized_input;
            continue;
          }
          int64_t exact_value = 0;
          if (loom_branch_facts_source_exact_integer(
                  rewriter, assume_set, referenced, &exact_value)) {
            predicate->arg_tags[argument_index] = LOOM_PRED_ARG_CONST;
            predicate->args[argument_index] = exact_value;
            candidate_changed = true;
          }
        }
      }
      if (candidate_changed) {
        IREE_RETURN_IF_ERROR(loom_branch_facts_set_predicates(
            rewriter, candidate, predicates, source.count));
        changed = true;
      }
    }
    if (!changed) {
      break;
    }
  }

  for (uint16_t candidate_index = 0;
       candidate_index < assume_set->candidate_count; ++candidate_index) {
    loom_branch_facts_edge_assume_candidate_t* candidate =
        &assume_set->candidates[candidate_index];
    if (candidate->kind != LOOM_BRANCH_FACTS_EDGE_ASSUME_PREDICATES ||
        !candidate->assume_op) {
      continue;
    }
    int64_t exact_value = 0;
    if (!loom_branch_facts_candidate_exact_integer(rewriter, candidate,
                                                   &exact_value)) {
      continue;
    }
    const loom_predicate_t exact_predicate = {
        .kind = LOOM_PREDICATE_EQ,
        .arg_count = 2,
        .arg_tags = {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_CONST,
                     LOOM_PRED_ARG_NONE},
        .args = {candidate->materialized_input, exact_value, 0},
    };
    const loom_attribute_t predicates =
        loom_op_const_attrs(candidate->assume_op)[0];
    if (predicates.count == 1 &&
        loom_branch_facts_predicate_equal(&predicates.predicate_list[0],
                                          &exact_predicate)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_branch_facts_set_predicates(rewriter, candidate,
                                                          &exact_predicate, 1));
  }
  return iree_ok_status();
}

static iree_status_t loom_branch_facts_materialize_edge_assumes(
    loom_rewriter_t* rewriter, loom_op_t* parent_op, loom_region_t* region,
    const loom_condition_fact_set_t* condition_facts,
    loom_branch_facts_edge_assume_set_t* assume_set, bool* out_changed) {
  *out_changed = false;
  if (!region || region->block_count == 0 || assume_set->candidate_count == 0) {
    return iree_ok_status();
  }

  loom_branch_facts_region_use_scan_t scan = {
      .rewriter = rewriter,
      .module = rewriter->module,
      .condition_facts = condition_facts,
      .assume_set = assume_set,
  };
  loom_walk_result_t walk_result = LOOM_WALK_CONTINUE;
  IREE_RETURN_IF_ERROR(loom_walk_region(
      rewriter->module, region, LOOM_WALK_PRE_ORDER,
      (loom_walk_callback_t){loom_branch_facts_scan_region_uses, &scan},
      &walk_result));

  loom_block_t* entry_block = loom_region_entry_block(region);
  if (!entry_block || !entry_block->first_op) {
    return iree_ok_status();
  }

  iree_status_t status = iree_ok_status();
  loom_builder_ip_t saved_ip = loom_builder_save(&rewriter->builder);
  for (uint16_t candidate_index = 0;
       iree_status_is_ok(status) &&
       candidate_index < assume_set->candidate_count;
       ++candidate_index) {
    loom_branch_facts_edge_assume_candidate_t* candidate =
        &assume_set->candidates[candidate_index];
    if (!candidate->has_region_use ||
        (candidate->kind == LOOM_BRANCH_FACTS_EDGE_ASSUME_PREDICATES &&
         candidate->refinement.predicates.predicate_count == 0)) {
      continue;
    }
    loom_value_id_t value = candidate->source;
    loom_op_t* insertion_anchor = NULL;
    for (uint16_t previous_count = candidate_index; previous_count > 0;
         --previous_count) {
      loom_branch_facts_edge_assume_candidate_t* previous =
          &assume_set->candidates[previous_count - 1];
      if (previous->source == candidate->source &&
          previous->replacement != LOOM_VALUE_ID_INVALID) {
        value = previous->replacement;
        insertion_anchor = previous->assume_op;
        break;
      }
    }
    if (insertion_anchor == NULL) {
      insertion_anchor = loom_branch_facts_edge_assume_insertion_anchor(
          rewriter->module, region, candidate->source);
    }
    candidate->materialized_input = value;
    if (insertion_anchor) {
      loom_builder_set_after(&rewriter->builder, insertion_anchor);
    } else {
      loom_builder_set_before(&rewriter->builder, entry_block->first_op);
    }
    if (candidate->kind == LOOM_BRANCH_FACTS_EDGE_ASSUME_PREDICATES) {
      loom_type_t result_type = candidate->type;
      uint16_t predicate_count =
          candidate->refinement.predicates.predicate_count;
      const loom_predicate_t* source_predicates =
          &assume_set
               ->predicates[candidate->refinement.predicates.predicate_offset];
      loom_predicate_t* predicates = NULL;
      status = iree_arena_allocate_array(
          &rewriter->module->arena, predicate_count, sizeof(loom_predicate_t),
          (void**)&predicates);
      if (!iree_status_is_ok(status)) {
        break;
      }
      memcpy(predicates, source_predicates,
             predicate_count * sizeof(loom_predicate_t));
      if (candidate->refinement.predicates.uses_index_assume) {
        status = loom_index_assume_build(
            &rewriter->builder, &value, 1, predicates, predicate_count,
            &result_type, 1, parent_op->location, &candidate->assume_op);
        if (!iree_status_is_ok(status)) {
          break;
        }
        candidate->replacement =
            loom_index_assume_results(candidate->assume_op).values[0];
      } else {
        status = loom_scalar_assume_build(
            &rewriter->builder, &value, 1, predicates, predicate_count,
            &result_type, 1, parent_op->location, &candidate->assume_op);
        if (!iree_status_is_ok(status)) {
          break;
        }
        candidate->replacement =
            loom_scalar_assume_results(candidate->assume_op).values[0];
      }
    } else {
      status = candidate->refinement.semantic.descriptor->materialize(
          rewriter, candidate->refinement.semantic.condition_op, value,
          candidate->refinement.semantic.assumed_truth,
          &candidate->replacement);
      if (!iree_status_is_ok(status)) {
        break;
      }
      IREE_ASSERT(candidate->replacement < rewriter->module->values.count);
      const loom_value_t* replacement_value =
          loom_module_value(rewriter->module, candidate->replacement);
      candidate->assume_op = loom_value_def_op(replacement_value);
      IREE_ASSERT(candidate->assume_op != NULL);
    }
    *out_changed = true;
  }
  loom_builder_restore(&rewriter->builder, saved_ip);
  if (!iree_status_is_ok(status)) {
    return status;
  }

  if (!*out_changed) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_branch_facts_normalize_exact_assumes(rewriter, assume_set));
  loom_branch_facts_region_replacement_t replacement = {
      .rewriter = rewriter,
      .assume_set = assume_set,
  };
  return loom_walk_region(
      rewriter->module, region, LOOM_WALK_PRE_ORDER,
      (loom_walk_callback_t){loom_branch_facts_replace_region_uses,
                             &replacement},
      &walk_result);
}

static iree_status_t loom_branch_facts_materialize_condition_facts_in_region(
    loom_rewriter_t* rewriter, loom_condition_query_t* condition_query,
    loom_op_t* parent_op, loom_region_t* region, loom_value_id_t condition,
    bool assumed_truth, bool* out_changed) {
  *out_changed = false;
  loom_condition_integer_relation_t
      relation_storage[LOOM_BRANCH_FACTS_EDGE_RELATION_CAPACITY];
  loom_condition_fact_set_t condition_facts;
  loom_condition_fact_set_initialize(
      relation_storage, IREE_ARRAYSIZE(relation_storage), &condition_facts);
  loom_condition_edge_refinement_t
      refinement_storage[LOOM_BRANCH_FACTS_EDGE_CANDIDATE_CAPACITY];
  loom_condition_edge_refinement_set_t condition_refinements;
  loom_condition_edge_refinement_set_initialize(
      refinement_storage, IREE_ARRAYSIZE(refinement_storage),
      &condition_refinements);
  bool complete = false;
  IREE_RETURN_IF_ERROR(loom_condition_facts_query_edge(
      condition_query, rewriter->fact_table, condition, assumed_truth,
      &condition_facts, &condition_refinements, &complete));
  if (!complete) {
    return iree_ok_status();
  }
  if (condition_facts.integer_relation_count == 0 &&
      condition_refinements.refinement_count == 0) {
    return iree_ok_status();
  }

  loom_branch_facts_edge_assume_set_t assume_set = {0};
  for (iree_host_size_t relation_index = 0;
       relation_index < condition_facts.integer_relation_count;
       ++relation_index) {
    IREE_RETURN_IF_ERROR(loom_branch_facts_edge_assume_set_append_relation(
        rewriter, &assume_set,
        &condition_facts.integer_relations[relation_index]));
  }
  for (iree_host_size_t refinement_index = 0;
       refinement_index < condition_refinements.refinement_count;
       ++refinement_index) {
    loom_branch_facts_edge_assume_set_append_semantic_refinement(
        &assume_set, &condition_refinements.refinements[refinement_index]);
  }
  return loom_branch_facts_materialize_edge_assumes(
      rewriter, parent_op, region, &condition_facts, &assume_set, out_changed);
}

static iree_status_t loom_branch_facts_materialize_selector_case_fact_in_region(
    loom_rewriter_t* rewriter, loom_op_t* parent_op, loom_region_t* region,
    loom_value_id_t selector, int64_t case_key, bool* out_changed) {
  *out_changed = false;
  loom_predicate_t predicate = {
      .kind = LOOM_PREDICATE_EQ,
      .arg_count = 2,
      .arg_tags = {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_CONST,
                   LOOM_PRED_ARG_NONE},
      .args = {selector, case_key, 0},
  };
  loom_branch_facts_edge_assume_set_t assume_set = {0};
  IREE_RETURN_IF_ERROR(loom_branch_facts_edge_assume_set_append_predicate(
      rewriter, &assume_set, selector, predicate));
  return loom_branch_facts_materialize_edge_assumes(rewriter, parent_op, region,
                                                    /*condition_facts=*/NULL,
                                                    &assume_set, out_changed);
}

static iree_status_t
loom_branch_facts_materialize_selector_default_facts_in_region(
    loom_rewriter_t* rewriter, loom_op_t* parent_op, loom_region_t* region,
    loom_value_id_t selector, loom_attribute_t case_keys, bool* out_changed) {
  *out_changed = false;
  loom_branch_facts_edge_assume_set_t assume_set = {0};
  for (uint16_t i = 0; i < case_keys.count; ++i) {
    loom_predicate_t predicate = {
        .kind = LOOM_PREDICATE_NE,
        .arg_count = 2,
        .arg_tags = {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_CONST,
                     LOOM_PRED_ARG_NONE},
        .args = {selector, case_keys.i64_array[i], 0},
    };
    IREE_RETURN_IF_ERROR(loom_branch_facts_edge_assume_set_append_predicate(
        rewriter, &assume_set, selector, predicate));
  }
  return loom_branch_facts_materialize_edge_assumes(rewriter, parent_op, region,
                                                    /*condition_facts=*/NULL,
                                                    &assume_set, out_changed);
}

static iree_status_t loom_branch_facts_materialize_if_edge_facts(
    loom_rewriter_t* rewriter, loom_condition_query_t* condition_query,
    loom_op_t* op, bool* out_changed) {
  *out_changed = false;
  bool then_changed = false;
  IREE_RETURN_IF_ERROR(loom_branch_facts_materialize_condition_facts_in_region(
      rewriter, condition_query, op, loom_scf_if_then_region(op),
      loom_scf_if_condition(op), true, &then_changed));
  bool else_changed = false;
  loom_region_t* else_region = loom_scf_if_else_region(op);
  if (else_region) {
    IREE_RETURN_IF_ERROR(
        loom_branch_facts_materialize_condition_facts_in_region(
            rewriter, condition_query, op, else_region,
            loom_scf_if_condition(op), false, &else_changed));
  }
  *out_changed = then_changed || else_changed;
  return iree_ok_status();
}

static iree_status_t loom_branch_facts_materialize_switch_edge_facts(
    loom_rewriter_t* rewriter, loom_op_t* op, bool* out_changed) {
  *out_changed = false;
  loom_attribute_t case_keys = loom_scf_switch_case_keys(op);
  if (case_keys.kind != LOOM_ATTR_I64_ARRAY ||
      (case_keys.count > 0 && !case_keys.i64_array)) {
    return iree_ok_status();
  }
  loom_region_slice_t case_regions = loom_scf_switch_case_regions(op);
  if (case_regions.count != case_keys.count) {
    return iree_ok_status();
  }

  loom_value_id_t selector = loom_scf_switch_selector(op);
  for (uint16_t i = 0; i < case_keys.count; ++i) {
    bool case_changed = false;
    IREE_RETURN_IF_ERROR(
        loom_branch_facts_materialize_selector_case_fact_in_region(
            rewriter, op, case_regions.regions[i], selector,
            case_keys.i64_array[i], &case_changed));
    *out_changed |= case_changed;
  }
  bool default_changed = false;
  IREE_RETURN_IF_ERROR(
      loom_branch_facts_materialize_selector_default_facts_in_region(
          rewriter, op, loom_scf_switch_default_region(op), selector, case_keys,
          &default_changed));
  *out_changed |= default_changed;
  return iree_ok_status();
}

static iree_status_t loom_scf_if_branch_fact_pattern(
    const loom_rewrite_pattern_t* pattern, void* context, loom_op_t* op,
    loom_rewriter_t* rewriter, bool* out_changed) {
  (void)pattern;
  loom_cleanup_pattern_context_t* cleanup_context =
      (loom_cleanup_pattern_context_t*)context;
  return loom_branch_facts_materialize_if_edge_facts(
      rewriter, &cleanup_context->symbolic_expression_context->condition_query,
      op, out_changed);
}

static iree_status_t loom_scf_switch_branch_fact_pattern(
    const loom_rewrite_pattern_t* pattern, void* context, loom_op_t* op,
    loom_rewriter_t* rewriter, bool* out_changed) {
  (void)pattern;
  (void)context;
  return loom_branch_facts_materialize_switch_edge_facts(rewriter, op,
                                                         out_changed);
}

static const loom_rewrite_pattern_t kScfBranchFactPatterns[] = {
    {
        .root_kind = LOOM_OP_SCF_IF,
        .match_and_rewrite = loom_scf_if_branch_fact_pattern,
    },
    {
        .root_kind = LOOM_OP_SCF_SWITCH,
        .match_and_rewrite = loom_scf_switch_branch_fact_pattern,
    },
};

const loom_rewrite_pattern_provider_t loom_scf_branch_fact_pattern_provider = {
    .name = IREE_SVL("scf-branch-facts"),
    .patterns = kScfBranchFactPatterns,
    .pattern_count = IREE_ARRAYSIZE(kScfBranchFactPatterns),
};

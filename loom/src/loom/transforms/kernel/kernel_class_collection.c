// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/kernel/kernel_class_collection.h"

#include <string.h>

#include "iree/base/internal/math.h"
#include "loom/decision/predicate.h"
#include "loom/ir/structural_hash.h"
#include "loom/util/adaptive_sort.h"

static iree_status_t loom_kernel_class_allocate_array(
    iree_arena_allocator_t* arena, iree_host_size_t count,
    iree_host_size_t element_size, void** out_values) {
  *out_values = NULL;
  if (count == 0) {
    return iree_ok_status();
  }
  return iree_arena_allocate_array(arena, count, element_size, out_values);
}

typedef struct loom_kernel_class_feature_context_t {
  // Immutable decision with its source outcomes and projected input queries.
  const loom_kernel_class_decision_t* decision;
  // Current launch site's facts in classifier projection order.
  const loom_value_facts_t* facts;
} loom_kernel_class_feature_context_t;

static loom_decision_truth_t loom_kernel_class_evaluate_feature(
    void* user_data, uint32_t feature_ordinal) {
  const loom_kernel_class_feature_context_t* context = user_data;
  const uint32_t outcome = context->decision->feature_outcomes[feature_ordinal];
  if (outcome < LOOM_KERNEL_CLASS_FEATURE_INPUT_BASE) {
    return (loom_decision_truth_t)outcome;
  }
  const loom_kernel_class_input_t* input =
      &context->decision
           ->inputs[outcome - LOOM_KERNEL_CLASS_FEATURE_INPUT_BASE];
  loom_decision_predicate_operand_t operands[3] = {0};
  for (uint8_t i = 0; i < 2; ++i) {
    const uint32_t projection = input->projections[i];
    operands[i].identity = projection;
    operands[i].facts = projection == UINT32_MAX
                            ? loom_value_facts_exact_i64(input->constants[i])
                            : context->facts[projection];
  }
  return loom_decision_predicate_evaluate(LOOM_PREDICATE_EQ, operands);
}

static iree_status_t loom_kernel_class_shape_values(
    loom_value_set_index_t* sets, loom_value_set_id_t values,
    iree_arena_allocator_t* arena, loom_kernel_class_shape_t* out_shape) {
  *out_shape = (loom_kernel_class_shape_t){0};
  loom_value_set_cursor_t cursor;
  loom_value_set_cursor_begin(sets, values, &cursor);
  while (loom_value_set_cursor_next(&cursor) != LOOM_VALUE_ID_INVALID) {
    ++out_shape->count;
  }
  uint32_t* storage = NULL;
  IREE_RETURN_IF_ERROR(loom_kernel_class_allocate_array(
      arena, out_shape->count, sizeof(*storage), (void**)&storage));
  out_shape->projections = storage;
  loom_value_set_cursor_begin(sets, values, &cursor);
  for (uint32_t i = 0; i < out_shape->count; ++i) {
    storage[i] = loom_value_set_cursor_next(&cursor);
  }
  return iree_ok_status();
}

static iree_status_t loom_kernel_class_union_shape(
    loom_value_set_index_t* sets, const loom_kernel_class_shape_t* shape,
    loom_value_set_id_t* inout_values) {
  for (uint32_t i = 0; i < shape->count; ++i) {
    IREE_RETURN_IF_ERROR(loom_value_set_index_add(
        sets, *inout_values, shape->projections[i], inout_values));
  }
  return iree_ok_status();
}

static bool loom_kernel_class_projection_ordinal_less(const uint32_t* lhs,
                                                      const uint32_t* rhs) {
  return *lhs < *rhs;
}

LOOM_DEFINE_ADAPTIVE_SORT(loom_kernel_class_sort_projection_ordinals, uint32_t,
                          loom_kernel_class_projection_ordinal_less)

static loom_value_facts_t loom_kernel_class_evaluate_projection_transfer(
    const loom_kernel_class_projection_t* projection,
    const loom_kernel_class_site_t* site) {
  loom_value_facts_t result = loom_value_facts_exact_i64(projection->constant);
  for (uint16_t i = 0; i < projection->term_count; ++i) {
    const loom_kernel_class_projection_term_t* term = &projection->terms[i];
    loom_value_facts_t contribution = loom_value_fact_table_lookup(
        site->facts, site->argument_values[term->argument_ordinal]);
    if (term->coefficient != 1) {
      const loom_value_facts_t coefficient =
          loom_value_facts_exact_i64(term->coefficient);
      loom_value_facts_muli(&contribution, &coefficient, &contribution);
    }
    loom_value_facts_addi(&result, &contribution, &result);
  }
  return result;
}

static loom_value_facts_t loom_kernel_class_evaluate_projection(
    const loom_kernel_class_projection_t* projection,
    const loom_kernel_class_site_t* site, uint16_t ranged_term_limit) {
  switch ((loom_kernel_class_projection_kind_t)projection->kind) {
    case LOOM_KERNEL_CLASS_PROJECTION_CONSTANT:
      return loom_value_facts_exact_i64(projection->constant);
    case LOOM_KERNEL_CLASS_PROJECTION_STATIC_FACTS:
      return projection->static_facts;
    case LOOM_KERNEL_CLASS_PROJECTION_AFFINE:
      break;
  }

  int64_t exact_result = projection->constant;
  for (uint16_t i = 0; i < projection->term_count; ++i) {
    const loom_kernel_class_projection_term_t* term = &projection->terms[i];
    const loom_value_facts_t input = loom_value_fact_table_lookup(
        site->facts, site->argument_values[term->argument_ordinal]);
    int64_t contribution = 0;
    if (loom_value_facts_is_float(input) || !loom_value_facts_is_exact(input) ||
        !iree_checked_mul_i64(input.range_lo, term->coefficient,
                              &contribution) ||
        !iree_checked_add_i64(exact_result, contribution, &exact_result)) {
      return projection->term_count <= ranged_term_limit
                 ? loom_kernel_class_evaluate_projection_transfer(projection,
                                                                  site)
                 : loom_value_facts_unknown();
    }
  }
  return loom_value_facts_exact_i64(exact_result);
}

static loom_kernel_class_decision_state_t
loom_kernel_class_skipped_decision_state(
    loom_kernel_class_decision_unavailable_reason_t reason) {
  switch (reason) {
    case LOOM_KERNEL_CLASS_DECISION_PATH_CONDITION:
      return LOOM_KERNEL_CLASS_DECISION_SKIPPED_PATH_CONDITION;
    case LOOM_KERNEL_CLASS_DECISION_UNPROJECTABLE_INPUT:
      return LOOM_KERNEL_CLASS_DECISION_SKIPPED_UNPROJECTABLE_INPUT;
    case LOOM_KERNEL_CLASS_DECISION_UNRESOLVED_TARGET:
      return LOOM_KERNEL_CLASS_DECISION_SKIPPED_UNRESOLVED_TARGET;
    case LOOM_KERNEL_CLASS_DECISION_AVAILABLE:
      break;
  }
  IREE_ASSERT_UNREACHABLE("available kernel decision cannot be skipped here");
  IREE_BUILTIN_UNREACHABLE();
}

static iree_status_t loom_kernel_class_unresolved_decision_status(
    uint32_t decision_ordinal, const loom_decision_program_result_t* result) {
  switch ((loom_decision_program_result_kind_t)result->kind) {
    case LOOM_DECISION_PROGRAM_RESULT_HARD_REJECT:
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "kernel decision %u rejects a launch semantic requirement",
          decision_ordinal);
    case LOOM_DECISION_PROGRAM_RESULT_UNRESOLVED:
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "kernel decision %u has no provable provider for a launch",
          decision_ordinal);
    case LOOM_DECISION_PROGRAM_RESULT_AMBIGUOUS:
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "kernel decision %u has ambiguous best-priority providers",
          decision_ordinal);
    case LOOM_DECISION_PROGRAM_RESULT_NO_MATCH:
      return iree_make_status(IREE_STATUS_NOT_FOUND,
                              "kernel decision %u has no matching provider",
                              decision_ordinal);
    case LOOM_DECISION_PROGRAM_RESULT_SELECTED:
      break;
  }
  IREE_ASSERT_UNREACHABLE("selected kernel decision requested failure status");
  IREE_BUILTIN_UNREACHABLE();
}

static bool loom_kernel_class_action_is_publishable(
    const loom_kernel_class_decision_t* decision, uint32_t action_ordinal) {
  IREE_ASSERT(action_ordinal < decision->model->program.choice_count);
  if (decision->generic_result.kind == LOOM_DECISION_PROGRAM_RESULT_SELECTED &&
      decision->generic_result.action_ordinal == action_ordinal) {
    return true;
  }
  const loom_kernel_class_contract_flags_t flags =
      decision->hard_requirement_flags |
      decision->actions[action_ordinal].contract_flags;
  return !iree_any_bit_set(flags,
                           LOOM_KERNEL_CLASS_CONTRACT_FLAG_RESULT_DEPENDENT);
}

// Finalized decisions refine a bounded prefix partition. Unprocessed optional
// decisions retain their generic structural requirements in a residual
// partition. Admission counts the observed intersection, including mandatory
// staged values, instead of charging provider choices before their shapes.
typedef struct loom_kernel_class_shape_binding_t {
  // Shared earlier structural bindings for this prefix class, or NULL.
  const struct loom_kernel_class_shape_binding_t* parent;
  // Newly captured structural facts owned by the collection arena.
  const loom_kernel_class_shape_value_t* values;
  // Number of facts introduced by this node.
  uint32_t count;
  // Number of distinct facts along the complete shared chain.
  uint32_t total_count;
  // Canonical projection membership for the complete shared chain.
  loom_value_set_id_t keys;
} loom_kernel_class_shape_binding_t;

typedef struct loom_kernel_class_residual_node_t {
  // Previous residual class before this structural distinction was added.
  uint32_t parent;
  // Last decision needing the distinction, or UINT32_MAX for the root.
  uint32_t decision;
  // Construction child count; collection visitation stamp after construction.
  uint32_t children;
} loom_kernel_class_residual_node_t;

typedef struct loom_kernel_class_residual_t {
  // Bounded split tree owned by the invocation's residual arena.
  loom_kernel_class_residual_node_t* nodes;
  // Current ancestor of each retained leaf after processed decisions retire.
  uint32_t* active_nodes;
  // Immutable leaf ordinal for each observed launch site.
  loom_decision_class_ordinal_t* site_classes;
  // Number of initialized tree nodes, at most twice the class limit minus one.
  uint32_t node_count;
  // Number of retained leaves, bounded by the class limit.
  uint32_t class_count;
  // Number of distinct active ancestors, retained by the retirement owner.
  uint32_t active_count;
  // First reverse refinement that overflowed, or UINT32_MAX when exact.
  uint32_t overflow_decision;
} loom_kernel_class_residual_t;

typedef struct loom_kernel_class_residual_group_t {
  // Tree node refined by this candidate scalar value.
  uint32_t parent;
  // Normalized exact integer or unknown structural facts.
  loom_value_facts_t facts;
} loom_kernel_class_residual_group_t;

static uint32_t loom_kernel_class_bucket_count(uint32_t capacity) {
  uint32_t count = 1;
  while (count < 2 * capacity) {
    count *= 2;
  }
  return count;
}

static loom_value_facts_t loom_kernel_class_evaluate_shape_projection(
    const loom_kernel_class_classifier_t* classifier,
    const loom_kernel_class_site_t* site, uint32_t projection,
    const loom_kernel_class_collection_options_t* options) {
  loom_value_facts_t facts = loom_kernel_class_evaluate_projection(
      &classifier->projections[projection], site,
      options->ranged_transfer_term_limit);
  return loom_value_facts_is_exact(facts) && !loom_value_facts_is_float(facts)
             ? loom_value_facts_exact_i64(facts.range_lo)
             : loom_value_facts_unknown();
}

static uint32_t loom_kernel_class_hash_shape_fact(uint32_t hash,
                                                  loom_value_facts_t facts) {
  bool exact = loom_value_facts_is_exact(facts);
  hash = loom_structural_hash_mix_u32(hash, exact);
  return exact ? loom_structural_hash_mix_u64(hash, facts.range_lo) : hash;
}

// Build generic requirements in reverse order. Equal requirements are charged
// only at their last use. Stop at the first overflow: earlier residuals cannot
// fit either, and removing that decision exposes the retained exact suffix.
// Splitting a leaf adds at least two children, bounding the tree by 2L-1 nodes.
static iree_status_t loom_kernel_class_residual_build(
    const loom_kernel_class_classifier_t* classifier,
    const loom_kernel_class_site_t* sites, iree_host_size_t site_count,
    const loom_kernel_class_collection_options_t* options,
    iree_arena_allocator_t* arena, iree_arena_allocator_t* scratch,
    loom_value_set_index_t* sets, loom_value_set_id_t covered, bool exact_count,
    loom_kernel_class_residual_t* residual) {
  *residual = (loom_kernel_class_residual_t){.overflow_decision = UINT32_MAX};
  if (!classifier->has_shape_requirements ||
      (!exact_count && site_count <= options->class_limit)) {
    return iree_ok_status();
  }
  uint32_t optional_count = 0;
  for (uint32_t i = 0; i < classifier->decision_count; ++i) {
    optional_count += classifier->decisions[i].unavailable_reason ==
                          LOOM_KERNEL_CLASS_DECISION_AVAILABLE &&
                      classifier->decisions[i].generic_result.kind ==
                          LOOM_DECISION_PROGRAM_RESULT_SELECTED;
  }
  // With one optional decision, the candidate itself is the complete plan.
  if (!exact_count && optional_count <= 1) {
    return iree_ok_status();
  }
  uint32_t* last = NULL;
  IREE_RETURN_IF_ERROR(loom_kernel_class_allocate_array(
      scratch, classifier->projection_count, sizeof(*last), (void**)&last));
  if (classifier->projection_count) {
    memset(last, 0xFF, classifier->projection_count * sizeof(*last));
  }
  bool any = false;
  for (uint32_t i = 0; i < classifier->decision_count; ++i) {
    const loom_kernel_class_decision_t* decision = &classifier->decisions[i];
    if (decision->unavailable_reason != LOOM_KERNEL_CLASS_DECISION_AVAILABLE ||
        decision->generic_result.kind !=
            LOOM_DECISION_PROGRAM_RESULT_SELECTED) {
      continue;
    }
    const loom_kernel_class_shape_t* shape = &decision->residual_shape;
    for (uint32_t j = 0; j < shape->count; ++j) {
      if (!loom_value_set_index_contains(sets, covered,
                                         shape->projections[j])) {
        last[shape->projections[j]] = i;
        any = true;
      }
    }
  }
  if (!any) {
    return iree_ok_status();
  }
  uint32_t capacity = (uint32_t)iree_min(site_count, options->class_limit);
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, 2 * capacity, sizeof(*residual->nodes), (void**)&residual->nodes));
  residual->nodes[0] =
      (loom_kernel_class_residual_node_t){.parent = 0, .decision = UINT32_MAX};
  residual->node_count = 1;
  residual->class_count = 1;
  residual->active_count = 1;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, capacity, sizeof(*residual->active_nodes),
      (void**)&residual->active_nodes));
  residual->active_nodes[0] = 0;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, site_count, sizeof(*residual->site_classes),
      (void**)&residual->site_classes));
  memset(residual->site_classes, 0,
         site_count * sizeof(*residual->site_classes));
  loom_decision_class_ordinal_t* next = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(scratch, site_count,
                                                 sizeof(*next), (void**)&next));
  loom_kernel_class_residual_group_t* groups = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch, capacity, sizeof(*groups), (void**)&groups));
  uint32_t* next_nodes = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch, capacity, sizeof(*next_nodes), (void**)&next_nodes));
  uint32_t bucket_count = loom_kernel_class_bucket_count(capacity);
  uint32_t* buckets = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch, bucket_count, sizeof(*buckets), (void**)&buckets));
  for (uint32_t ordinal = classifier->decision_count; ordinal != 0; --ordinal) {
    uint32_t decision_ordinal = ordinal - 1;
    const loom_kernel_class_decision_t* decision =
        &classifier->decisions[decision_ordinal];
    if (decision->unavailable_reason != LOOM_KERNEL_CLASS_DECISION_AVAILABLE ||
        decision->generic_result.kind !=
            LOOM_DECISION_PROGRAM_RESULT_SELECTED) {
      continue;
    }
    const loom_kernel_class_shape_t* shape = &decision->residual_shape;
    for (uint32_t j = 0; j < shape->count; ++j) {
      uint32_t projection = shape->projections[j];
      if (last[projection] != decision_ordinal) {
        continue;
      }
      memset(buckets, 0xFF, bucket_count * sizeof(*buckets));
      uint32_t group_count = 0;
      for (iree_host_size_t site = 0; site < site_count; ++site) {
        uint32_t parent = residual->active_nodes[residual->site_classes[site]];
        loom_value_facts_t facts = loom_kernel_class_evaluate_shape_projection(
            classifier, &sites[site], projection, options);
        uint32_t hash = loom_kernel_class_hash_shape_fact(
            loom_structural_hash_mix_u32(loom_structural_hash_initialize(),
                                         parent),
            facts);
        uint32_t bucket =
            loom_structural_hash_finalize(hash) & (bucket_count - 1);
        while (buckets[bucket] != UINT32_MAX) {
          const loom_kernel_class_residual_group_t* group =
              &groups[buckets[bucket]];
          if (group->parent == parent &&
              loom_value_facts_equal(group->facts, facts)) {
            break;
          }
          bucket = (bucket + 1) & (bucket_count - 1);
        }
        if (buckets[bucket] == UINT32_MAX) {
          if (group_count == capacity) {
            residual->overflow_decision = decision_ordinal;
            return iree_ok_status();
          }
          buckets[bucket] = group_count;
          groups[group_count++] = (loom_kernel_class_residual_group_t){
              .parent = parent, .facts = facts};
        }
        next[site] = (loom_decision_class_ordinal_t)buckets[bucket];
      }
      for (uint32_t group = 0; group < group_count; ++group) {
        ++residual->nodes[groups[group].parent].children;
      }
      for (uint32_t group = 0; group < group_count; ++group) {
        uint32_t parent = groups[group].parent;
        if (residual->nodes[parent].children == 1) {
          next_nodes[group] = parent;
        } else {
          uint32_t node = residual->node_count++;
          IREE_ASSERT(node < 2 * capacity);
          residual->nodes[node] = (loom_kernel_class_residual_node_t){
              .parent = parent, .decision = decision_ordinal};
          next_nodes[group] = node;
        }
      }
      for (uint32_t group = 0; group < group_count; ++group) {
        residual->nodes[groups[group].parent].children = 0;
      }
      memcpy(residual->site_classes, next, site_count * sizeof(*next));
      memcpy(residual->active_nodes, next_nodes,
             group_count * sizeof(*next_nodes));
      residual->class_count = group_count;
      residual->active_count = group_count;
    }
  }
  return iree_ok_status();
}

// Retire distinctions after their final generic consumer. Sites keep stable
// leaf IDs; only the bounded leaf-to-ancestor map changes.
static void loom_kernel_class_residual_advance(
    loom_kernel_class_residual_t* residual, uint32_t decision) {
  if (residual->overflow_decision != UINT32_MAX) {
    if (decision < residual->overflow_decision) {
      return;
    }
    residual->overflow_decision = UINT32_MAX;
  }
  residual->active_count = 0;
  for (uint32_t i = 0; i < residual->class_count; ++i) {
    uint32_t node = residual->active_nodes[i];
    while (residual->nodes[node].decision == decision) {
      node = residual->nodes[node].parent;
    }
    residual->active_nodes[i] = node;
    if (residual->nodes[node].children != decision + 1) {
      residual->nodes[node].children = decision + 1;
      ++residual->active_count;
    }
  }
}

static bool loom_kernel_class_completion_fits(
    const loom_kernel_class_residual_t* residual,
    const loom_decision_class_ordinal_t* prefix, iree_host_size_t site_count,
    uint32_t prefix_count, uint32_t capacity, uint64_t* buckets,
    uint32_t bucket_count) {
  if (residual->overflow_decision != UINT32_MAX) {
    return false;
  }
  if (residual->active_count <= 1 ||
      prefix_count * residual->active_count <= capacity) {
    return true;
  }
  memset(buckets, 0xFF, bucket_count * sizeof(*buckets));
  uint32_t count = 0;
  for (iree_host_size_t i = 0; i < site_count; ++i) {
    uint64_t key = ((uint64_t)prefix[i] << 32) |
                   residual->active_nodes[residual->site_classes[i]];
    uint32_t hash = loom_structural_hash_finalize(
        loom_structural_hash_mix_u64(loom_structural_hash_initialize(), key));
    uint32_t bucket = hash & (bucket_count - 1);
    while (buckets[bucket] != UINT64_MAX && buckets[bucket] != key) {
      bucket = (bucket + 1) & (bucket_count - 1);
    }
    if (buckets[bucket] == UINT64_MAX) {
      if (count == capacity) {
        return false;
      }
      ++count;
      buckets[bucket] = key;
    }
  }
  return true;
}

typedef struct loom_kernel_class_refinement_t {
  // Newly captured structural values owned by candidate scratch.
  const loom_kernel_class_shape_value_t* values;
  // Selected provider action, or zero for source-only shape refinement.
  uint32_t action;
  // Observed parent/action pair identifying this refinement's shape slice.
  uint32_t pair;
  // Number of new structural values.
  uint32_t count;
  // Current prefix class supplying earlier decisions and structural facts.
  loom_decision_class_ordinal_t parent;
} loom_kernel_class_refinement_t;

typedef struct loom_kernel_class_candidate_t {
  // Observed structural refinements; NULL when parent bindings are unchanged.
  loom_kernel_class_refinement_t* groups;
  // Number of complete prefix classes, including their structural variants.
  uint32_t count;
  // True when the candidate prefix fits the collection's class limit.
  bool fits;
  // True when every selected action can be published from input facts.
  bool publishable;
} loom_kernel_class_candidate_t;

static iree_status_t loom_kernel_class_refine(
    const loom_kernel_class_classifier_t* classifier,
    const loom_kernel_class_site_t* sites,
    const loom_kernel_class_collection_options_t* options,
    const loom_kernel_class_decision_t* decision, uint32_t decision_ordinal,
    const loom_kernel_class_shape_t* fixed_shape, bool evaluate,
    loom_value_set_index_t* sets,
    const loom_kernel_class_shape_binding_t* const* bindings,
    loom_value_facts_t* projection_facts,
    loom_decision_class_partition_t* partition, iree_arena_allocator_t* scratch,
    loom_kernel_class_candidate_t* candidate) {
  *candidate =
      (loom_kernel_class_candidate_t){.fits = true, .publishable = true};
  uint32_t capacity =
      (uint32_t)iree_min(partition->site_count, partition->class_limit);
  if (evaluate) {
    loom_decision_class_partition_begin(partition,
                                        decision->model->program.choice_count);
    for (iree_host_size_t i = 0; i < partition->site_count; ++i) {
      for (uint32_t j = 0; j < decision->projection_count; ++j) {
        uint32_t projection = decision->projection_ordinals[j];
        projection_facts[projection] = loom_kernel_class_evaluate_projection(
            &classifier->projections[projection], &sites[i],
            options->ranged_transfer_term_limit);
      }
      const loom_decision_program_indexed_binding_t binding = {
          .facts = projection_facts,
          .argument_values = decision->argument_values,
          .result_values = decision->result_values};
      const loom_kernel_class_feature_context_t feature_context = {
          .decision = decision,
          .facts = projection_facts,
      };
      const loom_decision_program_feature_evaluator_t feature_evaluator = {
          .fn = loom_kernel_class_evaluate_feature,
          .user_data = (void*)&feature_context,
      };
      loom_decision_program_result_t result;
      uint32_t live_count;
      loom_decision_program_evaluate_indexed(
          &decision->model->program, &binding, feature_evaluator,
          loom_decision_program_predicate_refiner_empty(),
          LOOM_DECISION_PROGRAM_SELECT_PROVEN, NULL, &live_count, &result);
      if (result.kind != LOOM_DECISION_PROGRAM_RESULT_SELECTED) {
        return loom_kernel_class_unresolved_decision_status(decision_ordinal,
                                                            &result);
      }
      if (!loom_kernel_class_action_is_publishable(decision,
                                                   result.action_ordinal)) {
        candidate->publishable = false;
        return iree_ok_status();
      }
      if (!loom_decision_class_partition_record(partition, i,
                                                result.action_ordinal)) {
        candidate->fits = false;
        return iree_ok_status();
      }
    }
  } else {
    partition->candidate_class_count = partition->class_count;
    memcpy(partition->candidate_site_classes, partition->site_classes,
           partition->site_count * sizeof(*partition->site_classes));
    for (uint32_t i = 0; i < partition->class_count; ++i) {
      partition->candidate_parent_classes[i] = (uint16_t)i;
      partition->candidate_outcomes[i] =
          decision ? decision->generic_result.action_ordinal : 0;
    }
  }
  uint32_t pair_count = partition->candidate_class_count;
  if (!classifier->has_shape_requirements) {
    candidate->count = pair_count;
    return iree_ok_status();
  }
  loom_kernel_class_shape_t* deltas = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch, pair_count, sizeof(*deltas), (void**)&deltas));
  uint32_t maximum_values = 0;
  for (uint32_t pair = 0; pair < pair_count; ++pair) {
    uint16_t parent = partition->candidate_parent_classes[pair];
    uint32_t action = partition->candidate_outcomes[pair];
    const loom_kernel_class_shape_t* shape =
        !decision  ? fixed_shape
        : evaluate ? &decision->action_shapes[action]
                   : &decision->residual_shape;
    loom_value_set_id_t keys = bindings[parent] ? bindings[parent]->keys : 0;
    uint32_t count = 0;
    for (uint32_t j = 0; j < shape->count; ++j) {
      count +=
          !loom_value_set_index_contains(sets, keys, shape->projections[j]);
    }
    uint32_t* projections = NULL;
    IREE_RETURN_IF_ERROR(loom_kernel_class_allocate_array(
        scratch, count, sizeof(*projections), (void**)&projections));
    uint32_t next = 0;
    for (uint32_t j = 0; j < shape->count; ++j) {
      uint32_t projection = shape->projections[j];
      if (!loom_value_set_index_contains(sets, keys, projection)) {
        projections[next++] = projection;
      }
    }
    deltas[pair] =
        (loom_kernel_class_shape_t){.projections = projections, .count = count};
    maximum_values = iree_max(maximum_values, count);
  }
  if (!maximum_values) {
    candidate->count = pair_count;
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(scratch, capacity,
                                                 sizeof(*candidate->groups),
                                                 (void**)&candidate->groups));
  uint32_t* direct_classes = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch, pair_count, sizeof(*direct_classes), (void**)&direct_classes));
  memset(direct_classes, 0xFF, pair_count * sizeof(*direct_classes));
  uint32_t bucket_count = loom_kernel_class_bucket_count(capacity);
  uint32_t* buckets = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch, bucket_count, sizeof(*buckets), (void**)&buckets));
  memset(buckets, 0xFF, bucket_count * sizeof(*buckets));
  loom_kernel_class_shape_value_t* values = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch, maximum_values, sizeof(*values), (void**)&values));
  for (iree_host_size_t i = 0; i < partition->site_count; ++i) {
    uint32_t pair = partition->candidate_site_classes[i];
    uint16_t parent = partition->candidate_parent_classes[pair];
    uint32_t action = partition->candidate_outcomes[pair];
    const loom_kernel_class_shape_t* delta = &deltas[pair];
    if (delta->count == 0) {
      if (direct_classes[pair] == UINT32_MAX) {
        if (candidate->count == capacity) {
          candidate->fits = false;
          return iree_ok_status();
        }
        direct_classes[pair] = candidate->count;
        candidate->groups[candidate->count++] =
            (loom_kernel_class_refinement_t){
                .parent = parent, .action = action, .pair = pair};
      }
      partition->candidate_site_classes[i] = (uint16_t)direct_classes[pair];
      continue;
    }
    uint32_t hash =
        loom_structural_hash_mix_u32(loom_structural_hash_initialize(), pair);
    for (uint32_t j = 0; j < delta->count; ++j) {
      uint32_t projection = delta->projections[j];
      values[j] = (loom_kernel_class_shape_value_t){
          .projection = projection,
          .facts = loom_kernel_class_evaluate_shape_projection(
              classifier, &sites[i], projection, options)};
      hash = loom_kernel_class_hash_shape_fact(hash, values[j].facts);
    }
    uint32_t bucket = loom_structural_hash_finalize(hash) & (bucket_count - 1);
    while (buckets[bucket] != UINT32_MAX) {
      const loom_kernel_class_refinement_t* group =
          &candidate->groups[buckets[bucket]];
      bool equal = group->pair == pair;
      for (uint32_t j = 0; equal && j < delta->count; ++j) {
        equal = loom_value_facts_equal(values[j].facts, group->values[j].facts);
      }
      if (equal) {
        break;
      }
      bucket = (bucket + 1) & (bucket_count - 1);
    }
    if (buckets[bucket] == UINT32_MAX) {
      if (candidate->count == capacity) {
        candidate->fits = false;
        return iree_ok_status();
      }
      loom_kernel_class_shape_value_t* retained = NULL;
      IREE_RETURN_IF_ERROR(loom_kernel_class_allocate_array(
          scratch, delta->count, sizeof(*retained), (void**)&retained));
      if (delta->count) {
        memcpy(retained, values, delta->count * sizeof(*retained));
      }
      buckets[bucket] = candidate->count;
      candidate->groups[candidate->count++] =
          (loom_kernel_class_refinement_t){.parent = parent,
                                           .action = action,
                                           .pair = pair,
                                           .values = retained,
                                           .count = delta->count};
    }
    partition->candidate_site_classes[i] = (uint16_t)buckets[bucket];
  }
  return iree_ok_status();
}

static iree_status_t loom_kernel_class_commit(
    const loom_kernel_class_candidate_t* candidate, uint32_t decision_ordinal,
    loom_value_set_index_t* sets,
    loom_kernel_class_shape_binding_t** current_bindings,
    loom_kernel_class_shape_binding_t** next_bindings,
    loom_kernel_class_trace_id_t* current_traces,
    loom_kernel_class_trace_id_t* next_traces,
    loom_kernel_class_trace_t** traces, uint32_t* trace_count,
    iree_host_size_t* trace_capacity,
    loom_decision_class_partition_t* partition, iree_arena_allocator_t* arena,
    iree_arena_allocator_t* scratch) {
  if (!current_bindings) {
    if (candidate->count > UINT32_MAX - *trace_count) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "kernel class trace table exceeds uint32_t");
    }
    if (decision_ordinal != UINT32_MAX) {
      void* storage = *traces;
      IREE_RETURN_IF_ERROR(iree_arena_grow_array(
          arena, *trace_count, *trace_count + candidate->count,
          sizeof(**traces), trace_capacity, &storage));
      *traces = storage;
      for (uint32_t i = 0; i < candidate->count; ++i) {
        (*traces)[*trace_count + i] = (loom_kernel_class_trace_t){
            .parent_trace_id =
                current_traces[partition->candidate_parent_classes[i]],
            .decision_ordinal = decision_ordinal,
            .action_ordinal = partition->candidate_outcomes[i]};
        next_traces[i] = *trace_count + i;
      }
      *trace_count += candidate->count;
    } else {
      for (uint32_t i = 0; i < candidate->count; ++i) {
        next_traces[i] = current_traces[partition->candidate_parent_classes[i]];
      }
    }
    memcpy(current_traces, next_traces,
           candidate->count * sizeof(*current_traces));
    loom_decision_class_partition_commit(partition);
    return iree_ok_status();
  }
  uint32_t bucket_count = loom_kernel_class_bucket_count(candidate->count);
  uint32_t* buckets = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch, bucket_count, sizeof(*buckets), (void**)&buckets));
  memset(buckets, 0xFF, bucket_count * sizeof(*buckets));
  loom_kernel_class_trace_t* new_traces = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch, candidate->count, sizeof(*new_traces), (void**)&new_traces));
  uint32_t added = 0;
  for (uint32_t i = 0; i < candidate->count; ++i) {
    loom_kernel_class_refinement_t unchanged;
    const loom_kernel_class_refinement_t* group;
    if (candidate->groups) {
      group = &candidate->groups[i];
    } else {
      unchanged = (loom_kernel_class_refinement_t){
          .parent = partition->candidate_parent_classes[i],
          .action = partition->candidate_outcomes[i],
      };
      group = &unchanged;
    }
    const loom_kernel_class_shape_binding_t* parent =
        current_bindings[group->parent];
    if (group->count == 0) {
      next_bindings[i] = current_bindings[group->parent];
    } else {
      loom_kernel_class_shape_binding_t* binding = NULL;
      IREE_RETURN_IF_ERROR(
          iree_arena_allocate(arena, sizeof(*binding), (void**)&binding));
      loom_kernel_class_shape_value_t* values = NULL;
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          arena, group->count, sizeof(*values), (void**)&values));
      memcpy(values, group->values, group->count * sizeof(*values));
      loom_value_set_id_t keys = parent ? parent->keys : 0;
      for (uint32_t j = 0; j < group->count; ++j) {
        IREE_RETURN_IF_ERROR(
            loom_value_set_index_add(sets, keys, values[j].projection, &keys));
      }
      *binding = (loom_kernel_class_shape_binding_t){
          .parent = parent,
          .values = values,
          .count = group->count,
          .total_count = group->count + (parent ? parent->total_count : 0),
          .keys = keys};
      next_bindings[i] = binding;
    }
    if (decision_ordinal == UINT32_MAX) {
      next_traces[i] = current_traces[group->parent];
    } else {
      uint32_t parent_trace = current_traces[group->parent];
      uint64_t key = ((uint64_t)parent_trace << 32) | group->action;
      uint32_t bucket =
          loom_structural_hash_finalize(loom_structural_hash_mix_u64(
              loom_structural_hash_initialize(), key)) &
          (bucket_count - 1);
      while (buckets[bucket] != UINT32_MAX) {
        const loom_kernel_class_trace_t* trace = &new_traces[buckets[bucket]];
        if (trace->parent_trace_id == parent_trace &&
            trace->action_ordinal == group->action) {
          break;
        }
        bucket = (bucket + 1) & (bucket_count - 1);
      }
      if (buckets[bucket] == UINT32_MAX) {
        buckets[bucket] = added;
        new_traces[added++] =
            (loom_kernel_class_trace_t){.parent_trace_id = parent_trace,
                                        .decision_ordinal = decision_ordinal,
                                        .action_ordinal = group->action};
      }
      next_traces[i] = *trace_count + buckets[bucket];
    }
  }
  if (added > UINT32_MAX - *trace_count) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "kernel class trace table exceeds uint32_t");
  }
  if (added) {
    void* storage = *traces;
    IREE_RETURN_IF_ERROR(
        iree_arena_grow_array(arena, *trace_count, *trace_count + added,
                              sizeof(**traces), trace_capacity, &storage));
    *traces = storage;
    memcpy(*traces + *trace_count, new_traces, added * sizeof(*new_traces));
    *trace_count += added;
  }
  memcpy(current_bindings, next_bindings,
         candidate->count * sizeof(*current_bindings));
  memcpy(current_traces, next_traces,
         candidate->count * sizeof(*current_traces));
  partition->candidate_class_count = (uint16_t)candidate->count;
  loom_decision_class_partition_commit(partition);
  return iree_ok_status();
}

// Working state for one immutable classifier and its complete launch slice.
typedef struct loom_kernel_class_collection_state_t {
  // Immutable projections borrowed through collection.
  const loom_kernel_class_classifier_t* classifier;
  // Complete borrowed launch slice for this group.
  const loom_kernel_class_site_t* sites;
  // Number of launches in the slice.
  iree_host_size_t site_count;
  // Collection policy bounded by this group's observed site count.
  loom_kernel_class_collection_options_t options;
  // Current prefix and tentative refined partition.
  loom_decision_class_partition_t partition;
  // Captured staged tuples borrowing the caller module.
  const loom_kernel_specialization_value_t** site_specializations;
  // Interned projection sets used by persistent shape bindings.
  loom_value_set_index_t* sets;
  // Source and unavailable-decision shape projections.
  loom_value_set_id_t common_values;
  // Current persistent shape binding for each prefix class.
  loom_kernel_class_shape_binding_t** bindings;
  // Scratch destinations for committed refinement bindings.
  loom_kernel_class_shape_binding_t** next_bindings;
  // Current trace tip for each prefix class.
  loom_kernel_class_trace_id_t* trace_ids;
  // Scratch destinations for committed refinement trace tips.
  loom_kernel_class_trace_id_t* next_trace_ids;
  // Per-decision acceptance or conservative residual outcome.
  loom_kernel_class_decision_result_t* results;
  // Reusable boundary facts for one launch evaluation.
  loom_value_facts_t* projection_facts;
  // Arena-owned accepted trace nodes.
  loom_kernel_class_trace_t* traces;
  // Number of live trace nodes.
  uint32_t trace_count;
  // Number of accepted source decisions.
  uint32_t accepted;
  // Number of decisions left in their generic form.
  uint32_t skipped;
  // Allocated trace-node capacity.
  iree_host_size_t trace_capacity;
  // Unprocessed generic structural requirements.
  loom_kernel_class_residual_t residual;
  // Reusable observed prefix/residual intersection table.
  uint64_t* completion_buckets;
  // Power-of-two bucket count for completion intersections.
  uint32_t bucket_count;
  // Current upper bound on this group's completed class count.
  uint32_t completion_count;
  // True when completion_count is the exact observed quotient.
  bool completion_exact;
  // True when this group already occupies the pending-resolution list.
  bool completion_pending;
} loom_kernel_class_collection_state_t;

static iree_status_t loom_kernel_class_collection_commit(
    loom_kernel_class_collection_state_t* state,
    const loom_kernel_class_candidate_t* candidate, uint32_t decision,
    iree_arena_allocator_t* arena, iree_arena_allocator_t* scratch) {
  return loom_kernel_class_commit(
      candidate, decision, state->sets, state->bindings, state->next_bindings,
      state->trace_ids, state->next_trace_ids, &state->traces,
      &state->trace_count, &state->trace_capacity, &state->partition, arena,
      scratch);
}

static iree_status_t loom_kernel_class_collection_refine(
    loom_kernel_class_collection_state_t* state, uint32_t ordinal,
    bool evaluate, iree_arena_allocator_t* scratch,
    loom_kernel_class_candidate_t* candidate) {
  return loom_kernel_class_refine(
      state->classifier, state->sites, &state->options,
      &state->classifier->decisions[ordinal], ordinal, NULL, evaluate,
      state->sets,
      (const loom_kernel_class_shape_binding_t* const*)state->bindings,
      state->projection_facts, &state->partition, scratch, candidate);
}

static uint32_t loom_kernel_class_completion_count(
    const loom_kernel_class_collection_state_t* state,
    const loom_decision_class_ordinal_t* classes, uint32_t class_count) {
  const loom_kernel_class_residual_t* residual = &state->residual;
  uint32_t capacity = state->options.class_limit;
  if (residual->overflow_decision != UINT32_MAX) {
    return capacity + 1;
  }
  if (residual->active_count <= 1) {
    return class_count;
  }
  if (class_count == 1) {
    return residual->active_count;
  }
  uint64_t* buckets = state->completion_buckets;
  memset(buckets, 0xFF, state->bucket_count * sizeof(*buckets));
  uint32_t count = 0;
  for (iree_host_size_t i = 0; i < state->site_count; ++i) {
    uint64_t key = ((uint64_t)classes[i] << 32) |
                   residual->active_nodes[residual->site_classes[i]];
    uint32_t hash = loom_structural_hash_finalize(
        loom_structural_hash_mix_u64(loom_structural_hash_initialize(), key));
    uint32_t bucket = hash & (state->bucket_count - 1);
    while (buckets[bucket] != UINT64_MAX && buckets[bucket] != key) {
      bucket = (bucket + 1) & (state->bucket_count - 1);
    }
    if (buckets[bucket] == UINT64_MAX) {
      if (count == capacity) {
        return capacity + 1;
      }
      buckets[bucket] = key;
      ++count;
    }
  }
  return count;
}

static iree_status_t loom_kernel_class_collection_initialize(
    const loom_kernel_class_group_t* group, const loom_module_t* site_module,
    const loom_kernel_class_collection_options_t* options,
    iree_arena_allocator_t* arena, iree_arena_allocator_t* scratch,
    loom_kernel_class_collection_state_t* state) {
  *state = (loom_kernel_class_collection_state_t){
      .classifier = group->classifier,
      .sites = group->sites,
      .site_count = group->site_count,
      .options = *options,
      .residual = {.overflow_decision = UINT32_MAX},
  };
  const loom_kernel_class_classifier_t* classifier = group->classifier;
  state->options.class_limit =
      (uint16_t)iree_min(group->site_count, options->class_limit);
  uint32_t capacity = state->options.class_limit;
  IREE_RETURN_IF_ERROR(loom_decision_class_partition_initialize(
      group->site_count, capacity,
      iree_max(1u, classifier->maximum_provider_count), arena,
      &state->partition));
  if (classifier->specialization_count) {
    IREE_RETURN_IF_ERROR(loom_kernel_specializations_collect(
        site_module, group->sites, classifier->specialization_count,
        &state->partition, arena, &state->site_specializations));
  }
  if (classifier->has_shape_requirements) {
    IREE_RETURN_IF_ERROR(loom_value_set_index_allocate(arena, &state->sets));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, capacity, sizeof(*state->bindings), (void**)&state->bindings));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, capacity, sizeof(*state->next_bindings),
        (void**)&state->next_bindings));
    memset(state->bindings, 0, capacity * sizeof(*state->bindings));
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, capacity, sizeof(*state->trace_ids), (void**)&state->trace_ids));
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(arena, capacity, sizeof(*state->next_trace_ids),
                                (void**)&state->next_trace_ids));
  memset(state->trace_ids, 0xFF, capacity * sizeof(*state->trace_ids));
  IREE_RETURN_IF_ERROR(loom_kernel_class_allocate_array(
      arena, classifier->decision_count, sizeof(*state->results),
      (void**)&state->results));
  if (classifier->decision_count) {
    memset(state->results, 0,
           classifier->decision_count * sizeof(*state->results));
  }
  IREE_RETURN_IF_ERROR(loom_kernel_class_allocate_array(
      arena, classifier->projection_count, sizeof(*state->projection_facts),
      (void**)&state->projection_facts));
  state->bucket_count = loom_kernel_class_bucket_count(capacity);
  if (classifier->has_shape_requirements) {
    IREE_RETURN_IF_ERROR(loom_kernel_class_union_shape(
        state->sets, &classifier->shape, &state->common_values));
  }
  for (uint32_t i = 0; i < classifier->decision_count; ++i) {
    const loom_kernel_class_decision_t* decision = &classifier->decisions[i];
    if (decision->unavailable_reason == LOOM_KERNEL_CLASS_DECISION_AVAILABLE) {
      continue;
    }
    if (decision->generic_result.kind !=
        LOOM_DECISION_PROGRAM_RESULT_SELECTED) {
      return loom_kernel_class_unresolved_decision_status(
          i, &decision->generic_result);
    }
    state->results[i].state =
        loom_kernel_class_skipped_decision_state(decision->unavailable_reason);
    ++state->skipped;
    if (classifier->has_shape_requirements) {
      IREE_RETURN_IF_ERROR(loom_kernel_class_union_shape(
          state->sets, &decision->residual_shape, &state->common_values));
    }
  }
  loom_kernel_class_shape_t common = {0};
  if (state->common_values) {
    IREE_RETURN_IF_ERROR(loom_kernel_class_shape_values(
        state->sets, state->common_values, scratch, &common));
  }
  loom_kernel_class_candidate_t candidate;
  if (common.count) {
    IREE_RETURN_IF_ERROR(loom_kernel_class_refine(
        classifier, group->sites, &state->options, NULL, UINT32_MAX, &common,
        false, state->sets,
        (const loom_kernel_class_shape_binding_t* const*)state->bindings,
        state->projection_facts, &state->partition, scratch, &candidate));
    if (!candidate.fits) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "mandatory kernel shapes exceed class limit");
    }
    IREE_RETURN_IF_ERROR(loom_kernel_class_collection_commit(
        state, &candidate, UINT32_MAX, arena, scratch));
  }
  iree_arena_reset(scratch);
  for (uint32_t i = 0; i < classifier->decision_count; ++i) {
    const loom_kernel_class_decision_t* decision = &classifier->decisions[i];
    if (decision->unavailable_reason != LOOM_KERNEL_CLASS_DECISION_AVAILABLE ||
        decision->generic_result.kind ==
            LOOM_DECISION_PROGRAM_RESULT_SELECTED) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_kernel_class_collection_refine(
        state, i, true, scratch, &candidate));
    if (!candidate.publishable) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "kernel decision %u selects a result-dependent specialization "
          "and has no generic residual",
          i);
    }
    if (!candidate.fits) {
      return iree_make_status(
          IREE_STATUS_RESOURCE_EXHAUSTED,
          "kernel decision %u exceeds the %u-class limit and has no generic "
          "residual",
          i, options->class_limit);
    }
    IREE_RETURN_IF_ERROR(loom_kernel_class_collection_commit(
        state, &candidate, i, arena, scratch));
    state->results[i].state = LOOM_KERNEL_CLASS_DECISION_ACCEPTED;
    ++state->accepted;
    iree_arena_reset(scratch);
  }
  return iree_ok_status();
}

static iree_status_t loom_kernel_class_collection_finalize(
    const loom_kernel_class_collection_state_t* state,
    const loom_module_t* site_module, iree_arena_allocator_t* arena,
    loom_kernel_class_collection_t* out_collection) {
  const loom_kernel_class_classifier_t* classifier = state->classifier;
  const loom_decision_class_partition_t* partition = &state->partition;
  loom_kernel_class_t* classes = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, partition->class_count, sizeof(*classes), (void**)&classes));
  memset(classes, 0, partition->class_count * sizeof(*classes));
  loom_value_facts_t* argument_facts = NULL;
  IREE_RETURN_IF_ERROR(loom_kernel_class_allocate_array(
      arena,
      (iree_host_size_t)partition->class_count *
          classifier->kernel_argument_count,
      sizeof(*argument_facts), (void**)&argument_facts));
  for (uint32_t i = 0; i < partition->class_count; ++i) {
    classes[i].trace_id = state->trace_ids[i];
    classes[i].argument_facts =
        argument_facts ? argument_facts + (iree_host_size_t)i *
                                              classifier->kernel_argument_count
                       : NULL;
    const loom_kernel_class_shape_binding_t* binding =
        state->bindings ? state->bindings[i] : NULL;
    if (!binding) {
      continue;
    }
    classes[i].shape_value_count = binding->total_count;
    loom_kernel_class_shape_value_t* values = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, binding->total_count, sizeof(*values), (void**)&values));
    classes[i].shape_values = values;
    for (; binding; binding = binding->parent) {
      memcpy(values, binding->values, binding->count * sizeof(*values));
      values += binding->count;
    }
  }
  for (iree_host_size_t i = 0; i < state->site_count; ++i) {
    loom_kernel_class_t* entry = &classes[partition->site_classes[i]];
    loom_value_facts_t* facts = (loom_value_facts_t*)entry->argument_facts;
    for (uint16_t j = 0; j < classifier->kernel_argument_count; ++j) {
      loom_value_facts_t observed = loom_value_fact_table_lookup(
          state->sites[i].facts, state->sites[i].argument_values[j]);
      observed.extension_id = LOOM_VALUE_FACT_EXTENSION_ID_NONE;
      if (!entry->member_count) {
        facts[j] = observed;
      } else {
        loom_value_facts_meet(&facts[j], &observed, &facts[j]);
      }
    }
    ++entry->member_count;
    entry->specializations =
        state->site_specializations ? state->site_specializations[i] : NULL;
  }
  *out_collection = (loom_kernel_class_collection_t){
      .specialization_module = site_module,
      .site_classes = partition->site_classes,
      .classes = classes,
      .traces = state->traces,
      .decision_results = state->results,
      .site_count = state->site_count,
      .class_count = partition->class_count,
      .trace_count = state->trace_count,
      .accepted_decision_count = state->accepted,
      .skipped_decision_count = state->skipped,
  };
  return iree_ok_status();
}

typedef struct loom_kernel_class_budget_t {
  // Working group states, including mandatory prefix reservations.
  loom_kernel_class_collection_state_t* states;
  // Groups whose completion estimate changed since the last exact count.
  uint32_t* pending;
  // Occupied entries in pending.
  uint32_t pending_count;
  // Total committed prefix classes across all groups.
  uint64_t prefix_count;
  // Sum of group completion counts, each exact or an upper bound.
  uint64_t complete_count;
  // One publication-wide class limit.
  uint32_t limit;
  // True once a complete publishable plan fits the shared limit.
  bool viable;
} loom_kernel_class_budget_t;

static uint32_t loom_kernel_class_completion_bound(
    loom_kernel_class_collection_state_t* state, uint32_t prefix_count,
    bool* exact) {
  *exact = true;
  if (state->residual.overflow_decision != UINT32_MAX) {
    return state->options.class_limit + 1;
  }
  uint32_t suffix_count = state->residual.active_count;
  if (suffix_count <= 1) {
    return prefix_count;
  }
  if (prefix_count == 1) {
    return suffix_count;
  }
  *exact = false;
  return (uint32_t)iree_min(
      state->options.class_limit + 1u,
      iree_min(state->site_count, (uint64_t)prefix_count * suffix_count));
}

static void loom_kernel_class_budget_update(loom_kernel_class_budget_t* budget,
                                            uint32_t group, uint32_t count,
                                            bool exact) {
  loom_kernel_class_collection_state_t* state = &budget->states[group];
  budget->complete_count =
      budget->complete_count - state->completion_count + count;
  state->completion_count = count;
  state->completion_exact = exact;
  if (!exact && !state->completion_pending) {
    budget->pending[budget->pending_count++] = group;
    state->completion_pending = true;
  }
}

// Each pending group has changed since its last exact count. The pending set
// prevents scans of unchanged foreign groups or of all groups at each decision.
static void loom_kernel_class_budget_resolve(
    loom_kernel_class_budget_t* budget, uint32_t candidate_group,
    const loom_decision_class_ordinal_t* candidate_classes,
    uint32_t candidate_count) {
  for (uint32_t i = 0; i < budget->pending_count; ++i) {
    uint32_t group = budget->pending[i];
    loom_kernel_class_collection_state_t* state = &budget->states[group];
    if (!state->completion_exact) {
      uint32_t count = loom_kernel_class_completion_count(
          state,
          group == candidate_group ? candidate_classes
                                   : state->partition.site_classes,
          group == candidate_group ? candidate_count
                                   : state->partition.class_count);
      budget->complete_count =
          budget->complete_count - state->completion_count + count;
      state->completion_count = count;
      state->completion_exact = true;
    }
    state->completion_pending = false;
  }
  budget->pending_count = 0;
}

static iree_status_t loom_kernel_class_collect_groups_impl(
    const loom_kernel_class_group_t* groups, uint32_t group_count,
    const loom_module_t* site_module,
    const loom_kernel_class_collection_options_t* options,
    iree_arena_allocator_t* arena, iree_arena_allocator_t* scratch,
    iree_arena_allocator_t* residual_arena,
    loom_kernel_class_collection_t* collections) {
  if (group_count > options->class_limit) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "mandatory tuples exceed class limit");
  }
  loom_kernel_class_collection_state_t single_state;
  loom_kernel_class_collection_state_t* states = &single_state;
  if (group_count > 1) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        residual_arena, group_count, sizeof(*states), (void**)&states));
  }
  loom_kernel_class_budget_t budget = {.states = states,
                                       .limit = options->class_limit};
  uint32_t observed = 0;
  for (uint32_t i = 0; i < group_count; ++i) {
    observed = (uint32_t)iree_min(
        options->class_limit + 1u,
        observed + iree_min(groups[i].site_count, options->class_limit + 1u));
    IREE_RETURN_IF_ERROR(loom_kernel_class_collection_initialize(
        &groups[i], site_module, options, arena, scratch, &states[i]));
    budget.prefix_count += states[i].partition.class_count;
    if (budget.prefix_count > options->class_limit) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "required classes exceed shared limit");
    }
  }
  bool bounded_by_sites = observed <= options->class_limit;
  bool joint = group_count > 1 && !bounded_by_sites;
  if (joint) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, group_count, sizeof(*budget.pending), (void**)&budget.pending));
  }
  for (uint32_t i = 0; i < group_count; ++i) {
    loom_kernel_class_collection_state_t* state = &states[i];
    if (!bounded_by_sites) {
      IREE_RETURN_IF_ERROR(loom_kernel_class_residual_build(
          state->classifier, state->sites, state->site_count, &state->options,
          residual_arena, scratch, state->sets, state->common_values, joint,
          &state->residual));
    }
    iree_arena_reset(scratch);
    if (state->residual.class_count > 1) {
      IREE_RETURN_IF_ERROR(
          iree_arena_allocate_array(residual_arena, state->bucket_count,
                                    sizeof(*state->completion_buckets),
                                    (void**)&state->completion_buckets));
    }
    if (joint) {
      bool exact;
      uint32_t count = loom_kernel_class_completion_bound(
          state, state->partition.class_count, &exact);
      loom_kernel_class_budget_update(&budget, i, count, exact);
    }
  }
  if (joint) {
    if (budget.complete_count > budget.limit) {
      loom_kernel_class_budget_resolve(&budget, UINT32_MAX, NULL, 0);
    }
    budget.viable = budget.complete_count <= budget.limit;
  } else {
    budget.viable =
        bounded_by_sites ||
        loom_kernel_class_completion_fits(
            &states[0].residual, states[0].partition.site_classes,
            states[0].site_count, states[0].partition.class_count, budget.limit,
            states[0].completion_buckets, states[0].bucket_count);
  }
  for (uint32_t group = 0; group < group_count; ++group) {
    loom_kernel_class_collection_state_t* state = &states[group];
    for (uint32_t i = 0; i < state->classifier->decision_count; ++i) {
      const loom_kernel_class_decision_t* decision =
          &state->classifier->decisions[i];
      if (decision->unavailable_reason !=
              LOOM_KERNEL_CLASS_DECISION_AVAILABLE ||
          decision->generic_result.kind !=
              LOOM_DECISION_PROGRAM_RESULT_SELECTED) {
        continue;
      }
      uint32_t previous_active = state->residual.active_count;
      uint32_t previous_complete = state->completion_count;
      bool previous_exact = state->completion_exact;
      loom_kernel_class_residual_advance(&state->residual, i);
      loom_kernel_class_candidate_t candidate;
      IREE_RETURN_IF_ERROR(loom_kernel_class_collection_refine(
          state, i, true, scratch, &candidate));
      uint32_t previous_prefix = state->partition.class_count;
      bool fits = candidate.fits && candidate.publishable &&
                  budget.prefix_count - previous_prefix + candidate.count <=
                      budget.limit;
      bool complete = false;
      if (fits && joint) {
        bool exact;
        uint32_t count =
            loom_kernel_class_completion_bound(state, candidate.count, &exact);
        if (candidate.count == previous_prefix && previous_complete < count) {
          count = previous_complete;
          exact =
              previous_exact && previous_active == state->residual.active_count;
        }
        loom_kernel_class_budget_update(&budget, group, count, exact);
        complete = budget.viable && candidate.count == previous_prefix;
        if (!complete && budget.complete_count > budget.limit) {
          loom_kernel_class_budget_resolve(
              &budget, group, state->partition.candidate_site_classes,
              candidate.count);
        }
        complete |= budget.complete_count <= budget.limit;
      } else if (fits) {
        complete =
            bounded_by_sites ||
            (budget.viable && candidate.count == previous_prefix) ||
            loom_kernel_class_completion_fits(
                &state->residual, state->partition.candidate_site_classes,
                state->site_count, candidate.count, budget.limit,
                state->completion_buckets, state->bucket_count);
      }
      bool take =
          fits &&
          (complete || (!budget.viable && (candidate.count == previous_prefix ||
                                           decision->can_reduce_structure)));
      if (take) {
        IREE_RETURN_IF_ERROR(loom_kernel_class_collection_commit(
            state, &candidate, i, arena, scratch));
        state->results[i].state = LOOM_KERNEL_CLASS_DECISION_ACCEPTED;
        ++state->accepted;
        budget.viable = complete;
      } else {
        if (fits && joint) {
          loom_kernel_class_budget_update(&budget, group, previous_complete,
                                          previous_exact);
        }
        state->results[i].state =
            candidate.publishable
                ? LOOM_KERNEL_CLASS_DECISION_SKIPPED_CLASS_LIMIT
                : LOOM_KERNEL_CLASS_DECISION_SKIPPED_RESULT_DEPENDENT;
        ++state->skipped;
        iree_arena_reset(scratch);
        IREE_RETURN_IF_ERROR(loom_kernel_class_collection_refine(
            state, i, false, scratch, &candidate));
        if (!candidate.fits ||
            budget.prefix_count - previous_prefix + candidate.count >
                budget.limit) {
          return iree_make_status(
              IREE_STATUS_RESOURCE_EXHAUSTED,
              "generic residual exceeds shared class limit");
        }
        IREE_RETURN_IF_ERROR(loom_kernel_class_collection_commit(
            state, &candidate, UINT32_MAX, arena, scratch));
      }
      budget.prefix_count += state->partition.class_count - previous_prefix;
      iree_arena_reset(scratch);
    }
  }
  for (uint32_t group = 0; group < group_count; ++group) {
    IREE_RETURN_IF_ERROR(loom_kernel_class_collection_finalize(
        &states[group], site_module, arena, &collections[group]));
  }
  return iree_ok_status();
}

iree_status_t loom_kernel_class_collect_groups(
    const loom_kernel_class_group_t* groups, uint32_t group_count,
    const loom_module_t* site_module,
    const loom_kernel_class_collection_options_t* options,
    iree_arena_allocator_t* arena,
    loom_kernel_class_collection_t* collections) {
  if (!group_count) {
    return iree_ok_status();
  }
  memset(collections, 0, group_count * sizeof(*collections));
  iree_arena_allocator_t scratch, residual_arena;
  iree_arena_initialize(arena->block_pool, &scratch);
  iree_arena_initialize(arena->block_pool, &residual_arena);
  iree_status_t status = loom_kernel_class_collect_groups_impl(
      groups, group_count, site_module, options, arena, &scratch,
      &residual_arena, collections);
  if (!iree_status_is_ok(status)) {
    memset(collections, 0, group_count * sizeof(*collections));
  }
  iree_arena_deinitialize(&scratch);
  iree_arena_deinitialize(&residual_arena);
  return status;
}

iree_status_t loom_kernel_class_classifier_collect(
    const loom_kernel_class_classifier_t* classifier,
    const loom_module_t* site_module, const loom_kernel_class_site_t* sites,
    iree_host_size_t site_count,
    const loom_kernel_class_collection_options_t* options,
    iree_arena_allocator_t* arena,
    loom_kernel_class_collection_t* out_collection) {
  *out_collection = (loom_kernel_class_collection_t){0};
  if (!site_count) {
    return iree_ok_status();
  }
  const loom_kernel_class_group_t group = {
      .classifier = classifier, .sites = sites, .site_count = site_count};
  return loom_kernel_class_collect_groups(&group, 1, site_module, options,
                                          arena, out_collection);
}

// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/representation_plan.h"

#include <stddef.h>
#include <string.h>

struct loom_low_representation_node_t {
  // Local value ordinal represented by this sparse node.
  loom_value_ordinal_t value_ordinal;
  // Union-find parent node ordinal.
  uint32_t parent;
  // Union-find rank used to keep trees shallow.
  uint16_t rank;
  // Selected physical representation, or NONE when unconstrained.
  loom_low_representation_id_t selected_representation;
  // First exact candidate constraint attached to this component.
  loom_low_representation_constraint_t* constraint_head;
  // Last constraint attached to this component.
  loom_low_representation_constraint_t* constraint_tail;
};

struct loom_low_representation_constraint_t {
  // Next constraint attached to the same component.
  loom_low_representation_constraint_t* next;
  // Number of exact candidates stored inline.
  uint16_t candidate_count;
  // Inline copied exact candidates.
  loom_low_representation_candidate_t candidates[];
};

typedef struct loom_low_representation_aggregate_cost_t {
  // Sum of target runtime costs across a component.
  uint64_t runtime;
  // Sum of target code-size costs across a component.
  uint64_t code_size;
} loom_low_representation_aggregate_cost_t;

static iree_status_t loom_low_representation_plan_prepare_node_map(
    loom_low_representation_plan_t* plan) {
  if (plan->node_ordinals != NULL || plan->value_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(plan->arena, plan->value_count,
                                                 sizeof(*plan->node_ordinals),
                                                 (void**)&plan->node_ordinals));
  memset(plan->node_ordinals, 0xFF,
         plan->value_count * sizeof(*plan->node_ordinals));
  return iree_ok_status();
}

static iree_status_t loom_low_representation_plan_node(
    loom_low_representation_plan_t* plan, loom_value_ordinal_t value_ordinal,
    uint32_t* out_node_ordinal) {
  IREE_ASSERT_LT(value_ordinal, plan->value_count);
  IREE_RETURN_IF_ERROR(loom_low_representation_plan_prepare_node_map(plan));
  uint32_t node_ordinal = plan->node_ordinals[value_ordinal];
  if (node_ordinal != UINT32_MAX) {
    *out_node_ordinal = node_ordinal;
    return iree_ok_status();
  }
  if (plan->node_count == plan->node_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        plan->arena, plan->node_count, (iree_host_size_t)plan->node_count + 1,
        sizeof(*plan->nodes), &plan->node_capacity, (void**)&plan->nodes));
  }
  node_ordinal = plan->node_count++;
  plan->nodes[node_ordinal] = (loom_low_representation_node_t){
      .value_ordinal = value_ordinal,
      .parent = node_ordinal,
      .selected_representation = LOOM_LOW_REPRESENTATION_ID_NONE,
  };
  plan->node_ordinals[value_ordinal] = node_ordinal;
  *out_node_ordinal = node_ordinal;
  return iree_ok_status();
}

IREE_ATTRIBUTE_NOINLINE static uint32_t loom_low_representation_plan_find_root(
    loom_low_representation_plan_t* plan, uint32_t node_ordinal) {
  loom_low_representation_node_t* node = &plan->nodes[node_ordinal];
  uint32_t root_ordinal = node_ordinal;
  while (plan->nodes[root_ordinal].parent != root_ordinal) {
    root_ordinal = plan->nodes[root_ordinal].parent;
  }
  while (node->parent != root_ordinal) {
    const uint32_t parent_ordinal = node->parent;
    node->parent = root_ordinal;
    node = &plan->nodes[parent_ordinal];
  }
  return root_ordinal;
}

static const loom_low_representation_candidate_t*
loom_low_representation_constraint_find_candidate(
    const loom_low_representation_constraint_t* constraint,
    loom_low_representation_id_t representation) {
  for (uint16_t i = 0; i < constraint->candidate_count; ++i) {
    if (constraint->candidates[i].representation == representation) {
      return &constraint->candidates[i];
    }
  }
  return NULL;
}

static bool loom_low_representation_aggregate_cost_less(
    loom_low_representation_aggregate_cost_t left,
    loom_low_representation_id_t left_representation,
    loom_low_representation_aggregate_cost_t right,
    loom_low_representation_id_t right_representation) {
  if (left.runtime != right.runtime) {
    return left.runtime < right.runtime;
  }
  if (left.code_size != right.code_size) {
    return left.code_size < right.code_size;
  }
  return left_representation < right_representation;
}

static bool loom_low_representation_plan_solve_component(
    loom_low_representation_node_t* root,
    loom_low_representation_conflict_t* out_conflict) {
  const loom_low_representation_constraint_t* first = root->constraint_head;
  if (first == NULL) {
    root->selected_representation = LOOM_LOW_REPRESENTATION_ID_NONE;
    return true;
  }

  loom_low_representation_id_t best_representation =
      LOOM_LOW_REPRESENTATION_ID_NONE;
  loom_low_representation_aggregate_cost_t best_cost = {UINT64_MAX, UINT64_MAX};
  for (uint16_t i = 0; i < first->candidate_count; ++i) {
    const loom_low_representation_candidate_t* first_candidate =
        &first->candidates[i];
    loom_low_representation_aggregate_cost_t aggregate_cost = {
        .runtime = first_candidate->cost.runtime,
        .code_size = first_candidate->cost.code_size,
    };
    bool exact = true;
    for (const loom_low_representation_constraint_t* constraint = first->next;
         constraint != NULL; constraint = constraint->next) {
      const loom_low_representation_candidate_t* candidate =
          loom_low_representation_constraint_find_candidate(
              constraint, first_candidate->representation);
      if (candidate == NULL) {
        exact = false;
        break;
      }
      aggregate_cost.runtime += candidate->cost.runtime;
      aggregate_cost.code_size += candidate->cost.code_size;
    }
    if (exact && (best_representation == LOOM_LOW_REPRESENTATION_ID_NONE ||
                  loom_low_representation_aggregate_cost_less(
                      aggregate_cost, first_candidate->representation,
                      best_cost, best_representation))) {
      best_representation = first_candidate->representation;
      best_cost = aggregate_cost;
    }
  }

  if (best_representation == LOOM_LOW_REPRESENTATION_ID_NONE) {
    if (out_conflict != NULL) {
      *out_conflict = (loom_low_representation_conflict_t){
          .value_ordinal = root->value_ordinal,
      };
    }
    return false;
  }
  root->selected_representation = best_representation;
  return true;
}

void loom_low_representation_plan_initialize(
    loom_value_ordinal_t value_count, iree_arena_allocator_t* arena,
    loom_low_representation_plan_t* out_plan) {
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_plan);
  *out_plan = (loom_low_representation_plan_t){
      .arena = arena,
      .value_count = value_count,
  };
}

iree_status_t loom_low_representation_plan_union(
    loom_low_representation_plan_t* plan, loom_value_ordinal_t left,
    loom_value_ordinal_t right) {
  IREE_ASSERT_ARGUMENT(plan);
  IREE_ASSERT(!plan->solved);
  uint32_t left_node_ordinal = UINT32_MAX;
  IREE_RETURN_IF_ERROR(
      loom_low_representation_plan_node(plan, left, &left_node_ordinal));
  uint32_t right_node_ordinal = UINT32_MAX;
  IREE_RETURN_IF_ERROR(
      loom_low_representation_plan_node(plan, right, &right_node_ordinal));
  uint32_t left_root =
      loom_low_representation_plan_find_root(plan, left_node_ordinal);
  uint32_t right_root =
      loom_low_representation_plan_find_root(plan, right_node_ordinal);
  if (left_root == right_root) {
    return iree_ok_status();
  }
  if (plan->nodes[left_root].rank < plan->nodes[right_root].rank) {
    const uint32_t temporary_root = left_root;
    left_root = right_root;
    right_root = temporary_root;
  }
  loom_low_representation_node_t* root = &plan->nodes[left_root];
  loom_low_representation_node_t* merged = &plan->nodes[right_root];
  merged->parent = left_root;
  if (root->rank == merged->rank) {
    ++root->rank;
  }
  if (root->constraint_tail != NULL) {
    root->constraint_tail->next = merged->constraint_head;
  } else {
    root->constraint_head = merged->constraint_head;
  }
  if (merged->constraint_tail != NULL) {
    root->constraint_tail = merged->constraint_tail;
  }
  return iree_ok_status();
}

iree_status_t loom_low_representation_plan_constrain(
    loom_low_representation_plan_t* plan, loom_value_ordinal_t value_ordinal,
    const loom_low_representation_candidate_t* candidates,
    iree_host_size_t candidate_count) {
  IREE_ASSERT_ARGUMENT(plan);
  IREE_ASSERT(!plan->solved);
  IREE_ASSERT_ARGUMENT(candidates);
  IREE_ASSERT_GT(candidate_count, 0u);
  IREE_ASSERT_LE(candidate_count, UINT16_MAX);
  for (iree_host_size_t i = 0; i < candidate_count; ++i) {
    IREE_ASSERT_NE(candidates[i].representation,
                   LOOM_LOW_REPRESENTATION_ID_NONE);
    for (iree_host_size_t j = 0; j < i; ++j) {
      IREE_ASSERT_NE(candidates[i].representation,
                     candidates[j].representation);
    }
  }

  uint32_t node_ordinal = UINT32_MAX;
  IREE_RETURN_IF_ERROR(
      loom_low_representation_plan_node(plan, value_ordinal, &node_ordinal));
  const uint32_t root_ordinal =
      loom_low_representation_plan_find_root(plan, node_ordinal);
  loom_low_representation_node_t* root = &plan->nodes[root_ordinal];
  const iree_host_size_t allocation_size =
      offsetof(loom_low_representation_constraint_t, candidates) +
      candidate_count * sizeof(*candidates);
  loom_low_representation_constraint_t* constraint = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(plan->arena, allocation_size, (void**)&constraint));
  constraint->next = NULL;
  constraint->candidate_count = (uint16_t)candidate_count;
  memcpy(constraint->candidates, candidates,
         candidate_count * sizeof(*candidates));
  if (root->constraint_tail != NULL) {
    root->constraint_tail->next = constraint;
  } else {
    root->constraint_head = constraint;
  }
  root->constraint_tail = constraint;
  return iree_ok_status();
}

bool loom_low_representation_plan_component_is_constrained(
    loom_low_representation_plan_t* plan, loom_value_ordinal_t value_ordinal) {
  IREE_ASSERT_ARGUMENT(plan);
  IREE_ASSERT_LT(value_ordinal, plan->value_count);
  if (plan->node_ordinals == NULL) return false;
  const uint32_t node_ordinal = plan->node_ordinals[value_ordinal];
  if (node_ordinal == UINT32_MAX) return false;
  const uint32_t root_ordinal =
      loom_low_representation_plan_find_root(plan, node_ordinal);
  return plan->nodes[root_ordinal].constraint_head != NULL;
}

bool loom_low_representation_plan_solve(
    loom_low_representation_plan_t* plan,
    loom_low_representation_conflict_t* out_conflict) {
  IREE_ASSERT_ARGUMENT(plan);
  IREE_ASSERT(!plan->solved);
  if (out_conflict != NULL) {
    *out_conflict = (loom_low_representation_conflict_t){
        .value_ordinal = LOOM_VALUE_ORDINAL_INVALID,
    };
  }
  plan->solved = true;
  for (uint32_t i = 0; i < plan->node_count; ++i) {
    if (plan->nodes[i].parent != i) {
      continue;
    }
    if (!loom_low_representation_plan_solve_component(&plan->nodes[i],
                                                      out_conflict)) {
      return false;
    }
  }
  return true;
}

bool loom_low_representation_plan_lookup(
    loom_low_representation_plan_t* plan, loom_value_ordinal_t value_ordinal,
    loom_low_representation_id_t* out_representation) {
  IREE_ASSERT_ARGUMENT(plan);
  IREE_ASSERT(plan->solved);
  IREE_ASSERT_ARGUMENT(out_representation);
  *out_representation = LOOM_LOW_REPRESENTATION_ID_NONE;
  IREE_ASSERT_LT(value_ordinal, plan->value_count);
  if (plan->node_ordinals == NULL) {
    return false;
  }
  const uint32_t node_ordinal = plan->node_ordinals[value_ordinal];
  if (node_ordinal == UINT32_MAX) {
    return false;
  }
  const uint32_t root_ordinal =
      loom_low_representation_plan_find_root(plan, node_ordinal);
  const loom_low_representation_id_t representation =
      plan->nodes[root_ordinal].selected_representation;
  if (representation == LOOM_LOW_REPRESENTATION_ID_NONE) {
    return false;
  }
  *out_representation = representation;
  return true;
}

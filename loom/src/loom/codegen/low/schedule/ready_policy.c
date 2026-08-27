// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/schedule/ready_policy.h"

#include <string.h>

#define LOOM_LOW_SCHEDULE_PAIR_LOOKAHEAD_CAPACITY 16u

typedef enum loom_low_schedule_ready_membership_change_e {
  LOOM_LOW_SCHEDULE_READY_MEMBERSHIP_REMOVE = 0,
  LOOM_LOW_SCHEDULE_READY_MEMBERSHIP_INSERT = 1,
} loom_low_schedule_ready_membership_change_t;

static bool loom_low_schedule_ready_policy_node_is_storage_setup(
    const loom_low_schedule_node_t* node) {
  return iree_any_bit_set(node->flags,
                          LOOM_LOW_SCHEDULE_NODE_FLAG_STORAGE_SETUP);
}

static void loom_low_schedule_ready_policy_update_setup_dependencies(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_ready_policy_t* policy, uint32_t node_index,
    loom_low_schedule_ready_membership_change_t change) {
  if (policy->setup_dependency_counts == NULL ||
      !loom_low_schedule_ready_policy_node_is_storage_setup(
          &state->nodes[node_index])) {
    return;
  }
  const loom_low_schedule_node_t* node = &state->nodes[node_index];
  const uint32_t group_begin = loom_low_schedule_dependency_index_group_begin(
      &state->dependency_index, node_index);
  const uint32_t group_end = loom_low_schedule_dependency_index_group_end(
      &state->dependency_index, node_index);
  for (uint32_t group_index = group_begin; group_index < group_end;
       ++group_index) {
    const loom_low_schedule_dependency_group_t* group =
        loom_low_schedule_dependency_index_group_at(&state->dependency_index,
                                                    group_index);
    const uint32_t consumer_node = group->consumer_node;
    if (state->nodes[consumer_node].block_index != node->block_index ||
        state->nodes[consumer_node].scheduled_ordinal !=
            LOOM_LOW_SCHEDULE_NODE_NONE) {
      continue;
    }
    uint32_t* count = &policy->setup_dependency_counts[consumer_node];
    if (change == LOOM_LOW_SCHEDULE_READY_MEMBERSHIP_INSERT) {
      IREE_ASSERT_LE(group->dependency_count, UINT32_MAX - *count);
      *count += group->dependency_count;
    } else {
      IREE_ASSERT_LE(group->dependency_count, *count);
      *count -= group->dependency_count;
    }
  }
}

iree_status_t loom_low_schedule_ready_policy_initialize(
    const loom_low_schedule_build_state_t* state, uint32_t node_count,
    uint8_t view_count, loom_low_schedule_ready_policy_t* out_policy) {
  *out_policy = (loom_low_schedule_ready_policy_t){0};
  IREE_RETURN_IF_ERROR(loom_low_schedule_ready_frontier_initialize(
      node_count, state->target.descriptor_set->descriptor_count, view_count,
      state->arena, &out_policy->frontier));
  if (node_count == 0 || state->pair_affinity_reverse_heads == NULL ||
      state->detached_transfer_node_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->arena, node_count, sizeof(*out_policy->setup_dependency_counts),
      (void**)&out_policy->setup_dependency_counts));
  memset(out_policy->setup_dependency_counts, 0,
         node_count * sizeof(*out_policy->setup_dependency_counts));
  return iree_ok_status();
}

void loom_low_schedule_ready_policy_insert(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_ready_policy_t* policy, uint32_t node_index,
    const loom_low_schedule_ready_keys_t* keys) {
  const loom_low_descriptor_t* descriptor = state->nodes[node_index].descriptor;
  const uint32_t descriptor_ordinal =
      descriptor != NULL ? loom_low_descriptor_set_descriptor_ordinal(
                               state->target.descriptor_set, descriptor)
                         : LOOM_LOW_SCHEDULE_READY_NODE_NONE;
  loom_low_schedule_ready_frontier_insert(&policy->frontier, node_index, keys,
                                          descriptor_ordinal);
  loom_low_schedule_ready_policy_update_setup_dependencies(
      state, policy, node_index, LOOM_LOW_SCHEDULE_READY_MEMBERSHIP_INSERT);
}

void loom_low_schedule_ready_policy_remove(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_ready_policy_t* policy, uint32_t node_index) {
  loom_low_schedule_ready_policy_update_setup_dependencies(
      state, policy, node_index, LOOM_LOW_SCHEDULE_READY_MEMBERSHIP_REMOVE);
  loom_low_schedule_ready_frontier_remove(&policy->frontier, node_index);
}

static bool loom_low_schedule_ready_policy_node_result_used_by(
    const loom_low_schedule_node_t* producer,
    const loom_low_schedule_node_t* consumer) {
  const loom_value_ordinal_t* result_ordinals =
      loom_low_schedule_node_const_result_ordinals(producer);
  const loom_value_ordinal_t* operand_ordinals =
      loom_low_schedule_node_const_operand_ordinals(consumer);
  for (uint16_t result_index = 0; result_index < producer->result_count;
       ++result_index) {
    const loom_value_ordinal_t result_ordinal = result_ordinals[result_index];
    for (uint16_t operand_index = 0; operand_index < consumer->operand_count;
         ++operand_index) {
      if (result_ordinal == operand_ordinals[operand_index]) return true;
    }
  }
  return false;
}

static bool loom_low_schedule_ready_policy_node_is_pair_transparent(
    const loom_low_schedule_node_t* node) {
  return iree_any_bit_set(node->flags,
                          LOOM_LOW_SCHEDULE_NODE_FLAG_PAIR_TRANSPARENT);
}

static const loom_low_schedule_pair_affinity_record_t*
loom_low_schedule_ready_policy_find_pair(
    const loom_low_schedule_build_state_t* state,
    const loom_low_schedule_node_t* first,
    const loom_low_schedule_node_t* second) {
  if (first == NULL || second == NULL || first->descriptor == NULL ||
      second->descriptor == NULL || state->pair_affinity_heads == NULL) {
    return NULL;
  }
  const uint32_t first_ordinal = loom_low_descriptor_set_descriptor_ordinal(
      state->target.descriptor_set, first->descriptor);
  const uint32_t second_ordinal = loom_low_descriptor_set_descriptor_ordinal(
      state->target.descriptor_set, second->descriptor);
  if (first_ordinal == LOOM_LOW_DESCRIPTOR_ORDINAL_NONE ||
      second_ordinal == LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) {
    return NULL;
  }
  for (uint32_t record_index = state->pair_affinity_heads[first_ordinal];
       record_index != LOOM_LOW_SCHEDULE_PAIR_AFFINITY_RECORD_NONE;
       record_index = state->pair_affinity_records[record_index].next_record) {
    const loom_low_schedule_pair_affinity_record_t* record =
        &state->pair_affinity_records[record_index];
    if (record->second_descriptor_ordinal == second_ordinal) return record;
  }
  return NULL;
}

static uint16_t loom_low_schedule_ready_policy_pair_priority(
    const loom_low_schedule_build_state_t* state,
    const loom_low_schedule_ready_policy_t* policy,
    uint32_t descriptor_ordinal) {
  if (descriptor_ordinal == LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) return 0;
  uint16_t priority = 0;
  for (uint32_t record_index = state->pair_affinity_heads[descriptor_ordinal];
       record_index != LOOM_LOW_SCHEDULE_PAIR_AFFINITY_RECORD_NONE;
       record_index = state->pair_affinity_records[record_index].next_record) {
    const loom_low_schedule_pair_affinity_record_t* record =
        &state->pair_affinity_records[record_index];
    if (loom_low_schedule_ready_frontier_descriptor_count(
            &policy->frontier, record->second_descriptor_ordinal) != 0) {
      priority = iree_max(priority, record->priority);
    }
  }
  for (uint32_t record_index =
           state->pair_affinity_reverse_heads[descriptor_ordinal];
       record_index != LOOM_LOW_SCHEDULE_PAIR_AFFINITY_RECORD_NONE;
       record_index =
           state->pair_affinity_records[record_index].reverse_next_record) {
    const loom_low_schedule_pair_affinity_record_t* record =
        &state->pair_affinity_records[record_index];
    if (loom_low_schedule_ready_frontier_descriptor_count(
            &policy->frontier, record->first_descriptor_ordinal) != 0) {
      priority = iree_max(priority, record->priority);
    }
  }
  return priority;
}

static bool loom_low_schedule_ready_policy_pair_is_preferred(
    const loom_low_schedule_build_state_t* state,
    const loom_low_schedule_node_t* first,
    const loom_low_schedule_node_t* second) {
  return state->preferred_pair_nodes != NULL &&
         state->preferred_pair_nodes[first->source_ordinal].successor_node ==
             second->source_ordinal;
}

static uint16_t loom_low_schedule_ready_policy_affinity_priority(
    const loom_low_schedule_build_state_t* state,
    const loom_low_schedule_node_t* first,
    const loom_low_schedule_node_t* second,
    uint16_t* out_placement_option_count) {
  *out_placement_option_count = 0;
  const loom_low_schedule_pair_affinity_record_t* record =
      loom_low_schedule_ready_policy_find_pair(state, first, second);
  if (record == NULL ||
      loom_low_schedule_ready_policy_node_result_used_by(first, second)) {
    return 0;
  }
  if (record->placement_recipe_index == LOOM_LOW_PLACEMENT_PAIR_RECIPE_NONE) {
    return loom_low_schedule_ready_policy_pair_is_preferred(state, first,
                                                            second)
               ? UINT16_MAX
               : record->priority;
  }
  const uint16_t recipe_index = (uint16_t)(record->placement_recipe_index - 1u);
  const loom_low_placement_pair_recipe_t* recipe =
      &state->options->pair_affinities.placement_recipes[recipe_index];
  const loom_low_placement_pair_use_t use = {
      .first_op = first->op,
      .second_op = second->op,
      .placement_recipe_index = record->placement_recipe_index,
      .priority = record->priority,
  };
  *out_placement_option_count =
      loom_low_placement_pair_possible_alternative_count(&use, recipe);
  if (*out_placement_option_count == 0) return 0;
  return loom_low_schedule_ready_policy_pair_is_preferred(state, first, second)
             ? UINT16_MAX
             : record->priority;
}

static bool loom_low_schedule_ready_policy_node_can_start_pair(
    const loom_low_schedule_build_state_t* state,
    const loom_low_schedule_node_t* node) {
  if (node == NULL || node->descriptor == NULL ||
      state->pair_affinity_heads == NULL) {
    return false;
  }
  const uint32_t descriptor_ordinal =
      loom_low_descriptor_set_descriptor_ordinal(state->target.descriptor_set,
                                                 node->descriptor);
  return descriptor_ordinal != LOOM_LOW_DESCRIPTOR_ORDINAL_NONE &&
         state->pair_affinity_heads[descriptor_ordinal] !=
             LOOM_LOW_SCHEDULE_PAIR_AFFINITY_RECORD_NONE;
}

static bool loom_low_schedule_ready_policy_node_can_unlock_pair(
    const loom_low_schedule_build_state_t* state,
    const loom_low_schedule_node_t* node) {
  return state->pending_pair_affinity_node == LOOM_LOW_SCHEDULE_NODE_NONE &&
         iree_any_bit_set(node->flags,
                          LOOM_LOW_SCHEDULE_NODE_FLAG_STORAGE_SETUP);
}

uint16_t loom_low_schedule_ready_policy_score_candidate_pair(
    const loom_low_schedule_build_state_t* state, uint32_t node_index,
    uint16_t* out_placement_option_count) {
  *out_placement_option_count = 0;
  if (loom_low_schedule_pair_affinity_list_is_empty(
          state->options->pair_affinities) ||
      state->pending_pair_affinity_node == LOOM_LOW_SCHEDULE_NODE_NONE) {
    return 0;
  }
  const loom_low_schedule_node_t* anchor =
      &state->nodes[state->pending_pair_affinity_node];
  const loom_low_schedule_node_t* candidate = &state->nodes[node_index];
  uint16_t priority = loom_low_schedule_ready_policy_affinity_priority(
      state, anchor, candidate, out_placement_option_count);
  if (priority != 0) {
    return priority > UINT16_MAX / 2u ? UINT16_MAX : (uint16_t)(priority * 2u);
  }

  if (!loom_low_schedule_ready_policy_node_is_pair_transparent(candidate)) {
    return 0;
  }
  const uint32_t group_begin = loom_low_schedule_dependency_index_group_begin(
      &state->dependency_index, node_index);
  const uint32_t group_end = loom_low_schedule_dependency_index_group_end(
      &state->dependency_index, node_index);
  const uint32_t lookahead_end =
      group_begin + iree_min(group_end - group_begin,
                             LOOM_LOW_SCHEDULE_PAIR_LOOKAHEAD_CAPACITY);
  for (uint32_t group_index = group_begin; group_index < lookahead_end;
       ++group_index) {
    if (!loom_low_schedule_dependency_index_group_has_ssa(
            &state->dependency_index, group_index)) {
      continue;
    }
    const loom_low_schedule_node_t* consumer =
        &state->nodes[loom_low_schedule_dependency_index_group_at(
                          &state->dependency_index, group_index)
                          ->consumer_node];
    uint16_t placement_option_count = 0;
    priority = loom_low_schedule_ready_policy_affinity_priority(
        state, anchor, consumer, &placement_option_count);
    if (priority != 0) {
      *out_placement_option_count = placement_option_count;
      return priority;
    }
  }
  return 0;
}

static uint16_t loom_low_schedule_ready_policy_preferred_member_priority(
    const loom_low_schedule_build_state_t* state, uint32_t node_index) {
  if (state->pending_pair_affinity_node != LOOM_LOW_SCHEDULE_NODE_NONE ||
      state->preferred_pair_nodes == NULL ||
      state->nodes[node_index].descriptor == NULL) {
    return 0;
  }
  const loom_low_schedule_preferred_pair_node_t* pair_node =
      &state->preferred_pair_nodes[node_index];
  return pair_node->predecessor_node != LOOM_LOW_SCHEDULE_NODE_NONE ||
                 pair_node->successor_node != LOOM_LOW_SCHEDULE_NODE_NONE
             ? UINT16_MAX
             : 0;
}

uint16_t loom_low_schedule_ready_policy_preferred_anchor_priority(
    const loom_low_schedule_build_state_t* state, const uint32_t* indegrees,
    uint32_t node_index) {
  if (loom_low_schedule_ready_policy_preferred_member_priority(
          state, node_index) == 0 ||
      indegrees == NULL) {
    return 0;
  }
  const uint32_t successor_node =
      state->preferred_pair_nodes[node_index].successor_node;
  if (successor_node == LOOM_LOW_SCHEDULE_NODE_NONE ||
      state->nodes[successor_node].scheduled_ordinal !=
          LOOM_LOW_SCHEDULE_NODE_NONE ||
      indegrees[successor_node] != 0) {
    return 0;
  }
  return UINT16_MAX;
}

uint32_t loom_low_schedule_ready_policy_pair_nominee(
    const loom_low_schedule_build_state_t* state,
    const loom_low_schedule_ready_policy_t* policy) {
  const uint32_t anchor_node = state->pending_pair_affinity_node;
  if (anchor_node == LOOM_LOW_SCHEDULE_NODE_NONE) {
    return LOOM_LOW_SCHEDULE_NODE_NONE;
  }
  if (state->preferred_pair_nodes != NULL) {
    const uint32_t preferred_node =
        state->preferred_pair_nodes[anchor_node].successor_node;
    if (preferred_node != LOOM_LOW_SCHEDULE_NODE_NONE &&
        loom_low_schedule_ready_frontier_contains(&policy->frontier,
                                                  preferred_node)) {
      return preferred_node;
    }
  }

  const loom_low_schedule_node_t* anchor = &state->nodes[anchor_node];
  if (anchor->descriptor == NULL || state->pair_affinity_heads == NULL) {
    return LOOM_LOW_SCHEDULE_NODE_NONE;
  }
  const uint32_t anchor_descriptor_ordinal =
      loom_low_descriptor_set_descriptor_ordinal(state->target.descriptor_set,
                                                 anchor->descriptor);
  if (anchor_descriptor_ordinal == LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) {
    return LOOM_LOW_SCHEDULE_NODE_NONE;
  }

  uint32_t best_node = LOOM_LOW_SCHEDULE_NODE_NONE;
  uint16_t best_priority = 0;
  for (uint32_t record_index =
           state->pair_affinity_heads[anchor_descriptor_ordinal];
       record_index != LOOM_LOW_SCHEDULE_PAIR_AFFINITY_RECORD_NONE;
       record_index = state->pair_affinity_records[record_index].next_record) {
    const loom_low_schedule_pair_affinity_record_t* record =
        &state->pair_affinity_records[record_index];
    const uint32_t candidate_node =
        loom_low_schedule_ready_frontier_descriptor_head(
            &policy->frontier, record->second_descriptor_ordinal);
    if (candidate_node == LOOM_LOW_SCHEDULE_READY_NODE_NONE) continue;
    if (record->priority > best_priority ||
        (record->priority == best_priority &&
         (best_node == LOOM_LOW_SCHEDULE_NODE_NONE ||
          state->nodes[candidate_node].source_ordinal <
              state->nodes[best_node].source_ordinal))) {
      best_node = candidate_node;
      best_priority = record->priority;
    }
  }
  return best_node;
}

uint16_t loom_low_schedule_ready_policy_score_setup_unlocks(
    const loom_low_schedule_build_state_t* state,
    const loom_low_schedule_ready_policy_t* policy, const uint32_t* indegrees,
    uint32_t node_index) {
  if (policy->setup_dependency_counts == NULL ||
      !loom_low_schedule_ready_policy_node_can_unlock_pair(
          state, &state->nodes[node_index])) {
    return 0;
  }
  uint16_t priority = 0;
  const uint32_t group_begin = loom_low_schedule_dependency_index_group_begin(
      &state->dependency_index, node_index);
  const uint32_t group_end = loom_low_schedule_dependency_index_group_end(
      &state->dependency_index, node_index);
  const uint32_t lookahead_end =
      group_begin + iree_min(group_end - group_begin,
                             LOOM_LOW_SCHEDULE_PAIR_LOOKAHEAD_CAPACITY);
  for (uint32_t group_index = group_begin; group_index < lookahead_end;
       ++group_index) {
    const uint32_t consumer_node = loom_low_schedule_dependency_index_group_at(
                                       &state->dependency_index, group_index)
                                       ->consumer_node;
    const loom_low_descriptor_t* consumer_descriptor =
        state->nodes[consumer_node].descriptor;
    if (consumer_descriptor == NULL ||
        policy->setup_dependency_counts[consumer_node] !=
            indegrees[consumer_node]) {
      continue;
    }
    priority = iree_max(
        priority, loom_low_schedule_ready_policy_preferred_member_priority(
                      state, consumer_node));
    priority = iree_max(
        priority, loom_low_schedule_ready_policy_pair_priority(
                      state, policy,
                      loom_low_descriptor_set_descriptor_ordinal(
                          state->target.descriptor_set, consumer_descriptor)));
  }
  return priority;
}

void loom_low_schedule_ready_policy_note_node_scheduled(
    loom_low_schedule_build_state_t* state, uint32_t node_index) {
  if (loom_low_schedule_pair_affinity_list_is_empty(
          state->options->pair_affinities)) {
    state->pending_pair_affinity_node = LOOM_LOW_SCHEDULE_NODE_NONE;
    return;
  }

  const loom_low_schedule_node_t* node = &state->nodes[node_index];
  if (node->descriptor != NULL) {
    if (state->pending_pair_affinity_node != LOOM_LOW_SCHEDULE_NODE_NONE) {
      const loom_low_schedule_node_t* anchor =
          &state->nodes[state->pending_pair_affinity_node];
      const loom_low_schedule_pair_affinity_record_t* affinity =
          loom_low_schedule_ready_policy_find_pair(state, anchor, node);
      if (affinity != NULL &&
          !loom_low_schedule_ready_policy_node_result_used_by(anchor, node)) {
        const uint16_t placement_recipe_index =
            affinity->placement_recipe_index;
        if (placement_recipe_index != LOOM_LOW_PLACEMENT_PAIR_RECIPE_NONE) {
          const uint16_t recipe_index = (uint16_t)(placement_recipe_index - 1u);
          IREE_ASSERT_LT(
              recipe_index,
              state->options->pair_affinities.placement_recipe_count);
          const loom_low_placement_pair_recipe_t* recipe =
              &state->options->pair_affinities.placement_recipes[recipe_index];
          IREE_ASSERT_NE(recipe->relation_count, 0);
          IREE_ASSERT_NE(recipe->alternative_count, 0);
          IREE_ASSERT(state->placement_pair_uses != NULL);
          IREE_ASSERT_LT(state->placement_pair_use_count,
                         state->scheduled_node_count / 2);
          state->placement_pair_uses[state->placement_pair_use_count++] =
              (loom_low_placement_pair_use_t){
                  .first_op = anchor->op,
                  .second_op = node->op,
                  .placement_recipe_index = placement_recipe_index,
                  .priority = affinity->priority,
              };
        }
        state->pending_pair_affinity_node = LOOM_LOW_SCHEDULE_NODE_NONE;
        return;
      }
    }
    state->pending_pair_affinity_node =
        loom_low_schedule_ready_policy_node_can_start_pair(state, node)
            ? node_index
            : LOOM_LOW_SCHEDULE_NODE_NONE;
    return;
  }

  if (loom_low_schedule_ready_policy_node_is_pair_transparent(node)) return;
  state->pending_pair_affinity_node = LOOM_LOW_SCHEDULE_NODE_NONE;
}

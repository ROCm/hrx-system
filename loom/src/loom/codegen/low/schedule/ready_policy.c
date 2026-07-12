// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/schedule/ready_policy.h"

#include <string.h>

typedef enum loom_low_schedule_ready_membership_change_e {
  LOOM_LOW_SCHEDULE_READY_MEMBERSHIP_REMOVE = 0,
  LOOM_LOW_SCHEDULE_READY_MEMBERSHIP_INSERT = 1,
} loom_low_schedule_ready_membership_change_t;

static bool loom_low_schedule_ready_policy_node_is_pair_setup(
    const loom_low_schedule_node_t* node) {
  return iree_any_bit_set(node->flags, LOOM_LOW_SCHEDULE_NODE_FLAG_PAIR_SETUP);
}

static void loom_low_schedule_ready_policy_update_setup_dependencies(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_ready_policy_t* policy, uint32_t node_index,
    loom_low_schedule_ready_membership_change_t change) {
  if (policy->setup_dependency_counts == NULL ||
      !loom_low_schedule_ready_policy_node_is_pair_setup(
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
      state->detached_copy_node_count == 0) {
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

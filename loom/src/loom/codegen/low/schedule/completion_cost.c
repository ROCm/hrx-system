// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/schedule/completion_cost.h"

uint16_t loom_low_schedule_completion_cost_query(
    const uint16_t* node_cycles,
    const loom_low_schedule_dependency_index_t* dependency_index,
    const loom_low_schedule_dependency_incoming_index_t* incoming_index,
    uint32_t consumer_node) {
  uint16_t wait_cycles = node_cycles[consumer_node];
  const uint32_t group_end =
      loom_low_schedule_dependency_incoming_index_group_end(incoming_index,
                                                            consumer_node);
  for (uint32_t incoming_group =
           loom_low_schedule_dependency_incoming_index_group_begin(
               incoming_index, consumer_node);
       incoming_group < group_end; ++incoming_group) {
    const uint32_t group_index =
        loom_low_schedule_dependency_incoming_index_group_at(incoming_index,
                                                             incoming_group);
    if (!loom_low_schedule_dependency_index_group_has_effect(dependency_index,
                                                             group_index)) {
      continue;
    }
    const uint32_t producer_node = loom_low_schedule_dependency_index_group_at(
                                       dependency_index, group_index)
                                       ->producer_node;
    wait_cycles = iree_max(wait_cycles, node_cycles[producer_node]);
  }
  return wait_cycles;
}

void loom_low_schedule_completion_cost_note_node_scheduled(
    uint16_t* node_cycles,
    const loom_low_schedule_dependency_index_t* dependency_index,
    const loom_low_schedule_dependency_incoming_index_t* incoming_index,
    uint32_t node_index, uint16_t opened_wait_cycles) {
  const uint32_t group_end =
      loom_low_schedule_dependency_incoming_index_group_end(incoming_index,
                                                            node_index);
  for (uint32_t incoming_group =
           loom_low_schedule_dependency_incoming_index_group_begin(
               incoming_index, node_index);
       incoming_group < group_end; ++incoming_group) {
    const uint32_t group_index =
        loom_low_schedule_dependency_incoming_index_group_at(incoming_index,
                                                             incoming_group);
    if (!loom_low_schedule_dependency_index_group_has_effect(dependency_index,
                                                             group_index)) {
      continue;
    }
    const uint32_t producer_node = loom_low_schedule_dependency_index_group_at(
                                       dependency_index, group_index)
                                       ->producer_node;
    node_cycles[producer_node] = 0;
  }
  node_cycles[node_index] = opened_wait_cycles;
}

void loom_low_schedule_completion_cost_publish_cross_block(
    uint16_t* node_cycles, uint32_t consumer_node, uint16_t wait_cycles) {
  node_cycles[consumer_node] =
      iree_max(node_cycles[consumer_node], wait_cycles);
}

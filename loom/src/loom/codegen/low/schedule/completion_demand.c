// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/schedule/completion_demand.h"

#include <string.h>

iree_status_t loom_low_schedule_completion_demand_initialize(
    const loom_low_schedule_dependency_index_t* index,
    const loom_low_schedule_node_t* nodes, uint16_t domain_count,
    iree_arena_allocator_t* arena,
    loom_low_schedule_completion_demand_t* out_demand) {
  memset(out_demand, 0, sizeof(*out_demand));
  const uint32_t node_count = index->node_count;
  if (node_count == 0 || domain_count == 0) return iree_ok_status();

  iree_host_size_t start_count = 0;
  if (!iree_host_size_checked_add(node_count, 1, &start_count)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "low schedule completion index size overflow");
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, start_count, sizeof(*out_demand->incoming_starts),
      (void**)&out_demand->incoming_starts));
  uint32_t* starts = out_demand->incoming_starts;
  memset(starts, 0, start_count * sizeof(*starts));
  for (uint32_t producer = 0; producer < node_count; ++producer) {
    const uint32_t end =
        loom_low_schedule_dependency_index_group_end(index, producer);
    for (uint32_t group =
             loom_low_schedule_dependency_index_group_begin(index, producer);
         group < end; ++group) {
      const uint32_t consumer =
          loom_low_schedule_dependency_index_group_at(index, group)
              ->consumer_node;
      if (loom_low_schedule_dependency_index_group_has_ssa(index, group) &&
          nodes[producer].block_index == nodes[consumer].block_index) {
        ++starts[consumer + 1];
      }
    }
  }
  for (uint32_t node = 0; node < node_count; ++node) {
    starts[node + 1] += starts[node];
  }
  const uint32_t producer_count = starts[node_count];
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, producer_count, sizeof(*out_demand->producers),
      (void**)&out_demand->producers));
  // Fill backwards from each row's end, reusing the offsets as cursors. The
  // decremented ends become the starts, avoiding a second node-sized array.
  for (uint32_t producer = 0; producer < node_count; ++producer) {
    const uint32_t end =
        loom_low_schedule_dependency_index_group_end(index, producer);
    for (uint32_t group =
             loom_low_schedule_dependency_index_group_begin(index, producer);
         group < end; ++group) {
      const uint32_t consumer =
          loom_low_schedule_dependency_index_group_at(index, group)
              ->consumer_node;
      if (loom_low_schedule_dependency_index_group_has_ssa(index, group) &&
          nodes[producer].block_index == nodes[consumer].block_index) {
        out_demand->producers[--starts[consumer + 1]] = producer;
      }
    }
  }
  memmove(starts, starts + 1, node_count * sizeof(*starts));
  starts[node_count] = producer_count;

  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, node_count, sizeof(*out_demand->worklist),
      (void**)&out_demand->worklist));
  out_demand->words_per_domain = node_count / 64 + (node_count % 64 != 0);
  iree_host_size_t word_count = 0;
  if (!iree_host_size_checked_mul(out_demand->words_per_domain, domain_count,
                                  &word_count)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "low schedule completion demand size overflow");
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, word_count, sizeof(*out_demand->demanded_bits),
      (void**)&out_demand->demanded_bits));
  memset(out_demand->demanded_bits, 0,
         word_count * sizeof(*out_demand->demanded_bits));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, domain_count,
                                                 sizeof(*out_demand->roots),
                                                 (void**)&out_demand->roots));
  memset(out_demand->roots, 0xFF, domain_count * sizeof(*out_demand->roots));
  return iree_ok_status();
}

void loom_low_schedule_completion_demand_select(
    loom_low_schedule_completion_demand_t* demand,
    const loom_low_schedule_node_t* nodes, uint16_t domain, uint32_t root) {
  const uint32_t previous_root = demand->roots[domain];
  if (previous_root != LOOM_LOW_SCHEDULE_NODE_NONE &&
      nodes[previous_root].scheduled_ordinal == LOOM_LOW_SCHEDULE_NODE_NONE) {
    return;
  }
  demand->roots[domain] = root;
  uint64_t* bits = demand->demanded_bits +
                   (iree_host_size_t)domain * demand->words_per_domain;
  bits[root / 64] |= UINT64_C(1) << (root % 64);
  uint32_t worklist_count = 1;
  demand->worklist[0] = root;
  while (worklist_count != 0) {
    const uint32_t node = demand->worklist[--worklist_count];
    const uint32_t end = demand->incoming_starts[node + 1];
    for (uint32_t i = demand->incoming_starts[node]; i < end; ++i) {
      const uint32_t producer = demand->producers[i];
      const uint64_t bit = UINT64_C(1) << (producer % 64);
      if (nodes[producer].scheduled_ordinal != LOOM_LOW_SCHEDULE_NODE_NONE ||
          (bits[producer / 64] & bit) != 0) {
        continue;
      }
      // Mark before pushing so shared ancestors occupy the worklist once.
      bits[producer / 64] |= bit;
      demand->worklist[worklist_count++] = producer;
    }
  }
}

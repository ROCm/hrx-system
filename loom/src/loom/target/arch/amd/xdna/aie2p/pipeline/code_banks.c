// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/pipeline/code_banks.h"

#include <string.h>

typedef struct loom_aie2p_code_bank_member_t {
  // Configuration load ordinal, in original order within a column.
  iree_host_size_t ordinal;
  // Selected for the first sender's bank.
  bool first;
} loom_aie2p_code_bank_member_t;

// Reachable totals in the first bank determine the demand left for the second.
// Recording the first catalog that reaches a total recovers a stable partition
// in O(catalog count * hardware bank capacity), without enumerating orders.
static bool loom_aie2p_code_bank_partition(
    const loom_aie2p_code_catalog_demand_t* demands,
    loom_aie2p_code_bank_member_t* members, iree_host_size_t count,
    uint16_t first, uint16_t second, iree_host_size_t* predecessor) {
  iree_host_size_t total = 0;
  iree_host_size_t prefix_count = 0;
  iree_host_size_t prefix = 0;
  for (iree_host_size_t i = 0; i < count; ++i) {
    const iree_host_size_t size = demands[members[i].ordinal].descriptor_count;
    if (size > first + second - total) {
      return false;
    }
    total += size;
    if (prefix_count == i && size <= first - prefix) {
      prefix += size;
      ++prefix_count;
    }
  }
  if (total - prefix <= second) {
    return false;
  }

  memset(predecessor, 0, (first + 1) * sizeof(*predecessor));
  for (iree_host_size_t i = 0; i < count; ++i) {
    const iree_host_size_t size = demands[members[i].ordinal].descriptor_count;
    for (iree_host_size_t slot = first; slot >= size; --slot) {
      if (!predecessor[slot] && (slot == size || predecessor[slot - size])) {
        predecessor[slot] = i + 1;
      }
    }
  }
  iree_host_size_t slot = first;
  while (slot && !predecessor[slot]) {
    --slot;
  }
  if (total - slot > second) {
    return false;
  }
  while (slot) {
    loom_aie2p_code_bank_member_t* member = &members[predecessor[slot] - 1];
    member->first = true;
    slot -= demands[member->ordinal].descriptor_count;
  }
  return true;
}

iree_status_t loom_aie2p_code_banks_order(
    const loom_aie2p_execution_plan_t* plan,
    const loom_aie2p_code_catalog_demand_t* demands, iree_host_size_t count,
    iree_arena_allocator_t* arena, iree_host_size_t** out_order) {
  *out_order = NULL;
  const loom_xdna_array_family_t* family = plan->routing.family;
  const loom_xdna_tile_facts_t* memory =
      loom_xdna_array_tile_kind_facts(family, LOOM_XDNA_TILE_KIND_MEMORY);
  const uint16_t capacity = memory->dma.buffer_descriptor_count / 2;
  iree_host_size_t* offsets = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, family->column_count + 1, sizeof(*offsets), (void**)&offsets));
  memset(offsets, 0, (family->column_count + 1) * sizeof(*offsets));
  for (iree_host_size_t i = 0; i < count; ++i) {
    offsets[demands[i].column] += demands[i].descriptor_count != 0;
  }
  for (uint16_t c = 0; c < family->column_count; ++c) {
    offsets[c + 1] += offsets[c];
  }
  loom_aie2p_code_bank_member_t* members = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(arena, offsets[family->column_count],
                                sizeof(*members), (void**)&members));
  for (iree_host_size_t i = count; i > 0; --i) {
    if (demands[i - 1].descriptor_count) {
      members[--offsets[demands[i - 1].column]] =
          (loom_aie2p_code_bank_member_t){.ordinal = i - 1};
    }
  }
  iree_host_size_t* predecessor = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, capacity + 1, sizeof(*predecessor), (void**)&predecessor));
  for (uint16_t c = 0; c < family->column_count; ++c) {
    const loom_aie2p_execution_tile_t* tile =
        &plan->tiles[loom_xdna_array_physical_tile_index(
            family, (loom_xdna_tile_coordinate_t){c, memory->first_row})];
    const iree_host_size_t length = offsets[c + 1] - offsets[c];
    // One worker can have one outstanding load. A two-sender partition must
    // fit the queue of either bank independently of its descriptor usage.
    if (length < 2 || length > memory->dma.task_queue_depth ||
        tile->mm2s_count + 1 >= memory->dma.channel_count_per_direction) {
      continue;
    }
    const unsigned bank = tile->mm2s_count & 1;
    loom_aie2p_code_bank_member_t* column = members + offsets[c];
    if (!loom_aie2p_code_bank_partition(
            demands, column, length, capacity - tile->descriptor_counts[bank],
            capacity - tile->descriptor_counts[bank ^ 1], predecessor)) {
      continue;
    }
    if (!*out_order) {
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          arena, count, sizeof(**out_order), (void**)out_order));
      for (iree_host_size_t i = 0; i < count; ++i) {
        (*out_order)[i] = i;
      }
    }
    iree_host_size_t position = 0;
    for (unsigned pass = 0; pass < 2; ++pass) {
      for (iree_host_size_t i = 0; i < length; ++i) {
        if (column[i].first == (pass == 0)) {
          (*out_order)[column[position++].ordinal] = column[i].ordinal;
        }
      }
    }
  }
  return iree_ok_status();
}

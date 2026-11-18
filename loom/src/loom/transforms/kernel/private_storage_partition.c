// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/kernel/private_storage_partition.h"

#include <stdlib.h>
#include <string.h>

#include "iree/base/internal/math.h"
#include "loom/ir/intern_table.h"
#include "loom/ir/structural_hash.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/view/ops.h"
#include "loom/transforms/kernel/private_storage_copy.h"

static int loom_private_storage_compare_intervals(const void* left_pointer,
                                                  const void* right_pointer) {
  const loom_private_storage_interval_t* left = left_pointer;
  const loom_private_storage_interval_t* right = right_pointer;
  if (left->allocation != right->allocation) {
    return (uintptr_t)left->allocation < (uintptr_t)right->allocation ? -1 : 1;
  }
  if (left->begin != right->begin) {
    return left->begin < right->begin ? -1 : 1;
  }
  if (left->end != right->end) {
    return left->end < right->end ? -1 : 1;
  }
  return (int)loom_type_element_type(left->type) -
         (int)loom_type_element_type(right->type);
}

static iree_status_t loom_private_storage_bind_cells(
    loom_private_storage_plan_t* plan,
    loom_private_storage_intervals_t* intervals) {
  if (!intervals->count) {
    return iree_ok_status();
  }
  qsort(intervals->values, intervals->count, sizeof(*intervals->values),
        loom_private_storage_compare_intervals);
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(plan->arena, intervals->count,
                                                 sizeof(*plan->cells),
                                                 (void**)&plan->cells));
  loom_private_storage_cell_t* previous = NULL;
  bool previous_has_access = false;
  for (iree_host_size_t i = 0; i < intervals->count; ++i) {
    const loom_private_storage_interval_t* interval = &intervals->values[i];
    if (!interval->allocation->selected || interval->allocation->write_only) {
      continue;
    }
    loom_private_storage_cell_t* cell = previous;
    if (previous && previous->allocation == interval->allocation &&
        interval->begin < previous->end) {
      if (interval->begin != previous->begin ||
          interval->end != previous->end ||
          !loom_type_equal(interval->type, previous->type)) {
        // A projected cell is optional. A missing source leaves the copy
        // unknown until another store overwrites it. Preserve every direct
        // access when choosing between incompatible representations.
        if (!interval->access) {
          continue;
        }
        if (previous_has_access) {
          interval->allocation->selected = false;
          continue;
        }
        cell->begin = interval->begin;
        cell->end = interval->end;
        cell->type = interval->type;
      }
    } else {
      cell = &plan->cells[plan->cell_count];
      if (!interval->allocation->cell_count) {
        interval->allocation->cell_begin = plan->cell_count;
      }
      ++interval->allocation->cell_count;
      *cell = (loom_private_storage_cell_t){.allocation = interval->allocation,
                                            .begin = interval->begin,
                                            .end = interval->end,
                                            .type = interval->type,
                                            .index = plan->cell_count++};
      previous = cell;
      previous_has_access = false;
    }
    if (interval->access) {
      previous_has_access = true;
      interval->access->cell = cell;
    }
  }
  return iree_ok_status();
}

// First observed cell ending after offset. Cells within an admitted allocation
// are sorted and disjoint.
static iree_host_size_t loom_private_storage_copy_first_cell(
    const loom_private_storage_plan_t* plan,
    const loom_private_storage_allocation_t* allocation, int64_t offset) {
  iree_host_size_t first = allocation->cell_begin;
  iree_host_size_t limit = first + allocation->cell_count;
  while (first < limit) {
    const iree_host_size_t middle = first + (limit - first) / 2;
    if (plan->cells[middle].end <= offset) {
      first = middle + 1;
    } else {
      limit = middle;
    }
  }
  return first;
}

static iree_status_t loom_private_storage_partition_copies(
    loom_private_storage_plan_t* plan) {
  for (loom_private_storage_copy_t* copy = plan->copies; copy;
       copy = copy->next) {
    if (!copy->target->selected) {
      continue;
    }
    const int64_t end = copy->target_offset + copy->byte_length;
    const iree_host_size_t limit =
        copy->target->cell_begin + copy->target->cell_count;
    copy->cell_begin = loom_private_storage_copy_first_cell(
        plan, copy->target, copy->target_offset);
    iree_host_size_t i = copy->cell_begin;
    while (i < limit && plan->cells[i].begin < end) {
      ++i;
    }
    copy->cell_count = i - copy->cell_begin;
    if (!copy->cell_count) {
      continue;
    }
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        plan->arena, copy->cell_count, sizeof(*copy->source_cells),
        (void**)&copy->source_cells));
    for (i = 0; i < copy->cell_count; ++i) {
      const loom_private_storage_cell_t* target =
          &plan->cells[copy->cell_begin + i];
      copy->source_cells[i] = NULL;
      if (target->begin < copy->target_offset || target->end > end) {
        continue;
      }
      const int64_t begin =
          copy->source_offset + (target->begin - copy->target_offset);
      const iree_host_size_t source_index =
          loom_private_storage_copy_first_cell(plan, copy->source, begin);
      if (source_index == copy->source->cell_begin + copy->source->cell_count) {
        continue;
      }
      const loom_private_storage_cell_t* source = &plan->cells[source_index];
      if (source->begin == begin &&
          source->end - begin == target->end - target->begin &&
          loom_type_equal(source->type, target->type)) {
        copy->source_cells[i] = source;
      }
    }
  }
  return iree_ok_status();
}

typedef struct loom_private_storage_demand_key_t {
  // Canonical intervals indexed by the scratch intern table.
  const loom_private_storage_intervals_t* intervals;
  // Typed source bytes required at the copy site.
  loom_private_storage_interval_t interval;
} loom_private_storage_demand_key_t;

static bool loom_private_storage_demand_equal(const void* context,
                                              uint32_t ordinal) {
  const loom_private_storage_demand_key_t* key = context;
  const loom_private_storage_interval_t* interval =
      &key->intervals->values[ordinal];
  return interval->allocation == key->interval.allocation &&
         interval->begin == key->interval.begin &&
         interval->end == key->interval.end &&
         loom_type_equal(interval->type, key->interval.type);
}

// The scalar type fixes the width of an admitted interval. Each new source
// interval is also a worklist entry for the transfers that wrote its bytes.
static iree_status_t loom_private_storage_demand_source(
    iree_arena_allocator_t* arena, loom_private_storage_intervals_t* intervals,
    loom_intern_table_t* seen, const loom_private_storage_copy_t* copy,
    const loom_private_storage_interval_t* target) {
  const int64_t begin =
      copy->source_offset + (target->begin - copy->target_offset);
  const loom_private_storage_demand_key_t key = {
      .intervals = intervals,
      .interval = {.allocation = copy->source,
                   .begin = begin,
                   .end = begin + (target->end - target->begin),
                   .type = target->type}};
  uint32_t hash = loom_structural_hash_mix_u64(
      loom_buffer_alloca_result(copy->source->op), (uint64_t)begin);
  hash = loom_structural_hash_mix_u32(hash, loom_type_hash(target->type));
  loom_intern_probe_t probe = loom_intern_table_probe(
      seen, hash, loom_private_storage_demand_equal, &key);
  if (probe.index != UINT32_MAX || intervals->count >= UINT32_MAX) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_intern_table_reserve_insert(arena, seen, hash, 1, &probe.slot));
  const uint32_t ordinal = (uint32_t)intervals->count;
  IREE_RETURN_IF_ERROR(
      loom_private_storage_intervals_append(arena, intervals, key.interval));
  loom_intern_table_insert(seen, probe.slot, hash, ordinal);
  return iree_ok_status();
}

// Construction-only reverse edges among allocations reached through copies.
typedef struct loom_private_storage_input_t {
  // Transfer that wrote these bytes into the destination.
  const loom_private_storage_copy_t* copy;
  // Next possible producer of the same destination.
  struct loom_private_storage_input_t* next;
} loom_private_storage_input_t;

typedef struct loom_private_storage_copy_node_t {
  // Transfers whose source is also written by a copy.
  loom_private_storage_input_t* inputs;
  // Number of incoming transfer edges across this representative's component.
  iree_host_size_t edge_count;
  // Remaining projections across this representative's edges.
  uint64_t work_remaining;
} loom_private_storage_copy_node_t;

// A -> B -> C requires a cell in B even if the program only addresses A and
// C directly. Project just the scalar reads backwards across whole-cell copies.
// Sources with no incoming copy already have all their possible scalar stores
// in the partition. The reaching-value solver still handles ordering and joins.
static iree_status_t loom_private_storage_collect_copy_demands(
    loom_private_storage_plan_t* plan,
    loom_private_storage_intervals_t* intervals,
    iree_arena_allocator_t* arena) {
  loom_private_storage_copy_node_t* nodes = NULL;
  for (const loom_private_storage_copy_t* copy = plan->copies; copy;
       copy = copy->next) {
    if (!copy->source->receives_copy ||
        !copy->target->copy_component->selected) {
      continue;
    }
    if (!nodes) {
      // Node zero represents allocations outside the copy graph. Single-hop
      // transfers finish the scan without allocating any projection scratch.
      const iree_host_size_t count =
          (iree_host_size_t)plan->copy_allocation_count + 1;
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          arena, count, sizeof(*nodes), (void**)&nodes));
      memset(nodes, 0, count * sizeof(*nodes));
    }
    loom_private_storage_copy_node_t* target =
        &nodes[copy->target->copy_node_ordinal];
    loom_private_storage_input_t* input = NULL;
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate(arena, sizeof(*input), (void**)&input));
    *input =
        (loom_private_storage_input_t){.copy = copy, .next = target->inputs};
    target->inputs = input;
    ++nodes[copy->target->copy_component->copy_node_ordinal].edge_count;
  }
  if (!nodes) {
    return iree_ok_status();
  }

  // One edge visit per original read and static relay edge suffices for
  // ordinary chains. Shifted cycles could otherwise enumerate a huge buffer
  // from a single observed word. A missing source cell stays unknown, so an
  // observed value that needs more work keeps its component in memory.
  for (iree_host_size_t i = 0; i < intervals->count; ++i) {
    const loom_private_storage_interval_t* interval = &intervals->values[i];
    if (nodes[interval->allocation->copy_node_ordinal].inputs &&
        loom_view_load_isa(interval->access->op)) {
      loom_private_storage_copy_node_t* component =
          &nodes[interval->allocation->copy_component->copy_node_ordinal];
      component->work_remaining = iree_math_saturating_add_u64(
          component->work_remaining, component->edge_count);
    }
  }
  loom_intern_table_t seen;
  IREE_RETURN_IF_ERROR(loom_intern_table_initialize(arena, 0, &seen));
  for (iree_host_size_t i = 0; i < intervals->count; ++i) {
    // Appending a newly required interval may relocate the array.
    const loom_private_storage_interval_t interval = intervals->values[i];
    const loom_private_storage_copy_node_t* node =
        &nodes[interval.allocation->copy_node_ordinal];
    if (!node->inputs ||
        (interval.access && !loom_view_load_isa(interval.access->op))) {
      continue;
    }
    loom_private_storage_copy_node_t* component =
        &nodes[interval.allocation->copy_component->copy_node_ordinal];
    for (const loom_private_storage_input_t* input = node->inputs;
         input && component->work_remaining; input = input->next) {
      --component->work_remaining;
      const loom_private_storage_copy_t* copy = input->copy;
      if (interval.begin < copy->target_offset ||
          interval.end > copy->target_offset + copy->byte_length) {
        continue;
      }
      IREE_RETURN_IF_ERROR(loom_private_storage_demand_source(
          arena, intervals, &seen, copy, &interval));
    }
  }
  return iree_ok_status();
}

iree_status_t loom_private_storage_partition(
    loom_private_storage_plan_t* plan,
    loom_private_storage_intervals_t* intervals,
    iree_arena_allocator_t* scratch_arena) {
  if (plan->copies) {
    loom_private_storage_finalize_copy_components(plan);
    if (intervals->count) {
      IREE_RETURN_IF_ERROR(loom_private_storage_collect_copy_demands(
          plan, intervals, scratch_arena));
    }
  }
  IREE_RETURN_IF_ERROR(loom_private_storage_bind_cells(plan, intervals));
  if (plan->copies) {
    loom_private_storage_resolve_copy_selection(plan);
    return loom_private_storage_partition_copies(plan);
  }
  return iree_ok_status();
}

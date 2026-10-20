// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/spill_storage.h"

#include "loom/analysis/liveness.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/util/adaptive_sort.h"
#include "loom/util/cfg_graph.h"

typedef struct loom_low_spill_storage_interval_t {
  // Layout and original value borrowed from this immutable spill batch.
  const loom_low_allocation_spill_plan_t* plan;
  // Original plan index, preserved through sorting.
  uint32_t plan_index;
  // First block whose execution may access this spilled value.
  uint16_t first_block;
  // Last block whose execution may access this spilled value, inclusive.
  uint16_t last_block;
} loom_low_spill_storage_interval_t;

typedef struct loom_low_spill_storage_active_t {
  // Slot ordinal indexing the representative plan array.
  uint32_t slot_index;
  // Inclusive last block occupied by the latest value assigned to this slot.
  uint16_t last_block;
} loom_low_spill_storage_active_t;

static bool loom_low_spill_storage_same_layout(
    const loom_low_allocation_spill_plan_t* lhs,
    const loom_low_allocation_spill_plan_t* rhs) {
  return lhs->slot_space == rhs->slot_space &&
         lhs->byte_size == rhs->byte_size &&
         lhs->byte_alignment == rhs->byte_alignment;
}

static bool loom_low_spill_storage_interval_less(
    const loom_low_spill_storage_interval_t* lhs,
    const loom_low_spill_storage_interval_t* rhs) {
  if (lhs->plan->slot_space != rhs->plan->slot_space) {
    return lhs->plan->slot_space < rhs->plan->slot_space;
  }
  if (lhs->plan->byte_size != rhs->plan->byte_size) {
    return lhs->plan->byte_size < rhs->plan->byte_size;
  }
  if (lhs->plan->byte_alignment != rhs->plan->byte_alignment) {
    return lhs->plan->byte_alignment < rhs->plan->byte_alignment;
  }
  if (lhs->first_block != rhs->first_block) {
    return lhs->first_block < rhs->first_block;
  }
  if (lhs->last_block != rhs->last_block) {
    return lhs->last_block < rhs->last_block;
  }
  return lhs->plan_index < rhs->plan_index;
}

LOOM_DEFINE_ADAPTIVE_SORT(loom_low_spill_storage_sort,
                          loom_low_spill_storage_interval_t,
                          loom_low_spill_storage_interval_less)

static loom_low_spill_storage_interval_t loom_low_spill_storage_interval(
    const loom_low_allocation_table_t* table,
    const loom_low_allocation_spill_plan_t* plan, uint32_t plan_index) {
  const loom_low_allocation_assignment_t* assignment =
      &table->assignments[plan->assignment_index];
  loom_low_spill_storage_interval_t interval = {
      .plan = plan,
      .plan_index = plan_index,
      .first_block = loom_liveness_block_index_at_point(
          &table->liveness, assignment->start_point),
      .last_block = loom_liveness_block_index_at_point(
          &table->liveness, assignment->end_point - 1u),
  };
  const loom_value_t* value = loom_module_value(table->module, plan->value_id);
  if (loom_value_is_block_arg(value)) {
    const uint16_t block_index =
        loom_block_region_index(loom_value_def_block(value));
    if (block_index == 0) {
      // Unique entry cells permit stores to join the incoming ABI transport.
      interval.first_block = 0;
      interval.last_block = UINT16_MAX;
    } else {
      const loom_cfg_block_index_span_t predecessors =
          loom_cfg_graph_predecessors(&table->cfg_graph, block_index);
      for (iree_host_size_t i = 0; i < predecessors.count; ++i) {
        interval.first_block =
            iree_min(interval.first_block, predecessors.values[i]);
        interval.last_block =
            iree_max(interval.last_block, predecessors.values[i]);
      }
    }
  } else if (loom_low_func_call_isa(loom_value_def_op(value))) {
    // Result stores can likewise join their call's outgoing ABI transport.
    interval.first_block = 0;
    interval.last_block = UINT16_MAX;
  }
  return interval;
}

static bool loom_low_spill_storage_active_less(
    loom_low_spill_storage_active_t lhs, loom_low_spill_storage_active_t rhs) {
  return lhs.last_block < rhs.last_block ||
         (lhs.last_block == rhs.last_block && lhs.slot_index < rhs.slot_index);
}

// Replaces the earliest-ending slot after extending its occupied interval.
static void loom_low_spill_storage_extend(
    loom_low_spill_storage_active_t* active, iree_host_size_t count,
    loom_low_spill_storage_active_t value) {
  iree_host_size_t index = 0;
  while (index < count / 2u) {
    iree_host_size_t child = index * 2u + 1u;
    if (child + 1u < count &&
        loom_low_spill_storage_active_less(active[child + 1u], active[child])) {
      ++child;
    }
    if (!loom_low_spill_storage_active_less(active[child], value)) {
      break;
    }
    active[index] = active[child];
    index = child;
  }
  active[index] = value;
}

static void loom_low_spill_storage_append(
    loom_low_spill_storage_active_t* active, iree_host_size_t count,
    loom_low_spill_storage_active_t value) {
  iree_host_size_t index = count;
  while (index > 0) {
    const iree_host_size_t parent = (index - 1u) / 2u;
    if (!loom_low_spill_storage_active_less(value, active[parent])) {
      break;
    }
    active[index] = active[parent];
    index = parent;
  }
  active[index] = value;
}

iree_status_t loom_low_allocation_spill_storage_plan(
    const loom_low_allocation_table_t* table,
    const loom_low_allocation_spill_plan_t* plans, iree_host_size_t plan_count,
    iree_arena_allocator_t* arena, const uint32_t** out_representatives) {
  *out_representatives = NULL;
  if (plan_count < 2 || table->liveness.block_count == 1) {
    return iree_ok_status();
  }
  loom_low_spill_storage_interval_t* intervals = NULL;
  loom_low_spill_storage_active_t* active = NULL;
  uint32_t* representatives = NULL;
  uint32_t* slot_representatives = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, plan_count, sizeof(*intervals), (void**)&intervals));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, plan_count, sizeof(*active), (void**)&active));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, plan_count, sizeof(*representatives), (void**)&representatives));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, plan_count, sizeof(*slot_representatives),
      (void**)&slot_representatives));
  for (iree_host_size_t i = 0; i < plan_count; ++i) {
    intervals[i] =
        loom_low_spill_storage_interval(table, &plans[i], (uint32_t)i);
  }
  loom_low_spill_storage_sort(intervals, plan_count);
  iree_host_size_t active_count = 0;
  uint32_t slot_count = 0;
  for (iree_host_size_t i = 0; i < plan_count; ++i) {
    const loom_low_spill_storage_interval_t* interval = &intervals[i];
    if (i > 0 && !loom_low_spill_storage_same_layout(intervals[i - 1].plan,
                                                     interval->plan)) {
      active_count = 0;
    }
    uint32_t slot_index;
    if (active_count > 0 && active[0].last_block < interval->first_block) {
      slot_index = active[0].slot_index;
      slot_representatives[slot_index] =
          iree_min(slot_representatives[slot_index], interval->plan_index);
      loom_low_spill_storage_extend(
          active, active_count,
          (loom_low_spill_storage_active_t){slot_index, interval->last_block});
    } else {
      slot_index = slot_count++;
      slot_representatives[slot_index] = interval->plan_index;
      loom_low_spill_storage_append(
          active, active_count++,
          (loom_low_spill_storage_active_t){slot_index, interval->last_block});
    }
    representatives[interval->plan_index] = slot_index;
  }
  for (iree_host_size_t i = 0; i < plan_count; ++i) {
    representatives[i] = slot_representatives[representatives[i]];
  }
  *out_representatives = representatives;
  return iree_ok_status();
}

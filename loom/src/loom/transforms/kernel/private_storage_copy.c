// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/kernel/private_storage_copy.h"

static loom_private_storage_allocation_t* loom_private_storage_copy_root(
    loom_private_storage_allocation_t* allocation) {
  while (allocation->copy_parent) {
    if (allocation->copy_parent->copy_parent) {
      allocation->copy_parent = allocation->copy_parent->copy_parent;
    }
    allocation = allocation->copy_parent;
  }
  return allocation;
}

iree_status_t loom_private_storage_record_copy(
    loom_private_storage_plan_t* plan, loom_op_t* op,
    loom_private_storage_allocation_t* source, int64_t source_offset,
    loom_private_storage_allocation_t* target, int64_t target_offset,
    int64_t byte_length, loom_private_storage_copy_t** out_copy) {
  *out_copy = NULL;
  if (!source || !target || !source->selected || !target->selected ||
      source_offset < 0 || target_offset < 0 || byte_length <= 0 ||
      source->byte_count - byte_length < source_offset ||
      target->byte_count - byte_length < target_offset ||
      (source == target && source_offset < target_offset + byte_length &&
       target_offset < source_offset + byte_length)) {
    return iree_ok_status();
  }
  loom_private_storage_copy_t* copy = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(plan->arena, sizeof(*copy), (void**)&copy));
  *copy = (loom_private_storage_copy_t){.op = op,
                                        .source = source,
                                        .target = target,
                                        .source_offset = source_offset,
                                        .target_offset = target_offset,
                                        .byte_length = byte_length,
                                        .next = plan->copies};
  plan->copies = copy;
  *out_copy = copy;
  // Erasing either endpoint's stores alone could change the other endpoint.
  source->write_only = target->write_only = false;
  source = loom_private_storage_copy_root(source);
  target = loom_private_storage_copy_root(target);
  if (source != target) {
    if (source->copy_rank < target->copy_rank) {
      source->copy_parent = target;
    } else {
      target->copy_parent = source;
      source->copy_rank += source->copy_rank == target->copy_rank;
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

iree_status_t loom_private_storage_partition_copies(
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

void loom_private_storage_resolve_copy_selection(
    loom_private_storage_plan_t* plan) {
  for (loom_private_storage_copy_t* copy = plan->copies; copy;
       copy = copy->next) {
    if (!copy->source->selected || !copy->target->selected) {
      loom_private_storage_copy_root(copy->source)->selected = false;
    }
  }
  for (loom_private_storage_copy_t* copy = plan->copies; copy;
       copy = copy->next) {
    if (!loom_private_storage_copy_root(copy->source)->selected) {
      copy->source->selected = copy->target->selected = false;
    }
  }
}

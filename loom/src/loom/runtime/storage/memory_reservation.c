// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/runtime/storage/memory_reservation.h"

static void loom_runtime_memory_add_class_demand(
    loom_runtime_memory_demand_t* demand, loom_runtime_backing_kind_t kind,
    uint64_t count, uint64_t bytes, uint64_t cost) {
  switch (kind) {
    case LOOM_RUNTIME_BACKING_PACKED:
      demand->packed_bytes += bytes;
      demand->packed_count += count;
      break;
    case LOOM_RUNTIME_BACKING_WEIGHT:
      demand->weight_bytes += bytes;
      break;
    default:
      demand->dedicated_bytes += bytes;
      demand->dedicated_cost += cost;
      break;
  }
}

void loom_runtime_memory_range_add_demand(
    loom_runtime_memory_demand_t* demand,
    const loom_runtime_virtual_buffer_t* buffer, iree_host_size_t index,
    iree_host_size_t count) {
  if (index + count == buffer->slab_count) {
    const iree_host_size_t tail = buffer->slab_count - 1;
    loom_runtime_memory_add_class_demand(
        demand, buffer->tail_kind, 1,
        loom_runtime_memory_slab_length(buffer, tail), buffer->tail_cost);
    --count;
  }
  loom_runtime_memory_add_class_demand(demand, buffer->regular_kind, count,
                                       count * buffer->geometry.mapping_size,
                                       count * buffer->regular_cost);
}

iree_status_t loom_runtime_memory_assign_backing(
    loom_runtime_virtual_buffer_t* buffer, iree_host_size_t index,
    iree_host_size_t count, loom_runtime_backing_slot_t slot) {
  const iree_host_size_t end = index + count;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = index; i < end && iree_status_is_ok(status);) {
    loom_runtime_backing_group_t* group = loom_runtime_backing_directory_find(
        &buffer->backing, i / LOOM_RUNTIME_BACKING_GROUP_SIZE);
    const uint64_t mask = loom_runtime_backing_group_mask(i, end);
    const loom_runtime_backing_slot_t first = {
        slot.parent, slot.offset + (i - index) * buffer->geometry.mapping_size};
    status = loom_runtime_backing_group_assign(
        group, buffer->backing.host_allocator, mask,
        buffer->geometry.mapping_size, first);
    i += iree_math_count_ones_u64(mask);
  }
  return status;
}

void loom_runtime_memory_clear_backing(loom_runtime_virtual_buffer_t* buffer,
                                       iree_host_size_t index,
                                       iree_host_size_t count) {
  const iree_host_size_t end = index + count;
  for (iree_host_size_t i = index; i < end;) {
    loom_runtime_backing_group_t* group = loom_runtime_backing_directory_find(
        &buffer->backing, i / LOOM_RUNTIME_BACKING_GROUP_SIZE);
    const uint64_t mask = loom_runtime_backing_group_mask(i, end);
    loom_runtime_backing_group_clear(group, buffer->backing.host_allocator,
                                     mask, buffer->geometry.mapping_size);
    i += iree_math_count_ones_u64(mask);
  }
}

iree_hal_buffer_t* loom_runtime_virtual_buffer_handle(
    const loom_runtime_virtual_buffer_t* buffer) {
  return buffer->handle;
}

iree_device_size_t loom_runtime_virtual_buffer_mapping_size(
    const loom_runtime_virtual_buffer_t* buffer) {
  return buffer->geometry.mapping_size;
}

iree_host_size_t loom_runtime_virtual_buffer_slab_count(
    const loom_runtime_virtual_buffer_t* buffer) {
  return buffer->slab_count;
}

iree_hal_buffer_binding_t loom_runtime_virtual_buffer_backing(
    const loom_runtime_virtual_buffer_t* buffer, iree_host_size_t slab_index) {
  const loom_runtime_backing_group_t* group =
      loom_runtime_backing_directory_find_const(
          &buffer->backing, slab_index / LOOM_RUNTIME_BACKING_GROUP_SIZE);
  const loom_runtime_backing_slot_t slot = loom_runtime_backing_group_slot(
      group, (int)(slab_index % LOOM_RUNTIME_BACKING_GROUP_SIZE),
      buffer->geometry.mapping_size);
  return (iree_hal_buffer_binding_t){
      loom_runtime_backing_pages_buffer(slot), slot.offset,
      loom_runtime_memory_slab_length(buffer, slab_index)};
}

loom_runtime_memory_demand_t loom_runtime_virtual_buffer_missing_demand(
    const loom_runtime_virtual_buffer_t* buffer) {
  loom_runtime_memory_demand_t demand = {0};
  const uint64_t regular_count =
      buffer->byte_length / buffer->geometry.mapping_size;
  const uint64_t regular_bytes =
      (regular_count - buffer->backed_regular_count) *
      buffer->geometry.mapping_size;
  switch (buffer->regular_kind) {
    case LOOM_RUNTIME_BACKING_PACKED:
      demand.packed_bytes = regular_bytes;
      demand.packed_count = regular_count - buffer->backed_regular_count;
      break;
    case LOOM_RUNTIME_BACKING_WEIGHT:
      demand.weight_bytes = regular_bytes;
      break;
    default:
      demand.dedicated_bytes = regular_bytes;
      demand.dedicated_cost =
          (regular_count - buffer->backed_regular_count) * buffer->regular_cost;
      break;
  }
  if (regular_count < buffer->slab_count) {
    const loom_runtime_backing_group_t* group =
        loom_runtime_backing_directory_find_const(
            &buffer->backing, regular_count / LOOM_RUNTIME_BACKING_GROUP_SIZE);
    const uint64_t bit = UINT64_C(1)
                         << (regular_count % LOOM_RUNTIME_BACKING_GROUP_SIZE);
    if (!group || !((group->published | group->in_flight) & bit)) {
      loom_runtime_memory_range_add_demand(&demand, buffer, regular_count, 1);
    }
  }
  return demand;
}

bool loom_runtime_virtual_buffer_is_backed(
    const loom_runtime_virtual_buffer_t* buffer, iree_device_size_t offset,
    iree_device_size_t length) {
  if (!length) {
    return true;
  }
  const uint64_t begin = offset / buffer->geometry.mapping_size;
  const uint64_t end =
      (offset + length - 1) / buffer->geometry.mapping_size + 1;
  for (iree_host_size_t index = begin; index < end;) {
    const loom_runtime_backing_group_t* group =
        loom_runtime_backing_directory_find_const(
            &buffer->backing, index / LOOM_RUNTIME_BACKING_GROUP_SIZE);
    const uint64_t mask = loom_runtime_backing_group_mask(index, end);
    if (!group || (group->published & mask) != mask) {
      return false;
    }
    index += iree_math_count_ones_u64(mask);
  }
  return true;
}

void loom_runtime_virtual_buffer_select_backing(
    loom_runtime_virtual_buffer_t* buffer, iree_device_size_t offset,
    iree_device_size_t length) {
  loom_runtime_backing_cursor_t cursor = loom_runtime_memory_range_iterate(
      &(loom_runtime_memory_range_t){buffer, offset, length});
  uint64_t mask;
  loom_runtime_backing_group_t* group;
  while ((group = loom_runtime_backing_directory_next(&cursor, &mask))) {
    group->selected |= mask & (group->published | group->in_flight);
  }
}

iree_status_t loom_runtime_virtual_buffer_visit_backing(
    loom_runtime_virtual_buffer_t* buffer,
    iree_status_t (*visitor)(void* user_data, iree_device_size_t offset,
                             iree_device_size_t length),
    void* user_data) {
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < buffer->backing.capacity; ++i) {
    loom_runtime_backing_group_t* group = &buffer->backing.groups[i];
    uint64_t selected = group->selected;
    group->selected = 0;
    while (selected && visitor && iree_status_is_ok(status)) {
      int position;
      const iree_host_size_t count =
          loom_runtime_backing_take_run(&selected, &position);
      const iree_host_size_t index =
          group->index * LOOM_RUNTIME_BACKING_GROUP_SIZE + position;
      status = visitor(user_data, index * buffer->geometry.mapping_size,
                       loom_runtime_memory_range_length(buffer, index, count));
    }
  }
  return status;
}

// A short final page contributes its actual byte length to every observation.
static uint64_t loom_runtime_memory_mask_bytes(
    const loom_runtime_virtual_buffer_t* buffer, iree_host_size_t group_index,
    uint64_t mask) {
  const uint64_t stride = buffer->geometry.mapping_size;
  const iree_host_size_t last = buffer->slab_count - 1;
  uint64_t bytes = (uint64_t)iree_math_count_ones_u64(mask) * stride;
  if (group_index == last / LOOM_RUNTIME_BACKING_GROUP_SIZE &&
      (mask & (UINT64_C(1) << (last % LOOM_RUNTIME_BACKING_GROUP_SIZE)))) {
    bytes -= stride - loom_runtime_memory_slab_length(buffer, last);
  }
  return bytes;
}

iree_status_t loom_runtime_memory_ranges_footprint(
    iree_host_size_t range_count, const loom_runtime_memory_range_t* ranges,
    loom_runtime_memory_footprint_t* out_footprint) {
  *out_footprint = (loom_runtime_memory_footprint_t){0};
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t r = 0; r < range_count && iree_status_is_ok(status);
       ++r) {
    loom_runtime_virtual_buffer_t* buffer = ranges[r].buffer;
    const iree_host_size_t end = loom_runtime_memory_range_end(&ranges[r]);
    for (iree_host_size_t i = ranges[r].offset / buffer->geometry.mapping_size;
         i < end && iree_status_is_ok(status);) {
      const uint64_t mask = loom_runtime_backing_group_mask(i, end);
      loom_runtime_backing_group_t* group = NULL;
      status = loom_runtime_backing_directory_insert(
          &buffer->backing, i / LOOM_RUNTIME_BACKING_GROUP_SIZE, &group);
      if (iree_status_is_ok(status)) {
        const uint64_t selected = mask & ~group->selected;
        group->selected |= selected;
        const uint64_t total =
            loom_runtime_memory_mask_bytes(buffer, group->index, selected);
        const uint64_t missing = loom_runtime_memory_mask_bytes(
            buffer, group->index,
            selected & ~(group->published | group->in_flight));
        out_footprint->total_bytes +=
            iree_min(total, UINT64_MAX - out_footprint->total_bytes);
        out_footprint->missing_bytes +=
            iree_min(missing, UINT64_MAX - out_footprint->missing_bytes);
      }
      i += iree_math_count_ones_u64(mask);
    }
  }
  for (iree_host_size_t r = 0; r < range_count; ++r) {
    loom_runtime_virtual_buffer_t* buffer = ranges[r].buffer;
    loom_runtime_backing_cursor_t cursor =
        loom_runtime_memory_range_iterate(&ranges[r]);
    loom_runtime_backing_group_t* group;
    uint64_t mask;
    while ((group = loom_runtime_backing_directory_next(&cursor, &mask))) {
      group->selected &= ~mask;
      loom_runtime_backing_directory_prune(&buffer->backing, group);
    }
    if (iree_status_is_ok(status)) {
      status = loom_runtime_backing_directory_shrink(&buffer->backing);
    }
  }
  return status;
}

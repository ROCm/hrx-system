// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/reservation_plan.h"

#include <stdlib.h>
#include <string.h>

const uint32_t* loom_serve_text_state_reservation_prefix(
    const loom_serve_text_state_reservation_t* reservation) {
  if (!reservation->preserved) {
    return NULL;
  }
  if (reservation->checkpoint) {
    return reservation->checkpoint->blocks;
  }
  const loom_serve_text_state_row_t* row = reservation->row;
  return row->owner->cache.maps +
         row->index * row->owner->storage.blocks_per_row;
}

static int loom_serve_text_requirement_compare(const void* lhs,
                                               const void* rhs) {
  const loom_runtime_snapshot_range_t* a = lhs;
  const loom_runtime_snapshot_range_t* b = rhs;
  if (a->buffer_index != b->buffer_index) {
    return a->buffer_index < b->buffer_index ? -1 : 1;
  }
  if (a->offset != b->offset) {
    return a->offset < b->offset ? -1 : 1;
  }
  return 0;
}

iree_status_t loom_serve_text_state_reservation_requirements(
    loom_serve_text_state_reservation_t* reservation) {
  if (iree_any_bit_set(reservation->flags,
                       LOOM_SERVE_TEXT_RESERVATION_REQUIREMENTS)) {
    return iree_ok_status();
  }
  loom_serve_text_state_row_t* row = reservation->row;
  loom_serve_text_state_t* state = row->owner;
  if (!state->memory.pool) {
    reservation->flags |= LOOM_SERVE_TEXT_RESERVATION_REQUIREMENTS;
    return iree_ok_status();
  }
  iree_hal_buffer_t* views[5];
  memcpy(views, &row->buffers[TEXT_CONTROL], sizeof(views));
  views[TEXT_RECURRENT - TEXT_CONTROL] =
      state->recurrent.buffers[reservation->writer];
  const loom_serve_text_storage_blocks_t spans[] = {
      {reservation->preserved,
       loom_serve_text_state_reservation_prefix(reservation)},
      {reservation->count - reservation->preserved, reservation->blocks},
  };
  iree_host_size_t row_count = 0;
  loom_runtime_snapshot_range_t* row_ranges = NULL;
  IREE_RETURN_IF_ERROR(loom_serve_text_storage_plan_snapshot(
      &state->storage, row->index, views, true, IREE_ARRAYSIZE(spans), spans,
      state->host_allocator, &row_count, &row_ranges));
  iree_host_size_t capacity = row_count + reservation->restore_count +
                              LOOM_SERVE_TEXT_STORAGE_ALLOCATION_COUNT + 1;
  if (reservation->tail != UINT32_MAX) {
    for (iree_host_size_t i = 0; i < state->storage.region_count; ++i) {
      capacity += state->storage.regions[i].blocks.count;
    }
  }
  loom_runtime_snapshot_range_t* ranges = NULL;
  iree_status_t status = iree_allocator_malloc_array(
      state->host_allocator, capacity, sizeof(*ranges), (void**)&ranges);
  iree_host_size_t count = 0;
  if (iree_status_is_ok(status)) {
    if (row_count) {
      memcpy(ranges, row_ranges, row_count * sizeof(*ranges));
      count += row_count;
    }
    if (reservation->restore_count) {
      memcpy(ranges + count, reservation->restore_ranges,
             reservation->restore_count * sizeof(*ranges));
      count += reservation->restore_count;
    }
    for (iree_host_size_t i = 0; i < LOOM_SERVE_TEXT_STORAGE_ALLOCATION_COUNT;
         ++i) {
      if (state->memory.private_lengths[i]) {
        ranges[count++] = (loom_runtime_snapshot_range_t){
            i, 0, state->memory.private_lengths[i]};
      }
    }
    if (reservation->reader != UINT32_MAX) {
      iree_hal_buffer_t* view = state->recurrent.buffers[reservation->reader];
      ranges[count++] =
          (loom_runtime_snapshot_range_t){1, iree_hal_buffer_byte_offset(view),
                                          iree_hal_buffer_byte_length(view)};
    }
    if (reservation->tail != UINT32_MAX) {
      for (iree_host_size_t i = 0; i < state->storage.region_count; ++i) {
        const loom_serve_text_cache_region_t* region =
            &state->storage.regions[i];
        for (iree_host_size_t plane = 0; plane < region->blocks.count;
             ++plane) {
          ranges[count++] = (loom_runtime_snapshot_range_t){
              region->allocation,
              region->blocks.origin + plane * region->blocks.stride +
                  reservation->tail * region->blocks.block_bytes,
              region->blocks.block_bytes};
        }
      }
    }
    // Requirements are a physical union. Snapshot descriptors retain their
    // independent logical byte order for capture/restore.
    iree_host_size_t physical_count = 0;
    for (iree_host_size_t i = 0; i < count; ++i) {
      loom_runtime_snapshot_range_t range = ranges[i];
      loom_runtime_virtual_buffer_t* buffer =
          state->memory.buffers[range.buffer_index];
      if (!buffer) {
        continue;
      }
      const uint64_t mapping = loom_runtime_virtual_buffer_mapping_size(buffer);
      const uint64_t end =
          iree_min(iree_device_align(range.offset + range.length, mapping),
                   iree_hal_buffer_byte_length(
                       loom_runtime_virtual_buffer_handle(buffer)));
      range.offset = range.offset / mapping * mapping;
      range.length = end - range.offset;
      ranges[physical_count++] = range;
    }
    qsort(ranges, physical_count, sizeof(*ranges),
          loom_serve_text_requirement_compare);
    count = 0;
    for (iree_host_size_t i = 0; i < physical_count; ++i) {
      const loom_runtime_snapshot_range_t range = ranges[i];
      if (count && ranges[count - 1].buffer_index == range.buffer_index &&
          ranges[count - 1].offset + ranges[count - 1].length >= range.offset) {
        ranges[count - 1].length =
            iree_max(ranges[count - 1].length,
                     range.offset + range.length - ranges[count - 1].offset);
      } else {
        ranges[count++] = range;
      }
    }
    uint64_t bytes = 0;
    for (iree_host_size_t i = 0; i < count; ++i) {
      bytes += iree_min(ranges[i].length, UINT64_MAX - bytes);
    }
    reservation->requirements = ranges;
    reservation->requirement_count = count;
    reservation->footprint_bytes = bytes;
    reservation->flags |= LOOM_SERVE_TEXT_RESERVATION_REQUIREMENTS;
  }
  iree_allocator_free(state->host_allocator, row_ranges);
  return status;
}

iree_status_t loom_serve_text_state_reservation_footprint(
    loom_serve_text_state_reservation_t* reservation, uint64_t* out_bytes) {
  *out_bytes = 0;
  IREE_RETURN_IF_ERROR(
      loom_serve_text_state_reservation_requirements(reservation));
  *out_bytes = reservation->footprint_bytes;
  return iree_ok_status();
}

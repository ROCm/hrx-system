// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/memory/fixed_block_pool_slab.h"

iree_status_t iree_hal_fixed_block_pool_slab_allocate(
    const iree_hal_pool_buffer_range_t* range,
    const iree_hal_fixed_block_pool_geometry_t* geometry,
    const iree_hal_asan_pool_options_t* asan,
    const iree_async_frontier_t* initial_frontier,
    iree_allocator_t host_allocator,
    iree_hal_fixed_block_pool_slab_t** out_slab) {
  *out_slab = NULL;
  const iree_device_size_t block_count =
      range->length / geometry->backing_block_size;
  if (!block_count ||
      block_count > IREE_HAL_MEMORY_FIXED_BLOCK_ALLOCATOR_MAX_BLOCKS) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "buffer range has an unsupported block count");
  }
  const bool asan_enabled = iree_hal_asan_pool_options_is_enabled(asan);
  iree_host_size_t total_size = 0;
  iree_host_size_t blocks_offset = 0;
  iree_host_size_t layouts_offset = 0;
  iree_host_size_t frontier_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(iree_hal_fixed_block_pool_slab_t), &total_size,
      IREE_STRUCT_FIELD(block_count, iree_hal_fixed_block_pool_block_t,
                        &blocks_offset),
      IREE_STRUCT_FIELD(asan_enabled ? block_count : 0,
                        iree_hal_asan_allocation_layout_t, &layouts_offset),
      IREE_STRUCT_FIELD_ALIGNED(1, iree_async_frontier_t,
                                IREE_ASYNC_FRONTIER_ALIGNMENT,
                                &frontier_offset),
      IREE_STRUCT_FIELD(geometry->frontier_capacity,
                        iree_async_frontier_entry_t, NULL)));
  iree_hal_fixed_block_pool_slab_t* slab = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total_size, (void**)&slab));
  slab->range = *range;
  iree_hal_buffer_retain(range->buffer);
  slab->block_count = (uint32_t)block_count;
  slab->blocks =
      (iree_hal_fixed_block_pool_block_t*)((uint8_t*)slab + blocks_offset);
  slab->asan_layouts =
      asan_enabled ? (iree_hal_asan_allocation_layout_t*)((uint8_t*)slab +
                                                          layouts_offset)
                   : NULL;
  slab->return_frontier =
      (iree_async_frontier_t*)((uint8_t*)slab + frontier_offset);
  for (uint32_t i = 0; i < slab->block_count; ++i) {
    slab->blocks[i].slab = slab;
  }
  const iree_hal_memory_fixed_block_allocator_options_t allocator_options = {
      .block_size = geometry->backing_block_size,
      .block_count = slab->block_count,
      .frontier_capacity = geometry->frontier_capacity,
      .initial_frontier = initial_frontier,
  };
  iree_status_t status = iree_hal_memory_fixed_block_allocator_allocate(
      allocator_options, host_allocator, &slab->allocator);
  if (iree_status_is_ok(status)) {
    if (initial_frontier) {
      memcpy(slab->return_frontier, initial_frontier,
             sizeof(iree_async_frontier_t) +
                 initial_frontier->entry_count *
                     sizeof(iree_async_frontier_entry_t));
    }
    *out_slab = slab;
  } else {
    iree_hal_fixed_block_pool_slab_free(slab, NULL, host_allocator);
  }
  return status;
}

iree_status_t iree_hal_fixed_block_pool_slab_prepare(
    iree_hal_pool_t* backing_pool,
    const iree_hal_pool_reservation_request_t* request,
    const iree_hal_fixed_block_pool_geometry_t* geometry,
    const iree_hal_asan_pool_options_t* asan,
    iree_hal_pool_reservation_t reservation, iree_hal_pool_acquire_info_t info,
    const iree_async_frontier_t* initial_frontier,
    iree_allocator_t host_allocator,
    iree_hal_fixed_block_pool_slab_t* reusable_slab,
    iree_hal_fixed_block_pool_slab_t** out_slab) {
  *out_slab = NULL;
  iree_hal_pool_buffer_range_t range = {0};
  iree_status_t status = iree_hal_pool_buffer_range_prepare(
      backing_pool, request, &reservation, geometry->alignment, asan, &range);
  range.memory.reuse_frontier = info.reuse_frontier;
  iree_hal_fixed_block_pool_slab_t* slab = reusable_slab;
  if (slab &&
      (!iree_status_is_ok(status) ||
       range.length / geometry->backing_block_size != slab->block_count)) {
    iree_hal_fixed_block_pool_slab_free(slab, NULL, host_allocator);
    slab = NULL;
  }
  if (iree_status_is_ok(status) && slab) {
    iree_hal_memory_fixed_block_allocator_reset(slab->allocator,
                                                initial_frontier);
    slab->range = range;
    range.buffer = NULL;
    if (initial_frontier) {
      memcpy(
          slab->return_frontier, initial_frontier,
          sizeof(*initial_frontier) + initial_frontier->entry_count *
                                          sizeof(initial_frontier->entries[0]));
    } else {
      iree_async_frontier_initialize(slab->return_frontier, 0);
    }
  } else if (iree_status_is_ok(status)) {
    status = iree_hal_fixed_block_pool_slab_allocate(
        &range, geometry, asan, initial_frontier, host_allocator, &slab);
  }
  if (iree_status_is_ok(status)) {
    slab->reservation = reservation;
    *out_slab = slab;
  }
  iree_hal_pool_buffer_range_deinitialize(&range);
  if (!iree_status_is_ok(status)) {
    iree_hal_pool_release_reservations(backing_pool, 1, &reservation,
                                       info.reuse_frontier);
  }
  return status;
}

bool iree_hal_fixed_block_pool_slab_merge_return(
    iree_hal_fixed_block_pool_slab_t* slab,
    const iree_hal_fixed_block_pool_geometry_t* geometry,
    iree_hal_pool_history_t* history) {
  if (!iree_hal_memory_fixed_block_allocator_merge_free_frontiers(
          slab->allocator, geometry->frontier_capacity,
          history->timeline ? iree_hal_pool_history_collapse : NULL, history,
          slab->return_frontier)) {
    return false;
  }
  const iree_device_size_t managed_length =
      slab->block_count * geometry->backing_block_size;
  if (managed_length != slab->reservation.byte_length &&
      slab->range.memory.reuse_frontier) {
    if (!iree_async_frontier_merge(slab->return_frontier,
                                   geometry->frontier_capacity,
                                   slab->range.memory.reuse_frontier)) {
      if (!history->timeline) {
        return false;
      }
      iree_hal_pool_history_collapse(history, slab->return_frontier,
                                     slab->range.memory.reuse_frontier);
    }
  }
  return true;
}

void iree_hal_fixed_block_pool_slab_release_backing(
    iree_hal_fixed_block_pool_slab_t* slab, iree_hal_pool_t* backing_pool) {
  iree_hal_pool_buffer_range_deinitialize(&slab->range);
  iree_hal_pool_release_reservations(
      backing_pool, 1, &slab->reservation,
      slab->return_frontier->entry_count ? slab->return_frontier : NULL);
  slab->reservation = (iree_hal_pool_reservation_t){0};
  slab->next = slab->previous = NULL;
  slab->candidate_next = slab->preparation_next = NULL;
}

void iree_hal_fixed_block_pool_slab_free(iree_hal_fixed_block_pool_slab_t* slab,
                                         iree_hal_pool_t* backing_pool,
                                         iree_allocator_t host_allocator) {
  if (backing_pool && slab->reservation.byte_length) {
    iree_hal_fixed_block_pool_slab_release_backing(slab, backing_pool);
  } else {
    iree_hal_pool_buffer_range_deinitialize(&slab->range);
  }
  iree_hal_memory_fixed_block_allocator_free(slab->allocator);
  iree_allocator_free(host_allocator, slab);
}

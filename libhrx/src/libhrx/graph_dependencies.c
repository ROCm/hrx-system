// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include "graph_dependencies.h"

void hrx_graph_dependency_cursor_initialize(
    const hrx_graph_schedule_t* schedule,
    hrx_graph_dependency_cursor_t* out_cursor) {
  uint32_t block_count = schedule->block_count;
  if (schedule->partition_count &&
      schedule->partitions[schedule->partition_count - 1].stream_count > 1) {
    ++block_count;
  }
  *out_cursor = (hrx_graph_dependency_cursor_t){
      .schedule = schedule,
      .block_count = block_count,
      .semaphore_count = block_count ? block_count - 1 : 0,
  };
}

bool hrx_graph_dependency_cursor_next(hrx_graph_dependency_cursor_t* cursor,
                                      hrx_graph_dependency_block_t* out_block) {
  const hrx_graph_schedule_t* schedule = cursor->schedule;
  if (cursor->partition_index < schedule->partition_count) {
    const hrx_graph_partition_t* partition =
        &schedule->partitions[cursor->partition_index];
    const uint8_t wait_count =
        cursor->partition_index
            ? schedule->partitions[cursor->partition_index - 1].stream_count
            : 0;
    *out_block = (hrx_graph_dependency_block_t){
        .partition = partition,
        .dependencies =
            {
                .wait_count = wait_count,
                .wait_distance = wait_count + cursor->stream_index,
            },
        .stream_index = cursor->stream_index,
    };
    if (++cursor->stream_index == partition->stream_count) {
      cursor->stream_index = 0;
      ++cursor->partition_index;
    }
    return true;
  }
  if (cursor->partition_index == schedule->partition_count &&
      cursor->block_count > schedule->block_count) {
    const uint8_t wait_count =
        schedule->partitions[schedule->partition_count - 1].stream_count;
    *out_block = (hrx_graph_dependency_block_t){
        .dependencies =
            {
                .wait_count = wait_count,
                .wait_distance = wait_count,
            },
    };
    ++cursor->partition_index;
    return true;
  }
  return false;
}

iree_status_t hrx_graph_dependencies_allocate(
    iree_hal_device_t* device, uint32_t semaphore_count,
    iree_arena_allocator_t* arena, iree_hal_resource_set_t* resource_set,
    iree_hal_semaphore_t*** out_semaphores) {
  *out_semaphores = NULL;
  if (!semaphore_count) {
    return iree_ok_status();
  }
  iree_hal_semaphore_t** semaphores = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(
      arena, semaphore_count * sizeof(*semaphores), (void**)&semaphores));
  iree_status_t status = iree_ok_status();
  for (uint32_t i = 0; iree_status_is_ok(status) && i < semaphore_count; ++i) {
    iree_hal_semaphore_t* semaphore = NULL;
    status =
        iree_hal_semaphore_create(device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
                                  IREE_HAL_SEMAPHORE_FLAG_NONE, &semaphore);
    if (iree_status_is_ok(status)) {
      status = iree_hal_resource_set_insert(resource_set, 1, &semaphore);
      if (iree_status_is_ok(status)) {
        semaphores[i] = semaphore;
      }
      iree_hal_semaphore_release(semaphore);
    }
  }
  if (iree_status_is_ok(status)) {
    *out_semaphores = semaphores;
  }
  return status;
}

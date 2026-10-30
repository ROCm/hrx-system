// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/timestamp.h"

iree_status_t iree_hal_amd_gpu_timestamp_writes_append(
    iree_hal_amd_gpu_recording_t* recording, iree_hal_buffer_ref_t target,
    iree_hal_amd_gpu_timestamp_writes_t* writes) {
  if (!writes->pending) {
    void* scratch = NULL;
    IREE_RETURN_IF_ERROR(iree_hal_amd_gpu_recording_data(
        recording, 4, 4, &writes->scratch_offset, &scratch));
  }
  if (writes->count == writes->capacity) {
    if (writes->capacity > UINT32_MAX / sizeof(*writes->pending) / 2) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "too many pending timestamp writes");
    }
    const uint32_t capacity = writes->capacity ? writes->capacity * 2 : 4;
    IREE_RETURN_IF_ERROR(iree_allocator_realloc(
        recording->host_allocator, capacity * sizeof(*writes->pending),
        (void**)&writes->pending));
    writes->capacity = capacity;
  }
  target.length = 8;
  writes->pending[writes->count++] = target;
  return iree_ok_status();
}

void iree_hal_amd_gpu_timestamp_writes_deinitialize(
    iree_allocator_t allocator, iree_hal_amd_gpu_timestamp_writes_t* writes) {
  iree_allocator_free(allocator, writes->pending);
  *writes = (iree_hal_amd_gpu_timestamp_writes_t){0};
}

static bool iree_hal_amd_gpu_timestamp_range_overlaps(
    iree_hal_buffer_ref_t timestamp, iree_hal_buffer_ref_t range) {
  if (!range.length) {
    return false;
  }
  if (timestamp.buffer && range.buffer) {
    // Distinct imported objects may name the same physical storage. Only a
    // shared logical allocation establishes comparable byte coordinates here.
    if (iree_hal_buffer_allocated_buffer(timestamp.buffer) !=
        iree_hal_buffer_allocated_buffer(range.buffer)) {
      return true;
    }
    return iree_hal_buffer_test_overlap(
               timestamp.buffer, timestamp.offset, 8, range.buffer,
               range.offset, range.length) != IREE_HAL_BUFFER_OVERLAP_DISJOINT;
  }
  if (timestamp.buffer || range.buffer ||
      timestamp.buffer_slot != range.buffer_slot) {
    return true;
  }
  // Subtraction avoids overflow for the unresolved whole-buffer sentinel.
  return timestamp.offset >= range.offset
             ? timestamp.offset - range.offset < range.length
             : range.offset - timestamp.offset < 8;
}

bool iree_hal_amd_gpu_timestamp_writes_overlap(
    const iree_hal_amd_gpu_timestamp_writes_t* writes,
    const iree_hal_barrier_t* barrier) {
  if (!writes->count ||
      (barrier->source_stage_mask &&
       !iree_any_bit_set(barrier->source_stage_mask,
                         IREE_HAL_EXECUTION_STAGE_TRANSFER |
                             IREE_HAL_EXECUTION_STAGE_COMMAND_PROCESS |
                             IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE))) {
    return false;
  }
  if (iree_any_bit_set(barrier->flags,
                       IREE_HAL_BARRIER_FLAG_RELEASE_SYSTEM_SCOPE) ||
      iree_any_bit_set(barrier->effects.bits,
                       IREE_HAL_MEMORY_EFFECT_GLOBAL_RELEASE_TO_SYSTEM)) {
    return true;
  }
  const iree_hal_access_scope_t scopes =
      IREE_HAL_ACCESS_SCOPE_TRANSFER_WRITE | IREE_HAL_ACCESS_SCOPE_MEMORY_WRITE;
  for (iree_host_size_t i = 0; i < barrier->memory_barrier_count; ++i) {
    if (iree_any_bit_set(barrier->memory_barriers[i].source_scope, scopes)) {
      return true;
    }
  }
  for (iree_host_size_t i = 0; i < barrier->buffer_barrier_count; ++i) {
    const iree_hal_buffer_barrier_t* range = &barrier->buffer_barriers[i];
    if (!iree_any_bit_set(range->source_scope, scopes) &&
        !(range->recipe &&
          iree_any_bit_set(range->recipe->effects.bits,
                           IREE_HAL_MEMORY_EFFECT_RANGE_RELEASE_TO_SYSTEM))) {
      continue;
    }
    for (uint32_t j = 0; j < writes->count; ++j) {
      if (iree_hal_amd_gpu_timestamp_range_overlaps(writes->pending[j],
                                                    range->buffer_ref)) {
        return true;
      }
    }
  }
  return false;
}

// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include "barrier.h"

#include "buffer.h"

#define HRX_CHECK_BARRIER_ENUM(name)                      \
  IREE_STATIC_ASSERT_ENUM_EQ(HRX_##name, IREE_HAL_##name, \
                             "barrier semantics must match HAL")
HRX_CHECK_BARRIER_ENUM(EXECUTION_STAGE_COMMAND_ISSUE);
HRX_CHECK_BARRIER_ENUM(EXECUTION_STAGE_COMMAND_PROCESS);
HRX_CHECK_BARRIER_ENUM(EXECUTION_STAGE_DISPATCH);
HRX_CHECK_BARRIER_ENUM(EXECUTION_STAGE_TRANSFER);
HRX_CHECK_BARRIER_ENUM(EXECUTION_STAGE_COMMAND_RETIRE);
HRX_CHECK_BARRIER_ENUM(EXECUTION_STAGE_HOST);
HRX_CHECK_BARRIER_ENUM(EXECUTION_STAGE_ATOMIC);
HRX_CHECK_BARRIER_ENUM(ACCESS_SCOPE_INDIRECT_COMMAND_READ);
HRX_CHECK_BARRIER_ENUM(ACCESS_SCOPE_CONSTANT_READ);
HRX_CHECK_BARRIER_ENUM(ACCESS_SCOPE_DISPATCH_READ);
HRX_CHECK_BARRIER_ENUM(ACCESS_SCOPE_DISPATCH_WRITE);
HRX_CHECK_BARRIER_ENUM(ACCESS_SCOPE_TRANSFER_READ);
HRX_CHECK_BARRIER_ENUM(ACCESS_SCOPE_TRANSFER_WRITE);
HRX_CHECK_BARRIER_ENUM(ACCESS_SCOPE_HOST_READ);
HRX_CHECK_BARRIER_ENUM(ACCESS_SCOPE_HOST_WRITE);
HRX_CHECK_BARRIER_ENUM(ACCESS_SCOPE_MEMORY_READ);
HRX_CHECK_BARRIER_ENUM(ACCESS_SCOPE_MEMORY_WRITE);
HRX_CHECK_BARRIER_ENUM(ACCESS_SCOPE_ATOMIC_READ);
HRX_CHECK_BARRIER_ENUM(ACCESS_SCOPE_ATOMIC_WRITE);
HRX_CHECK_BARRIER_ENUM(BARRIER_FLAG_NONE);
HRX_CHECK_BARRIER_ENUM(BARRIER_FLAG_ACQUIRE_SYSTEM_SCOPE);
HRX_CHECK_BARRIER_ENUM(BARRIER_FLAG_RELEASE_SYSTEM_SCOPE);
#undef HRX_CHECK_BARRIER_ENUM

iree_status_t hrx_barrier_storage_initialize(
    const hrx_queue_barriers_t* barriers, hrx_barrier_storage_t* out_storage) {
  out_storage->allocation = NULL;
  const hrx_barrier_list_t* lists[2] = {barriers->before, barriers->after};
  size_t barrier_count = 0;
  size_t memory_count = 0;
  size_t buffer_count = 0;
  for (size_t side = 0; side < 2; ++side) {
    if (!lists[side]) {
      continue;
    }
    if (lists[side]->count && !lists[side]->values) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "barrier list has no records");
    }
    if (!iree_host_size_checked_add(barrier_count, lists[side]->count,
                                    &barrier_count)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "barrier count overflows");
    }
    for (size_t i = 0; i < lists[side]->count; ++i) {
      const hrx_barrier_t* barrier = &lists[side]->values[i];
      if ((barrier->memory_barrier_count && !barrier->memory_barriers) ||
          (barrier->buffer_barrier_count && !barrier->buffer_barriers)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "barrier has no dependency storage");
      }
      if (!iree_host_size_checked_add(
              memory_count, barrier->memory_barrier_count, &memory_count) ||
          !iree_host_size_checked_add(
              buffer_count, barrier->buffer_barrier_count, &buffer_count)) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "barrier dependency count overflows");
      }
      for (size_t j = 0; j < barrier->buffer_barrier_count; ++j) {
        const hrx_buffer_t buffer =
            barrier->buffer_barriers[j].buffer_ref.buffer;
        if (!buffer || !buffer->hal_buffer) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "barrier requires a concrete native buffer");
        }
      }
    }
  }

  iree_hal_barrier_t* hal_barriers = out_storage->inline_barriers;
  iree_hal_memory_barrier_t* hal_memory = out_storage->inline_memory_barriers;
  iree_hal_buffer_barrier_t* hal_buffers = out_storage->inline_buffer_barriers;
  if (barrier_count > IREE_ARRAYSIZE(out_storage->inline_barriers) ||
      memory_count > IREE_ARRAYSIZE(out_storage->inline_memory_barriers) ||
      buffer_count > IREE_ARRAYSIZE(out_storage->inline_buffer_barriers)) {
    iree_host_size_t total_size = 0;
    iree_host_size_t barriers_offset = 0;
    iree_host_size_t memory_offset = 0;
    iree_host_size_t buffers_offset = 0;
    IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
        0, &total_size,
        IREE_STRUCT_FIELD(barrier_count, iree_hal_barrier_t, &barriers_offset),
        IREE_STRUCT_FIELD(memory_count, iree_hal_memory_barrier_t,
                          &memory_offset),
        IREE_STRUCT_FIELD(buffer_count, iree_hal_buffer_barrier_t,
                          &buffers_offset)));
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_uninitialized(
        iree_allocator_system(), total_size, &out_storage->allocation));
    uint8_t* base = out_storage->allocation;
    hal_barriers = (iree_hal_barrier_t*)(base + barriers_offset);
    hal_memory = (iree_hal_memory_barrier_t*)(base + memory_offset);
    hal_buffers = (iree_hal_buffer_barrier_t*)(base + buffers_offset);
  }
  out_storage->barriers = (iree_hal_queue_barriers_t){
      .before = lists[0] ? &out_storage->lists[0] : NULL,
      .after = lists[1] ? &out_storage->lists[1] : NULL,
  };
  for (size_t side = 0; side < 2; ++side) {
    if (!lists[side]) {
      continue;
    }
    out_storage->lists[side] = (iree_hal_barrier_list_t){
        .count = lists[side]->count,
        .values = hal_barriers,
    };
    for (size_t i = 0; i < lists[side]->count; ++i) {
      const hrx_barrier_t* barrier = &lists[side]->values[i];
      *hal_barriers++ = (iree_hal_barrier_t){
          .source_stage_mask = barrier->source_stage_mask,
          .target_stage_mask = barrier->target_stage_mask,
          .flags = barrier->flags,
          .effects = {barrier->effects.bits},
          .memory_barrier_count = barrier->memory_barrier_count,
          .memory_barriers = hal_memory,
          .buffer_barrier_count = barrier->buffer_barrier_count,
          .buffer_barriers = hal_buffers,
      };
      for (size_t j = 0; j < barrier->memory_barrier_count; ++j) {
        *hal_memory++ = (iree_hal_memory_barrier_t){
            .source_scope = barrier->memory_barriers[j].source_scope,
            .target_scope = barrier->memory_barriers[j].target_scope,
        };
      }
      for (size_t j = 0; j < barrier->buffer_barrier_count; ++j) {
        const hrx_buffer_barrier_t* range = &barrier->buffer_barriers[j];
        *hal_buffers++ = (iree_hal_buffer_barrier_t){
            .source_scope = range->source_scope,
            .target_scope = range->target_scope,
            .buffer_ref = iree_hal_make_buffer_ref(
                range->buffer_ref.buffer->hal_buffer, range->buffer_ref.offset,
                range->buffer_ref.length),
            .recipe = (const iree_hal_memory_transition_recipe_t*)range->recipe,
        };
      }
    }
  }
  return iree_ok_status();
}

void hrx_barrier_storage_deinitialize(hrx_barrier_storage_t* storage) {
  if (storage->allocation) {
    iree_allocator_free(iree_allocator_system(), storage->allocation);
  }
}

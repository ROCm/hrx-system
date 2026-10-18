// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/memory/buffer_range.h"

#include "iree/hal/memory_scope.h"

static iree_status_t iree_hal_pool_buffer_range_initialize_view(
    iree_hal_buffer_t* buffer, iree_device_size_t offset,
    iree_device_size_t length, iree_hal_buffer_memory_view_t memory,
    iree_device_size_t alignment, const iree_hal_asan_pool_options_t* asan,
    iree_hal_pool_buffer_range_t* out_range) {
  memset(out_range, 0, sizeof(*out_range));
  if (!memory.backing) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "buffer has no prepared native storage");
  }
  const iree_hal_buffer_backing_facts_t* backing = memory.backing;
  if (!backing->notification || !backing->tracker) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "buffer has no sealed-group progress services");
  }
  if (!iree_device_size_is_valid_alignment(alignment)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "range alignment must be a power of two");
  }
  alignment = alignment ? alignment : 1;
  if (alignment > backing->allocation_alignment) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "range alignment exceeds native backing guarantee");
  }
  // Maintenance is aligned in native storage coordinates. A larger granule
  // changes the managed endpoints without promising a stronger address
  // alignment than the caller requested from the backing.
  alignment = iree_max(alignment, backing->maintenance_alignment);
  IREE_RETURN_IF_ERROR(iree_hal_asan_pool_options_validate(asan));
  if (iree_hal_asan_pool_options_is_enabled(asan)) {
    if (!backing->advice) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "buffer has no native ASAN range advice");
    }
    IREE_RETURN_IF_ERROR(
        backing->advice->validate_asan(backing->advice->user_data, asan));
  }
  const iree_device_size_t padding = -memory.offset & (alignment - 1);
  if (padding >= length || length - padding < alignment) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "buffer range contains no aligned allocation");
  }
  out_range->offset = offset + padding;
  out_range->length = (length - padding) & ~(alignment - 1);
  memory.offset += padding;
  memory.binding_offset += padding;
  out_range->memory = memory;
  out_range->buffer = buffer;
  iree_hal_buffer_retain(buffer);
  return iree_ok_status();
}

iree_status_t iree_hal_pool_buffer_range_initialize(
    iree_hal_buffer_t* buffer, iree_device_size_t offset,
    iree_device_size_t length, iree_device_size_t alignment,
    const iree_hal_asan_pool_options_t* asan,
    iree_hal_pool_buffer_range_t* out_range) {
  memset(out_range, 0, sizeof(*out_range));
  IREE_RETURN_IF_ERROR(
      iree_hal_buffer_calculate_range(0, iree_hal_buffer_byte_length(buffer),
                                      offset, length, &offset, &length));
  iree_hal_buffer_memory_view_t memory = iree_hal_buffer_memory_view(buffer);
  memory.offset += offset;
  memory.binding_offset += offset;
  return iree_hal_pool_buffer_range_initialize_view(
      buffer, offset, length, memory, alignment, asan, out_range);
}

iree_status_t iree_hal_pool_buffer_range_prepare(
    iree_hal_pool_t* pool, const iree_hal_pool_reservation_request_t* request,
    const iree_hal_pool_reservation_t* reservation,
    iree_device_size_t alignment, const iree_hal_asan_pool_options_t* asan,
    iree_hal_pool_buffer_range_t* out_range) {
  iree_hal_pool_reservation_view_t view;
  if (iree_hal_pool_query_reservation_views(pool, 1, reservation, &view)) {
    return iree_hal_pool_buffer_range_initialize_view(
        view.buffer,
        view.byte_offset - iree_hal_buffer_byte_offset(view.buffer),
        view.byte_length, view.memory, alignment, asan, out_range);
  }
  memset(out_range, 0, sizeof(*out_range));
  iree_hal_buffer_t* buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_pool_materialize_reservations(
      pool, 1, request, reservation, IREE_HAL_POOL_MATERIALIZE_FLAG_NONE,
      &buffer));
  iree_status_t status = iree_hal_pool_buffer_range_initialize(
      buffer, 0, IREE_HAL_WHOLE_BUFFER, alignment, asan, out_range);
  iree_hal_buffer_release(buffer);
  return status;
}

void iree_hal_pool_buffer_range_deinitialize(
    iree_hal_pool_buffer_range_t* range) {
  iree_hal_buffer_release(range->buffer);
  memset(range, 0, sizeof(*range));
}

void iree_hal_pool_buffer_range_query_reservation_view(
    const iree_hal_pool_buffer_range_t* range, iree_device_size_t offset,
    iree_device_size_t length, const iree_async_frontier_t* reuse_frontier,
    iree_hal_pool_reservation_view_t* out_view) {
  *out_view = (iree_hal_pool_reservation_view_t){
      .buffer = range->buffer,
      .byte_offset =
          iree_hal_buffer_byte_offset(range->buffer) + range->offset + offset,
      .byte_length = length,
      .memory = range->memory,
  };
  out_view->memory.offset += offset;
  out_view->memory.binding_offset += offset;
  out_view->memory.reuse_frontier =
      reuse_frontier && reuse_frontier->entry_count ? reuse_frontier : NULL;
}

iree_status_t iree_hal_pool_buffer_range_materialize(
    const iree_hal_pool_buffer_range_t* range, iree_device_size_t offset,
    iree_device_size_t length, const iree_async_frontier_t* reuse_frontier,
    iree_hal_buffer_release_callback_t release_callback,
    iree_allocator_t host_allocator, iree_hal_buffer_t** out_buffer) {
  *out_buffer = NULL;
  iree_hal_memory_access_t allowed_access =
      iree_hal_buffer_allowed_access(range->buffer);
  iree_hal_buffer_usage_t allowed_usage =
      iree_hal_buffer_allowed_usage(range->buffer);
  if (range->memory.contract) {
    allowed_access = range->memory.contract->allowed_access;
    allowed_usage = range->memory.contract->allowed_usage;
  }
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_access(
      iree_hal_buffer_allowed_access(range->buffer), allowed_access));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_usage(
      iree_hal_buffer_allowed_usage(range->buffer), allowed_usage));
  IREE_RETURN_IF_ERROR(iree_hal_subspan_buffer_create_with_callback(
      range->buffer,
      iree_hal_buffer_byte_offset(range->buffer) + range->offset + offset,
      length, release_callback, host_allocator, out_buffer));
  (*out_buffer)->allowed_access = allowed_access;
  (*out_buffer)->allowed_usage = allowed_usage;
  (*out_buffer)->memory = range->memory;
  (*out_buffer)->memory.offset += offset;
  (*out_buffer)->memory.binding_offset += offset;
  (*out_buffer)->memory.reuse_frontier = reuse_frontier;
  return iree_ok_status();
}

void iree_hal_pool_buffer_range_advise_asan(
    const iree_hal_pool_buffer_range_t* range, iree_device_size_t offset,
    iree_hal_asan_range_advice_flags_t flags,
    const iree_hal_asan_allocation_layout_t* layout) {
  const iree_hal_buffer_range_advice_t* advice = range->memory.backing->advice;
  advice->advise_asan(advice->user_data, range->memory.offset + offset, flags,
                      layout);
}

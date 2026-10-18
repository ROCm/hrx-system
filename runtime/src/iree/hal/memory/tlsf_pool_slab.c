// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/memory/tlsf_pool_slab.h"

#include "iree/base/internal/math.h"

iree_status_t iree_hal_tlsf_pool_slab_initialize_range(
    iree_hal_pool_buffer_range_t range, iree_hal_memory_tlsf_options_t options,
    iree_allocator_t host_allocator, iree_hal_tlsf_pool_slab_t** out_slab) {
  iree_hal_tlsf_pool_slab_t* slab = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*slab), (void**)&slab));
  slab->range = range;
  options.range_length = range.length;
  if (!options.initial_frontier) {
    options.initial_frontier = range.memory.reuse_frontier;
  }
  iree_status_t status =
      iree_hal_memory_tlsf_initialize(options, host_allocator, &slab->tlsf);
  if (iree_status_is_ok(status)) {
    *out_slab = slab;
  } else {
    iree_allocator_free(host_allocator, slab);
  }
  return status;
}

iree_status_t iree_hal_tlsf_pool_slab_prepare(
    iree_hal_pool_t* backing_pool,
    const iree_hal_pool_reservation_request_t* request,
    iree_hal_pool_reservation_t reservation, iree_hal_pool_acquire_info_t info,
    iree_hal_memory_tlsf_options_t options,
    const iree_hal_asan_pool_options_t* asan, iree_allocator_t host_allocator,
    iree_hal_tlsf_pool_slab_t* reusable_slab,
    iree_hal_tlsf_pool_slab_t** out_slab) {
  iree_hal_pool_buffer_range_t range = {0};
  const iree_async_frontier_t* initial_frontier =
      options.initial_frontier ? options.initial_frontier : info.reuse_frontier;
  iree_status_t status =
      iree_hal_pool_buffer_range_prepare(backing_pool, request, &reservation,
                                         request->min_alignment, asan, &range);
  range.memory.reuse_frontier = info.reuse_frontier;
  iree_hal_tlsf_pool_slab_t* slab = reusable_slab;
  if (slab &&
      (!iree_status_is_ok(status) || range.length != slab->tlsf.range_length)) {
    iree_hal_tlsf_pool_slab_release(NULL, slab, host_allocator);
    slab = NULL;
  }
  if (iree_status_is_ok(status) && slab) {
    iree_hal_memory_tlsf_reset(&slab->tlsf, initial_frontier);
    slab->range = range;
  } else if (iree_status_is_ok(status)) {
    status = iree_hal_tlsf_pool_slab_initialize_range(range, options,
                                                      host_allocator, &slab);
  }
  if (iree_status_is_ok(status)) {
    slab->reservation = reservation;
    slab->release_frontier = info.reuse_frontier;
    *out_slab = slab;
  } else {
    iree_hal_pool_buffer_range_deinitialize(&range);
    iree_hal_pool_release_reservations(backing_pool, 1, &reservation,
                                       info.reuse_frontier);
  }
  return status;
}

void iree_hal_tlsf_pool_slab_release_backing(iree_hal_pool_t* backing_pool,
                                             iree_hal_tlsf_pool_slab_t* slab) {
  iree_hal_pool_buffer_range_deinitialize(&slab->range);
  iree_hal_pool_release_reservations(backing_pool, 1, &slab->reservation,
                                     slab->release_frontier);
  slab->reservation = (iree_hal_pool_reservation_t){0};
  slab->release_frontier = NULL;
  slab->next = slab->previous = NULL;
}

void iree_hal_tlsf_pool_slab_release(iree_hal_pool_t* backing_pool,
                                     iree_hal_tlsf_pool_slab_t* slab,
                                     iree_allocator_t host_allocator) {
  if (backing_pool && slab->reservation.byte_length) {
    iree_hal_tlsf_pool_slab_release_backing(backing_pool, slab);
  }
  iree_hal_memory_tlsf_deinitialize(&slab->tlsf);
  iree_allocator_free(host_allocator, slab);
}

// Returns the history of the entire parent reservation, including bytes that
// endpoint alignment excluded from the child's managed interval.
bool iree_hal_tlsf_pool_slab_query_full_free(
    iree_hal_tlsf_pool_slab_t* slab, const iree_async_frontier_t** out_frontier,
    iree_hal_memory_tlsf_block_flags_t* out_flags) {
  if (!iree_hal_memory_tlsf_query_full_free_block(&slab->tlsf, out_frontier,
                                                  out_flags)) {
    return false;
  }
  if (slab->reservation.byte_length &&
      slab->range.length != slab->reservation.byte_length) {
    iree_hal_memory_tlsf_merge_full_free_frontier(
        &slab->tlsf, slab->range.memory.reuse_frontier);
    iree_hal_memory_tlsf_query_full_free_block(&slab->tlsf, out_frontier,
                                               out_flags);
  }
  return true;
}

iree_status_t iree_hal_tlsf_pool_slab_calculate_geometry(
    const iree_hal_memory_tlsf_options_t* options,
    const iree_hal_asan_pool_options_t* asan, iree_device_size_t* out_alignment,
    iree_device_size_t* out_length) {
  iree_device_size_t alignment = options->alignment;
  if (!alignment) {
    alignment = IREE_HAL_MEMORY_TLSF_MIN_ALIGNMENT;
  }
  if (alignment < IREE_HAL_MEMORY_TLSF_MIN_ALIGNMENT ||
      !iree_device_size_is_power_of_two(alignment)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "TLSF alignment must be a power of two at least 16");
  }
  iree_device_size_t length = options->range_length;
  if (iree_hal_asan_pool_options_is_enabled(asan)) {
    alignment = iree_max(alignment, iree_max(asan->backing_alignment,
                                             asan->shadow_granule_size));
    iree_hal_asan_allocation_layout_t layout;
    IREE_RETURN_IF_ERROR(iree_hal_asan_calculate_allocation_layout(
        asan, length, alignment, &layout));
    length = layout.backing_length;
  }
  if (!iree_device_size_checked_align(length, alignment, &length)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "TLSF backing size overflows alignment");
  }
  *out_alignment = alignment;
  *out_length = length;
  return iree_ok_status();
}

iree_status_t iree_hal_tlsf_pool_slab_resolve_request(
    const iree_hal_pool_capabilities_t* capabilities,
    const iree_hal_memory_tlsf_options_t* options,
    const iree_hal_asan_pool_options_t* asan,
    iree_hal_pool_reservation_request_t* out_request) {
  iree_device_size_t alignment = 0;
  iree_device_size_t length = 0;
  IREE_RETURN_IF_ERROR(iree_hal_tlsf_pool_slab_calculate_geometry(
      options, asan, &alignment, &length));
  if (!options->range_length) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "TLSF range_length must be > 0");
  }
  if (alignment > capabilities->max_allocation_alignment) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "TLSF alignment exceeds backing pool support");
  }
  // Native address alignment and storage-relative maintenance endpoints are
  // independent. Reserve complete granules without asking the parent to
  // strengthen its native address guarantee.
  const iree_device_size_t granule =
      iree_max(alignment, capabilities->maintenance_alignment);
  if (!iree_device_size_checked_align(length, granule, &length)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "TLSF backing size overflows maintenance granule");
  }
  alignment = iree_min(granule, capabilities->max_allocation_alignment);
  if (capabilities->max_allocation_size &&
      length > capabilities->max_allocation_size) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "TLSF slab size exceeds backing pool support");
  }
  *out_request = (iree_hal_pool_reservation_request_t){
      .allocation_size = length,
      .min_alignment = alignment,
  };
  return iree_ok_status();
}

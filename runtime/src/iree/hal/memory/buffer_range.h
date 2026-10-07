// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_MEMORY_BUFFER_RANGE_H_
#define IREE_HAL_MEMORY_BUFFER_RANGE_H_

#include "iree/hal/pool.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Retained prepared range for an offset allocator. Ownership is one buffer
// reference; explicitly managed allocation epochs remain the caller's
// obligation, including a child pool holding an ordinary backing reservation.
typedef struct iree_hal_pool_buffer_range_t {
  // Source view retained once for the allocator's lifetime.
  iree_hal_buffer_t* buffer;
  // First managed byte relative to the source view.
  iree_device_size_t offset;
  // Managed extent after alignment of the supplied range's endpoints.
  iree_device_size_t length;
  // Captured storage facts with offset at the first managed byte.
  iree_hal_buffer_memory_view_t memory;
} iree_hal_pool_buffer_range_t;

// Qualifies and retains a prepared buffer range. Rounds its endpoints inward to
// alignment, including the backing's independent maintenance granule. No bytes
// outside the supplied range are used. The requested address alignment must be
// supported by the backing; the maintenance granule may be larger without
// implying stronger address alignment. Failure leaves an empty output.
iree_status_t iree_hal_pool_buffer_range_initialize(
    iree_hal_buffer_t* buffer, iree_device_size_t offset,
    iree_device_size_t length, iree_device_size_t alignment,
    const iree_hal_asan_pool_options_t* asan,
    iree_hal_pool_buffer_range_t* out_range);

// Releases the retained source after every child reservation has retired.
void iree_hal_pool_buffer_range_deinitialize(
    iree_hal_pool_buffer_range_t* range);

// Resolves a trusted live subrange to its existing prepared storage. The
// returned view borrows |range| and |reuse_frontier| for the reservation
// lifetime and performs no allocation or retention.
void iree_hal_pool_buffer_range_query_reservation_view(
    const iree_hal_pool_buffer_range_t* range, iree_device_size_t offset,
    iree_device_size_t length, const iree_async_frontier_t* reuse_frontier,
    iree_hal_pool_reservation_view_t* out_view);

// Materializes a reservation with narrowed permissions and exact reuse history.
// Offsets are relative to the managed range. Transfers the callback on success.
iree_status_t iree_hal_pool_buffer_range_materialize(
    const iree_hal_pool_buffer_range_t* range, iree_device_size_t offset,
    iree_device_size_t length, iree_hal_buffer_params_t params,
    const iree_async_frontier_t* reuse_frontier,
    iree_hal_buffer_release_callback_t release_callback,
    iree_allocator_t host_allocator, iree_hal_buffer_t** out_buffer);

// Applies qualified lifecycle advice in the original native backing
// coordinates.
void iree_hal_pool_buffer_range_advise_asan(
    const iree_hal_pool_buffer_range_t* range, iree_device_size_t offset,
    iree_hal_asan_range_advice_flags_t flags,
    const iree_hal_asan_allocation_layout_t* layout);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_MEMORY_BUFFER_RANGE_H_

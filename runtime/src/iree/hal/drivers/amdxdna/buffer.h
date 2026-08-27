// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDXDNA_BUFFER_H_
#define IREE_HAL_DRIVERS_AMDXDNA_BUFFER_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/amdxdna/native.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

iree_status_t iree_hal_amdxdna_buffer_wrap(
    iree_hal_amdxdna_native_buffer_t* native_buffer,
    iree_hal_buffer_placement_t placement, iree_hal_memory_type_t memory_type,
    iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_usage_t allowed_usage, iree_device_size_t allocation_size,
    iree_device_size_t byte_offset, iree_device_size_t byte_length,
    iree_hal_buffer_release_callback_t release_callback,
    iree_allocator_t host_allocator, iree_hal_buffer_t** out_buffer);

iree_hal_amdxdna_native_buffer_t* iree_hal_amdxdna_buffer_handle(
    iree_hal_buffer_t* base_buffer);

// Resolves a mapping/flush range onto the root native BO. Subspan buffers
// store a non-zero |byte_offset|; flush/invalidate/map must clflush that
// root offset rather than the local mapping offset.
iree_status_t iree_hal_amdxdna_buffer_resolve_root_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length,
    iree_device_size_t* out_root_byte_offset,
    iree_device_size_t* out_byte_length);

iree_status_t iree_hal_amdxdna_buffer_invalidate_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length);

iree_status_t iree_hal_amdxdna_buffer_flush_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length);

iree_hal_amdxdna_native_buffer_t* iree_hal_amdxdna_buffer_steal_native_buffer(
    iree_hal_buffer_t* base_buffer);

// Returns true if queue_dealloca has fired on this buffer and subsequent
// queue ops should fail with FAILED_PRECONDITION/INVALID_ARGUMENT.
bool iree_hal_amdxdna_buffer_is_deallocated(iree_hal_buffer_t* base_buffer);

// Marks the buffer as allocated. Called after allocator reuse to reset the
// deallocation marker on a recycled HAL/native buffer pair.
void iree_hal_amdxdna_buffer_mark_allocated(iree_hal_buffer_t* base_buffer);

// Marks the buffer as deallocated. Called by the queue_dealloca async task
// after its wait_semaphore_list is satisfied. Idempotent.
void iree_hal_amdxdna_buffer_mark_deallocated(iree_hal_buffer_t* base_buffer);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDXDNA_BUFFER_H_

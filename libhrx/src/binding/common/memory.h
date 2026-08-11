// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_EXPERIMENTAL_STREAMING_MEMORY_H_
#define IREE_EXPERIMENTAL_STREAMING_MEMORY_H_

#include "hrx_runtime.h"
#include "iree/base/api.h"
#include "iree/hal/buffer.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct iree_hal_streaming_buffer_t iree_hal_streaming_buffer_t;
typedef struct iree_hal_streaming_context_t iree_hal_streaming_context_t;
typedef struct iree_hal_streaming_stream_t iree_hal_streaming_stream_t;

// Allocates queue-visible host staging memory.
iree_status_t iree_hal_streaming_memory_allocate_host_staging(
    iree_hal_streaming_context_t* context, iree_host_size_t size,
    iree_hal_streaming_buffer_t** out_buffer);

// Enqueues a pitched host-to-device copy from queue-visible host memory.
// Returns IREE_STATUS_UNAVAILABLE when |src| is not represented by a host-local
// HAL buffer, allowing the caller to use an owned staging path instead.
iree_status_t iree_hal_streaming_memcpy_host_to_device_2d(
    iree_hal_streaming_context_t* context, uint64_t dst,
    iree_device_size_t dst_pitch, const void* src, iree_device_size_t src_pitch,
    iree_device_size_t width, iree_host_size_t height,
    iree_hal_streaming_stream_t* stream);

// Registers an HRX virtual-address reservation in the streaming pointer table.
// The wrapper retains both |context| and |virtual_buffer|. The reservation
// remains owned by the caller and must be released through the VMM allocator
// after the wrapper is removed.
iree_status_t iree_hal_streaming_memory_wrap_virtual_reservation(
    iree_hal_streaming_context_t* context, hrx_buffer_t virtual_buffer,
    iree_hal_streaming_buffer_t** out_buffer);

// Removes and releases a virtual reservation wrapper.
void iree_hal_streaming_memory_release_wrapped_buffer(
    iree_hal_streaming_buffer_t* buffer);

// Drops a virtual reservation wrapper's HRX/HAL reference before the allocator
// attempts to consume the reservation. The pointer-table entry remains in place
// until release succeeds, and concurrent use of an address being freed is not a
// valid operation. The prepared wrapper must be restored or released below.
void iree_hal_streaming_memory_prepare_virtual_reservation_release(
    iree_hal_streaming_buffer_t* buffer);

// Restores a prepared virtual reservation after the allocator rejects release.
void iree_hal_streaming_memory_restore_virtual_reservation(
    iree_hal_streaming_buffer_t* buffer, hrx_buffer_t virtual_buffer);

// Returns a HAL buffer representing |buffer| on |context|'s device. Imports
// the allocation once per device when the original buffer belongs to another
// context.
iree_status_t iree_hal_streaming_memory_import_buffer_for_context(
    iree_hal_streaming_context_t* context, iree_hal_streaming_buffer_t* buffer,
    iree_hal_buffer_t** out_buffer, uint64_t* out_device_ptr);

// Verifies that |accessor_device_ordinal| has all |required_access| bits for a
// pool allocation. Non-pool allocations have no allocation-specific policy.
iree_status_t iree_hal_streaming_memory_validate_pool_access(
    iree_hal_streaming_buffer_t* buffer,
    iree_host_size_t accessor_device_ordinal,
    hrx_memory_access_t required_access);

// Enqueues a device-to-host copy from a HAL buffer already represented on the
// stream's device. Runtime-owned staging keeps arbitrary host destinations off
// the device queue and publishes the completed bytes in stream order.
iree_status_t iree_hal_streaming_memcpy_buffer_to_host(
    iree_hal_streaming_stream_t* stream, void* dst,
    iree_hal_buffer_t* src_buffer, iree_device_size_t src_offset,
    iree_device_size_t size);

// Grants |accessor_context|'s device access to all ordinary device allocations
// currently published by |owner_context|. The caller must serialize this scan
// with ordinary device-allocation publication.
iree_status_t iree_hal_streaming_memory_grant_peer_access(
    iree_hal_streaming_context_t* owner_context,
    iree_hal_streaming_context_t* accessor_context);

// Records |access| for |accessor_device_ordinal| and imports all existing
// allocations owned by |pool| into that device. New pool allocations apply the
// same policy when they are published.
iree_status_t iree_hal_streaming_memory_set_pool_access(
    hrx_mem_pool_t pool, iree_host_size_t owner_device_ordinal,
    iree_host_size_t accessor_device_ordinal, hrx_memory_access_t access);

// Records a stream-ordered contiguous fill.
iree_status_t iree_hal_streaming_memory_memset(
    iree_hal_streaming_context_t* context, uint64_t dst,
    iree_device_size_t length, const void* pattern,
    iree_host_size_t pattern_length, iree_hal_streaming_stream_t* stream);

// Records a stream-ordered strided three-dimensional fill. |row_pitch| and
// |slice_pitch| are byte strides and |width| is the number of bytes filled in
// each row.
iree_status_t iree_hal_streaming_memory_memset_3d(
    iree_hal_streaming_context_t* context, uint64_t dst,
    iree_device_size_t row_pitch, iree_device_size_t slice_pitch,
    iree_device_size_t width, iree_host_size_t height, iree_host_size_t depth,
    const void* pattern, iree_host_size_t pattern_length,
    iree_hal_streaming_stream_t* stream);

// Records a stream-ordered strided three-dimensional copy between allocations
// owned by different contexts. Pitches and slice pitches are measured in
// bytes; only |width| bytes in each row are copied.
iree_status_t iree_hal_streaming_memcpy_peer_3d(
    iree_hal_streaming_context_t* dst_context, uint64_t dst,
    iree_device_size_t dst_row_pitch, iree_device_size_t dst_slice_pitch,
    iree_hal_streaming_context_t* src_context, uint64_t src,
    iree_device_size_t src_row_pitch, iree_device_size_t src_slice_pitch,
    iree_device_size_t width, iree_host_size_t height, iree_host_size_t depth,
    iree_hal_streaming_stream_t* stream);

// Applies synchronous memset completion semantics to an already-recorded
// destination range. Host-backed, managed, and offset destinations wait;
// base device allocations remain asynchronous with respect to the host.
iree_status_t iree_hal_streaming_memory_complete_synchronous_memset(
    iree_hal_streaming_context_t* context, uint64_t dst,
    iree_device_size_t length, iree_hal_streaming_stream_t* stream);

// Releases private host staging after all queue users have completed.
// Unlike a public host allocation, staging cannot escape through user-visible
// pointers, so its owner provides the lifetime synchronization.
#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_EXPERIMENTAL_STREAMING_MEMORY_H_

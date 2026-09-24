// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_EXPERIMENTAL_STREAMING_MEMORY_H_
#define IREE_EXPERIMENTAL_STREAMING_MEMORY_H_

#include "hrx_runtime.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct iree_hal_streaming_buffer_t iree_hal_streaming_buffer_t;
typedef struct iree_hal_streaming_context_t iree_hal_streaming_context_t;
typedef struct iree_hal_streaming_retained_buffer_ref_t
    iree_hal_streaming_retained_buffer_ref_t;
typedef struct iree_hal_streaming_stream_t iree_hal_streaming_stream_t;

typedef uint64_t iree_hal_streaming_deviceptr_t;

typedef struct iree_hal_streaming_memory_range_request_t {
  // First device or host address in the requested range.
  uint64_t address;
  // Requested range length in bytes.
  iree_device_size_t length;
} iree_hal_streaming_memory_range_request_t;

typedef struct iree_hal_streaming_memory_range_match_t {
  // Index into the returned retained references.
  iree_host_size_t ref_index;
  // Byte offset of the request into its retained reference.
  iree_device_size_t offset;
} iree_hal_streaming_memory_range_match_t;

// Returns the HAL buffer representing |buffer| in |execution_context|.
// Cross-context device-local imports are admitted only when the caller has
// already established peer access. The returned buffer is borrowed from the
// allocation and remains valid while the allocation remains live.
iree_status_t iree_hal_streaming_memory_buffer_for_context(
    iree_hal_streaming_context_t* execution_context,
    iree_hal_streaming_buffer_t* buffer, bool allow_peer_device_allocation,
    iree_hal_buffer_t** out_buffer);

// Resolves every requested range and retains each unique allocation once.
// Cross-context device allocations require enabled peer access. On success,
// each match names one initialized reference in |out_refs|. The function
// releases all partially initialized references before returning an error.
iree_status_t iree_hal_streaming_memory_lookup_ranges_retain_for_context(
    iree_hal_streaming_context_t* execution_context,
    iree_host_size_t request_count,
    const iree_hal_streaming_memory_range_request_t* requests,
    iree_host_size_t ref_capacity,
    iree_hal_streaming_retained_buffer_ref_t* out_refs,
    iree_host_size_t* out_ref_count,
    iree_hal_streaming_memory_range_match_t* out_matches);

// Allocates queue-visible host staging memory.
iree_status_t iree_hal_streaming_memory_allocate_host_staging(
    iree_hal_streaming_context_t* context, iree_host_size_t size,
    iree_hal_streaming_buffer_t** out_buffer);

// Enqueues a pitched H2D copy as one command-buffer transaction.
// Synchronization: stream-ordered.
iree_status_t iree_hal_streaming_memcpy_host_to_device_2d(
    iree_hal_streaming_context_t* context, iree_hal_streaming_deviceptr_t dst,
    iree_device_size_t dst_pitch, const void* src, iree_device_size_t src_pitch,
    iree_device_size_t width, iree_host_size_t height,
    iree_hal_streaming_stream_t* stream);

// Enqueues a pitched D2H copy through queue-visible staging. A stream-ordered
// host call scatters the packed staging rows into |dst| after the device copies
// complete.
// Synchronization: stream-ordered.
iree_status_t iree_hal_streaming_memcpy_device_to_host_2d(
    iree_hal_streaming_context_t* context, void* dst,
    iree_device_size_t dst_pitch, iree_hal_streaming_deviceptr_t src,
    iree_device_size_t src_pitch, iree_device_size_t width,
    iree_host_size_t height, iree_hal_streaming_stream_t* stream);

// Enqueues a pitched D2D copy as one command-buffer transaction.
// Synchronization: stream-ordered. If recording fails after accepting any
// rows, the accepted prefix completes before the original error is returned.
iree_status_t iree_hal_streaming_memcpy_device_to_device_2d(
    iree_hal_streaming_context_t* context, iree_hal_streaming_deviceptr_t dst,
    iree_device_size_t dst_pitch, iree_hal_streaming_deviceptr_t src,
    iree_device_size_t src_pitch, iree_device_size_t width,
    iree_host_size_t height, iree_hal_streaming_stream_t* stream);

// Registers an HRX virtual-address reservation in the streaming pointer table.
// The wrapper retains both |context| and |virtual_buffer|. The reservation
// remains owned by the caller and must be released through the VMM allocator
// after the wrapper is removed.
iree_status_t iree_hal_streaming_memory_wrap_virtual_reservation(
    iree_hal_streaming_context_t* context, hrx_buffer_t virtual_buffer,
    iree_hal_streaming_buffer_t** out_buffer);

// Publishes or removes an existing wrapper in its context's pointer table.
// These operations pair a wrapper's allocation-preparation admission state
// with table visibility and are used when a stable virtual reservation becomes
// live again across graph launches.
iree_status_t iree_hal_streaming_memory_publish_wrapped_buffer(
    iree_hal_streaming_buffer_t* buffer);
iree_status_t iree_hal_streaming_memory_unpublish_wrapped_buffer(
    iree_hal_streaming_buffer_t* buffer);

// Removes and releases an externally owned buffer wrapper.
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

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_EXPERIMENTAL_STREAMING_MEMORY_H_

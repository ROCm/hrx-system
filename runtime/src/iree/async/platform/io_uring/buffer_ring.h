// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// io_uring provided buffer ring (PBUF_RING) management.
//
// A buffer ring is a kernel-shared data structure for efficient buffer
// selection in multishot receive operations. The kernel consumes buffers from
// the ring during recv completions; userspace recycles them back after
// processing.
//
// This module uses the ring's registration lane for synchronous kernel
// registration and teardown. It does not depend on the proactor.
//
// ## Usage
//
//   // Allocate a ring with 64 buffers of 4KB each.
//   iree_io_uring_buffer_ring_t* ring = NULL;
//   iree_io_uring_buffer_ring_options_t options = {
//       .buffer_base = my_slab,
//       .buffer_size = 4096,
//       .buffer_count = 64,  // Must be power of 2.
//       .preferred_group_id = 0,
//   };
//   IREE_RETURN_IF_ERROR(iree_io_uring_buffer_ring_allocate(
//       registration, options, allocator, &ring));
//
//   // Use group_id in SQE for multishot recv:
//   //   sqe->flags |= IOSQE_BUFFER_SELECT;
//   //   sqe->buf_group = iree_io_uring_buffer_ring_group_id(ring);
//
//   // After processing a received buffer, recycle it:
//   iree_io_uring_buffer_ring_recycle(ring, buffer_index);
//
//   // Cleanup:
//   IREE_RETURN_IF_ERROR(iree_io_uring_buffer_ring_free(ring));
//
// ## Thread Safety
//
// Lifecycle operations may be called from any task while the registration
// owner is polling. Buffer recycling has one userspace producer and must be
// serialized by the caller, typically on the proactor completion task.
//
// ## Memory Layout
//
// The ring memory is page-aligned (required by kernel) and laid out as:
//   [0]: Ring header with tail index (16 bytes, overlaps first entry)
//   [0..buffer_count): Buffer entries (16 bytes each)
//
// Each entry contains: address (8 bytes), length (4 bytes), buffer_id (2),
// reserved (2). The kernel reads entries and advances an internal head;
// userspace writes entries and advances the tail.

#ifndef IREE_ASYNC_PLATFORM_IO_URING_BUFFER_RING_H_
#define IREE_ASYNC_PLATFORM_IO_URING_BUFFER_RING_H_

#include "iree/async/platform/io_uring/defs.h"
#include "iree/async/platform/io_uring/uring_registration.h"
#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Buffer ring types
//===----------------------------------------------------------------------===//

// Opaque handle to a registered buffer ring.
typedef struct iree_io_uring_buffer_ring_t iree_io_uring_buffer_ring_t;

// Options for buffer ring creation.
typedef struct iree_io_uring_buffer_ring_options_t {
  // Base address of the contiguous buffer slab. The slab must be at least
  // (buffer_size * buffer_count) bytes and remain valid for the ring's
  // lifetime. Caller retains ownership of the slab memory.
  void* buffer_base;

  // Size of each buffer in bytes. All buffers are the same size.
  iree_host_size_t buffer_size;

  // Number of buffers in the ring. MUST be a power of 2 (kernel requirement).
  // Maximum is 32768 (IREE_IO_URING_MAX_PBUF_RING_ENTRIES).
  iree_host_size_t buffer_count;

  // First buffer group ID to attempt. Registration scans the 16-bit ID space
  // on collisions, so callers need not track live group IDs.
  uint16_t preferred_group_id;

  // Page alignment for ring memory. 0 = use system normal page size.
  // For huge pages, set to 2MB (2097152) or 1GB (1073741824).
  // Falls back to normal pages if huge page allocation fails.
  iree_host_size_t page_alignment;

  // If true, hint to the kernel that transparent huge pages should be used
  // via madvise(MADV_HUGEPAGE). This is best-effort and does not guarantee
  // huge page backing.
  bool hint_transparent_huge_pages;
} iree_io_uring_buffer_ring_options_t;

// Returns default options with zero-initialized fields.
// Caller must set buffer_base, buffer_size, and buffer_count. The preferred
// group ID defaults to zero.
static inline iree_io_uring_buffer_ring_options_t
iree_io_uring_buffer_ring_options_default(void) {
  iree_io_uring_buffer_ring_options_t options = {0};
  options.hint_transparent_huge_pages = true;  // Good default for performance.
  return options;
}

//===----------------------------------------------------------------------===//
// Buffer ring lifecycle
//===----------------------------------------------------------------------===//

// Allocates and registers a provided buffer ring with the kernel.
//
// Allocates page-aligned ring memory, registers it via
// IORING_REGISTER_PBUF_RING, and populates all buffer entries as initially
// available.
//
// Returns IREE_STATUS_INVALID_ARGUMENT if buffer_count is not a power of 2.
// Kernel registration failures are propagated; the proactor probes feature
// availability separately before creating its slab rings.
iree_status_t iree_io_uring_buffer_ring_allocate(
    iree_io_uring_registration_t* registration,
    iree_io_uring_buffer_ring_options_t options, iree_allocator_t allocator,
    iree_io_uring_buffer_ring_t** out_ring);

// Unregisters and frees a buffer ring.
//
// Unregisters from the kernel (IORING_UNREGISTER_PBUF_RING), frees ring memory,
// and releases the ring structure. The ring pointer is invalid after this call.
//
// On failure the ring and its memory remain valid because the kernel may still
// reference them. Safe to call with NULL (no-op).
iree_status_t iree_io_uring_buffer_ring_free(iree_io_uring_buffer_ring_t* ring);

//===----------------------------------------------------------------------===//
// Buffer ring operations
//===----------------------------------------------------------------------===//

// Returns a buffer to the ring, making it available for kernel selection.
//
// Call this after processing a received buffer to recycle it for future
// receives. The buffer_index is obtained from the CQE flags:
//   uint16_t buffer_index = cqe->flags >> IORING_CQE_BUFFER_SHIFT;
//
// This operation is O(1), with release publication by the serialized producer.
void iree_io_uring_buffer_ring_recycle(iree_io_uring_buffer_ring_t* ring,
                                       uint16_t buffer_index);

//===----------------------------------------------------------------------===//
// Buffer ring queries
//===----------------------------------------------------------------------===//

// Returns the buffer group ID for use in SQE configuration.
uint16_t iree_io_uring_buffer_ring_group_id(
    const iree_io_uring_buffer_ring_t* ring);

// Returns the io_uring ring fd this buffer ring is registered with.
int iree_io_uring_buffer_ring_fd(const iree_io_uring_buffer_ring_t* ring);

// Returns the page alignment used for ring memory allocation.
iree_host_size_t iree_io_uring_buffer_ring_page_alignment(
    const iree_io_uring_buffer_ring_t* ring);

// Returns the total capacity (number of buffer slots) in the ring.
iree_host_size_t iree_io_uring_buffer_ring_capacity(
    const iree_io_uring_buffer_ring_t* ring);

// Returns the size of each buffer in bytes.
iree_host_size_t iree_io_uring_buffer_ring_buffer_size(
    const iree_io_uring_buffer_ring_t* ring);

// Returns the base address of the buffer slab.
void* iree_io_uring_buffer_ring_buffer_base(
    const iree_io_uring_buffer_ring_t* ring);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_ASYNC_PLATFORM_IO_URING_BUFFER_RING_H_

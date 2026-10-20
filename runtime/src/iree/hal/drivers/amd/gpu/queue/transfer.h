// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_GPU_QUEUE_TRANSFER_H_
#define IREE_HAL_DRIVERS_AMD_GPU_QUEUE_TRANSFER_H_

#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Captured address source. Native slots are qualified once at capture, while
// addresses are read only after dependencies permit queue-ordered commitment.
typedef union iree_hal_amd_gpu_transfer_address_t {
  // Retained buffer and its already-qualified queue access.
  struct {
    // Owned HAL buffer, independent of the caller's reference.
    iree_hal_buffer_t* buffer;
    // Byte offset relative to the buffer's logical beginning.
    iree_device_size_t offset;
    // Direct index into the buffer's immutable native binding layout.
    iree_hal_buffer_native_binding_slot_t slot;
  } device;
  // Captured update bytes or borrowed upload/download storage.
  void* host;
} iree_hal_amd_gpu_transfer_address_t;

// One nonempty admitted transfer. A 64-bit host needs one cache line per entry.
typedef struct iree_hal_amd_gpu_transfer_operation_t {
  // Active public operation family; zero-length operations are omitted.
  iree_hal_transfer_operation_type_t type;
  // Fill pattern expanded to four repeating bytes, zero for other operations.
  uint32_t pattern;
  // Device source for copy/download, host source for update/upload.
  iree_hal_amd_gpu_transfer_address_t source;
  // Device target except for a borrowed host download target.
  iree_hal_amd_gpu_transfer_address_t target;
  // Complete logical transfer extent in bytes.
  iree_device_size_t length;
} iree_hal_amd_gpu_transfer_operation_t;

// Checked inline storage requirements, established before capture allocation.
typedef struct iree_hal_amd_gpu_transfer_layout_t {
  // Number of nonempty operation descriptors.
  iree_host_size_t count;
  // Descriptor and captured update-byte extent, in bytes.
  iree_host_size_t byte_length;
} iree_hal_amd_gpu_transfer_layout_t;

// Exclusive publication cursor; retirement never reads or modifies it.
typedef struct iree_hal_amd_gpu_transfer_cursor_t {
  // Next operation, or the operation count when publication is complete.
  iree_host_size_t index;
  // Already published bytes of that operation.
  iree_device_size_t offset;
} iree_hal_amd_gpu_transfer_cursor_t;

// Captured transaction embedded in its submission owner.
typedef struct iree_hal_amd_gpu_transfer_t {
  // Initialized descriptors whose retained buffers belong to this capture.
  iree_host_size_t count;
  // Inline immutable descriptors followed by captured update bytes.
  iree_hal_amd_gpu_transfer_operation_t* operations;
  // Unpublished suffix, advanced only after a successful native publication.
  iree_hal_amd_gpu_transfer_cursor_t cursor;
  // Largest staged operation, capped at one bounded bulk window. Captured once
  // so publication can request service-owned growth without walking the batch.
  uint32_t staging_capacity;
  // Uploads read host bytes and therefore require completed host dependencies.
  bool host_wait_required;
  // Downloads finish on the host and cannot signal a native completion early.
  bool host_completion_required;
} iree_hal_amd_gpu_transfer_t;

// A download copy owned by one native retirement record. The queue keeps its
// staging source and captured transaction alive until this copy finishes.
typedef struct iree_hal_amd_gpu_download_t {
  // Coherent, mapped staging bytes written by the native transfer.
  const void* source;
  // Borrowed caller range receiving those bytes before terminal signaling.
  void* target;
  // Nonzero copy extent, or zero when this publication has no host copy.
  iree_host_size_t length;
} iree_hal_amd_gpu_download_t;

// Measures a transaction validated by the public queue API. Failure means its
// captured representation exceeds the host addressable extent.
iree_status_t iree_hal_amd_gpu_transfer_measure(
    iree_host_size_t count, const iree_hal_transfer_operation_t* operations,
    iree_hal_amd_gpu_transfer_layout_t* out_layout);

// Captures into caller-owned storage of the measured extent. Public validation
// already established ranges, usage, access, patterns, and borrowed lifetimes.
// The memory site selects the native access path; maximum_staging_capacity
// bounds the executor's reusable staging window. Native binding interfaces are
// qualified here without requiring committed addresses. Initializes
// out_transfer on every result; deinitialize releases its initialized prefix on
// failure as well as its complete capture on success.
iree_status_t iree_hal_amd_gpu_transfer_initialize(
    const iree_hal_queue_family_t* family, iree_hal_memory_site_kind_t site,
    uint32_t maximum_staging_capacity, iree_host_size_t count,
    const iree_hal_transfer_operation_t* operations,
    iree_hal_amd_gpu_transfer_layout_t layout, void* storage,
    iree_hal_amd_gpu_transfer_t* out_transfer);

// Releases retained buffers after the owning submission's last native use.
void iree_hal_amd_gpu_transfer_deinitialize(
    iree_hal_amd_gpu_transfer_t* transfer);

// Resolves a previously qualified device binding after its allocation waits.
// A pending allocation without a committed native address is a real failure.
iree_status_t iree_hal_amd_gpu_transfer_resolve(
    const iree_hal_amd_gpu_transfer_address_t* address, uint64_t* out_address);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_DRIVERS_AMD_GPU_QUEUE_TRANSFER_H_

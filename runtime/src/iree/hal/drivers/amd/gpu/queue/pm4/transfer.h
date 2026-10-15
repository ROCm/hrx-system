// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_GPU_QUEUE_PM4_TRANSFER_H_
#define IREE_HAL_DRIVERS_AMD_GPU_QUEUE_PM4_TRANSFER_H_

#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Captured address source. Native slots are qualified once at capture, while
// addresses are read only after dependencies permit queue-ordered commitment.
typedef union iree_hal_amd_gpu_pm4_transfer_address_t {
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
} iree_hal_amd_gpu_pm4_transfer_address_t;

// One nonempty admitted transfer. A 64-bit host needs one cache line per entry.
typedef struct iree_hal_amd_gpu_pm4_transfer_operation_t {
  // Active public operation family; zero-length operations are omitted.
  iree_hal_transfer_operation_type_t type;
  // Fill pattern expanded to four repeating bytes, zero for other operations.
  uint32_t pattern;
  // Device source for copy/download, host source for update/upload.
  iree_hal_amd_gpu_pm4_transfer_address_t source;
  // Device target except for a borrowed host download target.
  iree_hal_amd_gpu_pm4_transfer_address_t target;
  // Complete logical transfer extent in bytes.
  iree_device_size_t length;
} iree_hal_amd_gpu_pm4_transfer_operation_t;

// Checked inline storage requirements, established before capture allocation.
typedef struct iree_hal_amd_gpu_pm4_transfer_layout_t {
  // Number of nonempty operation descriptors.
  iree_host_size_t count;
  // Descriptor and captured update-byte extent, in bytes.
  iree_host_size_t byte_length;
} iree_hal_amd_gpu_pm4_transfer_layout_t;

// Exclusive publication cursor; retirement never reads or modifies it.
typedef struct iree_hal_amd_gpu_pm4_transfer_cursor_t {
  // Next operation, or the operation count when publication is complete.
  iree_host_size_t index;
  // Already published bytes of that operation.
  iree_device_size_t offset;
} iree_hal_amd_gpu_pm4_transfer_cursor_t;

// Captured transaction embedded in its submission owner.
typedef struct iree_hal_amd_gpu_pm4_transfer_t {
  // Initialized descriptors whose retained buffers belong to this capture.
  iree_host_size_t count;
  // Inline immutable descriptors followed by captured update bytes.
  iree_hal_amd_gpu_pm4_transfer_operation_t* operations;
  // Unpublished suffix, advanced only after a successful native publication.
  iree_hal_amd_gpu_pm4_transfer_cursor_t cursor;
  // Uploads read host bytes and therefore require completed host dependencies.
  bool host_wait_required;
  // Downloads finish on the host and cannot signal a native completion early.
  bool host_completion_required;
} iree_hal_amd_gpu_pm4_transfer_t;

// A download copy owned by one native retirement record. The queue keeps its
// staging source and captured transaction alive until this copy finishes.
typedef struct iree_hal_amd_gpu_pm4_download_t {
  // Coherent, mapped staging bytes written by the native transfer.
  const void* source;
  // Borrowed caller range receiving those bytes before terminal signaling.
  void* target;
  // Nonzero copy extent, or zero when this publication has no host copy.
  iree_host_size_t length;
} iree_hal_amd_gpu_pm4_download_t;

enum {
  IREE_HAL_AMD_GPU_PM4_TRANSFER_MAX_WORDS = 32 * 7,
  IREE_HAL_AMD_GPU_PM4_TRANSFER_MAX_PACKET_LENGTH = 32736,
};

// Admitted native DMA span, independent of direct or recorded ownership.
// Addresses already name the logical beginning of each device range. Host
// source bytes are immutable for this emission, and downloads use the caller's
// staging window as their native target.
typedef struct iree_hal_amd_gpu_pm4_transfer_span_t {
  // Device source for copies and downloads.
  uint64_t source_address;
  // Device target for copies, fills, updates, and uploads.
  uint64_t target_address;
  // Borrowed source bytes for updates and uploads.
  const uint8_t* source_bytes;
  // Complete logical extent, before applying the publication cursor.
  uint64_t length;
  // Fill value expanded to four repeating bytes.
  uint32_t pattern;
  // Admitted DMA family; no public HAL operation is replayed.
  iree_hal_transfer_operation_type_t type;
} iree_hal_amd_gpu_pm4_transfer_span_t;

typedef struct iree_hal_amd_gpu_pm4_transfer_packet_t {
  // Bytes covered by the emitted seven-DWORD packet; zero means no capacity.
  uint32_t length;
  // Staging bytes owned through this packet's DMA completion.
  uint32_t staging_length;
} iree_hal_amd_gpu_pm4_transfer_packet_t;

// Emits one native piece from a nonempty remaining span. Applies the firmware
// byte-count limit and stages unaligned fill edges or host bytes when needed.
// The caller supplies seven writable DWORDs and retains both ranges and staging
// through DMA completion. Capacity exhaustion emits nothing. This trusted
// transform performs no allocation, buffer lookup, or failure construction.
iree_hal_amd_gpu_pm4_transfer_packet_t
iree_hal_amd_gpu_pm4_transfer_emit_packet(
    const iree_hal_amd_gpu_pm4_transfer_span_t* span, uint64_t offset,
    iree_byte_span_t staging, uint64_t staging_address, uint32_t* words);

// One bounded candidate publication. A download ends the batch, so retirement
// needs one direct host copy instead of a list walk. No owner changes until the
// enclosing queue commits this candidate and its staging prefix.
typedef struct iree_hal_amd_gpu_pm4_transfer_chunk_t {
  // Native commands, with all addresses already resolved.
  uint32_t words[IREE_HAL_AMD_GPU_PM4_TRANSFER_MAX_WORDS];
  // Initialized command extent in DWORDs; zero means staging capacity is full.
  uint32_t word_count;
  // Contiguous staging bytes used by this candidate.
  uint32_t staging_length;
  // Cursor transferred to the publisher when the candidate is committed.
  iree_hal_amd_gpu_pm4_transfer_cursor_t next;
  // Optional retirement copy dominated by the candidate's staging prefix.
  iree_hal_amd_gpu_pm4_download_t download;
} iree_hal_amd_gpu_pm4_transfer_chunk_t;

// Measures a transaction validated by the public queue API. Failure means its
// captured representation exceeds the host addressable extent.
iree_status_t iree_hal_amd_gpu_pm4_transfer_measure(
    iree_host_size_t count, const iree_hal_transfer_operation_t* operations,
    iree_hal_amd_gpu_pm4_transfer_layout_t* out_layout);

// Captures into caller-owned storage of the measured extent. Public validation
// already established ranges, usage, access, patterns, and borrowed lifetimes.
// Native binding interfaces are qualified here without requiring committed
// addresses. Initializes out_transfer on every result; deinitialize releases
// its initialized prefix on failure as well as its complete capture on success.
iree_status_t iree_hal_amd_gpu_pm4_transfer_initialize(
    const iree_hal_queue_family_t* family, iree_host_size_t count,
    const iree_hal_transfer_operation_t* operations,
    iree_hal_amd_gpu_pm4_transfer_layout_t layout, void* storage,
    iree_hal_amd_gpu_pm4_transfer_t* out_transfer);

// Releases retained buffers after the owning submission's last native use.
void iree_hal_amd_gpu_pm4_transfer_deinitialize(
    iree_hal_amd_gpu_pm4_transfer_t* transfer);

// Plans at most word_capacity DWORDs and writes only into available staging.
// The transaction has an unpublished suffix and its required dependencies have
// been satisfied. Staging capacity is bounded by the queue and at most 32736
// bytes; no native allocation occurs. The caller alone owns staging writes.
// Capacity exhaustion is represented by an empty chunk, not an error status.
// Failure means a captured buffer still lacks its queue-ordered native address.
iree_status_t iree_hal_amd_gpu_pm4_transfer_prepare_chunk(
    const iree_hal_amd_gpu_pm4_transfer_t* transfer, uint32_t word_capacity,
    iree_byte_span_t staging, uint64_t staging_address,
    iree_hal_amd_gpu_pm4_transfer_chunk_t* out_chunk);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMD_GPU_QUEUE_PM4_TRANSFER_H_

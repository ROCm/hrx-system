// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVER_AMD_GPU_QUEUE_DMA_TRANSFER_H_
#define IREE_HAL_DRIVER_AMD_GPU_QUEUE_DMA_TRANSFER_H_

#include "iree/hal/driver/amd/gpu/queue/transfer.h"

#ifdef __cplusplus
extern "C" {
#endif

// Cold native packet selection. Both encoders consume qualified ranges and
// return the exclusive command end. Staging, captured ownership, and download
// retirement are independent of the packet protocol.
typedef struct iree_hal_amd_gpu_dma_transfer_encoding_t {
  // Native format features captured by the queue family.
  uint64_t features;
  // Maximum payload extent in a single copy or fill packet, in bytes.
  uint32_t max_packet_length;
  // Maximum command extent emitted by either encoder, in DWORDs.
  uint32_t max_packet_words;
  // Writes one nonempty native copy packet without accessing payload bytes.
  uint32_t* (*copy)(uint32_t* words, uint64_t features, uint64_t source,
                    uint64_t target, uint32_t length);
  // Writes one nonempty DWORD-aligned fill packet with a repeated DWORD.
  uint32_t* (*fill)(uint32_t* words, uint64_t features, uint64_t target,
                    uint32_t length, uint32_t pattern);
} iree_hal_amd_gpu_dma_transfer_encoding_t;

// Bounded CPU command preparation, independent of logical payload length.
enum { IREE_HAL_AMD_GPU_DMA_TRANSFER_MAX_WORDS = 32 * 7 };

// Admitted native DMA span, independent of direct or recorded ownership.
// Addresses already name the logical beginning of each device range. Host
// source bytes are immutable for this emission, and downloads use the caller's
// staging window as their native target.
typedef struct iree_hal_amd_gpu_dma_transfer_span_t {
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
} iree_hal_amd_gpu_dma_transfer_span_t;

typedef struct iree_hal_amd_gpu_dma_transfer_packet_t {
  // Native command extent written by the packet encoder, in DWORDs.
  uint32_t word_count;
  // Bytes covered by the emitted native packet; zero means no capacity.
  uint32_t length;
  // Staging bytes owned through this packet's DMA completion.
  uint32_t staging_length;
} iree_hal_amd_gpu_dma_transfer_packet_t;

// Emits one packet from a nonempty remaining span. Only host-source transfers
// and narrow fill edges write staging. Device copies and aligned fills never
// touch payload bytes on the host. Capacity exhaustion emits nothing. The
// caller retains staging through native completion.
iree_hal_amd_gpu_dma_transfer_packet_t
iree_hal_amd_gpu_dma_transfer_emit_packet(
    const iree_hal_amd_gpu_dma_transfer_encoding_t* encoding,
    const iree_hal_amd_gpu_dma_transfer_span_t* span, uint64_t offset,
    iree_byte_span_t staging, uint64_t staging_address, uint32_t* words);

// One bounded candidate publication. Consecutive downloads share contiguous
// staging and one retirement action; the next different operation ends the run.
// No owner changes until the enclosing queue commits this candidate and its
// staging prefix.
typedef struct iree_hal_amd_gpu_dma_transfer_chunk_t {
  // Native commands, with all addresses already resolved.
  uint32_t words[IREE_HAL_AMD_GPU_DMA_TRANSFER_MAX_WORDS];
  // Initialized command extent in DWORDs; zero means staging capacity is full.
  uint32_t word_count;
  // Contiguous staging bytes used by this candidate.
  uint32_t staging_length;
  // Cursor transferred to the publisher when the candidate is committed.
  iree_hal_amd_gpu_transfer_cursor_t next;
  // Optional retirement copy dominated by the candidate's staging prefix.
  iree_hal_amd_gpu_download_t download;
} iree_hal_amd_gpu_dma_transfer_chunk_t;

// Prepares at most word_capacity DWORDs using the native encoder. Inputs are
// already captured and dependency-ready; missing native commitment is the only
// address-resolution failure. Packet boundaries create no retirement action.
// The caller owns staging and commits the returned cursor after acceptance.
// word_capacity is at most DMA_TRANSFER_MAX_WORDS. Insufficient command or
// staging capacity produces an empty chunk without advancing its cursor.
iree_status_t iree_hal_amd_gpu_dma_transfer_prepare_chunk(
    const iree_hal_amd_gpu_dma_transfer_encoding_t* encoding,
    const iree_hal_amd_gpu_transfer_t* transfer, uint32_t word_capacity,
    iree_byte_span_t staging, uint64_t staging_address,
    iree_hal_amd_gpu_dma_transfer_chunk_t* out_chunk);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // IREE_HAL_DRIVER_AMD_GPU_QUEUE_DMA_TRANSFER_H_

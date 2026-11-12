// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVER_AMD_GPU_QUEUE_AQL_TRANSFER_H_
#define IREE_HAL_DRIVER_AMD_GPU_QUEUE_AQL_TRANSFER_H_

#include "iree/hal/driver/amd/gpu/queue/aql/packet.h"
#include "iree/hal/driver/amd/gpu/queue/blit.h"
#include "iree/hal/driver/amd/gpu/queue/transfer.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
  // Bounds caller publication work independently of the native ring size.
  IREE_HAL_AMD_GPU_AQL_TRANSFER_MAX_PACKETS = 16,
  // Short download retirements bound host-copy interference.
  IREE_HAL_AMD_GPU_AQL_TRANSFER_MAX_DOWNLOAD_LENGTH = 1024 * 1024,
  // Upload preparation overlaps native reads in two bounded regions.
  IREE_HAL_AMD_GPU_AQL_TRANSFER_MAX_UPLOAD_LENGTH = 16 * 1024 * 1024,
  // Only the blit ABI occupies argument storage, independently of payload size.
  IREE_HAL_AMD_GPU_AQL_TRANSFER_ARGUMENT_LENGTH =
      IREE_HAL_AMD_GPU_AQL_TRANSFER_MAX_PACKETS *
      sizeof(iree_hal_amd_gpu_blit_arguments_t),
};

// A bounded unpublished candidate. Argument and transport spans have distinct
// owners; the caller commits both prefixes only after successful admission.
typedef struct iree_hal_amd_gpu_aql_transfer_chunk_t {
  // Complete native dispatches, with admitted argument addresses and scopes.
  iree_hal_amd_gpu_aql_packet_t
      packets[IREE_HAL_AMD_GPU_AQL_TRANSFER_MAX_PACKETS];
  // Initialized native packets; zero means the supplied staging is full.
  uint32_t packet_count;
  // ABI extent retained through native completion.
  uint32_t argument_length;
  // Payload extent retained through native use and any host completion copy.
  uint32_t staging_length;
  // Cursor installed only after the native packets are accepted.
  iree_hal_amd_gpu_transfer_cursor_t next;
  // Consecutive host destinations, completed before their staging is reused.
  iree_hal_amd_gpu_download_t download;
} iree_hal_amd_gpu_aql_transfer_chunk_t;

// Complete native blit with no implicit application memory transition. length
// selects bounded parallelism, while the kernel itself traverses the full
// range.
iree_hal_amd_gpu_aql_packet_t iree_hal_amd_gpu_aql_transfer_packet(
    uint64_t kernel, uint64_t arguments, uint64_t length, uint32_t header);

// Returns the payload capacity for the next nonempty transfer piece.
// A zero capacity queries growth for one largest captured host operation.
// A fitting host source stays in one publication. Larger sources and downloads
// use two windows for device/host overlap. The cached maximum and current
// descriptor bound the request without scanning the capture.
uint64_t iree_hal_amd_gpu_aql_transfer_staging_length(
    const iree_hal_amd_gpu_transfer_t* transfer, uint64_t capacity);

// Resolves admitted buffer slots after dependencies. Device copies and fills
// consume no staging and need only one dispatch regardless of byte length.
// Host spans are split at available staging. Consecutive downloads share one
// retirement action; the next different operation ends that contiguous run.
// arguments has room for packet_capacity ABI records. staging is independent
// payload storage and may be empty for device copies/fills. Every initialized
// byte is published before the first valid header.
iree_status_t iree_hal_amd_gpu_aql_transfer_prepare_chunk(
    const iree_hal_amd_gpu_transfer_t* transfer, uint64_t kernel,
    uint32_t packet_capacity, iree_byte_span_t arguments,
    uint64_t argument_address, iree_byte_span_t staging,
    uint64_t staging_address, iree_hal_amd_gpu_aql_transfer_chunk_t* out_chunk);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_DRIVER_AMD_GPU_QUEUE_AQL_TRANSFER_H_

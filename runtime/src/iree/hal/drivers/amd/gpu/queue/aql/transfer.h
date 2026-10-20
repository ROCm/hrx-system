// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_GPU_QUEUE_AQL_TRANSFER_H_
#define IREE_HAL_DRIVERS_AMD_GPU_QUEUE_AQL_TRANSFER_H_

#include "iree/hal/drivers/amd/gpu/queue/aql/packet.h"
#include "iree/hal/drivers/amd/gpu/queue/blit.h"
#include "iree/hal/drivers/amd/gpu/queue/transfer.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
  // Bounds caller publication work independently of the native ring size.
  IREE_HAL_AMD_GPU_AQL_TRANSFER_MAX_PACKETS = 16,
  // Arguments and borrowed host staging together fit one power-of-two window.
  // File staging is separate and does not inherit this queue-upload policy.
  IREE_HAL_AMD_GPU_AQL_TRANSFER_MAX_STAGING_LENGTH =
      1024 * 1024 - IREE_HAL_AMD_GPU_AQL_TRANSFER_MAX_PACKETS *
                        sizeof(iree_hal_amd_gpu_blit_arguments_t),
};

// A bounded unpublished candidate. Its argument bytes and staging live in the
// queue's argument ring; the caller commits that prefix only after admission.
typedef struct iree_hal_amd_gpu_aql_transfer_chunk_t {
  // Complete native dispatches, with admitted argument addresses and scopes.
  iree_hal_amd_gpu_aql_packet_t
      packets[IREE_HAL_AMD_GPU_AQL_TRANSFER_MAX_PACKETS];
  // Initialized native packets; zero means the supplied staging is full.
  uint32_t packet_count;
  // Argument and staging extent retained through native and host completion.
  uint32_t argument_length;
  // Cursor installed only after the native packets are accepted.
  iree_hal_amd_gpu_transfer_cursor_t next;
  // At most one contiguous host copy, completed before its staging is reused.
  iree_hal_amd_gpu_download_t download;
} iree_hal_amd_gpu_aql_transfer_chunk_t;

// Complete native blit with no implicit application memory transition. length
// selects bounded parallelism, while the kernel itself traverses the full
// range.
iree_hal_amd_gpu_aql_packet_t iree_hal_amd_gpu_aql_transfer_packet(
    uint64_t kernel, uint64_t arguments, uint64_t length, uint32_t header);

// Resolves admitted buffer slots after dependencies. Device copies and fills
// consume no staging and need only one dispatch regardless of byte length.
// Host spans are split at available staging; a download ends the candidate so
// one retirement action can own its exact contiguous host copy.
// upload includes space for packet_capacity argument records followed by
// staging. Every initialized byte is published before the first valid header.
iree_status_t iree_hal_amd_gpu_aql_transfer_prepare_chunk(
    const iree_hal_amd_gpu_transfer_t* transfer, uint64_t kernel,
    uint32_t packet_capacity, iree_byte_span_t upload, uint64_t upload_address,
    iree_hal_amd_gpu_aql_transfer_chunk_t* out_chunk);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_DRIVERS_AMD_GPU_QUEUE_AQL_TRANSFER_H_

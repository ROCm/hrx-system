// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_GPU_QUEUE_AQL_PACKET_H_
#define IREE_HAL_DRIVERS_AMD_GPU_QUEUE_AQL_PACKET_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Native AMD HSA packet storage. The first DWORD is always published as one
// complete release store after the remaining fifteen DWORDs. The ring contains
// no host bookkeeping; even unused native fields belong to the packet ABI.
typedef struct iree_hal_amd_gpu_aql_packet_t {
  // Complete native representation, including header/setup at words[0].
  uint32_t words[16];
} iree_hal_amd_gpu_aql_packet_t;
static_assert(sizeof(iree_hal_amd_gpu_aql_packet_t) == 64,
              "AQL packets occupy exactly one native slot");

// Prepared native header fields. The recorder or direct-call boundary resolves
// execution ordering and memory effects; encoding never rediscovers them.
enum iree_hal_amd_gpu_aql_header_bits_e {
  IREE_HAL_AMD_GPU_AQL_HEADER_INVALID = 1u,
  IREE_HAL_AMD_GPU_AQL_HEADER_DISPATCH = 2u,
  IREE_HAL_AMD_GPU_AQL_HEADER_BARRIER_AND = 3u,
  IREE_HAL_AMD_GPU_AQL_HEADER_BARRIER_OR = 5u,
  IREE_HAL_AMD_GPU_AQL_HEADER_ORDERED = 1u << 8,
  IREE_HAL_AMD_GPU_AQL_HEADER_ACQUIRE_AGENT = 1u << 9,
  IREE_HAL_AMD_GPU_AQL_HEADER_ACQUIRE_SYSTEM = 2u << 9,
  IREE_HAL_AMD_GPU_AQL_HEADER_RELEASE_AGENT = 1u << 11,
  IREE_HAL_AMD_GPU_AQL_HEADER_RELEASE_SYSTEM = 2u << 11,
};

// Admitted kernel packet inputs. Geometry is in workitems, not group counts.
// The producer has checked native widths and resource/argument requirements.
typedef struct iree_hal_amd_gpu_aql_dispatch_t {
  // Complete header/setup DWORD, including the active dimension count.
  uint32_t header;
  // Workitems per workgroup on X/Y/Z; inactive dimensions are one.
  uint16_t workgroup_size[3];
  // Workitems in the full grid, including any partial final workgroup.
  uint32_t grid_size[3];
  // Private bytes per workitem, serviced by the queue's scratch owner.
  uint32_t private_segment_size;
  // Complete fixed and dynamic LDS bytes per workgroup.
  uint32_t group_segment_size;
  // GPU address of the placed AMDHSA kernel descriptor.
  uint64_t kernel_object;
  // GPU address of this invocation's native argument segment.
  uint64_t kernarg_address;
  // GPU address of a native USER signal, or zero without notification.
  uint64_t completion_signal;
} iree_hal_amd_gpu_aql_dispatch_t;

// Encodes a complete immutable template or directly published packet.
// All fields, including reserved words, are overwritten. No allocation occurs.
iree_hal_amd_gpu_aql_packet_t iree_hal_amd_gpu_aql_packet_dispatch(
    const iree_hal_amd_gpu_aql_dispatch_t* dispatch);

// Encodes an admitted AND/OR packet with up to five native signal addresses.
// Unused dependency entries are zero. A barrier blocks subsequent packet
// launches even without ORDERED; ORDERED also joins preceding execution.
iree_hal_amd_gpu_aql_packet_t iree_hal_amd_gpu_aql_packet_barrier(
    uint32_t header, uint32_t dependency_count, const uint64_t* dependencies,
    uint64_t completion_signal);

// AMD vendor-format-1 carrier for an admitted PM4 IB. Its dword-aligned extent
// fits the 48-bit address and positive 20-bit word count. The completion signal
// joins every participating XCC before the caller reuses the immutable IB.
iree_hal_amd_gpu_aql_packet_t iree_hal_amd_gpu_aql_packet_indirect_buffer(
    uint64_t address, uint32_t word_count, uint64_t completion_signal);

// Native AMD USER signal. The descriptor, not its value field, is named by
// dispatch completions and barrier dependencies. Host ownership lives
// elsewhere.
typedef struct iree_hal_amd_gpu_aql_signal_t {
  // AMD_SIGNAL_KIND_USER (1).
  iree_alignas(64) uint64_t kind;
  // Firmware decrements once on completion; a fresh binary producer starts
  // at 1.
  uint64_t value;
  // Same-device native event mailbox, or zero for a non-notifying signal.
  uint64_t event_mailbox;
  // Native interrupt context, not a HAL timeline value.
  uint32_t event_id;
  // Reserved event padding; always zero.
  uint32_t reserved0;
  // Firmware packet-start clock tick when queue profiling is enabled.
  uint64_t start_timestamp;
  // Firmware packet-end clock tick, published before completion.
  uint64_t end_timestamp;
  // Doorbell-only queue pointer; always zero for USER signals.
  uint64_t queue_address;
  // Reserved tail padding; always zero.
  uint64_t reserved1;
} iree_hal_amd_gpu_aql_signal_t;
static_assert(sizeof(iree_hal_amd_gpu_aql_signal_t) == 64,
              "native USER signals occupy exactly 64 bytes");

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMD_GPU_QUEUE_AQL_PACKET_H_

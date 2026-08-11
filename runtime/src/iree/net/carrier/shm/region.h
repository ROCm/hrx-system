// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Canonical shared-memory region layout for SHM carrier pairs.
//
// One creator initializes the outer header and two MPSC queues. Every opener
// calculates the same layout from the creator's ring capacity and normal-page
// alignment before mapping and validates the complete representation before
// binding queue handles.

#ifndef IREE_NET_CARRIER_SHM_REGION_H_
#define IREE_NET_CARRIER_SHM_REGION_H_

#include "iree/base/api.h"
#include "iree/base/internal/mpsc_queue.h"
#include "iree/base/internal/shm.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Magic number identifying an SHM carrier region header ("SHMC" in LE).
#define IREE_NET_SHM_REGION_MAGIC ((uint32_t)0x434D4853)

// Current SHM carrier region ABI version.
#define IREE_NET_SHM_REGION_VERSION ((uint32_t)1)

// Supported normal-page alignment range for transport regions.
#define IREE_NET_SHM_REGION_MIN_MAPPING_ALIGNMENT ((iree_host_size_t)4096)
#define IREE_NET_SHM_REGION_MAX_MAPPING_ALIGNMENT ((iree_host_size_t)65536)

// Byte offsets of shared state fields within the SHM region. Each mutable
// field occupies its own cache line to prevent false sharing.
//
// Layout:
//   0x0000  Region header (64B): magic, version, ring capacity, reserved.
//   0x0040  Reserved cache line.
//   0x0080  Reserved cache line.
//   0x00C0  Consumer A armed flag (64B).
//   0x0100  Consumer B armed flag (64B).
//   0x0140  Reserved padding (192B) to 0x0200.
//   0x0200  Ring A: MPSC queue (256B + ring capacity).
//   Ring B follows Ring A with the same size.
//
// Ring assignment:
//   Client TX = Ring B, Client RX = Ring A.
//   Server TX = Ring A, Server RX = Ring B.
#define IREE_NET_SHM_REGION_OFFSET_HEADER ((iree_host_size_t)0x0000)
#define IREE_NET_SHM_REGION_OFFSET_CONSUMER_A_ARMED ((iree_host_size_t)0x00C0)
#define IREE_NET_SHM_REGION_OFFSET_CONSUMER_B_ARMED ((iree_host_size_t)0x0100)
#define IREE_NET_SHM_REGION_OFFSET_RINGS ((iree_host_size_t)0x0200)

// Immutable header at offset zero of the SHM region.
typedef struct iree_net_shm_region_header_t {
  // Identifies the SHM carrier region representation.
  uint32_t magic;
  // Identifies the exact region ABI.
  uint32_t version;
  // MPSC queue data capacity in bytes.
  uint32_t ring_capacity;
  // Reserved bytes that must be zero.
  uint8_t reserved[52];
} iree_net_shm_region_header_t;
static_assert(sizeof(iree_net_shm_region_header_t) == 64, "");

// Canonical offsets and extents for one SHM carrier region.
//
// Instances must be produced by iree_net_shm_region_layout_calculate and may
// then be passed by value. They contain no pointers or owned resources.
typedef struct iree_net_shm_region_layout_t {
  // MPSC queue data capacity in bytes.
  uint32_t ring_capacity;
  // Creator normal-page alignment used for the mapped extent.
  iree_host_size_t mapping_alignment;
  // Total header and data size of either MPSC queue.
  iree_host_size_t ring_size;
  // Byte offset of Ring A from the mapping base.
  iree_host_size_t ring_a_offset;
  // Byte offset of Ring B from the mapping base.
  iree_host_size_t ring_b_offset;
  // Exact page-rounded mapping extent in bytes.
  iree_host_size_t region_size;
} iree_net_shm_region_layout_t;

// Calculates the canonical region layout for |ring_capacity| and the creator's
// |mapping_alignment|. The capacity must be a power of two at least
// IREE_MPSC_QUEUE_MIN_CAPACITY. The alignment must be a supported power-of-two
// normal-page size.
iree_status_t iree_net_shm_region_layout_calculate(
    uint32_t ring_capacity, iree_host_size_t mapping_alignment,
    iree_net_shm_region_layout_t* out_layout);

// Initializes |mapping| according to |layout| and returns queue handles for
// both rings. The mapping must have the exact calculated extent.
iree_status_t iree_net_shm_region_initialize(
    const iree_net_shm_region_layout_t* layout, iree_shm_mapping_t* mapping,
    iree_mpsc_queue_t* out_ring_a, iree_mpsc_queue_t* out_ring_b);

// Validates and opens an existing |mapping| according to |layout|. The mapping
// must have the exact calculated extent. No shared bytes are modified.
iree_status_t iree_net_shm_region_open(
    const iree_net_shm_region_layout_t* layout,
    const iree_shm_mapping_t* mapping, iree_mpsc_queue_t* out_ring_a,
    iree_mpsc_queue_t* out_ring_b);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_NET_CARRIER_SHM_REGION_H_

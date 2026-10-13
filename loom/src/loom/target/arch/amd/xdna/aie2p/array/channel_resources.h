// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 WITH LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Transactional AIE2P channel-resource selection.

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_ARRAY_CHANNEL_RESOURCES_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_ARRAY_CHANNEL_RESOURCES_H_

#include "loom/target/arch/amd/xdna/aie2p/array/local_memory.h"
#include "loom/target/arch/amd/xdna/aie2p/array/plan.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
  // Channel rings are bounded by the u8 topology carrier. Generated AIE2P
  // lock facts impose the tighter signed-credit limit before this component.
  LOOM_AIE2P_ARRAY_CHANNEL_PROPOSAL_MAX_RECORD_COUNT = UINT8_MAX,
  // Local-memory bank counts use a u8 generated-fact carrier.
  LOOM_AIE2P_ARRAY_CHANNEL_PROPOSAL_MAX_BANK_COUNT = UINT8_MAX,
  // Every channel ring allocation starts on a cache-line boundary.
  LOOM_AIE2P_ARRAY_CHANNEL_STORAGE_ALIGNMENT = 64,
};

typedef uint8_t loom_aie2p_array_tile_resource_flags_t;
enum loom_aie2p_array_tile_resource_flag_bits_e {
  // A resident worker owns the compute tile.
  LOOM_AIE2P_ARRAY_TILE_RESOURCE_FLAG_HAS_WORKER = 1u << 0,
  // A DMA row owns service-tile lifecycle programming.
  LOOM_AIE2P_ARRAY_TILE_RESOURCE_FLAG_HAS_DMA_SERVICE = 1u << 1,
};

// Mutable physical-resource cursors for one array tile.
typedef struct loom_aie2p_array_tile_resources_t {
  // Immutable generated facts for the tile.
  const loom_xdna_tile_facts_t* facts;
  // Next free byte in each local-memory bank, or NULL on shim tiles.
  uint32_t* bank_cursors;
  // First unallocated tile-local DMA buffer descriptor.
  uint16_t next_buffer_descriptor;
  // First unallocated memory-to-stream DMA channel.
  uint8_t next_memory_to_stream_channel;
  // First unallocated stream-to-memory DMA channel.
  uint8_t next_stream_to_memory_channel;
  // First unallocated hardware lock.
  uint8_t next_lock;
  // Round-robin bank considered first for the next channel ring.
  uint8_t next_bank;
  // Resident-worker and DMA-service ownership state.
  loom_aie2p_array_tile_resource_flags_t flags;
} loom_aie2p_array_tile_resources_t;

// Reason an exact channel-resource proposal could not be formed.
typedef enum loom_aie2p_array_channel_resource_failure_e {
  LOOM_AIE2P_ARRAY_CHANNEL_RESOURCE_FAILURE_NONE = 0,
  LOOM_AIE2P_ARRAY_CHANNEL_RESOURCE_FAILURE_RING_STORAGE = 1,
  LOOM_AIE2P_ARRAY_CHANNEL_RESOURCE_FAILURE_LOCK_PAIR = 2,
  LOOM_AIE2P_ARRAY_CHANNEL_RESOURCE_FAILURE_DMA_CHANNEL = 3,
  LOOM_AIE2P_ARRAY_CHANNEL_RESOURCE_FAILURE_DMA_DESCRIPTORS = 4,
  LOOM_AIE2P_ARRAY_CHANNEL_RESOURCE_FAILURE_DMA_LOOPBACK = 5,
} loom_aie2p_array_channel_resource_failure_t;

// One ring and synchronization pair selected without mutating tile state.
typedef struct loom_aie2p_array_ring_resource_proposal_t {
  // Physical tile owning the ring and lock pair.
  loom_xdna_tile_coordinate_t coordinate;
  // Admitted owner aperture supplying the endpoint's address and lock bases.
  const loom_xdna_address_window_t* load_window;
  // Byte offset selected for each ring record in allocation order.
  uint32_t owner_offsets[LOOM_AIE2P_ARRAY_CHANNEL_PROPOSAL_MAX_RECORD_COUNT];
  // Exact bank cursors following all ring allocations.
  uint32_t post_bank_cursors[LOOM_AIE2P_ARRAY_CHANNEL_PROPOSAL_MAX_BANK_COUNT];
  // Number of populated owner_offsets entries.
  uint8_t record_count;
  // Round-robin bank following all ring allocations.
  uint8_t post_next_bank;
  // Physical producer-credit lock ordinal.
  uint8_t credit_lock;
  // Physical consumer-ready lock ordinal.
  uint8_t ready_lock;
  // First unallocated lock following the selected pair.
  uint8_t post_next_lock;
} loom_aie2p_array_ring_resource_proposal_t;

typedef uint8_t loom_aie2p_array_compute_endpoint_request_flags_t;
enum loom_aie2p_array_compute_endpoint_request_flag_bits_e {
  // Require the receiver DMA to form a same-index loopback with its source.
  LOOM_AIE2P_ARRAY_COMPUTE_ENDPOINT_REQUEST_FLAG_REQUIRE_LOOPBACK = 1u << 0,
};

// Complete resource request for one compute DMA endpoint.
typedef struct loom_aie2p_array_compute_endpoint_request_t {
  // Physical compute tile considered for the endpoint.
  loom_xdna_tile_coordinate_t coordinate;
  // Admitted owner aperture visible to the endpoint's worker.
  const loom_xdna_address_window_t* load_window;
  // Number of records and DMA buffer descriptors in the ring.
  uint8_t record_count;
  // Byte length of each ring record.
  uint32_t record_byte_length;
  // DMA transfer direction relative to local memory.
  loom_xdna_dma_direction_t direction;
  // Source DMA channel required by a direct loopback request.
  uint8_t loopback_source_dma_channel;
  // Optional loopback relationship required by the request.
  loom_aie2p_array_compute_endpoint_request_flags_t flags;
} loom_aie2p_array_compute_endpoint_request_t;

typedef uint8_t loom_aie2p_array_compute_endpoint_proposal_flags_t;
enum loom_aie2p_array_compute_endpoint_proposal_flag_bits_e {
  // This endpoint acquires service-tile lifecycle ownership.
  LOOM_AIE2P_ARRAY_COMPUTE_ENDPOINT_PROPOSAL_FLAG_STARTS_DMA_SERVICE = 1u << 0,
};

// Exact compute DMA, ring, and lock transition retained for commit.
typedef struct loom_aie2p_array_compute_endpoint_proposal_t {
  // Ring storage and synchronization selected for the endpoint.
  loom_aie2p_array_ring_resource_proposal_t ring;
  // DMA transfer direction relative to local memory.
  loom_xdna_dma_direction_t direction;
  // Direction-local physical DMA channel ordinal.
  uint8_t dma_channel;
  // First tile-local buffer descriptor selected for the ring.
  uint16_t buffer_descriptor_start;
  // Number of contiguous buffer descriptors selected for the ring.
  uint16_t buffer_descriptor_count;
  // Optional lifecycle effect retained for commit.
  loom_aie2p_array_compute_endpoint_proposal_flags_t flags;
} loom_aie2p_array_compute_endpoint_proposal_t;

// Exact shim DMA transition retained for commit.
typedef struct loom_aie2p_array_shim_endpoint_proposal_t {
  // Physical shim tile owning the endpoint.
  loom_xdna_tile_coordinate_t coordinate;
  // DMA transfer direction relative to external memory.
  loom_xdna_dma_direction_t direction;
  // Direction-local physical DMA channel ordinal.
  uint8_t dma_channel;
  // First tile-local buffer descriptor selected for the endpoint.
  uint16_t buffer_descriptor_start;
  // Number of contiguous buffer descriptors selected for the endpoint.
  uint16_t buffer_descriptor_count;
} loom_aie2p_array_shim_endpoint_proposal_t;

// Selects one neighbor-memory ring and lock pair without mutating |resources|.
// The output is defined only when NONE is returned.
loom_aie2p_array_channel_resource_failure_t
loom_aie2p_array_channel_resources_propose_neighbor(
    const loom_aie2p_array_tile_resources_t* resources,
    loom_xdna_tile_coordinate_t coordinate,
    const loom_xdna_address_window_t* load_window, uint8_t record_count,
    uint32_t record_byte_length,
    loom_aie2p_array_ring_resource_proposal_t* out_proposal);

// Selects one compute DMA endpoint without mutating |resources|. When
// |predecessor| is non-NULL its retained transition is applied to the private
// selection snapshot first. The output is defined only when NONE is returned.
loom_aie2p_array_channel_resource_failure_t
loom_aie2p_array_channel_resources_propose_compute(
    const loom_aie2p_array_tile_resources_t* resources,
    const loom_aie2p_array_compute_endpoint_proposal_t* predecessor,
    const loom_aie2p_array_compute_endpoint_request_t* request,
    loom_aie2p_array_compute_endpoint_proposal_t* out_proposal);

// Selects one shim DMA endpoint without mutating |resources|. The output is
// defined only when NONE is returned.
loom_aie2p_array_channel_resource_failure_t
loom_aie2p_array_channel_resources_propose_shim(
    const loom_aie2p_array_tile_resources_t* resources,
    loom_xdna_tile_coordinate_t coordinate, loom_xdna_dma_direction_t direction,
    uint16_t descriptor_count,
    loom_aie2p_array_shim_endpoint_proposal_t* out_proposal);

// Applies a previously selected ring transition without placement work.
void loom_aie2p_array_channel_resources_commit_ring(
    const loom_aie2p_array_ring_resource_proposal_t* proposal,
    loom_aie2p_array_tile_resources_t* resources);

// Applies a previously selected compute-endpoint transition without placement
// work or resource checks.
void loom_aie2p_array_channel_resources_commit_compute(
    const loom_aie2p_array_compute_endpoint_proposal_t* proposal,
    loom_aie2p_array_tile_resources_t* resources);

// Applies a previously selected shim-endpoint transition without resource
// checks.
void loom_aie2p_array_channel_resources_commit_shim(
    const loom_aie2p_array_shim_endpoint_proposal_t* proposal,
    loom_aie2p_array_tile_resources_t* resources);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_ARRAY_CHANNEL_RESOURCES_H_

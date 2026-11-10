// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Deterministic allocation of AIE2P stream-switch routes.

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_ARRAY_ROUTE_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_ARRAY_ROUTE_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/target/arch/amd/xdna/aie2p/array/plan.h"

#ifdef __cplusplus
extern "C" {
#endif

// A unicast path whose producer admits each complete payload only after its
// receiver has granted storage. Control delivery through that tile can borrow
// its links without waiting behind a payload dependent on future computation.
// The owner frames payloads with IDs disjoint from the sharing control stream.
typedef struct loom_aie2p_array_route_sharing_t {
  // Final payload receiver; control can continue beyond this tile.
  loom_xdna_tile_coordinate_t destination;
  // Borrowable payload and control-continuation lane masks in
  // north/south/west/east order, indexed by tile.
  uint32_t* links;
  // At least one control route actually borrows this payload path.
  bool used;
} loom_aie2p_array_route_sharing_t;

// Mutable stream-network allocation state for one physical array plan.
typedef struct loom_aie2p_array_route_builder_t {
  // Immutable array-family topology and stream-port capacities.
  const loom_xdna_array_family_t* family;
  // Preallocated route records populated in traversal order.
  loom_aie2p_array_route_plan_t* routes;
  // Number of populated route records.
  iree_host_size_t route_count;
  // Populated route count per column-major tile, bounding switch admission
  // storage without rescanning the complete array's routes.
  iree_host_size_t* tile_route_counts;
  // Construction-owned multicast prefixes, indexed without route rescanning.
  struct {
    // Root node per canonical source channel, or IREE_HOST_SIZE_MAX before use.
    iree_host_size_t* roots;
    // Preallocated nodes retaining each physical arrival and direction child.
    struct loom_aie2p_array_route_node_t* nodes;
    // Number of initialized nodes.
    iree_host_size_t node_count;
  } prefixes;
  // Reusable suffix-planning storage, bounded by the array's Manhattan
  // diameter.
  struct loom_aie2p_array_route_step_t* steps;
  // Per-link occupied channel masks grouped by traversal direction.
  struct {
    // Occupied northbound channels for each vertical link.
    uint32_t* northbound;
    // Occupied southbound channels for each vertical link.
    uint32_t* southbound;
    // Occupied westbound channels for each horizontal link.
    uint32_t* westbound;
    // Occupied eastbound channels for each horizontal link.
    uint32_t* eastbound;
  } link_channels;
} loom_aie2p_array_route_builder_t;

// Initializes routing storage in |arena| for one validated physical plan.
// |route_capacity| is the checked per-channel Manhattan route bound and also
// bounds the prefix nodes. All allocations have the caller's arena lifetime.
iree_status_t loom_aie2p_array_route_builder_initialize(
    const loom_xdna_array_family_t* family, iree_host_size_t channel_count,
    iree_host_size_t route_capacity, iree_arena_allocator_t* arena,
    loom_aie2p_array_route_builder_t* out_builder);

// Connects one shim memory-to-stream DMA channel to a worker
// stream-to-memory DMA channel. |source_channel_index| is the topology-owned
// canonical source; its shared prefixes are established by the first branch.
// Returns false when no legal route has free channels. The builder is
// construction state and must be discarded on failure; only complete plans are
// published.
bool loom_aie2p_array_route_ingress(
    loom_aie2p_array_route_builder_t* builder, uint32_t channel_index,
    uint32_t source_channel_index, loom_xdna_tile_coordinate_t shim_coordinate,
    uint8_t shim_dma_channel, loom_xdna_tile_coordinate_t worker_coordinate,
    uint8_t worker_dma_channel);

// Connects one worker memory-to-stream DMA channel to a shim
// stream-to-memory DMA channel, reusing |source_channel_index|'s prefixes.
// Returns false on link exhaustion, invalidating the builder.
bool loom_aie2p_array_route_egress(
    loom_aie2p_array_route_builder_t* builder, uint32_t channel_index,
    uint32_t source_channel_index,
    loom_xdna_tile_coordinate_t worker_coordinate, uint8_t worker_dma_channel,
    loom_xdna_tile_coordinate_t shim_coordinate, uint8_t shim_dma_channel);

// Connects memory-to-stream and stream-to-memory DMA channels on two workers,
// reusing the topology-owned |source_channel_index|'s prefixes.
// Packet classes share that physical tree and retain complete paths for their
// filters; circuits retain only newly allocated edges.
// Returns false on link exhaustion, invalidating the builder.
bool loom_aie2p_array_route_workers(
    loom_aie2p_array_route_builder_t* builder, uint32_t channel_index,
    uint32_t source_channel_index,
    loom_xdna_tile_coordinate_t sender_coordinate, uint8_t sender_dma_channel,
    loom_xdna_tile_coordinate_t receiver_coordinate,
    uint8_t receiver_dma_channel, bool packet);

// Connects two programmable stream-switch endpoints. The same physical link
// allocator is used for DMA payload, core control packets, and task completion
// streams. The caller owns packet filtering where endpoints multiplex traffic.
// Packet routes include reused prefixes so each class can install its filters.
// Shim DMA endpoints include the physical shim mux connection; their channels
// name DMA engines, while other endpoints name stream-switch channels.
bool loom_aie2p_array_route_stream(loom_aie2p_array_route_builder_t* builder,
                                   uint32_t channel_index,
                                   uint32_t source_channel_index,
                                   loom_xdna_tile_coordinate_t source,
                                   loom_xdna_stream_port_t source_port,
                                   uint8_t source_channel,
                                   loom_xdna_tile_coordinate_t destination,
                                   loom_xdna_stream_port_t destination_port,
                                   uint8_t destination_channel, bool packet);

// Retains the directed lanes of the caller's already selected unicast route.
// The exact edge interval comes from its route construction, not a search.
// Admission and packet framing remain the caller's responsibility.
iree_status_t loom_aie2p_array_route_sharing_initialize(
    const loom_aie2p_array_route_builder_t* builder, iree_host_size_t first,
    iree_host_size_t count, loom_xdna_tile_coordinate_t destination,
    iree_arena_allocator_t* arena,
    loom_aie2p_array_route_sharing_t* out_sharing);

// Routes core control packets through a granted payload's receiver. Free lanes
// are preferred; occupied lanes may be borrowed only from |sharing|. Beyond
// its receiver, the route uses free lanes or this domain's control
// continuations. All participating sources share the caller-owned packet
// namespace. The destination's route must pass through the granted receiver.
bool loom_aie2p_array_route_control(loom_aie2p_array_route_builder_t* builder,
                                    uint32_t channel_index,
                                    uint32_t source_channel_index,
                                    loom_xdna_tile_coordinate_t source,
                                    loom_xdna_tile_coordinate_t destination,
                                    loom_aie2p_array_route_sharing_t* sharing);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_ARRAY_ROUTE_H_

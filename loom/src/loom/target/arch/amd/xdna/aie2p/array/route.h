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

// Prefix ownership belongs to one physical sender, independently of destination
// route labels. Zero-initialize before use and retain through all its branches.
typedef struct loom_aie2p_array_route_tree_t {
  // One-based first arrival in the builder's prefix inventory; zero is empty.
  // Identity survives relocation of both the source handle and node storage.
  iree_host_size_t root_index;
} loom_aie2p_array_route_tree_t;

typedef enum loom_aie2p_array_route_carrier_e {
  // Complete payloads have receiver credit. Free control lanes preserve
  // payload bandwidth; admitted payload lanes can be borrowed when necessary.
  LOOM_AIE2P_ARRAY_ROUTE_CARRIER_GRANTED_PAYLOAD = 0,
  // Bounded hardware-register commands have independent completion paths.
  // Early convergence preserves lanes and avoids downstream arbiter fan-in.
  LOOM_AIE2P_ARRAY_ROUTE_CARRIER_HARDWARE_COMMAND,
} loom_aie2p_array_route_carrier_t;

// An admitted unicast stream whose complete packets can retire independently
// of the new control sender's progress. All participating sources own disjoint
// packet identities, except identities naming the same complete destination.
typedef struct loom_aie2p_array_route_sharing_t {
  // Builder-local nonzero identity shared by this contract's borrowed lanes.
  iree_host_size_t domain;
  // Carrier progress contract and its physical convergence policy.
  loom_aie2p_array_route_carrier_t carrier;
  // Granted payload receiver or hardware-command convergence tile.
  loom_xdna_tile_coordinate_t destination;
  // At least one control route actually borrows this carrier path.
  bool used;
} loom_aie2p_array_route_sharing_t;

// Serialized SRAM senders with the same complete receiver set share one packet
// identity. Their horizontal backbone and descending delivery lanes may merge;
// northbound source prefixes remain private to preserve other packet classes.
// The caller proves exclusion and assigns identities before routing any member.
typedef struct loom_aie2p_array_route_multicast_t {
  // Builder-local nonzero identity of this serialized packet domain.
  iree_host_size_t domain;
} loom_aie2p_array_route_multicast_t;

// Mutable stream-network allocation state for one physical array plan.
typedef struct loom_aie2p_array_route_builder_t {
  // Owner of route inventories and every sharing domain's lane masks.
  iree_arena_allocator_t* arena;
  // Immutable array-family topology and stream-port capacities.
  const loom_xdna_array_family_t* family;
  // Preallocated route records populated in traversal order.
  loom_aie2p_array_route_plan_t* routes;
  // Number of populated route records.
  iree_host_size_t route_count;
  // Number of sharing identities assigned independently of contract storage.
  iree_host_size_t domain_count;
  // Owned lane masks indexed by nonzero domain identity minus one.
  struct loom_aie2p_array_route_domain_t* domains;
  // Allocated capacity of the domain inventory.
  iree_host_size_t domain_capacity;
  // Populated route count per column-major tile, bounding switch admission
  // storage without rescanning the complete array's routes.
  iree_host_size_t* tile_route_counts;
  // Construction-owned multicast nodes reached through sender-owned trees.
  struct {
    // Preallocated nodes retaining each physical arrival and direction child.
    struct loom_aie2p_array_route_node_t* nodes;
    // Number of initialized nodes.
    iree_host_size_t node_count;
  } prefixes;
  // Reusable suffix-planning storage for vertical/horizontal/vertical paths.
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
// |route_capacity| bounds all selected paths and their prefix nodes. All
// allocations have the caller's arena lifetime.
iree_status_t loom_aie2p_array_route_builder_initialize(
    const loom_xdna_array_family_t* family, iree_host_size_t route_capacity,
    iree_arena_allocator_t* arena,
    loom_aie2p_array_route_builder_t* out_builder);

// Copies selected routes, occupied lanes, sender prefixes, and sharing domains
// into an independent owner. Source and sharing handles retain their indexed
// identities in the copy. No route selection or source traversal is repeated;
// temporary suffix-planning storage starts empty. route_capacity covers both
// the retained inventory and any subsequent extension selected by the caller.
iree_status_t loom_aie2p_array_route_builder_clone(
    const loom_aie2p_array_route_builder_t* source,
    iree_host_size_t route_capacity, iree_arena_allocator_t* arena,
    loom_aie2p_array_route_builder_t* out_builder);

// Connects one shim memory-to-stream DMA channel to a worker
// stream-to-memory DMA channel. |tree| belongs to the physical shim sender;
// its shared prefixes are established by the first branch.
// Returns false when no legal route has free channels. The builder is
// construction state and must be discarded on failure; only complete plans are
// published.
bool loom_aie2p_array_route_ingress(
    loom_aie2p_array_route_builder_t* builder, uint32_t channel_index,
    loom_aie2p_array_route_tree_t* tree,
    loom_xdna_tile_coordinate_t shim_coordinate, uint8_t shim_dma_channel,
    loom_xdna_tile_coordinate_t worker_coordinate, uint8_t worker_dma_channel);

// Connects one worker memory-to-stream DMA channel to a shim
// stream-to-memory DMA channel, reusing the physical sender's |tree|.
// Returns false on link exhaustion, invalidating the builder.
bool loom_aie2p_array_route_egress(
    loom_aie2p_array_route_builder_t* builder, uint32_t channel_index,
    loom_aie2p_array_route_tree_t* tree,
    loom_xdna_tile_coordinate_t worker_coordinate, uint8_t worker_dma_channel,
    loom_xdna_tile_coordinate_t shim_coordinate, uint8_t shim_dma_channel);

// Connects memory-to-stream and stream-to-memory DMA channels on two workers,
// reusing the physical sender's |tree|.
// Packet classes share that physical tree and retain complete paths for their
// filters; circuits retain only newly allocated edges.
// Returns false on link exhaustion, invalidating the builder.
bool loom_aie2p_array_route_workers(
    loom_aie2p_array_route_builder_t* builder, uint32_t channel_index,
    loom_aie2p_array_route_tree_t* tree,
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
                                   loom_aie2p_array_route_tree_t* tree,
                                   loom_xdna_tile_coordinate_t source,
                                   loom_xdna_stream_port_t source_port,
                                   uint8_t source_channel,
                                   loom_xdna_tile_coordinate_t destination,
                                   loom_xdna_stream_port_t destination_port,
                                   uint8_t destination_channel, bool packet);

// Initializes a shared packet domain without reserving any physical lanes.
iree_status_t loom_aie2p_array_route_multicast_initialize(
    loom_aie2p_array_route_builder_t* builder,
    loom_aie2p_array_route_multicast_t* out_multicast);

// Routes one SRAM-to-core branch in a serialized multicast domain. Cross-column
// branches turn on the last compute row and descend to their receivers. Local
// receivers use the sender's private upward prefix. Members retain independent
// source trees and complete packet paths; other classes cannot use shared
// lanes.
bool loom_aie2p_array_route_multicast(
    loom_aie2p_array_route_builder_t* builder, uint32_t channel_index,
    loom_aie2p_array_route_tree_t* tree, loom_xdna_tile_coordinate_t source,
    uint8_t source_channel, loom_xdna_tile_coordinate_t destination,
    uint8_t destination_channel, loom_aie2p_array_route_multicast_t* multicast);

// Retains the directed lanes of the caller's already selected unicast route.
// The exact edge interval comes from its route construction, not a search.
// Admission and packet framing remain the caller's responsibility.
iree_status_t loom_aie2p_array_route_sharing_initialize(
    loom_aie2p_array_route_builder_t* builder, iree_host_size_t first,
    iree_host_size_t count, loom_xdna_tile_coordinate_t destination,
    loom_aie2p_array_route_carrier_t carrier,
    loom_aie2p_array_route_sharing_t* out_sharing);

// Routes complete control packets through the carrier's convergence tile.
// Core streams and DMA senders use their actual source port and channel.
// Granted payloads prefer free lanes; hardware commands converge as early as
// possible. Occupied lanes may be borrowed only from |sharing|. Beyond the
// convergence tile, free lanes or this domain's continuations remain available.
// All sources share the caller-owned packet namespace.
bool loom_aie2p_array_route_control(loom_aie2p_array_route_builder_t* builder,
                                    uint32_t channel_index,
                                    loom_aie2p_array_route_tree_t* tree,
                                    loom_xdna_tile_coordinate_t source,
                                    loom_xdna_stream_port_t source_port,
                                    uint8_t source_channel,
                                    loom_xdna_tile_coordinate_t destination,
                                    loom_aie2p_array_route_sharing_t* sharing);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_ARRAY_ROUTE_H_

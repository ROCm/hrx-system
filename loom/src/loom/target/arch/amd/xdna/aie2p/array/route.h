// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Deterministic allocation of AIE2P stream-switch routes.

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_ARRAY_ROUTE_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_ARRAY_ROUTE_H_

#include "iree/base/api.h"
#include "loom/target/arch/amd/xdna/aie2p/array/plan.h"

#ifdef __cplusplus
extern "C" {
#endif

// Mutable stream-network allocation state for one physical array plan.
typedef struct loom_aie2p_array_route_builder_t {
  // Immutable array-family topology and stream-port capacities.
  const loom_xdna_array_family_t* family;
  // Preallocated route records populated in traversal order.
  loom_aie2p_array_route_plan_t* routes;
  // Number of populated route records.
  iree_host_size_t route_count;
  // Per-link channel allocation cursors grouped by traversal direction.
  struct {
    // Next unallocated northbound channel for each vertical link.
    uint8_t* northbound;
    // Next unallocated southbound channel for each vertical link.
    uint8_t* southbound;
    // Next unallocated westbound channel for each horizontal link.
    uint8_t* westbound;
    // Next unallocated eastbound channel for each horizontal link.
    uint8_t* eastbound;
  } link_channels;
} loom_aie2p_array_route_builder_t;

// Connects one shim memory-to-stream DMA channel to a worker
// stream-to-memory DMA channel.
iree_status_t loom_aie2p_array_route_ingress(
    loom_aie2p_array_route_builder_t* builder, uint32_t channel_index,
    loom_xdna_tile_coordinate_t shim_coordinate, uint8_t shim_dma_channel,
    loom_xdna_tile_coordinate_t worker_coordinate, uint8_t worker_dma_channel);

// Connects one worker memory-to-stream DMA channel to a shim
// stream-to-memory DMA channel.
iree_status_t loom_aie2p_array_route_egress(
    loom_aie2p_array_route_builder_t* builder, uint32_t channel_index,
    loom_xdna_tile_coordinate_t worker_coordinate, uint8_t worker_dma_channel,
    loom_xdna_tile_coordinate_t shim_coordinate, uint8_t shim_dma_channel);

// Connects memory-to-stream and stream-to-memory DMA channels on two workers.
iree_status_t loom_aie2p_array_route_workers(
    loom_aie2p_array_route_builder_t* builder, uint32_t channel_index,
    loom_xdna_tile_coordinate_t sender_coordinate, uint8_t sender_dma_channel,
    loom_xdna_tile_coordinate_t receiver_coordinate,
    uint8_t receiver_dma_channel);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_ARRAY_ROUTE_H_

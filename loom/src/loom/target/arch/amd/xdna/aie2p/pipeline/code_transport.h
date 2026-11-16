// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PIPELINE_CODE_TRANSPORT_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PIPELINE_CODE_TRANSPORT_H_

#include "loom/target/arch/amd/xdna/aie2p/pipeline/execution_plan.h"

#ifdef __cplusplus
extern "C" {
#endif

// Physical loading path reserved independently of payload and channel state.
typedef struct loom_aie2p_code_transport_binding_t {
  // Compute tile receiving code and executing its resident loading helper.
  loom_xdna_tile_coordinate_t compute;
  // SRAM tile holding this program's code catalog.
  loom_xdna_tile_coordinate_t memory;
  // Memory-to-stream engine shared by a bounded set of sequential loaders.
  uint16_t engine;
  // First descriptor in the engine's parity bank; one per non-root code group.
  uint16_t first_descriptor;
  // Selected upper bound, reduced to the retained count when bound.
  uint16_t descriptor_count;
  // Core-sender packet identity addressing the SRAM queue.
  uint8_t queue_packet;
  // Column-wide DMA packet identity selecting this compute tile's program
  // memory.
  uint8_t code_packet;
  // Tile-control reply identity addressing the same core's completion stream.
  uint8_t reply_packet;
} loom_aie2p_code_transport_binding_t;

typedef struct loom_aie2p_code_transport_t {
  // Physical copy of the original selection, extended by code-delivery paths.
  // The common lifetime prefix borrows the unchanged original scope.
  loom_aie2p_execution_plan_t plan;
  // Mutable complete intended-delivery records backing plan.connections.
  loom_aie2p_native_connection_t* connections;
  // Mutable flat receivers backing plan.connections.
  loom_aie2p_native_connection_endpoint_t* destinations;
  // Current code sender per physical column, retaining its route prefix.
  struct loom_aie2p_code_transport_sender_t* senders;
} loom_aie2p_code_transport_t;

// Reserves extension storage once for all potentially oversized workers in one
// execution scope. Original physical ownership remains immutable. Native
// placement may add at most three control deliveries per worker.
iree_status_t loom_aie2p_code_transport_initialize(
    const loom_aie2p_execution_plan_t* source, iree_host_size_t worker_capacity,
    iree_arena_allocator_t* arena, loom_aie2p_code_transport_t* out_transport);

// Selects descriptor and packet identities without reserving resources. The
// returned binding remains valid until the next bind in this execution scope.
// Native compilation may discard the selection or reduce its descriptor count.
// Sequential loaders share a SRAM sender up to its task-queue depth and packet
// namespace capacity;
// each loader has at most one incomplete request. A full sender needs another
// free engine. Code engines share a complete-packet delivery carrier and
// destination namespace. Payload engines and descriptor ranges stay
// independent. Packet selection consumes the source producer's retained
// namespace. SRAM catalog bytes are placed once exact linked code is available.
// A resource rejection returns a static requirement and leaves transport
// intact.
void loom_aie2p_code_transport_select(
    const loom_aie2p_code_transport_t* transport,
    loom_xdna_tile_coordinate_t compute, iree_host_size_t descriptor_count,
    loom_aie2p_code_transport_binding_t* out_binding,
    iree_string_view_t* out_requirement);

// Reserves the selected routes and the retained prefix of its descriptor range.
// A zero count consumes no descriptors, engines, routes, or packet identities.
// The caller supplies a count no larger than the selected bound, before another
// binding changes this scope. A routing rejection invalidates the selection.
void loom_aie2p_code_transport_bind(
    loom_aie2p_code_transport_t* transport, uint16_t descriptor_count,
    loom_aie2p_code_transport_binding_t* binding,
    iree_string_view_t* out_requirement);

// Admits the complete switch bank once and independently audits every original
// payload/control delivery together with the added code paths.
iree_status_t loom_aie2p_code_transport_finish(
    loom_aie2p_code_transport_t* transport, iree_arena_allocator_t* arena,
    iree_string_view_t* out_requirement);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PIPELINE_CODE_TRANSPORT_H_

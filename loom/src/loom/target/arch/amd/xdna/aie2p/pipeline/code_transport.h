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
  // Memory-to-stream engine owned by this compute tile's sequential loader.
  uint16_t engine;
  // First descriptor in the engine's parity bank; one per non-root code group.
  uint16_t first_descriptor;
  // Number of descriptors reserved for the complete code catalog.
  uint16_t descriptor_count;
  // Core-sender packet identity addressing the SRAM queue.
  uint8_t queue_packet;
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
} loom_aie2p_code_transport_t;

// Reserves extension storage once for all potentially oversized workers in one
// execution scope. Original physical ownership remains immutable. Native
// placement may add at most three control deliveries per worker.
iree_status_t loom_aie2p_code_transport_initialize(
    const loom_aie2p_execution_plan_t* source, iree_host_size_t worker_capacity,
    iree_arena_allocator_t* arena, loom_aie2p_code_transport_t* out_transport);

// Reserves one free SRAM sender, descriptor range, and queue/data/reply routes.
// Packet selection consumes the source producer's retained namespace. SRAM
// catalog bytes are placed once exact linked code is available. A resource
// rejection returns a static requirement and invalidates this whole selection.
void loom_aie2p_code_transport_bind(
    loom_aie2p_code_transport_t* transport, loom_xdna_tile_coordinate_t compute,
    iree_host_size_t descriptor_count,
    loom_aie2p_code_transport_binding_t* out_binding,
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

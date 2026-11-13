// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PIPELINE_DMA_QUEUES_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PIPELINE_DMA_QUEUES_H_

#include "loom/target/arch/amd/xdna/aie2p/pipeline/native.h"

#ifdef __cplusplus
extern "C" {
#endif

// Completion routing belongs to the physical queue, independently of its
// descriptor geometries or the transport families reusing it.
typedef struct loom_aie2p_native_dma_control_t {
  // Initial completion destination, or UINT8_MAX when no task requests tokens.
  uint8_t controller;
  // Disjoint owners require different destinations; each restores its own.
  bool dynamic;
} loom_aie2p_native_dma_control_t;

// Assigns the complete formed endpoint demand through common construction
// lifetimes. Every transport family shares the same tile/direction pools;
// physical path interning and descriptor selection consume these assignments.
// Distribution workers retain their selection order and precede one-way
// transfers so same-tile copies can claim matching send and receive indices.
// External shim queues prefer their issuing column; overflow lifetime classes
// move together to the nearest free shim without changing local endpoints.
iree_status_t loom_aie2p_native_assign_dma_queues(
    loom_aie2p_native_context_t* context,
    const loom_pipeline_realization_t* realization,
    const iree_host_size_t* distribution_workers,
    iree_host_size_t distribution_count, bool* out_valid);

// Retains initial and varying queue-controller values after physical paths and
// completion routes have been selected. Token-free tasks leave them untouched.
void loom_aie2p_native_select_dma_controllers(
    loom_aie2p_native_context_t* context,
    const loom_pipeline_realization_t* realization);

// Register field shared by configuration and resident controller restoration.
loom_xdna_register_field_id_t loom_aie2p_native_dma_controller_field(
    const loom_aie2p_native_tile_t* tile, loom_xdna_dma_direction_t direction);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PIPELINE_DMA_QUEUES_H_

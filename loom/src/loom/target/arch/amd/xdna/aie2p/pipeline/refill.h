// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PIPELINE_REFILL_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PIPELINE_REFILL_H_

#include "loom/target/arch/amd/xdna/aie2p/pipeline/native.h"

#ifdef __cplusplus
extern "C" {
#endif

// One immutable DMA task queue replenished by observed reader progress.
typedef struct loom_aie2p_native_refill_queue_t {
  // Next queue driven by the same record and window.
  struct loom_aie2p_native_refill_queue_t* next;
  // Tile implementing the queue and its control-packet destination.
  const loom_aie2p_native_tile_t* tile;
  // Physical DMA channel within its direction's queue bank.
  uint8_t engine;
  // Transfer direction selecting queue and retirement registers.
  loom_xdna_dma_direction_t direction;
  // Task publication register relative to that tile's aperture.
  uint32_t address;
  // Task word for a complete window.
  uint32_t task;
  // Task word for the final, potentially partial window.
  uint32_t final_task;
  // Selected route's packet class.
  uint8_t packet;
  // Ordinary Low helper accepting the chosen task word.
  loom_symbol_ref_t submit;
} loom_aie2p_native_refill_queue_t;

// One private continuation counter pair shared by equivalent queue windows.
typedef struct loom_aie2p_native_refill_t {
  // Next group in this compute reader's continuation list.
  struct loom_aie2p_native_refill_t* next;
  // Next window driven by the same endpoint, assigned at materialization.
  const struct loom_aie2p_native_refill_t* next_reader;
  // Exact FIFO read stream proving each task's first record has arrived.
  const loom_aie2p_native_channel_t* progress;
  // Queues whose task windows have the same record ordinal boundaries.
  loom_aie2p_native_refill_queue_t* queues;
  // Number of records in a complete immutable task window.
  uint32_t window;
  // Number of tasks following the initially queued window.
  uint32_t remaining_tasks;
  // First of two private SSA state slots assigned at worker materialization.
  iree_host_size_t state_index;
} loom_aie2p_native_refill_t;

// Selects existing compute readers from retained one-to-one transport edges.
// This is physical continuation placement, not source discovery. Ordered
// replay, nonuniform batches and cyclic transport do not imply equal ordinals.
iree_status_t loom_aie2p_native_select_progress(
    loom_aie2p_native_context_t* context,
    const loom_pipeline_realization_t* realization);

// Builds a fixed-size window after startup-only admission exhausted the bank.
// Exact progress proves that every later publication has queue space. A false
// result preserves the original capacity rejection when no such proof exists.
iree_status_t loom_aie2p_native_select_refill(
    loom_aie2p_native_context_t* context, loom_aie2p_native_tile_t* tile,
    uint8_t engine, loom_xdna_dma_direction_t direction,
    const loom_aie2p_native_channel_t* channel, uint32_t repetitions,
    uint32_t capacity, uint32_t byte_offset, uint32_t byte_stride,
    uint32_t credit_interval, uint16_t first_descriptor,
    uint16_t available_descriptors, const uint32_t words[8],
    loom_aie2p_native_dma_sequence_t* sequence, iree_string_view_t* reason,
    bool* out_valid);

// Routes every selected refill before complete switch-bank admission.
iree_status_t loom_aie2p_native_route_refills(
    loom_aie2p_native_context_t* context,
    const loom_pipeline_realization_t* realization, uint32_t* next_source,
    bool* out_valid);

// Creates ordinary task-publication helpers in the current worker context.
iree_status_t loom_aie2p_native_prepare_refills(
    loom_aie2p_native_context_t* context, loom_aie2p_native_refill_t* refills);

// Advances one continuation after a successful read admission. The shared
// channel materializer threads the two returned SSA values through the CFG.
iree_status_t loom_aie2p_native_emit_refill(
    loom_builder_t* builder, const loom_aie2p_native_refill_t* refill,
    loom_value_id_t* state);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PIPELINE_REFILL_H_

// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Native DMA descriptor geometry and finite task programs.

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PIPELINE_DMA_PLAN_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PIPELINE_DMA_PLAN_H_

#include "iree/base/internal/arena.h"
#include "loom/analysis/storage_geometry.h"
#include "loom/target/arch/amd/xdna/array/registers.h"

#ifdef __cplusplus
extern "C" {
#endif

// Register layout of one on-chip DMA module. Compute data memory and SRAM
// retain the same channel ownership protocol with different wire encodings.
typedef struct loom_aie2p_native_memory_layout_t {
  // First descriptor register, used to publish a complete image.
  loom_xdna_register_field_id_t descriptor_base;
  // Number of descriptor words implemented by this module.
  uint8_t descriptor_word_count;
  // Word containing the DMA buffer address.
  uint8_t address_word;
  // DMA buffer-address field.
  loom_xdna_register_field_id_t address;
  // Word containing the iteration state.
  uint8_t iteration_word;
  // Iteration step, wrap, and current-position fields.
  struct {
    // Address-generator increment between records.
    loom_xdna_register_field_id_t step;
    // Number of physical ring slots minus one.
    loom_xdna_register_field_id_t wrap;
    // Current ring slot reset at invocation boundaries.
    loom_xdna_register_field_id_t current;
  } iteration;
  // Word and fields linking descriptors within one finite DMA task.
  struct {
    // Word containing both chain fields.
    uint8_t word;
    // Next descriptor ordinal.
    loom_xdna_register_field_id_t next;
    // The current task continues at the next descriptor.
    loom_xdna_register_field_id_t enable;
  } chain;
  // Word containing lock actions and descriptor validity.
  uint8_t lock_word;
  // Lock action and validity fields.
  struct {
    // Semaphore acquired before movement.
    loom_xdna_register_field_id_t acquire_id;
    // Credit count consumed by admission.
    loom_xdna_register_field_id_t acquire_value;
    // Descriptor has an admission action.
    loom_xdna_register_field_id_t acquire_enable;
    // Semaphore released after movement.
    loom_xdna_register_field_id_t release_id;
    // Credit count published by completion.
    loom_xdna_register_field_id_t release_value;
    // Descriptor is executable.
    loom_xdna_register_field_id_t valid;
  } locks;
  // Hardware semaphore initialization and observation register.
  loom_xdna_register_field_id_t lock_value;
  // Local SRAM bank's DMA address window; zero for compute data memory.
  uint32_t local_address_base;
  // Local SRAM bank's DMA lock-selector window; zero for compute memory.
  uint16_t local_lock_base;
  // Direction-indexed DMA control and task publication fields.
  struct {
    // Channel reset control.
    loom_xdna_register_field_id_t reset;
    // Task head descriptor ordinal.
    loom_xdna_register_field_id_t start;
    // Task execution count minus one.
    loom_xdna_register_field_id_t repeat;
    // Completion-poll fields: queue size, running, and four stall flags.
    loom_xdna_register_field_id_t status[6];
  } directions[2];
} loom_aie2p_native_memory_layout_t;

// Immutable on-chip descriptor layouts used by admission and emission.
extern const loom_aie2p_native_memory_layout_t loom_aie2p_compute_memory;
extern const loom_aie2p_native_memory_layout_t loom_aie2p_sram_memory;

// Complete finite descriptor and task program for one DMA endpoint.
typedef struct loom_aie2p_native_dma_sequence_t {
  // First descriptor in the contiguous allocation owned by this sequence.
  uint16_t descriptor;
  // Number of descriptors, including a separate prefix for a partial rotation.
  uint16_t descriptor_count;
  // Number of autonomous start tasks, bounded by the hardware queue depth.
  uint8_t task_count;
  // The single iterating descriptor ends away from the initial ring slot.
  bool rewind_iteration;
  // Arena-owned complete descriptor images in physical ordinal order.
  uint32_t (*words)[8];
  // Arena-owned encoded task words in FIFO publication order.
  uint32_t* tasks;
} loom_aie2p_native_dma_sequence_t;

// Encodes an endpoint's retained geometry, excluding its address and locks.
// Returns false with a static reason when hardware fields cannot express it.
// The caller supplies zero-initialized words and admitted record dimensions.
bool loom_aie2p_native_encode_descriptor(
    const loom_xdna_array_family_t* family,
    const loom_storage_geometry_t* geometry, const loom_xdna_dma_facts_t* dma,
    uint32_t record_bytes, uint32_t record_count, loom_xdna_tile_kind_t kind,
    uint32_t* words, iree_string_view_t* reason);

// Coalesces an immutable external sequence where native geometry permits it;
// otherwise retains fixed-source repetition or the short address iterator.
// Stable-source ownership and exact consumer completion precede this query.
// The caller supplies zero-initialized words and DMA-aligned record strides.
bool loom_aie2p_native_encode_ingress(const loom_xdna_array_family_t* family,
                                      const loom_storage_geometry_t* record,
                                      const loom_xdna_dma_facts_t* dma,
                                      uint32_t record_bytes,
                                      uint32_t byte_stride,
                                      uint32_t repetitions, uint32_t words[8],
                                      uint32_t* out_repetitions);

// Builds a finite program within the caller's available descriptor bank. All
// tasks fit the hardware queue before any completion wait. The layout is NULL
// for shim descriptors, whose external address is bound during invocation.
// Record capacity, alignment, and stride have already passed DMA admission.
// On success the caller owns the retained arena storage and commits the exact
// descriptor count. Unrepresentable programs return a static rejection reason.
iree_status_t loom_aie2p_native_build_sequence(
    const loom_xdna_dma_facts_t* dma,
    const loom_aie2p_native_memory_layout_t* layout, uint16_t first_descriptor,
    uint16_t available_descriptors, loom_xdna_dma_direction_t direction,
    uint32_t repetitions, uint32_t capacity, uint32_t byte_offset,
    uint32_t byte_stride, const uint32_t template_words[8],
    loom_aie2p_native_dma_sequence_t* out_sequence,
    iree_string_view_t* out_reason, bool* out_valid,
    iree_arena_allocator_t* arena);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PIPELINE_DMA_PLAN_H_

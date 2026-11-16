// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_BUNDLE_ALIGNMENT_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_BUNDLE_ALIGNMENT_H_

#include "iree/base/internal/arena.h"
#include "loom/target/arch/amd/xdna/aie2p/descriptors/encoding.h"
#include "loom/target/arch/amd/xdna/aie2p/emit/leaf_program.h"

#ifdef __cplusplus
extern "C" {
#endif

// Capacity and architectural alignment facts shared with plan allocation.
enum {
  LOOM_AIE2P_BUNDLE_ALIGNMENT_BOUNDARY_BYTES = 16,
  LOOM_AIE2P_BUNDLE_ALIGNMENT_CALL_PACKET_COUNT = 6,
  LOOM_AIE2P_BUNDLE_ALIGNMENT_LOOP_MAX_ADDED_SLOTS = 32,
};

// Native binding retained in parallel with each planned slot. Alignment only
// transports this record while rebuilding a region; physical issue interprets
// its descriptor or allocated move after the region is finalized.
typedef union loom_aie2p_slot_realization_t {
  // Selected descriptor for an ordinary instruction.
  uint32_t descriptor_ordinal;
  // Actual native registers after allocation-move decomposition.
  loom_aie2p_register_move_t move;
} loom_aie2p_slot_realization_t;

static_assert(sizeof(loom_aie2p_slot_realization_t) == sizeof(uint32_t),
              "native slot bindings must remain four bytes");

// Mutable packed-program storage presented to the alignment component. Arrays
// retain their addresses and capacities; successful alignment updates counts,
// offsets, formats, and parallel slot bindings in place.
typedef struct loom_aie2p_bundle_alignment_storage_t {
  // Build-lifetime arena used for transactional format selection.
  iree_arena_allocator_t* scratch_arena;
  // Caller-owned planned bundle records.
  loom_aie2p_planned_bundle_t* bundles;
  // Number of populated records in |bundles|.
  iree_host_size_t bundle_count;
  // Caller-owned planned slot records.
  loom_aie2p_planned_slot_t* slots;
  // Number of records available in |slots| and |slot_realizations|.
  iree_host_size_t slot_capacity;
  // Number of populated records in |slots| and |slot_realizations|.
  iree_host_size_t slot_count;
  // Caller-owned native binding record parallel to each slot.
  loom_aie2p_slot_realization_t* slot_realizations;
  // Exact byte length of the packed program.
  iree_host_size_t encoded_byte_length;
  // Contiguous NOP bytes after the final instruction with behavior.
  iree_host_size_t trailing_nop_byte_length;
} loom_aie2p_bundle_alignment_storage_t;

// Widens existing packets before |boundary_bundle_index| to place that bundle,
// or the stream endpoint, on a hardware-loop boundary after adding at least
// |minimum_added_byte_count| bytes. This also satisfies minimum hardware-loop
// setup-to-end distance without adding loop-control cycles. A legal selection
// never adds an issue position. |out_aligned| is false with storage unchanged
// when no bounded format selection exists.
iree_status_t loom_aie2p_bundle_alignment_try_loop_boundary(
    loom_aie2p_bundle_alignment_storage_t* storage,
    iree_host_size_t region_bundle_start,
    iree_host_size_t boundary_bundle_index, uint8_t minimum_added_byte_count,
    bool* out_aligned);

// Widens a direct call and its five architectural delay packets so the return
// address is aligned. When widening cannot reach the required residue, inserts
// the minimum implicit NOP prefix and shifts the closed call window. Returns
// that physical-cycle increase in |out_inserted_nop_count|.
iree_status_t loom_aie2p_bundle_alignment_align_call_continuation(
    loom_aie2p_bundle_alignment_storage_t* storage,
    iree_host_size_t call_bundle_index, uint32_t continuation_cycle,
    uint8_t* out_inserted_nop_count);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_BUNDLE_ALIGNMENT_H_

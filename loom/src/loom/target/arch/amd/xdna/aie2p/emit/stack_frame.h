// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AIE2P upward-growing function frames and worker-owned stack capacity.

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_STACK_FRAME_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_STACK_FRAME_H_

#include "loom/codegen/low/storage_layout.h"
#include "loom/error/emitter.h"
#include "loom/target/arch/amd/xdna/aie2p/emit/leaf_program.h"

#ifdef __cplusplus
extern "C" {
#endif

// Append-only target placement of the shared storage reservation inventory.
// Ordinary allocation repair preserves its prefix and appends new spill slots.
// Negative SP offsets therefore remain valid as the frame grows.
typedef struct loom_aie2p_stack_layout_t {
  // Arena whose lifetime spans every allocation repair and native planning.
  iree_arena_allocator_t* arena;
  // Distance below SP for each generic reservation ordinal; zero for other
  // spaces.
  uint64_t* depths;
  // Number of projected reservations, including slots planned before IR
  // insertion.
  iree_host_size_t count;
  // Allocated entries in depths.
  iree_host_size_t capacity;
  // Greatest occupied depth, excluding the frame's final alignment padding.
  uint64_t byte_length;
  // Strongest alignment of any stack reservation.
  uint64_t minimum_alignment;
} loom_aie2p_stack_layout_t;

// Appends one verified reservation shape and returns its distance below SP.
// Only allocation and aggregate byte-extent overflow can fail.
iree_status_t loom_aie2p_stack_layout_append(loom_storage_space_t space,
                                             uint64_t byte_size,
                                             uint64_t byte_alignment,
                                             loom_aie2p_stack_layout_t* layout,
                                             uint64_t* out_depth);

// Extends the target projection with the suffix of a newer shared layout.
// Previously projected records remain the unchanged reservation prefix.
iree_status_t loom_aie2p_stack_layout_extend(
    const loom_low_storage_layout_t* source, loom_aie2p_stack_layout_t* layout);

typedef struct loom_aie2p_stack_frame_t {
  // Aligned extent acquired on entry and released on every return.
  uint64_t byte_length;
  // Final storage positions, borrowing the shared flattened handle index.
  loom_low_storage_layout_t storage_layout;
  // Complete stack reservation owned by a worker; zero for a helper.
  loom_aie2p_leaf_storage_requirement_t storage;
} loom_aie2p_stack_frame_t;

// Finishes the native frame and its source-address projection. Worker backing
// covers its frame plus the largest direct helper frame; helpers borrow their
// caller's SP and own no separate physical domain. Final tile placement admits
// this backing together with all competing data and storage.
iree_status_t loom_aie2p_stack_frame_build(
    const loom_module_t* module, const loom_op_t* function_op,
    const loom_low_storage_layout_t* source, loom_aie2p_stack_layout_t* layout,
    uint64_t callee_byte_length, bool is_helper,
    iree_diagnostic_emitter_t emitter, bool* out_accepted,
    loom_aie2p_stack_frame_t* out_frame);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_STACK_FRAME_H_

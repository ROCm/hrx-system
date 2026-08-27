// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU SOPP branch-island layout over measured native packet offsets.

#ifndef LOOM_TARGET_EMIT_NATIVE_AMDGPU_BRANCH_LAYOUT_H_
#define LOOM_TARGET_EMIT_NATIVE_AMDGPU_BRANCH_LAYOUT_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"

#ifdef __cplusplus
extern "C" {
#endif

// Sentinel for an absent branch-island index.
#define LOOM_AMDGPU_BRANCH_ISLAND_NONE UINT32_MAX

// One measured basic-block start in the unrelaxed native stream.
typedef struct loom_amdgpu_branch_layout_block_t {
  // Byte offset of the block label in the unrelaxed stream.
  uint64_t byte_offset;
} loom_amdgpu_branch_layout_block_t;

// One emitted SOPP branch in native emission order.
typedef struct loom_amdgpu_branch_layout_input_edge_t {
  // Byte offset of the branch instruction in the unrelaxed stream.
  uint64_t source_byte_offset;
  // Index of the destination in the measured block table.
  uint32_t target_block_index;
} loom_amdgpu_branch_layout_input_edge_t;

// A legal insertion boundary before one scheduled packet.
typedef struct loom_amdgpu_branch_layout_anchor_t {
  // Byte offset before the packet in the unrelaxed stream.
  uint64_t byte_offset;
  // Packet index identifying the insertion boundary during emission.
  uint32_t packet_index;
} loom_amdgpu_branch_layout_anchor_t;

// Measured unrelaxed layout consumed by branch relaxation.
typedef struct loom_amdgpu_branch_layout_input_t {
  // Total byte length of the unrelaxed native instruction stream.
  uint64_t byte_length;
  // Basic-block start offsets in physical emission order.
  const loom_amdgpu_branch_layout_block_t* blocks;
  // Number of entries in |blocks|.
  iree_host_size_t block_count;
  // Emitted SOPP control-flow edges in physical emission order.
  const loom_amdgpu_branch_layout_input_edge_t* edges;
  // Number of entries in |edges|.
  iree_host_size_t edge_count;
  // Sorted legal branch-island insertion boundaries.
  const loom_amdgpu_branch_layout_anchor_t* anchors;
  // Number of entries in |anchors|.
  iree_host_size_t anchor_count;
} loom_amdgpu_branch_layout_input_t;

typedef uint8_t loom_amdgpu_branch_target_kind_t;
enum loom_amdgpu_branch_target_kind_e {
  // The target index names an original scheduled block.
  LOOM_AMDGPU_BRANCH_TARGET_BLOCK = 0,
  // The target index names a branch island in the relaxed plan.
  LOOM_AMDGPU_BRANCH_TARGET_ISLAND = 1,
};

// Symbolic destination shared by assembly and binary emission.
typedef struct loom_amdgpu_branch_target_t {
  // Namespace containing |index|.
  loom_amdgpu_branch_target_kind_t kind;
  // Block or island index selected by |kind|.
  uint32_t index;
} loom_amdgpu_branch_target_t;

// One original edge after final branch-island placement.
typedef struct loom_amdgpu_branch_layout_edge_t {
  // Direct block or first branch-island destination.
  loom_amdgpu_branch_target_t target;
  // Signed SOPP displacement from the instruction following the branch.
  int16_t relative_dword_offset;
} loom_amdgpu_branch_layout_edge_t;

// One unconditional branch island inserted into the native stream.
typedef struct loom_amdgpu_branch_layout_island_t {
  // Direct next-island or final-block destination.
  loom_amdgpu_branch_target_t target;
  // Signed SOPP displacement from the instruction following the island.
  int16_t relative_dword_offset;
} loom_amdgpu_branch_layout_island_t;

// Co-located islands inserted before one scheduled packet.
typedef struct loom_amdgpu_branch_layout_group_t {
  // Packet index identifying the insertion boundary during emission.
  uint32_t packet_index;
  // First entry owned by this group in the plan island table.
  uint32_t island_start;
  // Number of contiguous island entries owned by this group.
  uint32_t island_count;
} loom_amdgpu_branch_layout_group_t;

// Immutable exact layout shared by native assembly and binary emission.
typedef struct loom_amdgpu_branch_layout_t {
  // Final instruction-stream byte length including branch islands.
  uint64_t byte_length;
  // Original edges in native emission order.
  const loom_amdgpu_branch_layout_edge_t* edges;
  // Number of entries in |edges|.
  iree_host_size_t edge_count;
  // Inserted unconditional branch islands grouped by insertion boundary.
  const loom_amdgpu_branch_layout_island_t* islands;
  // Number of entries in |islands|.
  iree_host_size_t island_count;
  // Island groups in scheduled packet order.
  const loom_amdgpu_branch_layout_group_t* groups;
  // Number of entries in |groups|.
  iree_host_size_t group_count;
} loom_amdgpu_branch_layout_t;

// Builds an exact converged branch-island layout. Empty output means every
// measured edge is directly encodable and the original bytes remain unchanged.
iree_status_t loom_amdgpu_branch_layout_build(
    const loom_amdgpu_branch_layout_input_t* input,
    iree_arena_allocator_t* arena, loom_amdgpu_branch_layout_t* out_layout);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_NATIVE_AMDGPU_BRANCH_LAYOUT_H_

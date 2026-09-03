// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Final AIE2P compute-tile placement and native linking.

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_TILE_LINK_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_TILE_LINK_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/target/arch/amd/xdna/aie2p/emit/leaf_object.h"

#ifdef __cplusplus
extern "C" {
#endif

// Final core-visible placement for one function-local storage domain.
typedef struct loom_aie2p_tile_storage_placement_t {
  // Structural Low storage space selecting the leaf domain.
  loom_storage_space_t storage_space;
  // Address used by core loads and stores through the selected tile aperture.
  uint32_t load_address;
} loom_aie2p_tile_storage_placement_t;

// Final addresses assigned before native fixups and ELF serialization.
typedef struct loom_aie2p_tile_link_layout_t {
  // Core program-memory address assigned to the executable section.
  uint32_t program_address;
  // Program-memory bytes available beginning at |program_address|.
  uint32_t program_byte_capacity;
  // Function-local storage placements keyed by storage space.
  const loom_aie2p_tile_storage_placement_t* storage_placements;
  // Number of records in |storage_placements|.
  iree_host_size_t storage_placement_count;
} loom_aie2p_tile_link_layout_t;

// Fully placed and fixed-up native sections for one AIE2P compute tile.
//
// Every referenced table and section is arena-owned. The result is suitable
// for direct placement in a whole-array XDNA product.
typedef struct loom_aie2p_linked_tile_t {
  // Final assembled native sections with core-visible addresses.
  loom_native_section_contribution_assembly_t assembly;
  // Final layouts for every symbol in the source contribution.
  const loom_native_object_symbol_layout_t* symbol_layouts;
  // Number of records in |symbol_layouts|.
  iree_host_size_t symbol_layout_count;
  // Section containing the contribution entry point.
  iree_host_size_t entry_section_index;
  // Final core-visible entry address.
  uint32_t entry_address;
} loom_aie2p_linked_tile_t;

// Places and links one leaf contribution without serializing an object file.
//
// Section contributions are assembled, assigned the addresses from |layout|,
// and fixed up in that order. The result owns no source IR and all of
// its storage is allocated from |arena|.
iree_status_t loom_aie2p_tile_link(
    const loom_aie2p_leaf_contribution_t* contribution,
    const loom_aie2p_tile_link_layout_t* layout, iree_arena_allocator_t* arena,
    loom_aie2p_linked_tile_t* out_tile);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_TILE_LINK_H_

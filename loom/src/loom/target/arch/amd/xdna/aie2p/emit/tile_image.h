// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Final AIE2P compute-tile ELF image emission.

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_TILE_IMAGE_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_TILE_IMAGE_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/io/stream.h"
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
typedef struct loom_aie2p_tile_image_layout_t {
  // Core program-memory address assigned to the executable section.
  uint32_t program_address;
  // Program-memory bytes available beginning at |program_address|.
  uint32_t program_byte_capacity;
  // Function-local storage placements keyed by storage space.
  const loom_aie2p_tile_storage_placement_t* storage_placements;
  // Number of records in |storage_placements|.
  iree_host_size_t storage_placement_count;
} loom_aie2p_tile_image_layout_t;

// Writes one leaf contribution as an executable AIE2P tile ELF.
//
// The image uses the contribution entry symbol and retained target identity and
// keeps an ordinary ELF32 symbol table outside of load segments. Native fixups
// are applied after section assembly and |layout| establish final core-visible
// addresses. Every retained function-storage domain requires exactly one
// placement. Temporary section, symbol, fixup, and ELF layout storage is
// allocated from |scratch_arena|.
iree_status_t loom_aie2p_tile_image_write(
    const loom_aie2p_leaf_contribution_t* contribution,
    const loom_aie2p_tile_image_layout_t* layout, iree_io_stream_t* stream,
    iree_arena_allocator_t* scratch_arena);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_TILE_IMAGE_H_

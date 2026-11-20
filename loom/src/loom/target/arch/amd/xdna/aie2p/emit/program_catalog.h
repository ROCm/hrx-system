// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_PROGRAM_CATALOG_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_PROGRAM_CATALOG_H_

#include "loom/target/arch/amd/xdna/aie2p/emit/program.h"
#include "loom/target/arch/amd/xdna/aie2p/emit/program_layout.h"
#include "loom/target/arch/amd/xdna/aie2p/emit/tile_link.h"
#include "loom/target/arch/amd/xdna/aie2p/pipeline/code_transport.h"

#ifdef __cplusplus
extern "C" {
#endif

// Immutable, fully relocated control streams and their SRAM DMA descriptors.
typedef struct loom_aie2p_program_catalog_t {
  // SRAM tile receiving the framed code.
  loom_xdna_tile_coordinate_t memory;
  // Starting byte offset in the tile's local SRAM.
  uint32_t offset;
  // Framed code at its absolute firmware write address.
  loom_aie2p_program_register_block_write32_t contents;
  // Engine control register and reset value used to publish the descriptors.
  loom_aie2p_program_register_write32_t reset;
  // Complete SRAM descriptor writes in native group order.
  const loom_aie2p_program_record_t* descriptors;
  // Number of descriptors; zero for workers with no replacement code.
  iree_host_size_t descriptor_count;
} loom_aie2p_program_catalog_t;

// Reserves catalog storage after existing physical allocations, frames exact
// linked code, and materializes immutable DMA descriptors. The final ordered
// status read acknowledges all writes before the core calls the new body.
// Allocation is fallible; a physical capacity rejection returns a requirement.
iree_status_t loom_aie2p_program_catalog_build(
    loom_aie2p_code_transport_t* transport,
    const loom_aie2p_code_transport_binding_t* binding,
    const loom_aie2p_program_layout_t* layout,
    const loom_aie2p_linked_tile_t* linked, iree_arena_allocator_t* arena,
    loom_aie2p_program_catalog_t* out_catalog,
    iree_string_view_t* out_requirement);

// Completes a compiler-generated pipeline's initialization while its cores
// remain reset. All code-delivery paths have passed joint switch admission.
// first_record is the initial record boundary after all tile programs have
// been loaded and before application routes or DMA resources are configured.
iree_status_t loom_aie2p_program_catalog_install(
    const loom_aie2p_code_transport_t* transport,
    const loom_aie2p_program_catalog_t* catalogs,
    iree_host_size_t catalog_count, iree_host_size_t first_record,
    iree_arena_allocator_t* arena, loom_aie2p_array_program_t* program);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_PROGRAM_CATALOG_H_

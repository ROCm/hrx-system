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
  // Initialization records; the producer keeps all compute cores reset.
  const loom_aie2p_program_record_t* records;
  // Number of records needed for this worker's complete catalog.
  iree_host_size_t record_count;
  // Catalog offset within its admitted SRAM tile's allocation space.
  uint32_t offset;
  // Framed bytes retained across every invocation.
  uint32_t byte_length;
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
// Invocation records and their runtime relocation indices remain unchanged.
iree_status_t loom_aie2p_program_catalog_install(
    const loom_aie2p_code_transport_t* transport,
    const loom_aie2p_program_catalog_t* catalogs,
    iree_host_size_t catalog_count, iree_arena_allocator_t* arena,
    loom_aie2p_array_program_t* program);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_PROGRAM_CATALOG_H_

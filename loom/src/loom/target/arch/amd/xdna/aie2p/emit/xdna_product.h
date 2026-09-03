// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Canonical single-entry AIE2P XDNA ELF product emission.

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_XDNA_PRODUCT_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_XDNA_PRODUCT_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/io/stream.h"
#include "loom/target/arch/amd/xdna/aie2p/array/program.h"
#include "loom/target/arch/amd/xdna/aie2p/emit/tile_link.h"
#include "loom/target/arch/amd/xdna/device/profile.h"

#ifdef __cplusplus
extern "C" {
#endif

// First incompatible AIE2P placement and product-formation policy.
#define LOOM_AIE2P_XDNA_PRODUCT_POLICY_ID UINT64_C(0x413250504C414E01)

// One linked resident tile program and its physical placement.
typedef struct loom_aie2p_xdna_tile_t {
  // Physical compute tile executing the program.
  loom_xdna_tile_coordinate_t coordinate;
  // Detached contribution from which symbols and sizes are retained.
  const loom_aie2p_leaf_contribution_t* contribution;
  // Fully placed and fixed-up native sections.
  const loom_aie2p_linked_tile_t* linked_tile;
} loom_aie2p_xdna_tile_t;

// Complete inputs to one canonical single-entry AIE2P XDNA product.
typedef struct loom_aie2p_xdna_product_t {
  // Exact deployment profile serialized into the ABI note.
  const loom_xdna_device_profile_t* device_profile;
  // Diagnostic and runtime export name.
  iree_string_view_t entry_name;
  // Exact physical array plan defining bindings and placements.
  const loom_aie2p_array_plan_t* array_plan;
  // Typed array and invocation-control program awaiting final ordinals.
  const loom_aie2p_array_program_t* array_program;
  // Resident tile programs in worker order.
  const loom_aie2p_xdna_tile_t* tiles;
  // Number of records in |tiles|.
  iree_host_size_t tile_count;
} loom_aie2p_xdna_product_t;

// Writes one canonical ELF32LE `.xdna` product.
//
// The program-header table is the runtime directory. Tile programs are raw
// linked sections in the outer ELF; no nested object or vendor container is
// produced. Metadata tables, symbols, section names, and final ELF layout use
// |scratch_arena| and remain live only for the call.
iree_status_t loom_aie2p_xdna_product_write(
    const loom_aie2p_xdna_product_t* product, iree_io_stream_t* stream,
    iree_arena_allocator_t* scratch_arena);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_XDNA_PRODUCT_H_

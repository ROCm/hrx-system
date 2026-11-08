// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Private core IR and retained analysis products for native compilation.

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PROGRAM_CLONE_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PROGRAM_CLONE_H_

#include "loom/rewrite/remap.h"
#include "loom/target/arch/amd/xdna/aie2p/program_source.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_aie2p_program_clone_function_t {
  // Module-local symbol naming the cloned core definition.
  loom_symbol_ref_t entry;
  // Invocation target facts borrowed from the immutable source version.
  const loom_target_facts_t* function_target_facts;
  // Module-owned memory proofs translated to cloned operations.
  loom_low_memory_access_map_t* memory_accesses;
  // Module-owned loop facts translated to cloned controls.
  loom_low_loop_execution_map_t* loop_execution;
} loom_aie2p_program_clone_function_t;

// One source-symbol projection shared by an admitted set of core programs.
// Reserve every referenced data/function symbol before cloning definitions.
// The retained source inventories own reachability; this projection does not
// discover dependencies or change the source module.
typedef struct loom_aie2p_program_clone_t {
  // Immutable prepared source owning all retained inventories.
  const loom_module_t* source_module;
  // Private module receiving definitions and subsequent allocation repair.
  loom_module_t* module;
  // Invocation arena owning symbol correspondence and cloned-function records.
  iree_arena_allocator_t* arena;
  // Target symbol refs indexed by source-module symbol ID.
  loom_symbol_ref_t* symbols_by_source;
  // Lazily allocated cloned-function records indexed by source symbol ID.
  loom_aie2p_program_clone_function_t** functions_by_source;
  // Shared remap for target and immutable-data definitions without local SSA.
  // Its callback borrows this projection, whose address stays fixed after init.
  loom_ir_remap_t definition_remap;
} loom_aie2p_program_clone_t;

// Operation observers live through one body-clone invocation, including when
// the caller wraps the cloned blocks in an array resident protocol.
typedef struct loom_aie2p_program_clone_products_t {
  // Source-to-target memory-proof translator, or NULL without source proofs.
  loom_low_memory_access_clone_t* memory;
  // Source-to-target loop-fact translator, or NULL without source loop facts.
  loom_low_loop_execution_clone_t* loop_execution;
} loom_aie2p_program_clone_products_t;

iree_status_t loom_aie2p_program_clone_initialize(
    const loom_module_t* source_module, loom_module_t* module,
    iree_arena_allocator_t* arena, loom_aie2p_program_clone_t* out_clone);

// Reserves a function and its optional authored target witness.
iree_status_t loom_aie2p_program_clone_reserve_function(
    loom_aie2p_program_clone_t* clone,
    const loom_aie2p_program_source_function_t* source);

// Reserves all data symbols before any projected conflict set is cloned.
iree_status_t loom_aie2p_program_clone_reserve_data(
    loom_aie2p_program_clone_t* clone,
    const loom_low_read_only_data_set_t* data);

// Clones each definition once, projecting conflicts onto retained symbols.
iree_status_t loom_aie2p_program_clone_data(
    loom_aie2p_program_clone_t* clone,
    const loom_low_read_only_data_set_t* data);

// Translates a program inventory in its original physical-placement order.
iree_status_t loom_aie2p_program_clone_project_data(
    loom_aie2p_program_clone_t* clone,
    const loom_low_read_only_data_set_t* source_data,
    loom_low_read_only_data_set_t* out_data);

// Initializes a body remap that translates products at the clone boundary.
// |products| must outlive all cloning through |out_remap|.
iree_status_t loom_aie2p_program_clone_body_remap(
    loom_aie2p_program_clone_t* clone,
    const loom_aie2p_program_source_function_t* source,
    loom_aie2p_program_clone_products_t* products,
    loom_low_memory_access_map_t** out_memory_accesses,
    loom_low_loop_execution_map_t** out_loop_execution,
    loom_ir_remap_t* out_remap);

// Clones a complete core function and its target witness once. All referenced
// symbols have been reserved, and immutable data has already been cloned.
iree_status_t loom_aie2p_program_clone_function(
    loom_aie2p_program_clone_t* clone,
    const loom_aie2p_program_source_function_t* source);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PROGRAM_CLONE_H_

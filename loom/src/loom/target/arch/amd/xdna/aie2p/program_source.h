// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Artifact-scoped AIE2P core program sources.

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PROGRAM_SOURCE_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PROGRAM_SOURCE_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/analysis/symbol_facts.h"
#include "loom/codegen/low/function_requirements.h"
#include "loom/codegen/low/loop_execution.h"
#include "loom/codegen/low/memory_access.h"
#include "loom/codegen/low/target_binding.h"
#include "loom/error/emitter.h"
#include "loom/target/function_version.h"

#ifdef __cplusplus
extern "C" {
#endif

// One reachable AIE2P core function and its immutable compiler inputs.
typedef struct loom_aie2p_program_source_function_t {
  // Module-local function symbol.
  loom_symbol_ref_t symbol;
  // Source Low definition.
  const loom_op_t* function_op;
  // Resolved core target and representation contract.
  loom_low_resolved_target_t target;
  // Optional source memory proofs.
  const loom_low_memory_access_map_t* memory_accesses;
  // Optional source loop execution facts.
  const loom_low_loop_execution_map_t* loop_execution;
  // Source imports, storage, data, and direct calls.
  loom_low_function_requirements_t requirements;
  // Unique immutable definitions used by this entry and its admitted helpers.
  // Aliases the function inventory until helper admission adds definitions.
  const loom_low_read_only_data_set_t* program_data;
  // Whether the direct-leaf native call contract has been admitted.
  bool direct_helpers_admitted;
} loom_aie2p_program_source_function_t;

// Lazily populated source-function index for one immutable artifact module.
typedef struct loom_aie2p_program_source_table_t {
  // Prepared target-low module owning every indexed function.
  const loom_module_t* module;
  // Shared target-symbol fact cache.
  loom_symbol_fact_table_t* symbol_facts;
  // Concrete function versions indexed by the same module snapshot.
  const loom_target_function_version_snapshot_t* function_versions;
  // Registry resolving core representation contracts.
  const loom_low_descriptor_registry_t* descriptor_registry;
  // Target diagnostic destination.
  iree_diagnostic_emitter_t diagnostic_emitter;
  // Arena owning the index, records, and requirements.
  iree_arena_allocator_t* arena;
  // Stable function records indexed by module-local symbol ID.
  loom_aie2p_program_source_function_t** functions_by_symbol;
} loom_aie2p_program_source_table_t;

// Initializes an empty source table. The caller-owned table, module, and all
// supplied analysis state must remain alive and immutable through every plan
// borrowing the table.
iree_status_t loom_aie2p_program_source_table_initialize(
    const loom_module_t* module, loom_symbol_fact_table_t* symbol_facts,
    const loom_target_function_version_snapshot_t* function_versions,
    const loom_low_descriptor_registry_t* descriptor_registry,
    iree_diagnostic_emitter_t diagnostic_emitter, iree_arena_allocator_t* arena,
    loom_aie2p_program_source_table_t* out_table);

// Resolves one verified AIE2P core definition once. Authored target-binding
// rejection leaves |out_resolved| false; status reports infrastructure failure.
iree_status_t loom_aie2p_program_source_table_resolve(
    loom_aie2p_program_source_table_t* table, loom_symbol_ref_t symbol,
    bool* out_resolved, loom_aie2p_program_source_function_t** out_function);

// Returns an already resolved source record. |symbol| must have been admitted
// through resolve or direct-helper admission in the same table.
const loom_aie2p_program_source_function_t*
loom_aie2p_program_source_table_lookup(
    const loom_aie2p_program_source_table_t* table, loom_symbol_ref_t symbol);

// Admits the entry-to-direct-leaf native call closure of |entry|. A nested
// retained helper emits one XDNA diagnostic and leaves |out_admitted| false.
// The entry and every admitted helper remain indexed for materialization;
// program_data retains their union in entry-then-callee reference order.
iree_status_t loom_aie2p_program_source_admit_direct_helpers(
    loom_aie2p_program_source_table_t* table,
    loom_aie2p_program_source_function_t* entry, bool* out_admitted);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_PROGRAM_SOURCE_H_

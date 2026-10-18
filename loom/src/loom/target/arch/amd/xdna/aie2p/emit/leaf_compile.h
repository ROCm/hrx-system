// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Scheduling, allocation repair, and native emission of AIE2P core functions.

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_LEAF_COMPILE_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_LEAF_COMPILE_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/analysis/symbol_facts.h"
#include "loom/codegen/low/allocation/diagnostics.h"
#include "loom/codegen/low/allocation/target_constraints.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/ir/ir.h"
#include "loom/target/arch/amd/xdna/aie2p/emit/leaf_object.h"
#include "loom/target/reporting/report.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_low_memory_access_map_t loom_low_memory_access_map_t;
typedef struct loom_low_loop_execution_map_t loom_low_loop_execution_map_t;

typedef struct loom_aie2p_leaf_compile_options_t {
  // Descriptor registry used to resolve the core representation contract.
  const loom_low_descriptor_registry_t* descriptor_registry;
  // Optional invocation-refined target facts for this function version.
  const loom_target_facts_t* function_target_facts;
  // Captured source proofs bound to this core function's memory effects.
  const loom_low_memory_access_map_t* memory_accesses;
  // Captured execution facts bound to this core function's loop controls.
  const loom_low_loop_execution_map_t* loop_execution;
  // Borrowed SSA location constraints, valid for the duration of compilation.
  // Locations are constrained only over each value's live interval. Returned
  // values express state that must remain live through the leaf's exit.
  const loom_low_allocation_fixed_value_t* allocation_fixed_values;
  // Number of records in |allocation_fixed_values|.
  iree_host_size_t allocation_fixed_value_count;
  // Incoming register locations indexed by formal argument.
  const loom_low_allocation_abi_location_t* allocation_entry_locations;
  // Number of records in |allocation_entry_locations|.
  iree_host_size_t allocation_entry_location_count;
  // Outgoing register locations indexed by function result.
  const loom_low_allocation_abi_location_t* allocation_exit_locations;
  // Number of records in |allocation_exit_locations|.
  iree_host_size_t allocation_exit_location_count;
  // Direct-callee ABI locations and exact physical clobbers.
  loom_low_call_contract_provider_t call_contracts;
  // Whole-function registers owned by the native calling convention.
  const loom_low_allocation_reserved_range_t* allocation_reserved_ranges;
  // Number of records in |allocation_reserved_ranges|.
  iree_host_size_t allocation_reserved_range_count;
  // Whether the entry program preserves its incoming link across calls.
  bool preserve_link_register;
  // Reserved scalar register holding that incoming link.
  loom_aie2p_physical_register_id_t link_shadow_register;
  // Optional structured allocation and repair feedback to emit.
  loom_low_allocation_diagnostic_flags_t allocation_diagnostic_flags;
  // Diagnostic emitter receiving scheduling and allocation failures.
  iree_diagnostic_emitter_t diagnostic_emitter;
  // Optional compile report receiving exact Low planning evidence.
  loom_target_compile_report_t* compile_report;
} loom_aie2p_leaf_compile_options_t;

// Compiles one verified amd.xdna.aie2p.core Low function into an arena-owned
// detached native contribution and exact realization facts. Temporary planning
// storage is returned to the arena's block pool before this function returns.
// Structured rejection returns OK with |out_compiled| false and no
// contribution; infrastructure failures return a status.
iree_status_t loom_aie2p_leaf_compile(
    loom_module_t* module, loom_op_t* function_op,
    const loom_aie2p_leaf_compile_options_t* options,
    iree_arena_allocator_t* arena, bool* out_compiled,
    loom_aie2p_leaf_contribution_t* out_contribution);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_LEAF_COMPILE_H_

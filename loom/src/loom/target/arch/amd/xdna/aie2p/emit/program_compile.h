// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Direct-helper closure compilation for one AIE2P tile program.

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_PROGRAM_COMPILE_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_PROGRAM_COMPILE_H_

#include "loom/codegen/low/read_only_data.h"
#include "loom/target/arch/amd/xdna/aie2p/emit/leaf_compile.h"

#ifdef __cplusplus
extern "C" {
#endif

// Producer-retained inputs for one direct native helper.
typedef struct loom_aie2p_program_helper_t {
  // Module-local helper definition called by the entry function.
  loom_symbol_ref_t function;
  // Invocation-refined target facts owned by the helper's function version.
  const loom_target_facts_t* function_target_facts;
  // Source memory proofs bound to the helper's current operations.
  const loom_low_memory_access_map_t* memory_accesses;
  // Source execution proofs bound to the helper's current loop controls.
  const loom_low_loop_execution_map_t* loop_execution;
} loom_aie2p_program_helper_t;

typedef struct loom_aie2p_program_helper_list_t {
  // Direct helpers in producer-retained call inventory order.
  const loom_aie2p_program_helper_t* values;
  // Number of records in |values|.
  iree_host_size_t count;
} loom_aie2p_program_helper_list_t;

// Module-scoped owner of compiled native helper objects and their ABI/clobber
// contracts. Helper compilation may rewrite IR during allocation repair; every
// tile calling that helper consumes the same retained result. Source symbols,
// target versions, and descriptor registry remain fixed for this lifetime.
typedef struct loom_aie2p_program_compiler_t loom_aie2p_program_compiler_t;

// Allocates a compiler in |arena|, which also owns its retained helper objects,
// compile reports, and complete program contributions. The module outlives the
// compiler; |report_template| is borrowed only during allocation. Supplying a
// template fixes report retention for the compiler lifetime, and every program
// compiled with this instance must then supply a report with the same requested
// detail flags.
iree_status_t loom_aie2p_program_compiler_allocate(
    loom_module_t* module, iree_arena_allocator_t* arena,
    const loom_target_compile_report_t* report_template,
    loom_aie2p_program_compiler_t** out_compiler);

// Compiles |entry_function_op| and its producer-retained direct,
// nonrecursive, module-internal helper callees into one tile contribution.
// Helpers use scalar and pointer register ABI values, may reference immutable
// data, and cannot have imports, mutable storage, spills, or child calls.
// |program_data| is the admitted union of entry and helper definitions in this
// module. Its order agrees with the caller's physical placement plan.
// Call-free entries retain the exact ordinary leaf compilation path and ignore
// |program_data|. |may_return| comes from the entry's
// retained requirements or its producer's nonreturning construction contract.
// Only returning entries need to preserve their incoming link across calls.
iree_status_t loom_aie2p_program_compile(
    loom_aie2p_program_compiler_t* compiler, loom_op_t* entry_function_op,
    bool may_return, loom_aie2p_program_helper_list_t direct_helpers,
    const loom_low_read_only_data_set_t* program_data,
    const loom_aie2p_leaf_compile_options_t* entry_options, bool* out_compiled,
    loom_aie2p_leaf_contribution_t* out_contribution);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_PROGRAM_COMPILE_H_

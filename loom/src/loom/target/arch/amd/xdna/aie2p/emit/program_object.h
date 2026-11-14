// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// One-tile native object assembly from an entry and its helper closure.

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_PROGRAM_OBJECT_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_PROGRAM_OBJECT_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/target/arch/amd/xdna/aie2p/emit/leaf_object.h"

#ifdef __cplusplus
extern "C" {
#endif

// One immutable definition in the program's canonical source inventory.
typedef struct loom_aie2p_program_data_t {
  // Module-unique symbol name shared with the native contributions.
  iree_string_view_t name;
  // Conflicting definition ordinals in this complete program inventory.
  const uint32_t* bank_conflicts;
  // Number of records in |bank_conflicts|.
  iree_host_size_t bank_conflict_count;
} loom_aie2p_program_data_t;

// One logical function's identity in a detached complete program.
typedef struct loom_aie2p_program_function_t {
  // Detached function before merging, including its original local relocations.
  // Its owner outlives this program; native helper caches can share this
  // result.
  const loom_aie2p_leaf_contribution_t* contribution;
  // Entry symbol in the merged native object, including aliases of shared code.
  iree_host_size_t entry_symbol_index;
  // First equivalent function in this inventory. The program entry is unique.
  iree_host_size_t canonical_index;
} loom_aie2p_program_function_t;

// Physical program and the logical-to-physical identity established by merging.
typedef struct loom_aie2p_program_object_t {
  // Complete tile contribution, with sections and relocations owned by |arena|.
  loom_aie2p_leaf_contribution_t contribution;
  // Arena-owned functions in entry-then-helper input order. Logical identities
  // remain distinct when native bodies share a section, so execution lifetimes
  // can be projected without treating code equivalence as simultaneous use.
  const loom_aie2p_program_function_t* functions;
  // Number of logical functions, including the program entry.
  iree_host_size_t function_count;
} loom_aie2p_program_object_t;

// Joins an entry contribution followed by its reachable helper contributions.
// Source admission and native emission establish a nonempty complete closure
// of uniquely named functions with the same target and ABI. Each contribution
// retains its entry symbol and executable section; assembly trusts these facts.
//
// All executable fragments are normalized into the entry's code section. The
// entry remains first. Helpers with identical code and relocations share
// one body while retaining their symbols; placement of distinct bodies
// minimizes alignment padding with otherwise stable order. Ordinary
// contribution assembly performs the final aligned concatenation. Direct-call
// imports bind through an index of the retained function entries. Helpers must
// be module-internal functions reached only by direct calls, without observable
// code address identity. Immutable references participate in body equivalence.
// |data| lists the complete program's unique definitions and bank constraints;
// each definition contributes one section even when multiple functions use it.
// Names borrow the source module only for this call. The result copies conflict
// ordinals and retains the detached input contributions, which remain immutable
// and outlive the result.
// The entry continues to own all imports and mutable storage. The result
// retains its exact combined code requirement even when it exceeds one core's
// program memory; physical placement owns target capacity admission after
// composition. The result retains function-to-symbol and canonical identity
// without borrowing source IR or temporary layout storage. Scratch indexes
// need survive only through this call.
iree_status_t loom_aie2p_program_object_merge(
    const loom_aie2p_leaf_contribution_t* const* contributions,
    iree_host_size_t contribution_count, const loom_aie2p_program_data_t* data,
    iree_host_size_t data_count, iree_arena_allocator_t* arena,
    iree_arena_allocator_t* scratch_arena,
    loom_aie2p_program_object_t* out_program);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_PROGRAM_OBJECT_H_

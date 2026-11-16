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

// Exact native results before a placement owner assigns physical code groups.
typedef struct loom_aie2p_program_native_t {
  // Borrowed detached contributions in entry-then-helper order. The array and
  // its immutable contributions outlive native selection and composition.
  const loom_aie2p_leaf_contribution_t* const* contributions;
  // Number of logical functions, including the unique entry at index zero.
  iree_host_size_t function_count;
  // First equivalent native function, including its symbolic relocations.
  const iree_host_size_t* equivalent_indices;
  // Entry first, then helpers ordered by native equivalence and input ordinal.
  const iree_host_size_t* equivalence_order;
} loom_aie2p_program_native_t;

// One logical function's identity in a detached complete program.
typedef struct loom_aie2p_program_function_t {
  // Detached function before merging, including its original local relocations.
  // Its owner outlives this program; native helper caches can share this
  // result.
  const loom_aie2p_leaf_contribution_t* contribution;
  // Entry symbol in the merged native object, including aliases of shared code.
  iree_host_size_t entry_symbol_index;
  // First equivalent native function across all placement groups. Equality
  // includes symbolic relocations; the program entry remains unique.
  iree_host_size_t equivalent_index;
  // First equivalent function sharing this physical group's code and local
  // relocations. Equal bodies in other groups retain independent placement.
  iree_host_size_t canonical_index;
} loom_aie2p_program_function_t;

// Native equivalence and physical code placement established by composition.
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

// One independently placeable code section. The lifetime owner chooses which
// functions coexist in each group; composition only shares and packs its code.
typedef struct loom_aie2p_program_code_group_t {
  // Representative contribution identifying the group's assembled section.
  iree_host_size_t section_contribution_index;
  // Exact packed bytes and alignment, excluding other groups.
  loom_aie2p_leaf_storage_requirement_t code;
} loom_aie2p_program_code_group_t;

// Establishes native equivalence once, before physical groups are selected.
// Source admission supplies a nonempty closure with compatible targets/ABIs,
// uniquely named entries, and no observable helper code-address identity.
// The result borrows |contributions| and owns its equivalence indexes in arena.
iree_status_t loom_aie2p_program_native_initialize(
    const loom_aie2p_leaf_contribution_t* const* contributions,
    iree_host_size_t contribution_count, iree_arena_allocator_t* arena,
    loom_aie2p_program_native_t* out_native);

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
// composition. The result retains function-to-symbol bindings, global native
// equivalence, and physical body identities without borrowing source IR or
// temporary layout storage. Scratch indexes need survive only through this
// call.
iree_status_t loom_aie2p_program_object_merge(
    const loom_aie2p_program_native_t* native,
    const loom_aie2p_program_data_t* data, iree_host_size_t data_count,
    iree_arena_allocator_t* arena, iree_arena_allocator_t* scratch_arena,
    loom_aie2p_program_object_t* out_program);

// Composes the same admitted closure into independently placeable sections.
// |function_groups| assigns every input function to a dense, nonempty group;
// group zero contains the entry. Code sharing never crosses group boundaries:
// equal source bodies may require distinct addresses and local relocations.
// Global native equivalence remains available independently of that placement.
// Data and stack retain the complete program's ordinary ownership. The caller
// supplies |group_count| output slots; section names borrow detached inputs.
// The program's code requirement is the sum of packed group lengths, not its
// eventual physical footprint after lifetime placement.
iree_status_t loom_aie2p_program_object_partition(
    const loom_aie2p_program_native_t* native,
    const loom_aie2p_program_data_t* data, iree_host_size_t data_count,
    const iree_host_size_t* function_groups, iree_host_size_t group_count,
    iree_arena_allocator_t* arena, iree_arena_allocator_t* scratch_arena,
    loom_aie2p_program_object_t* out_program,
    loom_aie2p_program_code_group_t* out_groups);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_PROGRAM_OBJECT_H_

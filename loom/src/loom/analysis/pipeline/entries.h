// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_ANALYSIS_PIPELINE_ENTRIES_H_
#define LOOM_ANALYSIS_PIPELINE_ENTRIES_H_

#include "loom/analysis/pipeline/lifetimes.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_pipeline_entry_scope_t loom_pipeline_entry_scope_t;

// Static representation contract for compilation-owned execution state. A
// concrete scope embeds the common scope first and owns every additional fact
// needed by its execution owner. No source IR or pass storage may be borrowed.
typedef struct loom_pipeline_entry_scope_type_t {
  // Size of the complete concrete scope, including the common prefix.
  iree_host_size_t size;
  // Copies concrete state after the common lifetime facts have been cloned.
  // NULL for a scope containing only the common prefix. Allocation is the
  // only failure; the source is already admitted compiler-owned state.
  iree_status_t (*clone)(const loom_pipeline_entry_scope_t* source,
                         iree_arena_allocator_t* arena,
                         loom_pipeline_entry_scope_t* target);
} loom_pipeline_entry_scope_type_t;

// Source-independent execution lifetimes shared by the entries implementing
// one pipeline. Ordinary projected control owns runtime order and repetition;
// this retained tree owns coexistence and outer resource lifetimes. It contains
// no source operations, value IDs, or references to transient analysis storage.
struct loom_pipeline_entry_scope_t {
  // Static identity and independent-copy contract of the concrete owner.
  const loom_pipeline_entry_scope_type_t* type;
  // Complete indexed construction lifetimes, independent of the source IR.
  loom_pipeline_lifetimes_t lifetimes;
  // Number of construction owners in lifetimes.owners.
  iree_host_size_t owner_count;
};

// Representation containing only the common execution lifetime facts.
extern const loom_pipeline_entry_scope_type_t loom_pipeline_entry_scope_type;

typedef enum loom_pipeline_entry_role_e {
  // Runs this engine's complete contribution to a construction.
  LOOM_PIPELINE_ENTRY_COMPLETE,
  // Starts autonomous work before its instruction consumers execute.
  LOOM_PIPELINE_ENTRY_START,
  // Retires previously started autonomous work.
  LOOM_PIPELINE_ENTRY_RETIRE,
  // Initializes and retires the whole physical invocation.
  LOOM_PIPELINE_ENTRY_ROOT,
} loom_pipeline_entry_role_t;

// Binding retained by a concrete function version while lowering replaces its
// implementing function. Sharing a scope does not merge function ABIs or
// authorize overlap between constructions that require independent progress.
typedef struct loom_pipeline_entry_binding_t {
  // Immutable shared execution scope owned by the surrounding compilation.
  const loom_pipeline_entry_scope_t* scope;
  // Construction ordinal in scope->lifetimes.owners.
  iree_host_size_t owner;
  // Physical progress engine in this scope's admitted execution assignment.
  iree_host_size_t engine;
  // Part of the construction implemented by the bound function.
  loom_pipeline_entry_role_t role;
} loom_pipeline_entry_binding_t;

// Allocates a concrete scope and copies the producer's indexed lifetime facts.
// The caller initializes its concrete fields before publishing the scope to
// consumers. All fields and referenced storage belong to the compilation arena.
iree_status_t loom_pipeline_entry_scope_allocate(
    const loom_pipeline_entry_scope_type_t* type,
    const loom_pipeline_lifetimes_t* lifetimes, iree_host_size_t owner_count,
    iree_arena_allocator_t* arena, loom_pipeline_entry_scope_t** out_scope);

// Copies a complete scope into an independent compilation owner.
// No source traversal, ancestry reconstruction, or physical planning occurs.
// The result can outlive both source IR and its original analysis arena.
iree_status_t loom_pipeline_entry_scope_clone(
    const loom_pipeline_entry_scope_t* source, iree_arena_allocator_t* arena,
    const loom_pipeline_entry_scope_t** out_scope);

// Retained direct-call uses of one entry in a complete acyclic closure. The
// caller supplies callee-before-caller order from its admitted call graph;
// this projection never discovers edges or orders functions itself.
typedef struct loom_pipeline_entry_calls_t {
  // Construction bound to a phase entry, or IREE_HOST_SIZE_MAX for an ordinary
  // helper that inherits the common enclosing owner of all its callers.
  iree_host_size_t owner;
  // Direct callee indices in the same callee-before-caller inventory.
  const iree_host_size_t* callees;
  // Number of direct edges in callees.
  iree_host_size_t callee_count;
} loom_pipeline_entry_calls_t;

// Projects construction lifetimes onto a complete entry closure. The final
// entry is a bound root and reaches every preceding entry. Bound phase entries
// keep their source lifetimes; ordinary helpers inherit all caller uses,
// including transitive shared uses. The caller owns entry_count output slots.
// Allocation-free; each direct edge is visited once, with indexed common-owner
// joins independent of source nesting depth.
void loom_pipeline_entry_owners_project(
    const loom_pipeline_lifetimes_t* lifetimes,
    const loom_pipeline_entry_calls_t* entries, iree_host_size_t entry_count,
    iree_host_size_t* out_owners);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_ANALYSIS_PIPELINE_ENTRIES_H_

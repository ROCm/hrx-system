// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Memory spaces retained across structured scheduling rewrites.

#ifndef LOOM_TRANSFORMS_SCF_SCF_MEMORY_H_
#define LOOM_TRANSFORMS_SCF_SCF_MEMORY_H_

#include "iree/base/internal/arena.h"
#include "loom/ir/memory_space.h"
#include "loom/rewrite/remap.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_scf_memory_entry_t {
  // Original or cloned memory operation; NULL marks an unused table slot.
  const loom_op_t* op;
  // Concrete memory space, or UNKNOWN when the source analysis cannot prove it.
  uint8_t space;
  // Nearest source scheduling scope during initial fact resolution. Clones use
  // UINT32_MAX; reconstruction never queries this field.
  uint32_t scope;
} loom_scf_memory_entry_t;

// Invocation-owned spaces for every memory operation inside a selected scope.
// Discovery inserts original operations and their owning scope. The original
// value analysis fills their spaces before its invalidation. Reconstruction
// inserts clones through the rewriter's selected operation correspondence;
// spaces remain invariant even when iteration-dependent addresses change.
// Alias identities, coordinates and table-local extension IDs are not retained.
typedef struct loom_scf_memory_t {
  // Function-lifetime storage, independent of the invalidated value-fact owner.
  iree_arena_allocator_t* arena;
  // Open-addressed original and cloned access records, owned by the arena.
  loom_scf_memory_entry_t* entries;
  // Number of occupied entries.
  iree_host_size_t count;
  // Power-of-two table capacity, or zero before the first insertion.
  iree_host_size_t capacity;
} loom_scf_memory_t;

// Returns true when ordinary loads from |space| can form the read-ahead side
// of a workgroup-ordered pipeline. These concrete global-like spaces cannot
// alias workgroup storage; unknown and target-generic storage remain
// conservative.
bool loom_scf_memory_space_is_read_ahead_source(
    loom_value_fact_memory_space_t space);

// Inserts one distinct operation. Original operations are collected once;
// emitted clones have distinct identities and retain their source space.
iree_status_t loom_scf_memory_insert(loom_scf_memory_t* memory,
                                     const loom_op_t* op, uint8_t space,
                                     uint32_t scope);

// Returns the space retained for an operation in a selected scheduling scope.
// The discovery/projection owner has already inserted every queried operation.
uint8_t loom_scf_memory_lookup(const loom_scf_memory_t* memory,
                               const loom_op_t* op);

// Retains spaces for a completed selected clone correspondence and clears it
// from |remap|. Source operations remain indexed for subsequent iterations.
iree_status_t loom_scf_memory_project(loom_scf_memory_t* memory,
                                      loom_ir_remap_t* remap);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TRANSFORMS_SCF_SCF_MEMORY_H_

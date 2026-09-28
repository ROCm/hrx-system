// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_FORMAT_BYTECODE_WRITER_TYPE_INDEX_H_
#define LOOM_FORMAT_BYTECODE_WRITER_TYPE_INDEX_H_

#include "iree/base/internal/arena.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

// Serialization facts indexed by canonical module type ID. The module owns
// type identity and payloads; the writer owns the dependency slice and
// bindings.
typedef struct loom_bytecode_type_node_t {
  // Ordered immediate module type IDs, including repeated occurrences.
  struct {
    // Beginning of the slice in the index dependency array.
    iree_host_size_t begin;
    // Number of immediate dependencies.
    iree_host_size_t count;
    // Explicit child slots at the start of the slice, excluding scalar closure.
    iree_host_size_t explicit_count;
  } dependencies;
  // Whether this node or a child contains an actual SSA binding.
  bool has_bindings;
  // Scope generation which owns the completed binding ordinal.
  uint32_t binding_generation;
  // One-based completed binding ordinal in that scope.
  uint32_t binding;
} loom_bytecode_type_node_t;

// Explicit postorder continuation retained from graph construction.
typedef struct loom_bytecode_type_frame_t {
  // Canonical module type being completed.
  loom_type_id_t type_id;
  // Next immediate edge.
  iree_host_size_t next_dependency;
} loom_bytecode_type_frame_t;

// Invocation-owned serialization state indexed by canonical module type ID.
// Each type is analyzed once. Static types use global wire IDs; SSA-dependent
// types use scope-local records without shape folding. The immutable module
// outlives the index, and the supplied scratch arena owns all index storage.
typedef struct loom_bytecode_type_index_t {
  // Borrowed source module owning canonical type identities and payloads.
  const loom_module_t* module;
  // Serialization facts for every entry in the source module's type table.
  loom_bytecode_type_node_t* nodes;
  // Exact-storage lookup slots containing module IDs, or LOOM_TYPE_ID_INVALID.
  loom_type_id_t* slots;
  // Power-of-two slot count, fixed for the immutable source module.
  iree_host_size_t slot_capacity;
  // Ordered immediate module type IDs, owned by the scratch arena.
  loom_type_id_t* dependencies;
  // Reusable explicit traversal stack.
  loom_bytecode_type_frame_t* stack;
  // Completed nodes in the current extension batch, allocated lazily.
  loom_type_id_t* pending;
  // Monotonic generation assigned to each independent value scope.
  uint32_t binding_generation;
} loom_bytecode_type_index_t;

// Builds immediate dependency slices without changing module or wire order.
// Fallibility is limited to scratch allocation.
iree_status_t loom_bytecode_type_index_initialize(
    const loom_module_t* module, iree_arena_allocator_t* arena,
    loom_bytecode_type_index_t* out_index);

// Resolves a canonical by-value source type without traversing child payloads.
// Returns LOOM_TYPE_ID_INVALID when the type is absent from the source module.
loom_type_id_t loom_bytecode_type_index_lookup(
    const loom_bytecode_type_index_t* index, loom_type_t type);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_FORMAT_BYTECODE_WRITER_TYPE_INDEX_H_

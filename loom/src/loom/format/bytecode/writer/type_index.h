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

// Exact storage node and its completed wire-equivalence class. The module owns
// all borrowed type payloads; the index owns the ordered dependency slice.
typedef struct loom_bytecode_type_node_t {
  // Borrowed by-value type retaining the first-use physical payload.
  loom_type_t type;
  // Hash of exact storage, including payload addresses and SSA identities.
  uint32_t storage_hash;
  // Earliest equivalent module index, retained on the wire representative.
  loom_type_id_t module_index;
  // Ordered immediate child-node references, including repeated occurrences.
  struct {
    // Beginning of the slice in the index dependency array.
    iree_host_size_t begin;
    // Number of immediate dependencies.
    iree_host_size_t count;
  } dependencies;
  // Shallow structural hash using completed child equivalence classes.
  uint32_t wire_hash;
  // Wire-equivalent storage node, or UINT32_MAX during index construction.
  uint32_t representative;
} loom_bytecode_type_node_t;

// Invocation-owned projection from module type storage to wire equivalence.
// Structural wire identity omits scoped SSA dimension/encoding bindings. Each
// physically shared type is analyzed once; child equivalence is retained rather
// than recursively recovered during numbering. The immutable module outlives
// the index, and the supplied scratch arena owns all index storage.
typedef struct loom_bytecode_type_index_t {
  // Immutable module owning all borrowed type and attribute payloads.
  const loom_module_t* module;
  // Distinct by-value storage nodes and their retained wire representatives.
  loom_bytecode_type_node_t* nodes;
  // Number of populated storage nodes.
  iree_host_size_t count;
  // Allocated storage-node capacity.
  iree_host_size_t capacity;
  // Storage-node IDs indexed by exact representation hash, or UINT32_MAX.
  uint32_t* slots;
  // Power-of-two capacity of slots.
  iree_host_size_t slot_capacity;
  // Retained ordered immediate dependency node IDs, owned by the scratch arena.
  uint32_t* dependencies;
} loom_bytecode_type_index_t;

// Builds the wire-equivalence projection without changing module or wire order.
// Fallibility is limited to scratch allocation and index-size representation.
iree_status_t loom_bytecode_type_index_initialize(
    const loom_module_t* module, iree_arena_allocator_t* arena,
    loom_bytecode_type_index_t* out_index);

// Returns the exact physical node, or NULL when the type has no retained
// storage. Its representative owns the canonical module index used on wire.
const loom_bytecode_type_node_t* loom_bytecode_type_index_lookup_node(
    const loom_bytecode_type_index_t* index, loom_type_t type);

// Returns the earliest module-table entry equivalent to a retained type, or
// LOOM_TYPE_ID_INVALID when the type is not in the module's retained closure.
// No structural traversal or allocation occurs during lookup.
loom_type_id_t loom_bytecode_type_index_lookup(
    const loom_bytecode_type_index_t* index, loom_type_t type);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_FORMAT_BYTECODE_WRITER_TYPE_INDEX_H_

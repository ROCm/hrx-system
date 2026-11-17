// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Sparse identities for source operations participating in source graphs.

#ifndef LOOM_CODEGEN_LOW_LOWER_SOURCE_GRAPH_NODES_H_
#define LOOM_CODEGEN_LOW_LOWER_SOURCE_GRAPH_NODES_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/local_value_domain.h"
#include "loom/ir/module.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_low_lower_source_graph_node_t {
  // Next participating node in insertion order, or NULL at the end.
  struct loom_low_lower_source_graph_node_t* next;
  // Immutable source operation represented by this node.
  const loom_op_t* source_op;
  // Caller-owned payload associated with this node.
  void* user_data;
  // Dense identity assigned in insertion order.
  uint32_t index;
} loom_low_lower_source_graph_node_t;

typedef struct loom_low_lower_source_graph_source_op_entry_t {
  // Source operation without a value-domain ordinal, or NULL for an empty slot.
  const loom_op_t* source_op;
  // Sparse source graph node for source_op.
  loom_low_lower_source_graph_node_t* node;
} loom_low_lower_source_graph_source_op_entry_t;

typedef struct loom_low_lower_source_graph_node_index_t {
  // First participating node in insertion order.
  loom_low_lower_source_graph_node_t* first;
  // Last participating node in insertion order.
  loom_low_lower_source_graph_node_t* last;
  // Optional function-local identities. NULL keeps every node pointer-indexed
  // when a sparse consumer cannot amortize a function-sized value table.
  const loom_local_value_domain_t* value_domain;
  // Arena owning nodes and index storage.
  iree_arena_allocator_t* arena;
  // Source-value-ordinal indexed nodes for operations producing results.
  loom_low_lower_source_graph_node_t** value_nodes;
  // Number of entries in value_nodes.
  loom_value_ordinal_t value_node_count;
  // Pointer-keyed nodes for operations without a value-domain ordinal.
  loom_low_lower_source_graph_source_op_entry_t* source_op_entries;
  // Number of populated source_op_entries.
  iree_host_size_t source_op_entry_count;
  // Power-of-two capacity of source_op_entries.
  iree_host_size_t source_op_entry_capacity;
  // Number of participating source nodes.
  uint32_t count;
} loom_low_lower_source_graph_node_index_t;

// Initializes an empty sparse node index borrowing optional value_domain.
void loom_low_lower_source_graph_node_index_initialize(
    const loom_local_value_domain_t* value_domain,
    iree_arena_allocator_t* arena,
    loom_low_lower_source_graph_node_index_t* out_index);

// Looks up source_op without changing the index.
loom_low_lower_source_graph_node_t*
loom_low_lower_source_graph_node_index_lookup(
    const loom_low_lower_source_graph_node_index_t* index,
    const loom_op_t* source_op);

// Returns source_op's node, assigning its dense identity on first use.
iree_status_t loom_low_lower_source_graph_node_index_get_or_add(
    loom_low_lower_source_graph_node_index_t* index, const loom_op_t* source_op,
    loom_low_lower_source_graph_node_t** out_node);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_LOWER_SOURCE_GRAPH_NODES_H_

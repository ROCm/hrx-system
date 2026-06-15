// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Rebuildable module symbol dependency table.
//
// This analysis is the canonical symbol-use substrate for module-level
// analyses. It records occurrence edges from the symbol whose definition owns
// an operation to each module-local symbol referenced by that operation, nested
// attributes, static encoding attributes, and value types. The table is
// intentionally rebuilt from immutable IR instead of maintained on every IR
// mutation path; callers that rewrite symbols should rebuild after mutation.

#ifndef LOOM_ANALYSIS_SYMBOL_DEPENDENCIES_H_
#define LOOM_ANALYSIS_SYMBOL_DEPENDENCIES_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/analysis/scc.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

// Index into a symbol dependency table's edge array.
typedef uint32_t loom_symbol_dependency_edge_id_t;
#define LOOM_SYMBOL_DEPENDENCY_EDGE_ID_INVALID \
  ((loom_symbol_dependency_edge_id_t)UINT32_MAX)

// Sentinel for edges that are not attached to a concrete op attribute.
#define LOOM_SYMBOL_DEPENDENCY_ATTR_INDEX_NONE ((uint16_t)UINT16_MAX)

// Classifies where a symbol dependency occurrence was found.
typedef enum loom_symbol_dependency_edge_kind_e {
  // Invalid default used only for zero-initialized storage.
  LOOM_SYMBOL_DEPENDENCY_EDGE_NONE = 0,
  // Direct generated symbol-reference op attribute.
  LOOM_SYMBOL_DEPENDENCY_EDGE_SYMBOL_ATTR = 1,
  // Direct call-like callee attribute.
  LOOM_SYMBOL_DEPENDENCY_EDGE_CALL = 2,
  // Direct global-like load/store/reference attribute.
  LOOM_SYMBOL_DEPENDENCY_EDGE_GLOBAL_ACCESS = 3,
  // Symbol reference nested inside a DICT attribute.
  LOOM_SYMBOL_DEPENDENCY_EDGE_NESTED_ATTR = 4,
  // Symbol reference reached through an AttrType static encoding.
  LOOM_SYMBOL_DEPENDENCY_EDGE_TYPE_ATTR = 5,
  // Symbol reference reached through an AttrEncoding static encoding.
  LOOM_SYMBOL_DEPENDENCY_EDGE_ENCODING_ATTR = 6,
  // Symbol reference reached through an SSA value type's static encoding.
  LOOM_SYMBOL_DEPENDENCY_EDGE_VALUE_TYPE = 7,
  // Symbol reference reached through a module encoding table record.
  LOOM_SYMBOL_DEPENDENCY_EDGE_MODULE_ENCODING = 8,
} loom_symbol_dependency_edge_kind_t;

// One symbol reference occurrence.
typedef struct loom_symbol_dependency_edge_t {
  // Symbol that owns this reference, or LOOM_SYMBOL_ID_INVALID for module-root
  // records such as static module encodings.
  loom_symbol_id_t source_symbol_id;
  // Module-local symbol referenced by this occurrence.
  loom_symbol_id_t target_symbol_id;
  // Classified source of the occurrence.
  loom_symbol_dependency_edge_kind_t kind;
  // Attribute index on user_op when the occurrence came from an op attribute.
  uint16_t attr_index;
  // Operation that owns the occurrence, or NULL for module-root records.
  const loom_op_t* user_op;
  // Next occurrence with the same source symbol.
  loom_symbol_dependency_edge_id_t next_outgoing_edge_id;
  // Next occurrence with the same target symbol.
  loom_symbol_dependency_edge_id_t next_incoming_edge_id;
} loom_symbol_dependency_edge_t;

// Incoming/outgoing edge-list heads for one symbol.
typedef struct loom_symbol_dependency_symbol_edges_t {
  // First edge whose source_symbol_id is this symbol.
  loom_symbol_dependency_edge_id_t first_outgoing_edge_id;
  // First edge whose target_symbol_id is this symbol.
  loom_symbol_dependency_edge_id_t first_incoming_edge_id;
  // Number of outgoing occurrence edges.
  uint32_t outgoing_count;
  // Number of incoming occurrence edges.
  uint32_t incoming_count;
} loom_symbol_dependency_symbol_edges_t;

// Built dependency table for one module snapshot.
typedef struct loom_symbol_dependency_table_t {
  // Module this table was built from.
  const loom_module_t* module;
  // Dense per-symbol incoming/outgoing lists.
  const loom_symbol_dependency_symbol_edges_t* symbols;
  // Number of entries in symbols.
  iree_host_size_t symbol_count;
  // Occurrence edges owned by the caller-provided arena.
  const loom_symbol_dependency_edge_t* edges;
  // Number of entries in edges.
  iree_host_size_t edge_count;
  // First module-root edge.
  loom_symbol_dependency_edge_id_t first_module_edge_id;
  // Number of module-root edges.
  uint32_t module_edge_count;
} loom_symbol_dependency_table_t;

// Builds the symbol dependency table for |module| into |arena|.
iree_status_t loom_symbol_dependency_table_build(
    const loom_module_t* module, iree_arena_allocator_t* arena,
    loom_symbol_dependency_table_t* out_table);

// Returns an SCC graph adapter whose node ordinals are module symbol IDs.
loom_scc_graph_t loom_symbol_dependency_scc_graph(
    const loom_symbol_dependency_table_t* table);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_ANALYSIS_SYMBOL_DEPENDENCIES_H_

// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Rebuildable module symbol reference table.
//
// This analysis is the canonical symbol-use substrate for module-level
// analyses. It records occurrences from the symbol whose definition owns
// an operation to each module-local symbol referenced by that operation, nested
// attributes, static encoding attributes, and each SSA value definition's
// type. The table is intentionally rebuilt from immutable IR instead of
// maintained on every IR mutation path; callers that rewrite symbols should
// rebuild after mutation.

#ifndef LOOM_ANALYSIS_SYMBOL_REFERENCES_H_
#define LOOM_ANALYSIS_SYMBOL_REFERENCES_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/analysis/scc.h"
#include "loom/ir/attribute_schema.h"
#include "loom/ir/ir.h"
#include "loom/util/segmented_storage.h"

#ifdef __cplusplus
extern "C" {
#endif

// Dense ordinal in a symbol reference table's stable occurrence storage.
typedef uint32_t loom_symbol_reference_occurrence_id_t;
#define LOOM_SYMBOL_REFERENCE_OCCURRENCE_ID_INVALID \
  ((loom_symbol_reference_occurrence_id_t)UINT32_MAX)

// Index into a symbol reference table's template-demand array.
typedef uint32_t loom_template_demand_id_t;
#define LOOM_TEMPLATE_DEMAND_ID_INVALID ((loom_template_demand_id_t)UINT32_MAX)

// Sentinel for occurrences not attached to a concrete op attribute.
#define LOOM_SYMBOL_REFERENCE_ATTR_INDEX_NONE ((uint8_t)UINT8_MAX)

// Independent execution region ordinal plus one in execution_targets. Zero
// names the owning function's context, including its specialized version.
// A nonzero scope enters that environment's independently executing worker
// contract, optionally selecting a different target first.
typedef uint32_t loom_symbol_reference_execution_scope_id_t;

// Classifies where a symbol reference occurrence was found.
typedef enum loom_symbol_reference_occurrence_kind_e {
  // Invalid default used only for zero-initialized storage.
  LOOM_SYMBOL_REFERENCE_OCCURRENCE_NONE = 0,
  // Direct generated symbol-reference op attribute.
  LOOM_SYMBOL_REFERENCE_OCCURRENCE_SYMBOL_ATTR = 1,
  // Direct call-like callee attribute.
  LOOM_SYMBOL_REFERENCE_OCCURRENCE_CALL = 2,
  // Direct global-like load/store/reference attribute.
  LOOM_SYMBOL_REFERENCE_OCCURRENCE_GLOBAL_ACCESS = 3,
  // Symbol reference nested inside a DICT attribute.
  LOOM_SYMBOL_REFERENCE_OCCURRENCE_NESTED_ATTR = 4,
  // Symbol reference reached through an AttrType static encoding.
  LOOM_SYMBOL_REFERENCE_OCCURRENCE_TYPE_ATTR = 5,
  // Symbol reference reached through an AttrEncoding static encoding.
  LOOM_SYMBOL_REFERENCE_OCCURRENCE_ENCODING_ATTR = 6,
  // Symbol reference reached through an SSA value type's static encoding.
  LOOM_SYMBOL_REFERENCE_OCCURRENCE_VALUE_TYPE = 7,
  // Symbol reference reached through a module encoding table record.
  LOOM_SYMBOL_REFERENCE_OCCURRENCE_MODULE_ENCODING = 8,
} loom_symbol_reference_occurrence_kind_t;

// One symbol reference occurrence.
typedef struct loom_symbol_reference_occurrence_t {
  // Symbol that owns this reference, or LOOM_SYMBOL_ID_INVALID for module-root
  // records such as static module encodings.
  loom_symbol_id_t source_symbol_id;
  // Module-local symbol referenced by this occurrence.
  loom_symbol_id_t target_symbol_id;
  // Target interfaces accepted by the owning attribute schema, or zero when
  // the reference has no structural interface constraint.
  loom_symbol_interface_flags_t target_interfaces;
  // Next occurrence with the same source symbol.
  loom_symbol_reference_occurrence_id_t next_outgoing_occurrence_id;
  // Next occurrence with the same target symbol.
  loom_symbol_reference_occurrence_id_t next_incoming_occurrence_id;
  // Attribute index on user_op when the occurrence came from an op attribute.
  uint8_t attr_index;
  // Classified source of the occurrence. Shares one byte with its join fact.
  uint8_t kind : 7;
  // This composition completes before its enclosing resident schedule advances.
  uint8_t joins_composition : 1;
  // Compile-time graph role declared by the owning attribute schema.
  loom_symbol_reference_role_t role;
  // Root region slot on the source symbol plus one, or zero for its contract.
  uint8_t source_root_region_index_plus_one;
  // Execution context retained by the structural reference traversal.
  loom_symbol_reference_execution_scope_id_t execution_scope;
  // Operation that owns the occurrence, or NULL for module-root records.
  const loom_op_t* user_op;
} loom_symbol_reference_occurrence_t;

static_assert(sizeof(loom_symbol_reference_occurrence_t) ==
                  (IREE_PTR_SIZE == 8 ? 32 : 28),
              "symbol reference occurrences must remain compact");

// Shift mapping an occurrence ID to its fixed-size segment.
#define LOOM_SYMBOL_REFERENCE_OCCURRENCE_SEGMENT_SHIFT 6u

// Number of occurrence rows in each arena-owned segment.
#define LOOM_SYMBOL_REFERENCE_OCCURRENCE_SEGMENT_CAPACITY \
  (1u << LOOM_SYMBOL_REFERENCE_OCCURRENCE_SEGMENT_SHIFT)

static_assert((uint64_t)LOOM_SYMBOL_REFERENCE_OCCURRENCE_SEGMENT_CAPACITY *
                      LOOM_SEGMENTED_STORAGE_MAX_SEGMENT_COUNT >=
                  (uint64_t)LOOM_SYMBOL_REFERENCE_OCCURRENCE_ID_INVALID,
              "occurrence storage must cover the full occurrence ID domain");

// Returns true when |occurrence| contributes to reachability and link closure.
static inline bool loom_symbol_reference_occurrence_is_dependency(
    const loom_symbol_reference_occurrence_t* occurrence) {
  return loom_symbol_reference_role_is_dependency(occurrence->role);
}

// One abstract template.apply provider demand.
typedef struct loom_template_demand_t {
  // template.apply operation carrying this demand.
  const loom_op_t* apply_op;
  // Next demand owned by the same source symbol.
  loom_template_demand_id_t next_source_demand_id;
  // Module-local template family requested by the application.
  loom_symbol_id_t family_symbol_id;
  // Symbol whose definition owns the application.
  loom_symbol_id_t source_symbol_id;
  // Root region slot on source_symbol_id plus one, or zero for its contract.
  uint8_t source_root_region_index_plus_one;

  // True under structured conditions or in a non-entry CFG block. Applicability
  // may depend on path facts that are not available at the function boundary.
  bool has_path_condition;
  // Execution context at the application, independent of captured-value facts.
  loom_symbol_reference_execution_scope_id_t execution_scope;
} loom_template_demand_t;

static_assert(sizeof(loom_template_demand_t) == (IREE_PTR_SIZE == 8 ? 24 : 20),
              "template demands must remain compact");

// Incoming/outgoing occurrence-list heads for one referenced symbol.
typedef struct loom_symbol_reference_symbol_occurrences_t {
  // First occurrence whose source_symbol_id is this symbol.
  loom_symbol_reference_occurrence_id_t first_outgoing_occurrence_id;
  // First occurrence whose target_symbol_id is this symbol.
  loom_symbol_reference_occurrence_id_t first_incoming_occurrence_id;
  // First abstract provider demand owned by this symbol.
  loom_template_demand_id_t first_template_demand_id;
  // Number of abstract provider demands owned by this symbol.
  uint32_t template_demand_count;
} loom_symbol_reference_symbol_occurrences_t;

static_assert(sizeof(loom_symbol_reference_symbol_occurrences_t) == 16,
              "symbol occurrence heads must remain 16 bytes");

// Shift mapping a symbol ID to its lazily allocated row segment.
#define LOOM_SYMBOL_REFERENCE_SYMBOL_SEGMENT_SHIFT 7u

// Number of symbol occurrence rows in each arena-owned segment.
#define LOOM_SYMBOL_REFERENCE_SYMBOL_SEGMENT_CAPACITY \
  (1u << LOOM_SYMBOL_REFERENCE_SYMBOL_SEGMENT_SHIFT)

// Direct call occurrences classified during reference publication.
typedef struct loom_symbol_reference_call_counts_t {
  // Number of direct calls across all call kinds and graph roles.
  uint32_t count;
  // Subset of |count| for exact template.call operations. Abstract
  // template.apply demands are counted separately.
  uint32_t template_count;
} loom_symbol_reference_call_counts_t;

// Built reference table for one module snapshot.
typedef struct loom_symbol_reference_table_t {
  // Module this table was built from.
  const loom_module_t* module;
  // Direct directory of lazily allocated symbol occurrence row segments.
  const loom_symbol_reference_symbol_occurrences_t* const* symbol_segments;
  // Number of logical symbol rows addressable through symbol_segments.
  iree_host_size_t symbol_count;
  // Stable occurrence segments owned by the caller-provided arena.
  loom_segmented_storage_t occurrences;
  // Number of published occurrences; unused segment rows are uninitialized.
  iree_host_size_t occurrence_count;
  // First module-root occurrence.
  loom_symbol_reference_occurrence_id_t first_module_occurrence_id;
  // Number of module-root occurrences.
  uint32_t module_occurrence_count;
  // Call counts retained by the reference producer for plan sizing.
  loom_symbol_reference_call_counts_t calls;
  // Number of valid module-local template providers.
  uint32_t template_provider_count;
  // Dense membership of symbols containing contextual operations that inherit
  // their target environment. Explicit nested execution targets are excluded.
  // NULL when there are no such operations; storage belongs to the table arena.
  const uint64_t* contextual_symbols;
  // Dense membership of symbols directly owning func.exit operations. Calls
  // do not propagate membership: each bit describes only the symbol body
  // visited by this structural analysis. NULL when no callable exits exist.
  const uint64_t* callable_exit_symbols;
  // Target environments selected by independently executing regions. Nested
  // regions inheriting an existing worker environment share its scope.
  struct {
    // References in traversal order. Null inherits the owning function's
    // environment; an explicit unresolved reference selects an unbound worker.
    const loom_symbol_ref_t* values;
    // Number of distinct execution scopes.
    iree_host_size_t count;
  } execution_targets;
  // Abstract template.apply provider demands owned by module symbols.
  struct {
    // Demand records owned by the caller-provided arena.
    const loom_template_demand_t* values;
    // Number of entries in values.
    iree_host_size_t count;

    // Unique demanded family symbol IDs in ascending order.
    const loom_symbol_id_t* family_symbol_ids;

    // Number of entries in family_symbol_ids.
    iree_host_size_t family_count;

    // Dense bitset indexed by module symbol ID for demanded families, or NULL
    // when the module contains no template demands.
    const uint64_t* family_bits;
  } template_demands;

} loom_symbol_reference_table_t;

// Returns whether this symbol directly contains source expansion depending on
// its inherited target context. Calls contribute through the consumer's graph;
// the structural producer does not recursively inspect their definitions.
static inline bool loom_symbol_reference_symbol_is_contextual(
    const loom_symbol_reference_table_t* table, loom_symbol_id_t symbol_id) {
  return table->contextual_symbols &&
         (table->contextual_symbols[symbol_id >> 6] &
          (UINT64_C(1) << (symbol_id & 63u))) != 0;
}

// Returns whether this symbol directly owns a callable exit. Transitive call
// closure remains the responsibility of the consuming call-graph analysis.
static inline bool loom_symbol_reference_symbol_has_callable_exit(
    const loom_symbol_reference_table_t* table, loom_symbol_id_t symbol_id) {
  return table->callable_exit_symbols &&
         (table->callable_exit_symbols[symbol_id >> 6] &
          (UINT64_C(1) << (symbol_id & 63u))) != 0;
}

// Returns occurrence heads for a valid module symbol ID. Symbols with no
// incoming, outgoing, or template-demand occurrences return an empty row by
// value and require no arena storage.
static inline loom_symbol_reference_symbol_occurrences_t
loom_symbol_reference_table_symbol(const loom_symbol_reference_table_t* table,
                                   loom_symbol_id_t symbol_id) {
  const loom_symbol_reference_symbol_occurrences_t* segment =
      table->symbol_segments
          ? table->symbol_segments[symbol_id >>
                                   LOOM_SYMBOL_REFERENCE_SYMBOL_SEGMENT_SHIFT]
          : NULL;
  if (!segment) {
    return (loom_symbol_reference_symbol_occurrences_t){
        /*.first_outgoing_occurrence_id=*/
        LOOM_SYMBOL_REFERENCE_OCCURRENCE_ID_INVALID,
        /*.first_incoming_occurrence_id=*/
        LOOM_SYMBOL_REFERENCE_OCCURRENCE_ID_INVALID,
        /*.first_template_demand_id=*/LOOM_TEMPLATE_DEMAND_ID_INVALID,
        /*.template_demand_count=*/0,
    };
  }
  return segment[symbol_id &
                 (LOOM_SYMBOL_REFERENCE_SYMBOL_SEGMENT_CAPACITY - 1u)];
}

// Returns an occurrence by its valid ID in this immutable analysis snapshot.
// The row borrows the table's arena; user_op also borrows the analyzed module.
static inline const loom_symbol_reference_occurrence_t*
loom_symbol_reference_table_occurrence(
    const loom_symbol_reference_table_t* table,
    loom_symbol_reference_occurrence_id_t occurrence_id) {
  const loom_symbol_reference_occurrence_t* segment =
      (const loom_symbol_reference_occurrence_t*)
          loom_segmented_storage_const_segment(
              &table->occurrences,
              occurrence_id >> LOOM_SYMBOL_REFERENCE_OCCURRENCE_SEGMENT_SHIFT);
  return &segment[occurrence_id &
                  (LOOM_SYMBOL_REFERENCE_OCCURRENCE_SEGMENT_CAPACITY - 1u)];
}

// Returns true when at least one template.apply demands |family_symbol_id|.
// |family_symbol_id| must be valid in the table's module symbol table.
static inline bool loom_symbol_reference_template_family_is_demanded(
    const loom_symbol_reference_table_t* table,
    loom_symbol_id_t family_symbol_id) {
  return table->template_demands.family_bits &&
         (table->template_demands.family_bits[family_symbol_id >> 6] &
          (UINT64_C(1) << (family_symbol_id & 63u))) != 0;
}

// Builds an immutable reference snapshot borrowing |module| and storage owned
// by |arena|. On failure, |out_table| is empty; any partial allocations remain
// owned by |arena| until reset. No occurrence storage is retained by the
// module.
iree_status_t loom_symbol_reference_table_build(
    const loom_module_t* module, iree_arena_allocator_t* arena,
    loom_symbol_reference_table_t* out_table);

// Returns the dependency SCC graph whose node ordinals are module symbol IDs.
loom_scc_graph_t loom_symbol_reference_dependency_scc_graph(
    const loom_symbol_reference_table_t* table);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_ANALYSIS_SYMBOL_REFERENCES_H_

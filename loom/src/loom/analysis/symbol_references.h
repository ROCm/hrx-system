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

#ifdef __cplusplus
extern "C" {
#endif

// Index into a symbol reference table's occurrence array.
typedef uint32_t loom_symbol_reference_occurrence_id_t;
#define LOOM_SYMBOL_REFERENCE_OCCURRENCE_ID_INVALID \
  ((loom_symbol_reference_occurrence_id_t)UINT32_MAX)

// Index into a symbol reference table's template-demand array.
typedef uint32_t loom_template_demand_id_t;
#define LOOM_TEMPLATE_DEMAND_ID_INVALID ((loom_template_demand_id_t)UINT32_MAX)

// Index into a symbol reference table's template-provider array.
typedef uint32_t loom_template_provider_reference_id_t;
#define LOOM_TEMPLATE_PROVIDER_REFERENCE_ID_INVALID \
  ((loom_template_provider_reference_id_t)UINT32_MAX)

// Sentinel for occurrences not attached to a concrete op attribute.
#define LOOM_SYMBOL_REFERENCE_ATTR_INDEX_NONE ((uint16_t)UINT16_MAX)

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
  // Classified source of the occurrence.
  loom_symbol_reference_occurrence_kind_t kind;
  // Compile-time graph role declared by the owning attribute schema.
  loom_symbol_reference_role_t role;
  // Attribute index on user_op when the occurrence came from an op attribute.
  uint16_t attr_index;
  // Operation that owns the occurrence, or NULL for module-root records.
  const loom_op_t* user_op;
  // Next occurrence with the same source symbol.
  loom_symbol_reference_occurrence_id_t next_outgoing_occurrence_id;
  // Next occurrence with the same target symbol.
  loom_symbol_reference_occurrence_id_t next_incoming_occurrence_id;
} loom_symbol_reference_occurrence_t;

static_assert(sizeof(loom_symbol_reference_occurrence_t) == 32,
              "symbol reference occurrences must remain 32 bytes");

// Returns true when |occurrence| contributes to reachability and link closure.
static inline bool loom_symbol_reference_occurrence_is_dependency(
    const loom_symbol_reference_occurrence_t* occurrence) {
  return occurrence->role == LOOM_SYMBOL_REFERENCE_ROLE_DEPENDENCY;
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
} loom_template_demand_t;

static_assert(sizeof(loom_template_demand_t) == 16,
              "template demands must remain 16 bytes");

// One available template provider indexed by its implemented family.
typedef struct loom_template_provider_reference_t {
  // Module-local provider symbol.
  loom_symbol_id_t symbol_id;

  // Next provider implementing the same family.
  loom_template_provider_reference_id_t next_family_provider_id;
} loom_template_provider_reference_t;

static_assert(sizeof(loom_template_provider_reference_t) == 8,
              "template provider references must remain 8 bytes");

// Incoming/outgoing occurrence-list heads for one symbol.
typedef struct loom_symbol_reference_symbol_occurrences_t {
  // First occurrence whose source_symbol_id is this symbol.
  loom_symbol_reference_occurrence_id_t first_outgoing_occurrence_id;
  // First occurrence whose target_symbol_id is this symbol.
  loom_symbol_reference_occurrence_id_t first_incoming_occurrence_id;
  // Number of outgoing occurrences across all graph roles.
  uint32_t outgoing_count;
  // Number of incoming occurrences across all graph roles.
  uint32_t incoming_count;
  // First abstract provider demand owned by this symbol.
  loom_template_demand_id_t first_template_demand_id;
  // Number of abstract provider demands owned by this symbol.
  uint32_t template_demand_count;
} loom_symbol_reference_symbol_occurrences_t;

// Built reference table for one module snapshot.
typedef struct loom_symbol_reference_table_t {
  // Module this table was built from.
  const loom_module_t* module;
  // Dense per-symbol incoming/outgoing lists.
  const loom_symbol_reference_symbol_occurrences_t* symbols;
  // Number of entries in symbols.
  iree_host_size_t symbol_count;
  // Occurrences owned by the caller-provided arena.
  const loom_symbol_reference_occurrence_t* occurrences;
  // Number of entries in occurrences.
  iree_host_size_t occurrence_count;
  // First module-root occurrence.
  loom_symbol_reference_occurrence_id_t first_module_occurrence_id;
  // Number of module-root occurrences.
  uint32_t module_occurrence_count;
  // Abstract template.apply provider demands owned by module symbols.
  struct {
    // Demand records owned by the caller-provided arena.
    const loom_template_demand_t* values;
    // Number of entries in values.
    iree_host_size_t count;
    // Dense bitset indexed by module symbol ID for demanded families, or NULL
    // when the module contains no template demands.
    const uint64_t* family_bits;
  } template_demands;

  // Available template providers indexed by implemented family.
  struct {
    // Provider records owned by the caller-provided arena.
    const loom_template_provider_reference_t* values;

    // Number of entries in values.
    iree_host_size_t count;

    // First provider record for each module-local family symbol.
    const loom_template_provider_reference_id_t* first_by_family_symbol_id;
  } template_providers;
} loom_symbol_reference_table_t;

// Returns true when at least one template.apply demands |family_symbol_id|.
// |family_symbol_id| must be valid in the table's module symbol table.
static inline bool loom_symbol_reference_template_family_is_demanded(
    const loom_symbol_reference_table_t* table,
    loom_symbol_id_t family_symbol_id) {
  return table->template_demands.family_bits &&
         (table->template_demands.family_bits[family_symbol_id >> 6] &
          (UINT64_C(1) << (family_symbol_id & 63u))) != 0;
}

// Builds the symbol reference table for |module| into |arena|.
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

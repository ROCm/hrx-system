// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Reached-only shared-table materialization for selected bytecode symbols.

#ifndef LOOM_FORMAT_BYTECODE_READER_SELECTED_TABLES_H_
#define LOOM_FORMAT_BYTECODE_READER_SELECTED_TABLES_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/format/bytecode/index.h"
#include "loom/format/bytecode/reader/attribute.h"
#include "loom/format/bytecode/reader/decoder.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/util/ordinal_map.h"

#ifdef __cplusplus
extern "C" {
#endif

// Identity domains requiring memoization during selective materialization.
//
// Strings are projected directly through the output interner and registered
// operations resolve directly through the context. Every other source-table
// identity receives an entry only when reached.
typedef enum loom_bytecode_selected_projection_domain_e {
  LOOM_BYTECODE_SELECTED_PROJECTION_DOMAIN_SYMBOL_NAME = 0,
  LOOM_BYTECODE_SELECTED_PROJECTION_DOMAIN_SOURCE = 1,
  LOOM_BYTECODE_SELECTED_PROJECTION_DOMAIN_ENCODING = 2,
  LOOM_BYTECODE_SELECTED_PROJECTION_DOMAIN_TYPE = 3,
  LOOM_BYTECODE_SELECTED_PROJECTION_DOMAIN_LOCATION = 4,
  LOOM_BYTECODE_SELECTED_PROJECTION_DOMAIN_COUNT = 5,
} loom_bytecode_selected_projection_domain_t;

// One shared-table identity waiting to be materialized.
typedef enum loom_bytecode_selected_table_kind_e {
  LOOM_BYTECODE_SELECTED_TABLE_ENCODING = 0,
  LOOM_BYTECODE_SELECTED_TABLE_TYPE = 1,
  LOOM_BYTECODE_SELECTED_TABLE_LOCATION = 2,
} loom_bytecode_selected_table_kind_t;

// Resolves one module-local source symbol ordinal into the output module.
typedef iree_status_t (*loom_bytecode_selected_symbol_resolver_fn_t)(
    void* user_data, uint32_t source_symbol_ordinal,
    loom_symbol_ref_t* out_target_symbol_ref);

// Optional external symbol projection used by metadata-only readers.
typedef struct loom_bytecode_selected_symbol_resolver_t {
  // Resolver invoked when a symbol was not predeclared by this materializer.
  loom_bytecode_selected_symbol_resolver_fn_t fn;
  // Caller-owned payload passed to fn.
  void* user_data;
} loom_bytecode_selected_symbol_resolver_t;

// Returns an empty external symbol resolver.
static inline loom_bytecode_selected_symbol_resolver_t
loom_bytecode_selected_symbol_resolver_empty(void) {
  return (loom_bytecode_selected_symbol_resolver_t){0};
}

// Returns a resolver wrapping fn and user_data.
static inline loom_bytecode_selected_symbol_resolver_t
loom_bytecode_selected_symbol_resolver_make(
    loom_bytecode_selected_symbol_resolver_fn_t fn, void* user_data) {
  return (loom_bytecode_selected_symbol_resolver_t){
      /*.fn=*/fn,
      /*.user_data=*/user_data,
  };
}

// Invocation-local state for reached-only shared-table materialization.
//
// Exact entry ranges and borrowed strings come from |metadata|. Completed
// source identities are memoized in |projection|, while the arena-owned
// worklist replaces recursive C calls across prior-entry dependency chains.
typedef struct loom_bytecode_selected_table_materializer_t {
  // Bounded decoder and structured diagnostic state.
  loom_bytecode_reader_decoder_t* decoder;
  // Complete source bytecode containing indexed entry spans.
  iree_const_byte_span_t bytecode;
  // Finalized dialect and registry context.
  loom_context_t* context;
  // Immutable retained metadata for the selected source module.
  const loom_bytecode_module_metadata_t* metadata;
  // Resettable storage for decoded facts and temporary payloads.
  iree_arena_allocator_t* scratch_arena;
  // Module receiving compact canonical table entries.
  loom_module_t* output_module;
  // Optional projection for references to symbols outside the selection.
  loom_bytecode_selected_symbol_resolver_t symbol_resolver;
  // Retained projections and worklist storage, independent of fact checkpoints.
  iree_arena_allocator_t retained_arena;
  // Reached source identity to compact output identity map.
  loom_ordinal_map_t projection;
  // Number of distinct source names reached by this materializer. Equality
  // with the output source count proves every output name is already reached,
  // so another source from the validated unique input can append directly.
  iree_host_size_t projected_source_count;
  // Explicit dependency stack reused across root resolutions.
  struct {
    // First retained-arena-owned chunk, reused after each root completes.
    struct loom_bytecode_selected_table_chunk_t* first;
    // Chunk containing the current top frame, or the empty first chunk.
    struct loom_bytecode_selected_table_chunk_t* current;
    // Number of live frames.
    iree_host_size_t count;
  } worklist;
} loom_bytecode_selected_table_materializer_t;

// Constructs completed scoped types without interning decoding templates.
iree_status_t loom_bytecode_selected_type_project_completed(
    loom_bytecode_selected_table_materializer_t* materializer,
    loom_bytecode_reader_cursor_t* cursor,
    const loom_bytecode_attribute_ssa_materialization_scope_t* scope,
    uint64_t reference, loom_type_id_t* out_type);

// Constructs completed scoped types without interning decoding templates.
iree_status_t loom_bytecode_selected_type_materialize_bindings(
    loom_bytecode_selected_table_materializer_t* materializer,
    loom_bytecode_reader_cursor_t* cursor,
    const loom_bytecode_attribute_ssa_materialization_scope_t* scope,
    uint64_t source_type_id, loom_type_id_t* out_type_id);

// Initializes an empty reached-only table materializer.
//
// Reached source names reuse the current output module's canonical identities,
// including names added by other readers after initialization. A standalone
// reader appends validated unique source names without a lookup or index.
// The initialized materializer remains at its original address until teardown.
void loom_bytecode_selected_table_materializer_initialize(
    loom_bytecode_reader_decoder_t* decoder, iree_const_byte_span_t bytecode,
    loom_context_t* context, const loom_bytecode_module_metadata_t* metadata,
    iree_arena_allocator_t* scratch_arena, loom_module_t* output_module,
    loom_bytecode_selected_symbol_resolver_t symbol_resolver,
    loom_bytecode_selected_table_materializer_t* out_materializer);

// Releases all transient storage owned by |materializer|.
void loom_bytecode_selected_table_materializer_deinitialize(
    loom_bytecode_selected_table_materializer_t* materializer);

// Records a selected source symbol name after its compact target symbol has
// been predeclared. Source symbol-name references use STRINGS ordinals.
iree_status_t loom_bytecode_selected_table_bind_symbol(
    loom_bytecode_selected_table_materializer_t* materializer,
    uint32_t source_name_ordinal, uint16_t target_symbol_id);

// Interns one reached source STRINGS ordinal into the output module.
iree_status_t loom_bytecode_selected_table_intern_string(
    loom_bytecode_selected_table_materializer_t* materializer,
    uint32_t source_string_ordinal, loom_string_id_t* out_target_string_id);

// Looks up a predeclared selected symbol by its source STRINGS ordinal.
bool loom_bytecode_selected_table_lookup_symbol(
    const loom_bytecode_selected_table_materializer_t* materializer,
    uint32_t source_name_ordinal, loom_symbol_ref_t* out_target_symbol_ref);

// Resolves a source symbol by STRINGS ordinal.
//
// Predeclared bindings are returned directly. Otherwise the validated dense
// string-to-symbol projection identifies the exact source symbol passed to the
// optional external resolver. |out_found| is false when no binding or resolver
// exists; resolver and allocation failures propagate as statuses.
iree_status_t loom_bytecode_selected_table_resolve_symbol(
    loom_bytecode_selected_table_materializer_t* materializer,
    uint32_t source_name_ordinal, loom_symbol_ref_t* out_target_symbol_ref,
    bool* out_found);

// Projects one source encoding reference, scheduling it when not yet reached.
// The destination slot remains valid until the enclosing root completes.
iree_status_t loom_bytecode_selected_table_project_encoding(
    loom_bytecode_selected_table_materializer_t* materializer,
    uint16_t source_encoding_id, uint32_t* out_target_encoding_id);

// Projects one source type reference, scheduling it when not yet reached.
// The destination slot remains valid until the enclosing root completes.
iree_status_t loom_bytecode_selected_table_project_type(
    loom_bytecode_selected_table_materializer_t* materializer,
    loom_type_id_t source_type_id, loom_type_id_t* out_target_type_id);

// Projects one source location reference, scheduling it when not yet reached.
// The destination slot remains valid until the enclosing root completes.
iree_status_t loom_bytecode_selected_table_project_location(
    loom_bytecode_selected_table_materializer_t* materializer,
    loom_location_id_t source_location_id,
    loom_location_id_t* out_target_location_id);

// Materializes the closure of one source encoding and returns its compact ID.
iree_status_t loom_bytecode_selected_table_materialize_encoding(
    loom_bytecode_selected_table_materializer_t* materializer,
    uint16_t source_encoding_id, uint16_t* out_target_encoding_id);

// Materializes the closure of one source type and returns its compact ID.
iree_status_t loom_bytecode_selected_table_materialize_type(
    loom_bytecode_selected_table_materializer_t* materializer,
    loom_type_id_t source_type_id, loom_type_id_t* out_target_type_id);

// Materializes the closure of one source location and returns its compact ID.
iree_status_t loom_bytecode_selected_table_materialize_location(
    loom_bytecode_selected_table_materializer_t* materializer,
    loom_location_id_t source_location_id,
    loom_location_id_t* out_target_location_id);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_FORMAT_BYTECODE_READER_SELECTED_TABLES_H_

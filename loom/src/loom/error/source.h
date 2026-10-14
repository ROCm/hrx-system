// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Borrowed source resolution for diagnostics and source-aware compilation.

#ifndef LOOM_ERROR_SOURCE_H_
#define LOOM_ERROR_SOURCE_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/error/diagnostic.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

// Resolves a module-local location to a borrowed source range. Returns false
// when the location cannot be resolved; |out_range| is only valid on success.
// The resolver owner keeps the returned filename and source bytes alive for
// the duration of their use. Consumers retaining source data must copy it.
typedef bool (*loom_source_resolver_fn_t)(void* user_data,
                                          const loom_module_t* module,
                                          loom_location_id_t location,
                                          loom_source_range_t* out_range);

typedef struct loom_source_resolver_t {
  // Optional enrichment callback. Without it, module locations still resolve.
  loom_source_resolver_fn_t fn;
  // Borrowed callback context, live through the final resolution call.
  void* user_data;
} loom_source_resolver_t;

// Resolves |location| using the callback when available, otherwise retaining
// the module's recorded filename and coordinates with UNAVAILABLE_SOURCE
// provenance. Success identifies a location, not the availability of text.
// Tagged locations retain their child's identity; locations without a single
// file origin return false. Callback ranges borrow the resolver owner's
// storage; metadata-only ranges borrow |module|. Retaining consumers must copy
// both.
bool loom_source_resolve(loom_source_resolver_t resolver,
                         const loom_module_t* module,
                         loom_location_id_t location,
                         loom_source_range_t* out_range);

// Resolves a trusted operation diagnostic to borrowed source ranges. Explicit
// module overrides select each operation's source namespace; absent overrides
// use |module|. Unresolved operations contribute no source range. Resolved
// related locations beyond LOOM_DIAGNOSTIC_MAX_RELATED_LOCATIONS are counted as
// omitted. Field-token highlighting and printed-IR fallback belong to callers
// that provide those richer presentations.
//
// |related_locations| supplies LOOM_DIAGNOSTIC_MAX_RELATED_LOCATIONS entries
// and must outlive |out_diagnostic|, along with the emission, modules and
// resolver. This operation allocates nothing and does not invoke a diagnostic
// sink.
void loom_source_resolve_diagnostic(
    const loom_module_t* module, loom_source_resolver_t resolver,
    loom_emitter_t emitter, const loom_diagnostic_emission_t* emission,
    loom_diagnostic_related_location_t* related_locations,
    loom_diagnostic_t* out_diagnostic);

// Source bytes associated with a source identity in the module being resolved.
typedef struct loom_source_entry_t {
  // Module-local source identity, or LOOM_SOURCE_ID_INVALID for an empty entry.
  loom_source_id_t source_id;
  // Borrowed original source bytes.
  iree_string_view_t source;
  // Borrowed source filename.
  iree_string_view_t filename;
} loom_source_entry_t;

// Borrowed source entries for loom_source_table_resolve. Entries are indexed
// by source ID; missing snapshots have an invalid source ID. Linking projects
// IDs into the target module before its locations are resolved against this
// table.
typedef struct loom_source_table_resolver_t {
  // Borrowed module whose source IDs identify these exact snapshots.
  const loom_module_t* module;
  // Source-ID-indexed borrowed entries and their strings, live through the
  // final resolution use.
  const loom_source_entry_t* entries;
  // Number of entries, including any empty entries.
  iree_host_size_t count;
} loom_source_table_resolver_t;

// Borrowed snapshots following module replacements during one compilation.
typedef struct loom_source_table_projection_t {
  // Current module's entries; filenames and text remain borrowed from the
  // input.
  loom_source_table_resolver_t table;
  // Owns projected entry arrays through the final resolution call.
  iree_arena_allocator_t* arena;
} loom_source_table_projection_t;

// Immutable source snapshots owned independently of frontend and input-module
// lifetimes. Keep this object at a stable address while using its resolver.
typedef struct loom_source_storage_t {
  // Source-ID-indexed entries. Missing snapshots have an invalid source ID.
  loom_source_table_resolver_t table;
  // Allocator owning the table and one compact block per snapshot.
  iree_allocator_t allocator;
  // Allocated entries in the table.
  iree_host_size_t capacity;
} loom_source_storage_t;

// Copies one link input's source snapshots into target-owned storage.
typedef struct loom_source_storage_projection_t {
  // Input snapshots borrowed for the duration of the linker callback.
  const loom_source_table_resolver_t* source;
  // Destination storage owned by the linked module.
  loom_source_storage_t* target;
} loom_source_storage_projection_t;

// Initializes empty owned source storage using |allocator|.
void loom_source_storage_initialize(iree_allocator_t allocator,
                                    loom_source_storage_t* out_storage);

// Releases all source snapshots and table storage.
void loom_source_storage_deinitialize(loom_source_storage_t* storage);

// Copies a source snapshot into |storage| at a module-owned source ID.
// Repeated admission of the same identity requires identical name and bytes.
iree_status_t loom_source_storage_insert(loom_source_storage_t* storage,
                                         loom_source_id_t source_id,
                                         iree_string_view_t filename,
                                         iree_string_view_t source);

// Linker source callback for a loom_source_storage_projection_t. Copies
// snapshots through the producer-owned source correspondence. Inputs without
// snapshots allocate nothing; invalid target IDs represent omitted sources.
iree_status_t loom_source_storage_project(
    void* user_data, const loom_module_t* source_module,
    const loom_module_t* target_module, const loom_source_id_t* target_sources);

// Returns a resolver borrowing |storage| and its source snapshots.
loom_source_resolver_t loom_source_storage_resolver(
    const loom_source_storage_t* storage);

// Linker source callback for a loom_source_table_projection_t. Projects entries
// through the producer's source-ID map without copying their filenames or text.
// Inputs without snapshots allocate nothing. The output table is indexed by
// target source ID, with invalid IDs for missing snapshots.
iree_status_t loom_source_table_project(void* user_data,
                                        const loom_module_t* source_module,
                                        const loom_module_t* target_module,
                                        const loom_source_id_t* target_sources);

// Resolves file locations against a loom_source_table_resolver_t passed as
// |user_data|. Other modules, unknown, non-file, and missing sources return
// false, as do unavailable, reversed, or out-of-snapshot coordinates. Success
// denotes an exact original range; coordinates are never clamped into different
// spelling.
bool loom_source_table_resolve(void* user_data, const loom_module_t* module,
                               loom_location_id_t location,
                               loom_source_range_t* out_range);

// Computes the byte offset of a one-based line and Unicode code-point column.
// Line zero resolves to offset zero. A column at or before one resolves to the
// line's start; a column past the line resolves to its end. A line past the
// source resolves to source.size. This matches Loom text tokenizer coordinates.
iree_host_size_t loom_source_byte_offset(iree_string_view_t source,
                                         uint32_t line, uint32_t column);

// Computes a byte offset starting from |range| when the requested coordinate
// follows its start. Coordinates before the range retain the same clamping
// semantics as loom_source_byte_offset.
iree_host_size_t loom_source_range_byte_offset(const loom_source_range_t* range,
                                               uint32_t line, uint32_t column);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_ERROR_SOURCE_H_

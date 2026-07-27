// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Dense per-symbol facts.
//
// Symbol facts are analysis-owned payloads indexed by module symbol ID. The
// symbol table stores identity and def-use state; dialect-defined facts live
// here so whole-program metadata stays queryable without adding ad hoc globals,
// attr-dict walks, or target-specific side tables.

#ifndef LOOM_ANALYSIS_SYMBOL_FACTS_H_
#define LOOM_ANALYSIS_SYMBOL_FACTS_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_symbol_fact_context_t loom_symbol_fact_context_t;
typedef struct loom_symbol_fact_domain_t loom_symbol_fact_domain_t;
typedef struct loom_symbol_fact_resource_list_t
    loom_symbol_fact_resource_list_t;
typedef struct loom_symbol_fact_resource_t loom_symbol_fact_resource_t;
typedef struct loom_symbol_fact_table_t loom_symbol_fact_table_t;
typedef struct loom_symbol_fact_table_options_t
    loom_symbol_fact_table_options_t;
typedef struct loom_symbol_facts_base_t loom_symbol_facts_base_t;

// Generic symbol fact flags shared by all domains.
typedef uint32_t loom_symbol_fact_flags_t;

// Common header at offset zero of every typed symbol-fact payload.
typedef struct loom_symbol_facts_base_t {
  // Fact domain that owns the payload layout after this base header.
  const loom_symbol_fact_domain_t* domain;

  // Symbol kind copied from loom_symbol_t for cheap generic filtering.
  loom_symbol_kind_t symbol_kind;

  // Reserved padding; must be zero.
  uint8_t reserved;

  // Generic flags shared by all fact domains.
  loom_symbol_fact_flags_t flags;
} loom_symbol_facts_base_t;

// Computes facts for one symbol in a domain.
typedef iree_status_t (*loom_symbol_fact_compute_fn_t)(
    const loom_symbol_fact_domain_t* domain,
    loom_symbol_fact_context_t* context, const loom_module_t* module,
    loom_symbol_id_t symbol_id, const loom_symbol_t* symbol,
    const loom_symbol_facts_base_t** out_facts);

// Dialect-owned symbol fact domain descriptor.
typedef struct loom_symbol_fact_domain_t {
  // Computes and arena-allocates the facts for a symbol in this domain.
  loom_symbol_fact_compute_fn_t compute;
} loom_symbol_fact_domain_t;

// Injected resource available to symbol fact domains during computation.
typedef struct loom_symbol_fact_resource_t {
  // Process-local pointer key owned by the resource provider.
  const void* key;

  // Borrowed typed resource payload owned by the provider.
  const void* value;
} loom_symbol_fact_resource_t;

// Borrowed list of injected symbol fact resources.
typedef struct loom_symbol_fact_resource_list_t {
  // Borrowed resource entries.
  const loom_symbol_fact_resource_t* values;

  // Number of resource entries in values.
  iree_host_size_t count;
} loom_symbol_fact_resource_list_t;

static inline loom_symbol_fact_resource_list_t
loom_symbol_fact_resource_list_empty(void) {
  loom_symbol_fact_resource_list_t list = {0};
  return list;
}

static inline loom_symbol_fact_resource_list_t
loom_make_symbol_fact_resource_list(const loom_symbol_fact_resource_t* values,
                                    iree_host_size_t count) {
  loom_symbol_fact_resource_list_t list = {
      /*.values=*/count > 0 ? values : NULL,
      /*.count=*/count,
  };
  return list;
}

static inline bool loom_symbol_fact_resource_list_is_empty(
    loom_symbol_fact_resource_list_t list) {
  return list.count == 0;
}

// Configuration for a symbol fact table.
typedef struct loom_symbol_fact_table_options_t {
  // Borrowed resources visible to fact domains during computation.
  loom_symbol_fact_resource_list_t resources;
} loom_symbol_fact_table_options_t;

// Dense per-module symbol-fact table.
//
// A zero-initialized table is valid. Call initialize to attach an arena before
// lookup. Entries are cached by symbol ID and reset as a group when IR mutation
// invalidates symbol facts.
typedef struct loom_symbol_fact_table_t {
  // Module whose symbols are cached in this table, or NULL before first lookup.
  const loom_module_t* module;

  // Borrowed resources visible to fact domains during computation.
  loom_symbol_fact_resource_list_t resources;

  // Arena used for table storage and domain-owned fact payloads.
  iree_arena_allocator_t* arena;

  // Cached fact pointers indexed by loom_symbol_id_t. NULL means either
  // uncomputed or a symbol with no fact domain; state disambiguates them.
  const loom_symbol_facts_base_t** entries;

  // Per-symbol computation state indexed by loom_symbol_id_t.
  uint8_t* states;

  // Number of live slots in entries/states.
  iree_host_size_t count;

  // Allocated slot capacity in entries/states.
  iree_host_size_t capacity;
} loom_symbol_fact_table_t;

// Initializes |table| to use |arena| for all storage.
void loom_symbol_fact_table_initialize(loom_symbol_fact_table_t* table,
                                       iree_arena_allocator_t* arena);

// Initializes |table| with explicit options.
void loom_symbol_fact_table_initialize_with_options(
    loom_symbol_fact_table_t* table,
    const loom_symbol_fact_table_options_t* options,
    iree_arena_allocator_t* arena);

// Clears cached facts while preserving allocated table capacity.
void loom_symbol_fact_table_reset(loom_symbol_fact_table_t* table);

// Looks up facts for |symbol_id|, computing them through the symbol's generated
// definition descriptor if needed. Symbols without a fact domain return NULL.
iree_status_t loom_symbol_fact_table_lookup(
    loom_symbol_fact_table_t* table, const loom_module_t* module,
    loom_symbol_id_t symbol_id, const loom_symbol_facts_base_t** out_facts);

// Looks up facts for a module-local symbol reference.
iree_status_t loom_symbol_fact_table_lookup_ref(
    loom_symbol_fact_table_t* table, const loom_module_t* module,
    loom_symbol_ref_t symbol_ref, const loom_symbol_facts_base_t** out_facts);

// Allocates domain-owned storage in the fact table arena.
iree_status_t loom_symbol_fact_context_allocate(
    loom_symbol_fact_context_t* context, iree_host_size_t byte_length,
    void** out_ptr);

// Looks up an injected resource by provider-owned pointer key.
iree_status_t loom_symbol_fact_context_lookup_resource(
    loom_symbol_fact_context_t* context, const void* key,
    const void** out_value);

// Recursively looks up another symbol's facts while computing a domain payload.
iree_status_t loom_symbol_fact_context_lookup(
    loom_symbol_fact_context_t* context, loom_symbol_id_t symbol_id,
    const loom_symbol_facts_base_t** out_facts);

// Recursively looks up another symbol reference while computing a domain
// payload.
iree_status_t loom_symbol_fact_context_lookup_ref(
    loom_symbol_fact_context_t* context, loom_symbol_ref_t symbol_ref,
    const loom_symbol_facts_base_t** out_facts);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_ANALYSIS_SYMBOL_FACTS_H_

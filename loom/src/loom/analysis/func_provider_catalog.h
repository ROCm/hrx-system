// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Rebuildable func implementation-provider catalog.
//
// The catalog is a query view over available func.template/func.ukernel
// providers keyed by implementation contract. Local providers are discovered
// from the module symbol table and summarized through func symbol facts.
// Selection plans own per-apply match state; this catalog only owns provider
// availability.

#ifndef LOOM_ANALYSIS_FUNC_PROVIDER_CATALOG_H_
#define LOOM_ANALYSIS_FUNC_PROVIDER_CATALOG_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/analysis/symbol_facts.h"
#include "loom/ir/ir.h"
#include "loom/ops/func_symbol_facts.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef uint8_t loom_func_provider_kind_t;

typedef enum loom_func_provider_kind_e {
  LOOM_FUNC_PROVIDER_KIND_NONE = 0,
  LOOM_FUNC_PROVIDER_KIND_TEMPLATE = 1,
  LOOM_FUNC_PROVIDER_KIND_UKERNEL = 2,
} loom_func_provider_kind_e;

typedef uint8_t loom_func_provider_origin_t;

typedef enum loom_func_provider_origin_e {
  LOOM_FUNC_PROVIDER_ORIGIN_NONE = 0,
  LOOM_FUNC_PROVIDER_ORIGIN_LOCAL = 1,
  LOOM_FUNC_PROVIDER_ORIGIN_EXTERNAL = 2,
} loom_func_provider_origin_e;

typedef struct loom_func_provider_summary_t loom_func_provider_summary_t;
typedef struct loom_func_provider_slice_t loom_func_provider_slice_t;
typedef struct loom_func_provider_catalog_bucket_t
    loom_func_provider_catalog_bucket_t;
typedef struct loom_func_provider_catalog_t loom_func_provider_catalog_t;

// Compact immutable provider summary.
typedef struct loom_func_provider_summary_t {
  // Provider identity stable within origin. Local providers use symbol ID.
  uint64_t provider_id;

  // Storage/materialization origin for this provider.
  loom_func_provider_origin_t origin;

  // Provider implementation kind.
  loom_func_provider_kind_t kind;

  // True when the provider is visible outside the owning module.
  bool is_public;

  // True when the provider has a materialized body.
  bool has_body;

  // Calling convention enum value.
  uint8_t calling_convention;

  // Purity enum value.
  uint8_t purity;

  // Execution temperature enum value.
  uint8_t temperature;

  // Inline policy enum value.
  uint8_t inline_policy;

  // Local module symbol reference, or null for external providers.
  loom_symbol_ref_t symbol;

  // Module-local target applicability symbol, or null for target-independent
  // providers.
  loom_symbol_ref_t target_symbol;

  // Local function-like operation, or empty for external providers.
  loom_func_like_t function;

  // Local function facts backing this summary, or NULL for external providers.
  const loom_func_symbol_facts_t* func_facts;

  // Interned implementation contract key in the local module.
  loom_string_id_t contract_id;

  // Number of argument types.
  uint16_t argument_count;

  // Number of result types.
  uint16_t result_count;

  // Number of predicate entries.
  uint16_t predicate_count;

  // Borrowed implementation contract key text.
  iree_string_view_t contract;

  // Borrowed provider symbol name.
  iree_string_view_t name;

  // Provider priority. Larger values sort before smaller values.
  int64_t priority;

  // Borrowed argument type list in signature order, or NULL when empty.
  const loom_type_t* argument_types;

  // Borrowed result type list in signature order, or NULL when empty.
  const loom_type_t* result_types;

  // Borrowed predicate list for feasibility checks.
  const loom_predicate_t* predicates;
} loom_func_provider_summary_t;

// Borrowed provider result range.
typedef struct loom_func_provider_slice_t {
  // Borrowed provider summaries in deterministic selection order.
  const loom_func_provider_summary_t* providers;

  // Number of provider summaries.
  iree_host_size_t count;
} loom_func_provider_slice_t;

static inline loom_func_provider_slice_t loom_func_provider_slice_empty(void) {
  loom_func_provider_slice_t slice = {0};
  return slice;
}

// Internal contiguous range for one contract key.
typedef struct loom_func_provider_catalog_bucket_t {
  // First provider index for the key, or UINT32_MAX when absent.
  uint32_t first_provider_index;

  // Number of providers for the key.
  uint32_t provider_count;
} loom_func_provider_catalog_bucket_t;

// Rebuildable local provider overlay.
typedef struct loom_func_provider_catalog_t {
  // Module this catalog summarizes, or NULL before build/reset.
  const loom_module_t* module;

  // Arena used for provider and bucket storage.
  iree_arena_allocator_t* arena;

  // Provider summaries sorted by contract key and provider ordering.
  const loom_func_provider_summary_t* providers;

  // Number of provider summaries.
  iree_host_size_t provider_count;

  // Buckets indexed by module string ID.
  const loom_func_provider_catalog_bucket_t* buckets_by_string_id;

  // Number of bucket entries.
  iree_host_size_t bucket_count;
} loom_func_provider_catalog_t;

// Initializes |catalog| to allocate from |arena|.
void loom_func_provider_catalog_initialize(
    loom_func_provider_catalog_t* catalog, iree_arena_allocator_t* arena);

// Invalidates |catalog| while preserving the configured arena.
void loom_func_provider_catalog_reset(loom_func_provider_catalog_t* catalog);

// Rebuilds |catalog| from local func.template/func.ukernel symbols.
iree_status_t loom_func_provider_catalog_build_local(
    loom_func_provider_catalog_t* catalog, const loom_module_t* module,
    loom_symbol_fact_table_t* fact_table);

// Returns local providers for an interned contract key.
loom_func_provider_slice_t loom_func_provider_catalog_lookup(
    const loom_func_provider_catalog_t* catalog, loom_string_id_t contract_id);

// Returns local providers for a contract key string.
loom_func_provider_slice_t loom_func_provider_catalog_lookup_key(
    const loom_func_provider_catalog_t* catalog, iree_string_view_t contract);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_ANALYSIS_FUNC_PROVIDER_CATALOG_H_

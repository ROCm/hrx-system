// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Rebuildable template implementation-provider catalog.
//
// The catalog is a query view over available template.def/template.ukernel
// providers keyed by the module-local template family symbol. Selection plans
// own per-apply match state; this catalog only owns provider availability.
// Provider bodies never contribute applicability facts.

#ifndef LOOM_ANALYSIS_TEMPLATE_PROVIDER_CATALOG_H_
#define LOOM_ANALYSIS_TEMPLATE_PROVIDER_CATALOG_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/analysis/symbol_facts.h"
#include "loom/ir/ir.h"
#include "loom/ops/func_symbol_facts.h"
#include "loom/target/facts.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef uint8_t loom_template_provider_kind_t;

typedef enum loom_template_provider_kind_e {
  LOOM_TEMPLATE_PROVIDER_KIND_NONE = 0,
  LOOM_TEMPLATE_PROVIDER_KIND_DEF = 1,
  LOOM_TEMPLATE_PROVIDER_KIND_UKERNEL = 2,
} loom_template_provider_kind_e;

typedef struct loom_template_provider_summary_t
    loom_template_provider_summary_t;
typedef struct loom_template_provider_contract_t
    loom_template_provider_contract_t;
typedef struct loom_template_provider_slice_t loom_template_provider_slice_t;
typedef struct loom_template_provider_catalog_bucket_t
    loom_template_provider_catalog_bucket_t;
typedef struct loom_template_provider_catalog_t
    loom_template_provider_catalog_t;

// Compact immutable provider summary.
typedef struct loom_template_provider_summary_t {
  // Module whose value/type/string domain owns this summary's IR payloads.
  const loom_module_t* module;

  // Provider implementation kind.
  loom_template_provider_kind_t kind;

  // True when the provider has a materialized body.
  bool has_body;

  // Local module symbol reference, or null for external providers.
  loom_symbol_ref_t symbol;

  // Module-local target applicability symbol, or null for target-independent
  // providers.
  loom_symbol_ref_t target_symbol;

  // Local function-like operation, or empty for external providers.
  loom_func_like_t function;

  // Local function facts backing this summary, or NULL for external providers.
  const loom_func_symbol_facts_t* func_facts;

  // Caller-defined stable source identity, or IREE_HOST_SIZE_MAX when absent.
  iree_host_size_t origin_ordinal;

  // Module-local template family symbol.
  loom_symbol_ref_t family;

  // Number of argument types.
  uint16_t argument_count;

  // Number of result types.
  uint16_t result_count;

  // Number of predicate entries.
  uint16_t predicate_count;

  // Number of resolved target-condition entries.
  uint16_t target_condition_count;

  // Borrowed template family symbol name.
  iree_string_view_t family_name;

  // Borrowed provider symbol name.
  iree_string_view_t name;

  // Provider priority used to rank applicable implementations.
  int64_t priority;

  // Borrowed argument value IDs in the summary module's value domain.
  const loom_value_id_t* argument_ids;

  // Borrowed result value IDs in the summary module's value domain.
  const loom_value_id_t* result_ids;

  // Borrowed predicate list for feasibility checks.
  const loom_predicate_t* predicates;

  // Borrowed resolved target-condition conjunction.
  const loom_target_condition_t* target_conditions;

  // Immutable target identity requirement, or NULL when target-independent.
  const loom_target_facts_t* target_facts;
} loom_template_provider_summary_t;

// Representation-independent applicability contract for an external provider.
//
// Predicate VALUE arguments are signature ordinals: arguments first, followed
// by results. Binding translates them into the linked family's value domain.
typedef struct loom_template_provider_contract_t {
  // Provider implementation kind.
  loom_template_provider_kind_t kind;

  // True when the provider has an implementation body.
  bool has_body;

  // Number of signature arguments.
  uint16_t argument_count;

  // Number of signature results.
  uint16_t result_count;

  // Number of signature-local predicates.
  uint16_t predicate_count;

  // Number of resolved target conditions.
  uint16_t target_condition_count;

  // Borrowed provider symbol name.
  iree_string_view_t name;

  // Provider priority used to rank applicable implementations.
  int64_t priority;

  // Borrowed predicates whose VALUE arguments are signature ordinals.
  const loom_predicate_t* predicates;

  // Borrowed resolved target-condition conjunction.
  const loom_target_condition_t* target_conditions;

  // Immutable target identity requirement, or NULL when target-independent.
  const loom_target_facts_t* target_facts;
} loom_template_provider_contract_t;

// Borrowed provider result range.
typedef struct loom_template_provider_slice_t {
  // Borrowed provider summaries in stable source order.
  const loom_template_provider_summary_t* providers;

  // Number of provider summaries.
  iree_host_size_t count;
} loom_template_provider_slice_t;

static inline loom_template_provider_slice_t loom_template_provider_slice_empty(
    void) {
  loom_template_provider_slice_t slice = {0};
  return slice;
}

// Internal contiguous range for one template family symbol.
typedef struct loom_template_provider_catalog_bucket_t {
  // First provider index for the key, or UINT32_MAX when absent.
  uint32_t first_provider_index;

  // Number of providers for the key.
  uint32_t provider_count;
} loom_template_provider_catalog_bucket_t;

// Rebuildable local provider overlay.
typedef struct loom_template_provider_catalog_t {
  // Module this catalog summarizes, or NULL before build/reset.
  const loom_module_t* module;

  // Arena used for provider and bucket storage.
  iree_arena_allocator_t* arena;

  // Provider summaries grouped by family in stable source order.
  const loom_template_provider_summary_t* providers;

  // Number of provider summaries.
  iree_host_size_t provider_count;

  // Buckets indexed by module symbol ID.
  const loom_template_provider_catalog_bucket_t* buckets_by_symbol_id;

  // Number of bucket entries.
  iree_host_size_t bucket_count;
} loom_template_provider_catalog_t;

// Initializes |catalog| to allocate from |arena|.
void loom_template_provider_catalog_initialize(
    loom_template_provider_catalog_t* catalog, iree_arena_allocator_t* arena);

// Invalidates |catalog| while preserving the configured arena.
void loom_template_provider_catalog_reset(
    loom_template_provider_catalog_t* catalog);

// Rebuilds |catalog| from local template.def/template.ukernel symbols.
iree_status_t loom_template_provider_catalog_build_local(
    loom_template_provider_catalog_t* catalog, const loom_module_t* module,
    loom_symbol_fact_table_t* fact_table);

// Rebuilds |catalog| from local providers plus external summaries.
//
// Every external provider must already be bound to |module|'s family and
// signature domain. Borrowed source-owned names, conditions, and target facts
// must remain live for the catalog lifetime.
iree_status_t loom_template_provider_catalog_build(
    loom_template_provider_catalog_t* catalog, const loom_module_t* module,
    loom_symbol_fact_table_t* fact_table,
    const loom_template_provider_summary_t* external_providers,
    iree_host_size_t external_provider_count);

// Binds one provider contract to an already-linked template family.
//
// The family owns the canonical signature used for applicability. Provider
// predicates are translated from |source|'s signature values to the matching
// family values; no values, types, symbols, or operations are added to
// |target_module|. |target_symbol| is the already-linked target witness when
// available and may be null when |source| carries immutable target facts.
// Source-owned names, conditions, and target facts must remain live while the
// bound summary is queried.
iree_status_t loom_template_provider_summary_bind_family(
    const loom_template_provider_summary_t* source,
    const loom_module_t* target_module, loom_symbol_ref_t target_family,
    loom_symbol_ref_t target_symbol, iree_host_size_t origin_ordinal,
    iree_arena_allocator_t* arena,
    loom_template_provider_summary_t* out_provider);

// Binds a representation-independent provider contract to a linked family.
//
// |target_symbol| is the already-linked target witness when available and may
// be null when |contract| carries immutable target facts. Contract-owned names,
// conditions, and target facts must remain live while the result is queried.
iree_status_t loom_template_provider_contract_bind_family(
    const loom_template_provider_contract_t* contract,
    const loom_module_t* target_module, loom_symbol_ref_t target_family,
    loom_symbol_ref_t target_symbol, iree_host_size_t origin_ordinal,
    iree_arena_allocator_t* arena,
    loom_template_provider_summary_t* out_provider);

// Returns local providers for |family|.
loom_template_provider_slice_t loom_template_provider_catalog_lookup(
    const loom_template_provider_catalog_t* catalog, loom_symbol_ref_t family);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_ANALYSIS_TEMPLATE_PROVIDER_CATALOG_H_

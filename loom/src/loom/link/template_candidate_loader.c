// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/link/template_candidate_loader.h"

#include <string.h>

#include "loom/analysis/symbol_facts.h"
#include "loom/ir/module.h"
#include "loom/link/bytecode_template_contract.h"
#include "loom/ops/func_symbol_facts.h"

typedef struct loom_link_template_candidate_module_t {
  // Indexed source module described by this cache.
  const loom_link_module_index_module_t* source_module;

  // Representation-specific candidate source state.
  union {
    // State derived from an already-materialized source module.
    struct {
      // Cached source-module symbol facts.
      loom_symbol_fact_table_t fact_table;
      // Source-module provider summaries.
      loom_template_provider_catalog_t catalog;
      // Provider summaries indexed by module-local source symbol ordinal.
      struct {
        // Arena-owned pointers into catalog storage.
        const loom_template_provider_summary_t** values;
        // Number of source symbol slots.
        iree_host_size_t count;
      } summaries;
    } materialized;
    // State derived directly from validated bytecode metadata.
    struct {
      // Lazy body-blind applicability contract reader.
      loom_link_bytecode_template_contract_reader_t contracts;
    } bytecode;
  } source;
  // True after this module cache has been initialized.
  bool initialized;
} loom_link_template_candidate_module_t;

struct loom_link_template_candidate_loader_t {
  // Borrowed provider-backed source index.
  const loom_link_module_index_t* index;

  // Finalized context used to decode bytecode provider contracts.
  loom_context_t* context;

  // Host allocator used for loader storage.
  iree_allocator_t allocator;

  // Persistent source-summary and fact storage.
  iree_arena_allocator_t arena;

  // One lazy cache per indexed module.
  struct {
    // Arena-owned module cache records allocated on first template demand.
    loom_link_template_candidate_module_t* values;
    // Number of indexed modules.
    iree_host_size_t count;
  } modules;
};

static iree_status_t loom_link_template_candidate_initialize_bytecode_module(
    loom_link_template_candidate_loader_t* loader,
    loom_link_template_candidate_module_t* cache) {
  const loom_link_module_index_module_t* source_module = cache->source_module;
  const loom_link_module_index_provider_t* provider =
      loom_link_module_index_provider_at(loader->index,
                                         source_module->provider_ordinal);
  IREE_ASSERT(provider->kind == LOOM_LINK_PROVIDER_BYTECODE);
  IREE_ASSERT(source_module->provider_module_ordinal <
              provider->bytecode.metadata.module_count);
  const loom_bytecode_module_metadata_t* metadata =
      &provider->bytecode.metadata
           .modules[source_module->provider_module_ordinal];
  return loom_link_bytecode_template_contract_reader_initialize(
      provider->bytecode.contents, loader->context, metadata, &loader->arena,
      &cache->source.bytecode.contracts);
}

static iree_status_t
loom_link_template_candidate_initialize_materialized_module(
    loom_link_template_candidate_loader_t* loader,
    loom_link_template_candidate_module_t* cache) {
  const loom_module_t* module = cache->source_module->materialized_module;
  IREE_ASSERT(module != NULL);
  loom_symbol_fact_table_initialize(&cache->source.materialized.fact_table,
                                    &loader->arena);
  loom_template_provider_catalog_initialize(&cache->source.materialized.catalog,
                                            &loader->arena);
  IREE_RETURN_IF_ERROR(loom_template_provider_catalog_build_local(
      &cache->source.materialized.catalog, module,
      &cache->source.materialized.fact_table));
  for (iree_host_size_t i = 0;
       i < cache->source.materialized.catalog.provider_count; ++i) {
    const loom_template_provider_summary_t* provider =
        &cache->source.materialized.catalog.providers[i];
    IREE_ASSERT(loom_symbol_ref_is_valid(provider->symbol));
    IREE_ASSERT(provider->symbol.module_id == 0);
    IREE_ASSERT(provider->symbol.symbol_id <
                cache->source.materialized.summaries.count);
    cache->source.materialized.summaries.values[provider->symbol.symbol_id] =
        provider;
  }
  return iree_ok_status();
}

static iree_status_t loom_link_template_candidate_loader_ensure_modules(
    loom_link_template_candidate_loader_t* loader) {
  if (loader->modules.values != NULL || loader->modules.count == 0) {
    return iree_ok_status();
  }
  return iree_allocator_malloc_array(
      iree_arena_allocator(&loader->arena), loader->modules.count,
      sizeof(*loader->modules.values), (void**)&loader->modules.values);
}

static iree_status_t loom_link_template_candidate_initialize_module(
    loom_link_template_candidate_loader_t* loader,
    const loom_link_module_index_module_t* source_module,
    loom_link_template_candidate_module_t** out_cache) {
  IREE_ASSERT(source_module->ordinal < loader->modules.count);
  loom_link_template_candidate_module_t* cache =
      &loader->modules.values[source_module->ordinal];
  if (cache->initialized) {
    *out_cache = cache;
    return iree_ok_status();
  }

  cache->source_module = source_module;
  if (source_module->materialized_module != NULL) {
    cache->source.materialized.summaries.count = source_module->symbol_count;
    if (source_module->symbol_count != 0) {
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          &loader->arena, source_module->symbol_count,
          sizeof(*cache->source.materialized.summaries.values),
          (void**)&cache->source.materialized.summaries.values));
      memset(cache->source.materialized.summaries.values, 0,
             source_module->symbol_count *
                 sizeof(*cache->source.materialized.summaries.values));
    }
    IREE_RETURN_IF_ERROR(
        loom_link_template_candidate_initialize_materialized_module(loader,
                                                                    cache));
  } else {
    IREE_RETURN_IF_ERROR(
        loom_link_template_candidate_initialize_bytecode_module(loader, cache));
  }
  cache->initialized = true;
  *out_cache = cache;
  return iree_ok_status();
}

static const loom_template_provider_summary_t*
loom_link_template_candidate_materialized_summary(
    const loom_link_template_candidate_module_t* cache,
    const loom_link_module_index_symbol_t* source_symbol) {
  IREE_ASSERT(cache->source_module->materialized_module != NULL);
  IREE_ASSERT(source_symbol->module_symbol_ordinal <
              cache->source.materialized.summaries.count);
  const loom_template_provider_summary_t* summary =
      cache->source.materialized.summaries
          .values[source_symbol->module_symbol_ordinal];
  IREE_ASSERT(summary != NULL);
  return summary;
}

static bool loom_link_template_candidate_lookup_linked_global(
    const loom_link_template_candidate_loader_t* loader,
    const loom_link_plan_materialization_t* materialization,
    const loom_link_module_index_symbol_t* source_symbol,
    loom_symbol_ref_t* out_target_ref) {
  const loom_link_module_index_symbol_t* candidate =
      loom_link_module_index_lookup_global(loader->index, source_symbol->name);
  while (candidate != NULL) {
    const loom_symbol_ref_t target_ref =
        materialization->target_symbols.values[candidate->ordinal];
    if (loom_symbol_ref_is_valid(target_ref)) {
      *out_target_ref = target_ref;
      return true;
    }
    candidate =
        loom_link_module_index_next_global_duplicate(loader->index, candidate);
  }
  return false;
}

static loom_symbol_ref_t loom_link_template_candidate_lookup_linked_symbol(
    const loom_link_template_candidate_loader_t* loader,
    const loom_link_module_index_module_t* source_module,
    const loom_link_plan_materialization_t* materialization,
    uint32_t source_symbol_ordinal) {
  loom_symbol_ref_t target_ref = loom_symbol_ref_null();
  if (source_symbol_ordinal == UINT32_MAX) return target_ref;
  IREE_ASSERT(source_symbol_ordinal < source_module->symbol_count);
  const iree_host_size_t source_ordinal =
      source_module->symbol_start_ordinal + source_symbol_ordinal;
  IREE_ASSERT(source_ordinal < materialization->target_symbols.count);
  target_ref = materialization->target_symbols.values[source_ordinal];
  if (loom_symbol_ref_is_valid(target_ref)) {
    return target_ref;
  }

  const loom_link_module_index_symbol_t* source_symbol =
      loom_link_module_index_symbol_at(loader->index, source_ordinal);
  IREE_ASSERT(source_symbol != NULL);
  if (source_symbol->identity == LOOM_LINK_SYMBOL_IDENTITY_GLOBAL) {
    (void)loom_link_template_candidate_lookup_linked_global(
        loader, materialization, source_symbol, &target_ref);
  }
  return target_ref;
}

static iree_status_t loom_link_template_candidate_capacity(
    const loom_link_template_candidate_loader_t* loader,
    const loom_link_plan_t* plan, iree_host_size_t* out_capacity) {
  *out_capacity = 0;
  const iree_host_size_t family_count =
      loom_link_plan_demanded_template_family_count(plan);
  for (iree_host_size_t i = 0; i < family_count; ++i) {
    const loom_link_template_family_ordinal_t family_ordinal =
        loom_link_plan_demanded_template_family_at(plan, i);
    const loom_link_module_index_template_family_t* family =
        loom_link_module_index_template_family_at(loader->index,
                                                  family_ordinal);
    IREE_ASSERT(family != NULL);
    iree_host_size_t capacity = 0;
    if (!iree_host_size_checked_add(*out_capacity, family->providers.count,
                                    &capacity)) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "template candidate capacity overflow");
    }
    *out_capacity = capacity;
  }
  return iree_ok_status();
}

iree_status_t loom_link_template_candidate_loader_allocate(
    const loom_link_module_index_t* index,
    const loom_link_plan_materialization_environment_t* environment,
    loom_link_template_candidate_loader_t** out_loader) {
  IREE_ASSERT_ARGUMENT(index);
  IREE_ASSERT_ARGUMENT(environment);
  IREE_ASSERT_ARGUMENT(environment->context);
  IREE_ASSERT_ARGUMENT(environment->block_pool);
  IREE_ASSERT_ARGUMENT(out_loader);
  *out_loader = NULL;

  loom_link_template_candidate_loader_t* loader = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(environment->allocator,
                                             sizeof(*loader), (void**)&loader));
  loader->index = index;
  loader->context = environment->context;
  loader->allocator = environment->allocator;
  iree_arena_initialize(environment->block_pool, &loader->arena);
  loader->modules.count = loom_link_module_index_module_count(index);
  *out_loader = loader;
  return iree_ok_status();
}

void loom_link_template_candidate_loader_free(
    loom_link_template_candidate_loader_t* loader) {
  if (loader == NULL) {
    return;
  }
  const iree_allocator_t allocator = loader->allocator;
  iree_arena_deinitialize(&loader->arena);
  iree_allocator_free(allocator, loader);
}

iree_status_t loom_link_template_candidate_loader_load(
    loom_link_template_candidate_loader_t* loader, const loom_link_plan_t* plan,
    loom_link_plan_materialization_t* materialization,
    iree_arena_allocator_t* arena,
    loom_template_provider_slice_t* out_candidates) {
  IREE_ASSERT_ARGUMENT(loader);
  IREE_ASSERT_ARGUMENT(plan);
  IREE_ASSERT_ARGUMENT(materialization);
  IREE_ASSERT_ARGUMENT(materialization->module);
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_candidates);
  *out_candidates = loom_template_provider_slice_empty();
  iree_host_size_t candidate_capacity = 0;
  IREE_RETURN_IF_ERROR(
      loom_link_template_candidate_capacity(loader, plan, &candidate_capacity));
  if (candidate_capacity == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_link_template_candidate_loader_ensure_modules(loader));
  loom_template_provider_summary_t* candidates = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, candidate_capacity, sizeof(*candidates), (void**)&candidates));

  iree_host_size_t candidate_count = 0;
  const iree_host_size_t family_count =
      loom_link_plan_demanded_template_family_count(plan);
  for (iree_host_size_t i = 0; i < family_count; ++i) {
    const loom_link_template_family_ordinal_t family_ordinal =
        loom_link_plan_demanded_template_family_at(plan, i);
    const loom_link_module_index_template_family_t* family =
        loom_link_module_index_template_family_at(loader->index,
                                                  family_ordinal);
    IREE_ASSERT(family != NULL);
    IREE_ASSERT(family_ordinal <
                materialization->target_template_families.count);
    const loom_symbol_ref_t target_family =
        materialization->target_template_families.values[family_ordinal];
    IREE_ASSERT(loom_symbol_ref_is_valid(target_family));

    iree_host_size_t provider_ordinal = family->providers.first_symbol_ordinal;
    while (provider_ordinal != LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL) {
      const loom_link_module_index_symbol_t* provider =
          loom_link_module_index_symbol_at(loader->index, provider_ordinal);
      IREE_ASSERT(provider != NULL);
      if (!loom_link_plan_contains_symbol(plan, provider_ordinal)) {
        const loom_link_module_index_module_t* source_module =
            loom_link_module_index_symbol_module(loader->index, provider);
        IREE_ASSERT(source_module != NULL);
        loom_link_template_candidate_module_t* cache = NULL;
        IREE_RETURN_IF_ERROR(loom_link_template_candidate_initialize_module(
            loader, source_module, &cache));
        if (source_module->materialized_module != NULL) {
          const loom_template_provider_summary_t* source_summary =
              loom_link_template_candidate_materialized_summary(cache,
                                                                provider);
          const loom_symbol_ref_t target_symbol =
              loom_link_template_candidate_lookup_linked_symbol(
                  loader, source_module, materialization,
                  loom_symbol_ref_is_valid(source_summary->target_symbol)
                      ? source_summary->target_symbol.symbol_id
                      : UINT32_MAX);
          IREE_RETURN_IF_ERROR(loom_template_provider_summary_bind_family(
              source_summary, materialization->module, target_family,
              target_symbol, provider_ordinal, arena,
              &candidates[candidate_count++]));
        } else {
          const loom_link_bytecode_template_contract_t* contract = NULL;
          IREE_RETURN_IF_ERROR(loom_link_bytecode_template_contract_reader_load(
              &cache->source.bytecode.contracts,
              (uint32_t)provider->module_symbol_ordinal, &contract));
          const loom_symbol_ref_t target_symbol =
              loom_link_template_candidate_lookup_linked_symbol(
                  loader, source_module, materialization,
                  contract->target_symbol_ordinal);
          IREE_RETURN_IF_ERROR(loom_template_provider_contract_bind_family(
              &contract->provider, materialization->module, target_family,
              target_symbol, provider_ordinal, arena,
              &candidates[candidate_count++]));
        }
      }
      provider_ordinal = provider->next.template_provider_ordinal;
    }
  }
  IREE_ASSERT(candidate_count <= candidate_capacity);
  *out_candidates = (loom_template_provider_slice_t){
      .providers = candidates,
      .count = candidate_count,
  };
  return iree_ok_status();
}

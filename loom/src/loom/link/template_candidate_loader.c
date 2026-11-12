// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/link/template_candidate_loader.h"

#include <string.h>

#include "loom/analysis/symbol_facts.h"
#include "loom/ir/intern_table.h"
#include "loom/ir/module.h"
#include "loom/ir/structural_hash.h"
#include "loom/link/bytecode_template_contract.h"
#include "loom/link/template_input_binding.h"
#include "loom/ops/func_symbol_facts.h"
#include "loom/ops/op_defs.h"
#include "loom/util/segmented_storage.h"

typedef struct loom_link_template_candidate_module_t {
  // Indexed source module described by this cache.
  const loom_link_module_index_module_t* source_module;

  // Immutable index providing native source input requirements.
  const loom_link_module_index_t* index;

  // Representation-specific candidate source state.
  union {
    // State derived from an already-materialized source module.
    struct {
      // Borrowed source input requirements injected into symbol facts.
      loom_func_input_contract_source_t inputs;
      // Stable resource entry retained for the fact table lifetime.
      loom_symbol_fact_resource_t input_resource;
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

typedef struct loom_link_template_candidate_family_t {
  // Immutable index-wide family identity.
  loom_link_template_family_ordinal_t ordinal;
  // Fixed-size projection chunks in the index's stable provider-chain order.
  loom_link_bytecode_template_provider_projection_t** projections;
} loom_link_template_candidate_family_t;

typedef struct loom_link_template_candidate_families_t {
  // Lookup over demanded families, independent of the library symbol span.
  loom_intern_table_t index;
  // Stable family records, allocated in fixed-size segments.
  loom_segmented_storage_t entries;
} loom_link_template_candidate_families_t;

struct loom_link_template_candidate_loader_t {
  // Borrowed provider-backed source index.
  const loom_link_module_index_t* index;

  // Finalized context used to decode bytecode provider contracts.
  loom_context_t* context;

  // Host allocator used for loader storage.
  iree_allocator_t allocator;

  // Persistent source-summary and fact storage.
  iree_arena_allocator_t arena;

  // Lazily allocated records for demanded template families.
  loom_link_template_candidate_families_t* families;

  // One lazy cache per indexed module.
  struct {
    // Arena-owned module cache records allocated on first template demand.
    loom_link_template_candidate_module_t* values;
    // Number of indexed modules.
    iree_host_size_t count;
  } modules;
};

#define LOOM_LINK_TEMPLATE_CANDIDATE_FAMILY_SEGMENT_CAPACITY 128u
#define LOOM_LINK_TEMPLATE_CANDIDATE_PROJECTION_SEGMENT_CAPACITY 128u

typedef struct loom_link_template_candidate_family_query_t {
  // Stable records owning each occupied hash bucket.
  const loom_link_template_candidate_families_t* families;
  // Index-wide identity being requested.
  loom_link_template_family_ordinal_t ordinal;
} loom_link_template_candidate_family_query_t;

static const loom_link_template_candidate_family_t*
loom_link_template_candidate_family_at(
    const loom_link_template_candidate_families_t* families, uint32_t index) {
  const loom_link_template_candidate_family_t* segment =
      loom_segmented_storage_const_segment(
          &families->entries,
          index / LOOM_LINK_TEMPLATE_CANDIDATE_FAMILY_SEGMENT_CAPACITY);
  return &segment[index % LOOM_LINK_TEMPLATE_CANDIDATE_FAMILY_SEGMENT_CAPACITY];
}

static bool loom_link_template_candidate_family_equal(const void* opaque,
                                                      uint32_t index) {
  const loom_link_template_candidate_family_query_t* query = opaque;
  return loom_link_template_candidate_family_at(query->families, index)
             ->ordinal == query->ordinal;
}

// Keep first-use family construction outside the per-provider loop, including
// its allocation scratch, so it does not enlarge the live-module binding path.
static IREE_ATTRIBUTE_NOINLINE iree_status_t
loom_link_template_candidate_family_projections(
    loom_link_template_candidate_loader_t* loader,
    const loom_link_module_index_template_family_t* family,
    loom_link_bytecode_template_provider_projection_t*** out_projections) {
  if (!loader->families) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate(
        &loader->arena, sizeof(*loader->families), (void**)&loader->families));
    IREE_RETURN_IF_ERROR(loom_intern_table_initialize(
        &loader->arena, 0, &loader->families->index));
    loom_segmented_storage_initialize(
        LOOM_LINK_TEMPLATE_CANDIDATE_FAMILY_SEGMENT_CAPACITY *
            sizeof(loom_link_template_candidate_family_t),
        iree_alignof(loom_link_template_candidate_family_t),
        &loader->families->entries);
  }
  loom_link_template_candidate_families_t* families = loader->families;
  const loom_link_template_candidate_family_query_t query = {
      .families = families,
      .ordinal = family->ordinal,
  };
  const uint32_t hash =
      loom_structural_hash_finalize(loom_structural_hash_mix_u32(
          loom_structural_hash_initialize(), family->ordinal));
  loom_intern_probe_t probe = loom_intern_table_probe(
      &families->index, hash, loom_link_template_candidate_family_equal,
      &query);
  if (probe.index != UINT32_MAX) {
    *out_projections =
        loom_link_template_candidate_family_at(families, probe.index)
            ->projections;
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_intern_table_reserve_insert(
      &loader->arena, &families->index, hash, 1, &probe.slot));
  const uint32_t index = (uint32_t)families->index.count;
  loom_link_template_candidate_family_t* segment = NULL;
  if (index / LOOM_LINK_TEMPLATE_CANDIDATE_FAMILY_SEGMENT_CAPACITY ==
      families->entries.segment_count) {
    IREE_RETURN_IF_ERROR(loom_segmented_storage_append(
        &families->entries, &loader->arena, (void**)&segment));
  } else {
    segment = loom_segmented_storage_segment(
        &families->entries,
        index / LOOM_LINK_TEMPLATE_CANDIDATE_FAMILY_SEGMENT_CAPACITY);
  }
  loom_link_template_candidate_family_t* entry =
      &segment[index % LOOM_LINK_TEMPLATE_CANDIDATE_FAMILY_SEGMENT_CAPACITY];
  const iree_host_size_t chunk_count = iree_host_size_ceil_div(
      family->providers.count,
      LOOM_LINK_TEMPLATE_CANDIDATE_PROJECTION_SEGMENT_CAPACITY);
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(&loader->arena, chunk_count,
                                                 sizeof(*entry->projections),
                                                 (void**)&entry->projections));
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < chunk_count && iree_status_is_ok(status);
       ++i) {
    const iree_host_size_t projection_count = iree_min(
        family->providers.count -
            i * LOOM_LINK_TEMPLATE_CANDIDATE_PROJECTION_SEGMENT_CAPACITY,
        LOOM_LINK_TEMPLATE_CANDIDATE_PROJECTION_SEGMENT_CAPACITY);
    status = iree_allocator_malloc_array(
        iree_arena_allocator(&loader->arena), projection_count,
        sizeof(**entry->projections), (void**)&entry->projections[i]);
  }
  IREE_RETURN_IF_ERROR(status);
  entry->ordinal = family->ordinal;
  loom_intern_table_insert(&families->index, probe.slot, hash, index);
  *out_projections = entry->projections;
  return iree_ok_status();
}

static void loom_link_template_candidate_initialize_bytecode_module(
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
  loom_link_bytecode_template_contract_reader_initialize(
      provider->bytecode.contents, loader->context, metadata, &loader->arena,
      &cache->source.bytecode.contracts);
}

static const loom_input_contract_t* loom_link_template_candidate_source_inputs(
    const void* self, loom_symbol_id_t symbol_id) {
  const loom_link_template_candidate_module_t* cache = self;
  const loom_link_module_index_symbol_t* symbol =
      loom_link_module_index_symbol_at(
          cache->index, cache->source_module->symbol_start_ordinal + symbol_id);
  return symbol->input_contract_ordinal_plus_one
             ? cache->source_module
                   ->input_contracts[symbol->input_contract_ordinal_plus_one -
                                     1]
             : NULL;
}

static iree_status_t
loom_link_template_candidate_initialize_materialized_module(
    loom_link_template_candidate_loader_t* loader,
    loom_link_template_candidate_module_t* cache) {
  const loom_module_t* module = cache->source_module->materialized_module;
  IREE_ASSERT(module != NULL);
  cache->index = loader->index;
  cache->source.materialized.inputs = (loom_func_input_contract_source_t){
      .self = cache,
      .lookup = loom_link_template_candidate_source_inputs,
  };
  cache->source.materialized.input_resource = (loom_symbol_fact_resource_t){
      .key = &loom_func_input_contract_resource_key,
      .value = &cache->source.materialized.inputs,
  };
  const loom_symbol_fact_table_options_t options = {
      .resources = {.values = &cache->source.materialized.input_resource,
                    .count = 1},
  };
  loom_symbol_fact_table_initialize_with_options(
      &cache->source.materialized.fact_table, &options, &loader->arena);
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
    loom_link_template_candidate_initialize_bytecode_module(loader, cache);
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

static iree_status_t loom_link_template_candidate_loader_load_impl(
    loom_link_template_candidate_loader_t* loader, const loom_link_plan_t* plan,
    loom_link_plan_materialization_t* materialization,
    iree_arena_allocator_t* arena,
    loom_link_template_input_binding_t* input_binding,
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
    const loom_func_like_t target_function = loom_func_like_cast(
        materialization->module,
        materialization->module->symbols.entries[target_family.symbol_id]
            .defining_op);
    uint16_t argument_count = 0;
    loom_func_like_arg_ids(target_function, &argument_count);
    const uint16_t result_count = target_function.op->result_count;

    loom_link_bytecode_template_provider_projection_t** projections = NULL;
    iree_host_size_t provider_index = 0;
    const loom_link_module_index_symbol_t* provider =
        loom_link_module_index_symbol_at(
            loader->index, family->providers.first_symbol_ordinal);
    while (provider != NULL) {
      if (!loom_link_plan_contains_symbol(plan, provider->ordinal)) {
        loom_template_provider_summary_t* candidate =
            &candidates[candidate_count];
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
              loom_link_plan_materialization_lookup_symbol(
                  loader->index, source_module, materialization,
                  loom_symbol_ref_is_valid(source_summary->target_symbol)
                      ? source_summary->target_symbol.symbol_id
                      : UINT32_MAX);
          loom_template_provider_summary_bind_family(
              source_summary, materialization->module, target_family,
              target_symbol, provider->ordinal, candidate);
        } else {
          if (!projections && family->requires_bytecode_projection) {
            IREE_RETURN_IF_ERROR(
                loom_link_template_candidate_family_projections(loader, family,
                                                                &projections));
          }
          loom_link_bytecode_template_contract_t contract;
          IREE_RETURN_IF_ERROR(loom_link_bytecode_template_contract_reader_load(
              &cache->source.bytecode.contracts,
              (uint32_t)provider->module_symbol_ordinal,
              projections
                  ? &projections
                        [provider_index /
                         LOOM_LINK_TEMPLATE_CANDIDATE_PROJECTION_SEGMENT_CAPACITY]
                        [provider_index %
                         LOOM_LINK_TEMPLATE_CANDIDATE_PROJECTION_SEGMENT_CAPACITY]
                  : NULL,
              &contract));
          const loom_symbol_ref_t target_symbol =
              loom_link_plan_materialization_lookup_symbol(
                  loader->index, source_module, materialization,
                  contract.target_symbol_ordinal);
          loom_template_provider_contract_bind_family(
              &contract.provider, materialization->module, target_family,
              target_symbol, provider->ordinal, candidate);
        }
        IREE_RETURN_IF_ERROR(loom_link_template_input_bind(
            input_binding, provider, &candidate->input_contract));
        if (candidate->argument_count != argument_count ||
            candidate->result_count != result_count) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "provider signature does not match its bound template family");
        }
        ++candidate_count;
      }
      ++provider_index;
      provider = loom_link_module_index_next_template_provider(loader->index,
                                                               provider);
    }
  }
  IREE_ASSERT(candidate_count <= candidate_capacity);
  *out_candidates = (loom_template_provider_slice_t){
      .providers = candidates,
      .count = candidate_count,
  };
  return iree_ok_status();
}

iree_status_t loom_link_template_candidate_loader_load(
    loom_link_template_candidate_loader_t* loader, const loom_link_plan_t* plan,
    loom_link_plan_materialization_t* materialization,
    iree_arena_allocator_t* arena,
    loom_template_provider_slice_t* out_candidates) {
  loom_link_template_input_binding_t inputs = {
      .index = loader->index,
      .materialization = materialization,
      .arena = arena,
  };
  const iree_status_t status = loom_link_template_candidate_loader_load_impl(
      loader, plan, materialization, arena, &inputs, out_candidates);
  loom_link_template_input_binding_deinitialize(&inputs);
  return status;
}

// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/link/template_candidate_loader.h"

#include <string.h>

#include "loom/analysis/symbol_facts.h"
#include "loom/format/bytecode/symbol_header_reader.h"
#include "loom/ir/module.h"
#include "loom/ops/func_symbol_facts.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/target/facts.h"
#include "loom/target/condition.h"

typedef struct loom_link_template_candidate_module_t {
  // Indexed source module described by this cache.
  const loom_link_module_index_module_t* source_module;
  // Lazy bytecode header reader, or NULL for materialized modules.
  loom_bytecode_symbol_header_reader_t* header_reader;
  // Cached source-module symbol facts.
  loom_symbol_fact_table_t fact_table;
  // Materialized-module provider summaries, empty for bytecode modules.
  loom_template_provider_catalog_t catalog;
  // Provider summaries indexed by module-local source symbol ordinal.
  struct {
    // Arena-owned pointers into catalog or lazy bytecode summaries.
    const loom_template_provider_summary_t** values;
    // Number of source symbol slots.
    iree_host_size_t count;
  } summaries;
  // True after this module cache has been initialized.
  bool initialized;
} loom_link_template_candidate_module_t;

struct loom_link_template_candidate_loader_t {
  // Borrowed provider-backed source index.
  const loom_link_module_index_t* index;
  // Borrowed materialization environment shared with the specializing link.
  const loom_link_plan_materialization_environment_t* environment;
  // Persistent source-summary and fact storage.
  iree_arena_allocator_t arena;
  // One lazy cache per indexed module.
  struct {
    // Arena-owned module cache records.
    loom_link_template_candidate_module_t* values;
    // Number of indexed modules.
    iree_host_size_t count;
  } modules;
};

static bool loom_link_template_provider_membership_contains(
    loom_link_template_provider_membership_t membership,
    iree_host_size_t symbol_ordinal) {
  IREE_ASSERT(symbol_ordinal < membership.symbol_count);
  return membership.words != NULL &&
         (membership.words[symbol_ordinal >> 6] &
          (UINT64_C(1) << (symbol_ordinal & 63u))) != 0;
}

static loom_diagnostic_sink_t loom_link_template_candidate_diagnostic_sink(
    const loom_link_template_candidate_loader_t* loader,
    const loom_link_module_index_provider_t* provider) {
  if (loader->environment->diagnostic_sink == NULL) {
    return (loom_diagnostic_sink_t){0};
  }
  return loader->environment->diagnostic_sink(loader->environment->user_data,
                                              provider);
}

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
  const loom_bytecode_symbol_header_reader_options_t options = {
      .diagnostic_sink =
          loom_link_template_candidate_diagnostic_sink(loader, provider),
      .low_repr_environment = loader->environment->low_repr_environment,
  };
  IREE_RETURN_IF_ERROR(loom_bytecode_symbol_header_reader_allocate(
      provider->bytecode.contents, provider->bytecode.filename,
      loader->environment->context, loader->environment->block_pool, metadata,
      &options, loader->environment->allocator, &cache->header_reader));
  loom_symbol_fact_table_initialize(&cache->fact_table, &loader->arena);
  return iree_ok_status();
}

static iree_status_t
loom_link_template_candidate_initialize_materialized_module(
    loom_link_template_candidate_loader_t* loader,
    loom_link_template_candidate_module_t* cache) {
  const loom_module_t* module = cache->source_module->materialized_module;
  IREE_ASSERT(module != NULL);
  loom_symbol_fact_table_initialize(&cache->fact_table, &loader->arena);
  loom_template_provider_catalog_initialize(&cache->catalog, &loader->arena);
  IREE_RETURN_IF_ERROR(loom_template_provider_catalog_build_local(
      &cache->catalog, module, &cache->fact_table));
  for (iree_host_size_t i = 0; i < cache->catalog.provider_count; ++i) {
    const loom_template_provider_summary_t* provider =
        &cache->catalog.providers[i];
    IREE_ASSERT(loom_symbol_ref_is_valid(provider->symbol));
    IREE_ASSERT(provider->symbol.module_id == 0);
    IREE_ASSERT(provider->symbol.symbol_id < cache->summaries.count);
    cache->summaries.values[provider->symbol.symbol_id] = provider;
  }
  return iree_ok_status();
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
  cache->summaries.count = source_module->symbol_count;
  if (cache->summaries.count != 0) {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
        iree_arena_allocator(&loader->arena), cache->summaries.count,
        sizeof(*cache->summaries.values), (void**)&cache->summaries.values));
  }
  if (source_module->materialized_module != NULL) {
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

static iree_status_t loom_link_template_candidate_copy_signature_types(
    const loom_module_t* module, const loom_bytecode_function_header_t* header,
    iree_arena_allocator_t* arena, loom_template_provider_summary_t* summary) {
  const iree_host_size_t type_count =
      (iree_host_size_t)header->argument_count + header->result_count;
  if (type_count == 0) {
    return iree_ok_status();
  }
  loom_type_t* types = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, type_count, sizeof(*types), (void**)&types));
  const loom_value_id_t* arguments =
      header->signature_values + header->workload_argument_count;
  const loom_value_id_t* results = arguments + header->argument_count;
  for (uint16_t i = 0; i < header->argument_count; ++i) {
    types[i] = loom_module_value_type(module, arguments[i]);
  }
  for (uint16_t i = 0; i < header->result_count; ++i) {
    types[header->argument_count + i] =
        loom_module_value_type(module, results[i]);
  }
  summary->argument_types = header->argument_count != 0 ? types : NULL;
  summary->result_types =
      header->result_count != 0 ? types + header->argument_count : NULL;
  return iree_ok_status();
}

static iree_status_t loom_link_template_candidate_resolve_conditions(
    const loom_module_t* module, loom_parameterized_attr_array_t requirements,
    iree_arena_allocator_t* arena, loom_template_provider_summary_t* summary) {
  if (requirements.count == 0) {
    return iree_ok_status();
  }
  loom_target_condition_t* conditions = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, requirements.count, sizeof(*conditions), (void**)&conditions));
  for (iree_host_size_t i = 0; i < requirements.count; ++i) {
    const loom_target_condition_descriptor_t* descriptor = NULL;
    IREE_RETURN_IF_ERROR(loom_target_condition_resolve(
        module->context, requirements.values[i], &descriptor));
    conditions[i] = (loom_target_condition_t){
        .descriptor = descriptor,
        .value = requirements.values[i],
    };
  }
  summary->target_conditions = conditions;
  summary->target_condition_count = (uint16_t)requirements.count;
  return iree_ok_status();
}

static iree_status_t loom_link_template_candidate_lookup_target_facts(
    loom_link_template_candidate_module_t* cache,
    loom_symbol_ref_t target_symbol,
    const loom_target_facts_t** out_target_facts) {
  *out_target_facts = NULL;
  if (!loom_symbol_ref_is_valid(target_symbol)) {
    return iree_ok_status();
  }
  IREE_ASSERT(cache->header_reader != NULL);
  IREE_ASSERT(target_symbol.module_id == 0);
  IREE_RETURN_IF_ERROR(
      loom_bytecode_symbol_header_reader_materialize_bodyless_symbol(
          cache->header_reader, target_symbol.symbol_id));
  const loom_module_t* module =
      loom_bytecode_symbol_header_reader_module(cache->header_reader);
  const loom_symbol_facts_base_t* base_facts = NULL;
  IREE_RETURN_IF_ERROR(loom_symbol_fact_table_lookup_ref(
      &cache->fact_table, module, target_symbol, &base_facts));
  const loom_target_symbol_facts_t* target_facts =
      loom_target_symbol_facts_cast(base_facts);
  if (target_facts != NULL) {
    *out_target_facts = target_facts->projection;
  }
  return iree_ok_status();
}

static loom_template_provider_kind_t loom_link_template_candidate_kind(
    loom_symbol_kind_t kind) {
  switch ((loom_symbol_kind_e)kind) {
    case LOOM_SYMBOL_TEMPLATE_DEF:
      return LOOM_TEMPLATE_PROVIDER_KIND_DEF;
    case LOOM_SYMBOL_TEMPLATE_UKERNEL:
      return LOOM_TEMPLATE_PROVIDER_KIND_UKERNEL;
    default:
      IREE_ASSERT_UNREACHABLE("indexed candidate is not a template provider");
      IREE_BUILTIN_UNREACHABLE();
  }
}

static iree_status_t loom_link_template_candidate_build_bytecode_summary(
    loom_link_template_candidate_loader_t* loader,
    loom_link_template_candidate_module_t* cache,
    const loom_link_module_index_symbol_t* source_symbol,
    const loom_template_provider_summary_t** out_summary) {
  loom_bytecode_function_header_t header;
  IREE_RETURN_IF_ERROR(loom_bytecode_symbol_header_reader_read_function(
      cache->header_reader, (uint32_t)source_symbol->module_symbol_ordinal,
      &header));
  if (header.workload_argument_count != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "template provider '@%.*s' has a kernel workload signature",
        (int)source_symbol->name.size, source_symbol->name.data);
  }
  const loom_module_t* module =
      loom_bytecode_symbol_header_reader_module(cache->header_reader);
  const loom_func_like_vtable_t* func_like = header.func_like;
  IREE_ASSERT(func_like->template_family_attr_index != LOOM_ATTR_INDEX_NONE);
  const loom_symbol_ref_t family = loom_attr_as_symbol(
      header.attributes[func_like->template_family_attr_index]);
  IREE_ASSERT(loom_symbol_ref_is_valid(family));

  loom_template_provider_summary_t* summary = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(&loader->arena, sizeof(*summary), (void**)&summary));
  *summary = (loom_template_provider_summary_t){
      .module = module,
      .kind = loom_link_template_candidate_kind(source_symbol->kind),
      .has_body = header.body_region_payload_ordinal_plus_one != 0,
      .symbol = {0, header.symbol_id},
      .target_symbol = loom_symbol_ref_null(),
      .origin_ordinal = source_symbol->ordinal,
      .family = family,
      .argument_count = header.argument_count,
      .result_count = header.result_count,
      .family_name =
          module->strings
              .entries[module->symbols.entries[family.symbol_id].name_id],
      .name = header.name,
      .argument_ids = header.signature_values,
      .result_ids = header.signature_values + header.argument_count,
  };
  if (func_like->priority_attr_index != LOOM_ATTR_INDEX_NONE) {
    summary->priority =
        loom_attr_as_i64(header.attributes[func_like->priority_attr_index]);
  }
  if (func_like->predicates_attr_index != LOOM_ATTR_INDEX_NONE) {
    const loom_attribute_t predicates =
        header.attributes[func_like->predicates_attr_index];
    summary->predicates = predicates.predicate_list;
    summary->predicate_count = predicates.count;
  }
  if (func_like->target_attr_index != LOOM_ATTR_INDEX_NONE) {
    const loom_attribute_t target =
        header.attributes[func_like->target_attr_index];
    if (!loom_attr_is_absent(target)) {
      summary->target_symbol = loom_attr_as_symbol(target);
    }
  }
  if (func_like->requires_attr_index != LOOM_ATTR_INDEX_NONE) {
    const loom_attribute_t requires =
        header.attributes[func_like->requires_attr_index];
    if (!loom_attr_is_absent(requires)) {
      IREE_RETURN_IF_ERROR(loom_link_template_candidate_resolve_conditions(
          module, loom_attr_as_parameterized_array(requires), &loader->arena,
          summary));
    }
  }
  IREE_RETURN_IF_ERROR(loom_link_template_candidate_copy_signature_types(
      module, &header, &loader->arena, summary));
  IREE_RETURN_IF_ERROR(loom_link_template_candidate_lookup_target_facts(
      cache, summary->target_symbol, &summary->target_facts));
  *out_summary = summary;
  return iree_ok_status();
}

static iree_status_t loom_link_template_candidate_source_summary(
    loom_link_template_candidate_loader_t* loader,
    const loom_link_module_index_symbol_t* source_symbol,
    const loom_template_provider_summary_t** out_summary) {
  const loom_link_module_index_module_t* source_module =
      loom_link_module_index_symbol_module(loader->index, source_symbol);
  IREE_ASSERT(source_module != NULL);
  loom_link_template_candidate_module_t* cache = NULL;
  IREE_RETURN_IF_ERROR(loom_link_template_candidate_initialize_module(
      loader, source_module, &cache));
  IREE_ASSERT(source_symbol->module_symbol_ordinal < cache->summaries.count);
  const loom_template_provider_summary_t** summary_slot =
      &cache->summaries.values[source_symbol->module_symbol_ordinal];
  if (*summary_slot == NULL && cache->header_reader != NULL) {
    IREE_RETURN_IF_ERROR(loom_link_template_candidate_build_bytecode_summary(
        loader, cache, source_symbol, summary_slot));
  }
  IREE_ASSERT(*summary_slot != NULL);
  *out_summary = *summary_slot;
  return iree_ok_status();
}

typedef struct loom_link_template_candidate_projection_t {
  // Provider-backed source index.
  const loom_link_module_index_t* index;
  // Indexed module owning the summary being projected.
  const loom_link_module_index_module_t* source_module;
  // IR module owning source summary payloads.
  const loom_module_t* source_ir_module;
  // Disposable analysis module receiving placeholders.
  loom_module_t* target_module;
  // Mutable index-wide source-to-analysis symbol projection.
  loom_symbol_ref_t* target_symbols;
  // Number of target_symbols entries.
  iree_host_size_t target_symbol_count;
} loom_link_template_candidate_projection_t;

static bool loom_link_template_candidate_projection_lookup_global(
    const loom_link_template_candidate_projection_t* projection,
    const loom_link_module_index_symbol_t* source_symbol,
    loom_symbol_ref_t* out_target_ref) {
  const loom_link_module_index_symbol_t* candidate =
      loom_link_module_index_lookup_global(projection->index,
                                           source_symbol->name);
  while (candidate != NULL) {
    const loom_symbol_ref_t target_ref =
        projection->target_symbols[candidate->ordinal];
    if (loom_symbol_ref_is_valid(target_ref)) {
      *out_target_ref = target_ref;
      return true;
    }
    candidate = loom_link_module_index_next_global_duplicate(projection->index,
                                                             candidate);
  }
  return false;
}

static iree_status_t loom_link_template_candidate_projection_add_placeholder(
    loom_link_template_candidate_projection_t* projection,
    const loom_link_module_index_symbol_t* source_symbol,
    loom_symbol_ref_t* out_target_ref) {
  loom_string_id_t target_name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_module_intern_string(
      projection->target_module, source_symbol->name, &target_name_id));
  loom_symbol_id_t target_symbol_id = LOOM_SYMBOL_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_module_add_symbol(
      projection->target_module, target_name_id, &target_symbol_id));
  projection->target_module->symbols.entries[target_symbol_id].kind =
      source_symbol->kind;
  *out_target_ref = (loom_symbol_ref_t){0, target_symbol_id};
  projection->target_symbols[source_symbol->ordinal] = *out_target_ref;
  return iree_ok_status();
}

static iree_status_t loom_link_template_candidate_remap_symbol(
    void* user_data, const loom_module_t* source_module,
    loom_module_t* target_module, loom_symbol_ref_t source_ref,
    loom_symbol_ref_t* out_target_ref) {
  loom_link_template_candidate_projection_t* projection =
      (loom_link_template_candidate_projection_t*)user_data;
  IREE_ASSERT(source_module == projection->source_ir_module);
  IREE_ASSERT(target_module == projection->target_module);
  if (!loom_symbol_ref_is_valid(source_ref)) {
    *out_target_ref = loom_symbol_ref_null();
    return iree_ok_status();
  }
  if (source_ref.module_id != 0 ||
      source_ref.symbol_id >= projection->source_module->symbol_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "candidate provider has an invalid symbol ref");
  }
  const iree_host_size_t source_ordinal =
      projection->source_module->symbol_start_ordinal + source_ref.symbol_id;
  IREE_ASSERT(source_ordinal < projection->target_symbol_count);
  loom_symbol_ref_t target_ref = projection->target_symbols[source_ordinal];
  if (loom_symbol_ref_is_valid(target_ref)) {
    *out_target_ref = target_ref;
    return iree_ok_status();
  }

  const loom_link_module_index_symbol_t* source_symbol =
      loom_link_module_index_symbol_at(projection->index, source_ordinal);
  IREE_ASSERT(source_symbol != NULL);
  if (source_symbol->identity == LOOM_LINK_SYMBOL_IDENTITY_GLOBAL &&
      loom_link_template_candidate_projection_lookup_global(
          projection, source_symbol, &target_ref)) {
    projection->target_symbols[source_ordinal] = target_ref;
    *out_target_ref = target_ref;
    return iree_ok_status();
  }
  return loom_link_template_candidate_projection_add_placeholder(
      projection, source_symbol, out_target_ref);
}

static iree_status_t loom_link_template_candidate_count(
    const loom_link_template_candidate_loader_t* loader,
    const loom_link_plan_t* plan,
    loom_link_template_provider_membership_t selected_providers,
    iree_host_size_t* out_count) {
  *out_count = 0;
  const iree_host_size_t family_count =
      loom_link_plan_demanded_template_family_count(plan);
  for (iree_host_size_t i = 0; i < family_count; ++i) {
    const loom_link_template_family_ordinal_t family_ordinal =
        loom_link_plan_demanded_template_family_at(plan, i);
    const loom_link_module_index_template_family_t* family =
        loom_link_module_index_template_family_at(loader->index,
                                                  family_ordinal);
    IREE_ASSERT(family != NULL);
    iree_host_size_t provider_ordinal = family->providers.first_symbol_ordinal;
    while (provider_ordinal != LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL) {
      if (!loom_link_template_provider_membership_contains(selected_providers,
                                                           provider_ordinal)) {
        iree_host_size_t count = 0;
        if (!iree_host_size_checked_add(*out_count, 1, &count)) {
          return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                  "template candidate count overflow");
        }
        *out_count = count;
      }
      const loom_link_module_index_symbol_t* provider =
          loom_link_module_index_symbol_at(loader->index, provider_ordinal);
      IREE_ASSERT(provider != NULL);
      provider_ordinal = provider->next.template_provider_ordinal;
    }
  }
  return iree_ok_status();
}

iree_status_t loom_link_template_candidate_loader_allocate(
    const loom_link_module_index_t* index,
    const loom_link_plan_materialization_environment_t* environment,
    loom_link_template_candidate_loader_t** out_loader) {
  IREE_ASSERT_ARGUMENT(index);
  IREE_ASSERT_ARGUMENT(environment);
  IREE_ASSERT_ARGUMENT(environment->block_pool);
  IREE_ASSERT_ARGUMENT(out_loader);
  *out_loader = NULL;

  loom_link_template_candidate_loader_t* loader = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(environment->allocator,
                                             sizeof(*loader), (void**)&loader));
  loader->index = index;
  loader->environment = environment;
  iree_arena_initialize(environment->block_pool, &loader->arena);
  loader->modules.count = loom_link_module_index_module_count(index);
  iree_status_t status = iree_ok_status();
  if (loader->modules.count != 0) {
    status = iree_allocator_malloc_array(
        iree_arena_allocator(&loader->arena), loader->modules.count,
        sizeof(*loader->modules.values), (void**)&loader->modules.values);
  }
  if (iree_status_is_ok(status)) {
    *out_loader = loader;
  } else {
    loom_link_template_candidate_loader_free(loader);
  }
  return status;
}

void loom_link_template_candidate_loader_free(
    loom_link_template_candidate_loader_t* loader) {
  if (loader == NULL) {
    return;
  }
  const iree_allocator_t allocator = loader->environment->allocator;
  for (iree_host_size_t i = 0; i < loader->modules.count; ++i) {
    loom_bytecode_symbol_header_reader_free(
        loader->modules.values[i].header_reader);
  }
  iree_arena_deinitialize(&loader->arena);
  iree_allocator_free(allocator, loader);
}

iree_status_t loom_link_template_candidate_loader_project(
    loom_link_template_candidate_loader_t* loader, const loom_link_plan_t* plan,
    loom_link_plan_materialization_t* materialization,
    loom_link_template_provider_membership_t selected_providers,
    iree_arena_allocator_t* arena,
    loom_template_provider_slice_t* out_candidates) {
  IREE_ASSERT_ARGUMENT(loader);
  IREE_ASSERT_ARGUMENT(plan);
  IREE_ASSERT_ARGUMENT(materialization);
  IREE_ASSERT_ARGUMENT(materialization->module);
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_candidates);
  *out_candidates = loom_template_provider_slice_empty();
  IREE_ASSERT(selected_providers.symbol_count ==
              loom_link_module_index_symbol_count(loader->index));

  iree_host_size_t candidate_capacity = 0;
  IREE_RETURN_IF_ERROR(loom_link_template_candidate_count(
      loader, plan, selected_providers, &candidate_capacity));
  if (candidate_capacity == 0) {
    return iree_ok_status();
  }
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
      if (!loom_link_template_provider_membership_contains(selected_providers,
                                                           provider_ordinal)) {
        const loom_template_provider_summary_t* source_summary = NULL;
        IREE_RETURN_IF_ERROR(loom_link_template_candidate_source_summary(
            loader, provider, &source_summary));
        const loom_link_module_index_module_t* source_module =
            loom_link_module_index_symbol_module(loader->index, provider);
        loom_link_template_candidate_projection_t projection = {
            .index = loader->index,
            .source_module = source_module,
            .source_ir_module = source_summary->module,
            .target_module = materialization->module,
            .target_symbols = materialization->target_symbols.values,
            .target_symbol_count = materialization->target_symbols.count,
        };
        IREE_RETURN_IF_ERROR(loom_template_provider_summary_project(
            source_summary, materialization->module, target_family,
            provider_ordinal,
            loom_ir_remap_symbol_callback_make(
                loom_link_template_candidate_remap_symbol, &projection),
            arena, &candidates[candidate_count++]));
      }
      provider_ordinal = provider->next.template_provider_ordinal;
    }
  }
  IREE_ASSERT(candidate_count == candidate_capacity);
  *out_candidates = (loom_template_provider_slice_t){
      .providers = candidates,
      .count = candidate_count,
  };
  return iree_ok_status();
}

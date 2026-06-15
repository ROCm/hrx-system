// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/func_provider_catalog.h"

#include <stdlib.h>
#include <string.h>

#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"

#define LOOM_FUNC_PROVIDER_INDEX_INVALID ((uint32_t)UINT32_MAX)

static bool loom_func_provider_symbol_kind(
    loom_symbol_kind_t symbol_kind, loom_func_provider_kind_t* out_kind) {
  switch ((loom_symbol_kind_e)symbol_kind) {
    case LOOM_SYMBOL_FUNC_TEMPLATE:
      *out_kind = LOOM_FUNC_PROVIDER_KIND_TEMPLATE;
      return true;
    case LOOM_SYMBOL_FUNC_UKERNEL:
      *out_kind = LOOM_FUNC_PROVIDER_KIND_UKERNEL;
      return true;
    default:
      *out_kind = LOOM_FUNC_PROVIDER_KIND_NONE;
      return false;
  }
}

static bool loom_func_provider_symbol_is_live_provider(
    const loom_symbol_t* symbol, loom_func_provider_kind_t* out_kind) {
  if (!loom_func_provider_symbol_kind(symbol->kind, out_kind)) return false;
  return symbol->defining_op &&
         !iree_any_bit_set(symbol->defining_op->flags, LOOM_OP_FLAG_DEAD);
}

static int loom_func_provider_summary_compare(const void* lhs,
                                              const void* rhs) {
  const loom_func_provider_summary_t* left =
      (const loom_func_provider_summary_t*)lhs;
  const loom_func_provider_summary_t* right =
      (const loom_func_provider_summary_t*)rhs;
  if (left->contract_id != right->contract_id) {
    return left->contract_id < right->contract_id ? -1 : 1;
  }
  if (left->priority != right->priority) {
    return left->priority > right->priority ? -1 : 1;
  }
  if (left->kind != right->kind) {
    return left->kind < right->kind ? -1 : 1;
  }
  if (left->origin != right->origin) {
    return left->origin < right->origin ? -1 : 1;
  }
  if (left->provider_id == right->provider_id) return 0;
  return left->provider_id < right->provider_id ? -1 : 1;
}

static iree_status_t loom_func_provider_catalog_count_local(
    const loom_module_t* module, iree_host_size_t* out_provider_count) {
  *out_provider_count = 0;
  if (module->symbols.count > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "func provider symbol table exceeds uint32_t "
                            "index range");
  }
  loom_symbol_t* symbol = NULL;
  loom_module_for_each_symbol(module, symbol) {
    loom_func_provider_kind_t kind = LOOM_FUNC_PROVIDER_KIND_NONE;
    if (!loom_func_provider_symbol_is_live_provider(symbol, &kind)) continue;
    ++*out_provider_count;
  }
  return iree_ok_status();
}

static iree_status_t loom_func_provider_catalog_lookup_facts(
    const loom_module_t* module, loom_symbol_fact_table_t* fact_table,
    loom_symbol_id_t symbol_id, const loom_func_symbol_facts_t** out_facts) {
  *out_facts = NULL;
  const loom_symbol_facts_base_t* base_facts = NULL;
  IREE_RETURN_IF_ERROR(loom_symbol_fact_table_lookup(fact_table, module,
                                                     symbol_id, &base_facts));
  const loom_func_symbol_facts_t* facts =
      loom_func_symbol_facts_cast(base_facts);
  if (!facts) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "func provider symbol has no func facts");
  }
  if (facts->implements_id == LOOM_STRING_ID_INVALID ||
      facts->implements_id >= module->strings.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "func provider @%.*s has no valid implementation "
                            "contract key",
                            (int)facts->name.size, facts->name.data);
  }
  *out_facts = facts;
  return iree_ok_status();
}

static iree_status_t loom_func_provider_catalog_copy_value_types(
    const loom_module_t* module, const loom_value_id_t* value_ids,
    uint16_t value_count, loom_type_t* types) {
  for (uint16_t i = 0; i < value_count; ++i) {
    const loom_value_id_t value_id = value_ids[i];
    if (value_id >= module->values.count) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "func provider signature value id %u is outside "
                              "the module value table",
                              (uint32_t)value_id);
    }
    types[i] = loom_module_value_type(module, value_id);
  }
  return iree_ok_status();
}

static iree_status_t loom_func_provider_catalog_populate_signature(
    loom_func_provider_catalog_t* catalog, const loom_module_t* module,
    const loom_func_symbol_facts_t* facts,
    loom_func_provider_summary_t* provider) {
  provider->argument_count = facts->argument_count;
  provider->result_count = facts->result_count;
  const iree_host_size_t type_count =
      (iree_host_size_t)facts->argument_count + facts->result_count;
  if (type_count == 0) return iree_ok_status();

  loom_type_t* types = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      catalog->arena, type_count, sizeof(*types), (void**)&types));
  IREE_RETURN_IF_ERROR(loom_func_provider_catalog_copy_value_types(
      module, facts->argument_ids, facts->argument_count, types));
  IREE_RETURN_IF_ERROR(loom_func_provider_catalog_copy_value_types(
      module, facts->result_ids, facts->result_count,
      types + facts->argument_count));
  provider->argument_types = facts->argument_count > 0 ? types : NULL;
  provider->result_types =
      facts->result_count > 0 ? types + facts->argument_count : NULL;
  return iree_ok_status();
}

static iree_status_t loom_func_provider_catalog_populate_local(
    loom_func_provider_catalog_t* catalog, const loom_module_t* module,
    loom_symbol_fact_table_t* fact_table,
    loom_func_provider_summary_t* providers) {
  iree_host_size_t provider_index = 0;
  for (iree_host_size_t i = 0; i < module->symbols.count; ++i) {
    const loom_symbol_t* symbol = &module->symbols.entries[i];
    loom_func_provider_kind_t kind = LOOM_FUNC_PROVIDER_KIND_NONE;
    if (!loom_func_provider_symbol_is_live_provider(symbol, &kind)) continue;

    const loom_func_symbol_facts_t* facts = NULL;
    IREE_RETURN_IF_ERROR(loom_func_provider_catalog_lookup_facts(
        module, fact_table, (loom_symbol_id_t)i, &facts));
    loom_func_like_t function =
        loom_func_like_cast(module, symbol->defining_op);
    if (!loom_func_like_isa(function)) {
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "func provider symbol is not function-like");
    }

    loom_func_provider_summary_t* provider = &providers[provider_index];
    *provider = (loom_func_provider_summary_t){
        .provider_id = i,
        .origin = LOOM_FUNC_PROVIDER_ORIGIN_LOCAL,
        .kind = kind,
        .is_public = facts->visibility != 0,
        .has_body = facts->has_body,
        .calling_convention = facts->calling_convention,
        .purity = facts->purity,
        .temperature = facts->temperature,
        .inline_policy = facts->inline_policy,
        .symbol =
            (loom_symbol_ref_t){
                .module_id = 0,
                .symbol_id = (loom_symbol_id_t)i,
            },
        .target_symbol = facts->target_symbol,
        .function = function,
        .func_facts = facts,
        .contract_id = facts->implements_id,
        .contract = facts->implements,
        .name = facts->name,
        .priority = facts->priority,
        .predicates = facts->predicates,
        .predicate_count = facts->predicate_count,
    };
    IREE_RETURN_IF_ERROR(loom_func_provider_catalog_populate_signature(
        catalog, module, facts, provider));
    ++provider_index;
  }
  catalog->provider_count = provider_index;
  if (provider_index > 1) {
    qsort(providers, provider_index, sizeof(*providers),
          loom_func_provider_summary_compare);
  }
  return iree_ok_status();
}

static void loom_func_provider_catalog_initialize_buckets(
    loom_func_provider_catalog_bucket_t* buckets,
    iree_host_size_t bucket_count) {
  for (iree_host_size_t i = 0; i < bucket_count; ++i) {
    buckets[i] = (loom_func_provider_catalog_bucket_t){
        .first_provider_index = LOOM_FUNC_PROVIDER_INDEX_INVALID,
    };
  }
}

static iree_status_t loom_func_provider_catalog_build_buckets(
    loom_func_provider_catalog_t* catalog,
    loom_func_provider_catalog_bucket_t* buckets) {
  const loom_func_provider_summary_t* providers = catalog->providers;
  for (iree_host_size_t i = 0; i < catalog->provider_count; ++i) {
    const loom_string_id_t contract_id = providers[i].contract_id;
    if (contract_id >= catalog->bucket_count) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "func provider contract id %u is outside the "
                              "module string table",
                              (uint32_t)contract_id);
    }
    loom_func_provider_catalog_bucket_t* bucket = &buckets[contract_id];
    if (bucket->first_provider_index == LOOM_FUNC_PROVIDER_INDEX_INVALID) {
      bucket->first_provider_index = (uint32_t)i;
    }
    ++bucket->provider_count;
  }
  return iree_ok_status();
}

void loom_func_provider_catalog_initialize(
    loom_func_provider_catalog_t* catalog, iree_arena_allocator_t* arena) {
  IREE_ASSERT_ARGUMENT(catalog);
  memset(catalog, 0, sizeof(*catalog));
  catalog->arena = arena;
}

void loom_func_provider_catalog_reset(loom_func_provider_catalog_t* catalog) {
  IREE_ASSERT_ARGUMENT(catalog);
  iree_arena_allocator_t* arena = catalog->arena;
  memset(catalog, 0, sizeof(*catalog));
  catalog->arena = arena;
}

iree_status_t loom_func_provider_catalog_build_local(
    loom_func_provider_catalog_t* catalog, const loom_module_t* module,
    loom_symbol_fact_table_t* fact_table) {
  IREE_ASSERT_ARGUMENT(catalog);
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(fact_table);
  if (!catalog->arena) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "func provider catalog has no arena");
  }

  loom_func_provider_catalog_reset(catalog);
  catalog->module = module;
  catalog->bucket_count = module->strings.count;

  iree_host_size_t provider_count = 0;
  IREE_RETURN_IF_ERROR(
      loom_func_provider_catalog_count_local(module, &provider_count));

  loom_func_provider_summary_t* providers = NULL;
  if (provider_count > 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(catalog->arena, provider_count,
                                  sizeof(*providers), (void**)&providers));
  }
  catalog->providers = providers;

  loom_func_provider_catalog_bucket_t* buckets = NULL;
  if (catalog->bucket_count > 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(catalog->arena, catalog->bucket_count,
                                  sizeof(*buckets), (void**)&buckets));
    loom_func_provider_catalog_initialize_buckets(buckets,
                                                  catalog->bucket_count);
  }
  catalog->buckets_by_string_id = buckets;

  IREE_RETURN_IF_ERROR(loom_func_provider_catalog_populate_local(
      catalog, module, fact_table, providers));
  IREE_RETURN_IF_ERROR(
      loom_func_provider_catalog_build_buckets(catalog, buckets));
  return iree_ok_status();
}

loom_func_provider_slice_t loom_func_provider_catalog_lookup(
    const loom_func_provider_catalog_t* catalog, loom_string_id_t contract_id) {
  IREE_ASSERT_ARGUMENT(catalog);
  if (!catalog->providers || !catalog->buckets_by_string_id ||
      contract_id >= catalog->bucket_count) {
    return loom_func_provider_slice_empty();
  }
  const loom_func_provider_catalog_bucket_t* bucket =
      &catalog->buckets_by_string_id[contract_id];
  if (bucket->first_provider_index == LOOM_FUNC_PROVIDER_INDEX_INVALID ||
      bucket->provider_count == 0) {
    return loom_func_provider_slice_empty();
  }
  return (loom_func_provider_slice_t){
      .providers = catalog->providers + bucket->first_provider_index,
      .count = bucket->provider_count,
  };
}

loom_func_provider_slice_t loom_func_provider_catalog_lookup_key(
    const loom_func_provider_catalog_t* catalog, iree_string_view_t contract) {
  IREE_ASSERT_ARGUMENT(catalog);
  if (!catalog->module) return loom_func_provider_slice_empty();
  loom_string_id_t contract_id =
      loom_module_lookup_string(catalog->module, contract);
  if (contract_id == LOOM_STRING_ID_INVALID) {
    return loom_func_provider_slice_empty();
  }
  return loom_func_provider_catalog_lookup(catalog, contract_id);
}

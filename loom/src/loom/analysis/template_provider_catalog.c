// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/template_provider_catalog.h"

#include <string.h>

#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/target/facts.h"
#include "loom/rewrite/remap.h"

#define LOOM_TEMPLATE_PROVIDER_INDEX_INVALID ((uint32_t)UINT32_MAX)

static bool loom_template_provider_symbol_kind(
    loom_symbol_kind_t symbol_kind, loom_template_provider_kind_t* out_kind) {
  switch ((loom_symbol_kind_e)symbol_kind) {
    case LOOM_SYMBOL_TEMPLATE_DEF:
      *out_kind = LOOM_TEMPLATE_PROVIDER_KIND_DEF;
      return true;
    case LOOM_SYMBOL_TEMPLATE_UKERNEL:
      *out_kind = LOOM_TEMPLATE_PROVIDER_KIND_UKERNEL;
      return true;
    default:
      *out_kind = LOOM_TEMPLATE_PROVIDER_KIND_NONE;
      return false;
  }
}

static bool loom_template_provider_symbol_is_live_provider(
    const loom_symbol_t* symbol, loom_template_provider_kind_t* out_kind) {
  if (!loom_template_provider_symbol_kind(symbol->kind, out_kind)) {
    return false;
  }
  return symbol->defining_op &&
         !iree_any_bit_set(symbol->defining_op->flags, LOOM_OP_FLAG_DEAD);
}

static iree_status_t loom_template_provider_catalog_count_local(
    const loom_module_t* module,
    loom_template_provider_catalog_bucket_t* buckets,
    iree_host_size_t* out_provider_count) {
  *out_provider_count = 0;
  if (module->symbols.count > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "template provider symbol table exceeds uint32_t "
                            "index range");
  }
  loom_symbol_t* symbol = NULL;
  loom_module_for_each_symbol(module, symbol) {
    loom_template_provider_kind_t kind = LOOM_TEMPLATE_PROVIDER_KIND_NONE;
    if (!loom_template_provider_symbol_is_live_provider(symbol, &kind)) {
      continue;
    }

    loom_func_like_t function =
        loom_func_like_cast(module, symbol->defining_op);
    if (!loom_func_like_isa(function)) {
      IREE_ASSERT_UNREACHABLE("template provider symbol is not function-like");
      IREE_BUILTIN_UNREACHABLE();
    }
    const loom_symbol_ref_t family = loom_func_like_template_family(function);
    IREE_ASSERT(loom_symbol_ref_is_valid(family));
    IREE_ASSERT(family.module_id == 0);
    IREE_ASSERT(family.symbol_id < module->symbols.count);
    loom_template_provider_catalog_bucket_t* bucket =
        &buckets[family.symbol_id];
    if (bucket->provider_count == UINT32_MAX) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "template family provider count overflow");
    }
    ++bucket->provider_count;
    ++*out_provider_count;
  }
  return iree_ok_status();
}

static iree_status_t loom_template_provider_catalog_lookup_facts(
    const loom_module_t* module, loom_symbol_fact_table_t* fact_table,
    loom_symbol_id_t symbol_id, const loom_func_symbol_facts_t** out_facts) {
  *out_facts = NULL;
  const loom_symbol_facts_base_t* base_facts = NULL;
  IREE_RETURN_IF_ERROR(loom_symbol_fact_table_lookup(fact_table, module,
                                                     symbol_id, &base_facts));
  const loom_func_symbol_facts_t* facts =
      loom_func_symbol_facts_cast(base_facts);
  if (!facts) {
    IREE_ASSERT_UNREACHABLE("template provider symbol has no function facts");
    IREE_BUILTIN_UNREACHABLE();
  }
  IREE_ASSERT(loom_symbol_ref_is_valid(facts->template_family));
  IREE_ASSERT(facts->template_family.module_id == 0);
  IREE_ASSERT(facts->template_family.symbol_id < module->symbols.count);
  *out_facts = facts;
  return iree_ok_status();
}

static iree_status_t loom_template_provider_catalog_copy_value_types(
    const loom_module_t* module, const loom_value_id_t* value_ids,
    uint16_t value_count, loom_type_t* types) {
  for (uint16_t i = 0; i < value_count; ++i) {
    const loom_value_id_t value_id = value_ids[i];
    if (value_id >= module->values.count) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "template provider signature value id %u is "
                              "outside "
                              "the module value table",
                              (uint32_t)value_id);
    }
    types[i] = loom_module_value_type(module, value_id);
  }
  return iree_ok_status();
}

static iree_status_t loom_template_provider_catalog_populate_signature(
    loom_template_provider_catalog_t* catalog, const loom_module_t* module,
    const loom_func_symbol_facts_t* facts,
    loom_template_provider_summary_t* provider) {
  provider->argument_count = facts->argument_count;
  provider->result_count = facts->result_count;
  provider->argument_ids = facts->argument_ids;
  provider->result_ids = facts->result_ids;
  const iree_host_size_t type_count =
      (iree_host_size_t)facts->argument_count + facts->result_count;
  provider->signature_is_module_independent = true;
  if (type_count == 0) {
    return iree_ok_status();
  }

  loom_type_t* types = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      catalog->arena, type_count, sizeof(*types), (void**)&types));
  IREE_RETURN_IF_ERROR(loom_template_provider_catalog_copy_value_types(
      module, facts->argument_ids, facts->argument_count, types));
  IREE_RETURN_IF_ERROR(loom_template_provider_catalog_copy_value_types(
      module, facts->result_ids, facts->result_count,
      types + facts->argument_count));
  provider->argument_types = facts->argument_count > 0 ? types : NULL;
  provider->result_types =
      facts->result_count > 0 ? types + facts->argument_count : NULL;
  provider->signature_is_module_independent =
      loom_type_sequence_is_module_independent(types, type_count);
  return iree_ok_status();
}

static iree_status_t loom_template_provider_catalog_target_facts(
    const loom_module_t* module, loom_symbol_fact_table_t* fact_table,
    loom_symbol_ref_t target_symbol,
    const loom_target_facts_t** out_target_facts) {
  *out_target_facts = NULL;
  if (!loom_symbol_ref_is_valid(target_symbol)) {
    return iree_ok_status();
  }
  const loom_symbol_facts_base_t* base_facts = NULL;
  IREE_RETURN_IF_ERROR(loom_symbol_fact_table_lookup_ref(
      fact_table, module, target_symbol, &base_facts));
  const loom_target_symbol_facts_t* target_facts =
      loom_target_symbol_facts_cast(base_facts);
  if (target_facts != NULL) {
    *out_target_facts = target_facts->projection;
  }
  return iree_ok_status();
}

static iree_status_t loom_template_provider_catalog_populate_local(
    loom_template_provider_catalog_t* catalog, const loom_module_t* module,
    loom_symbol_fact_table_t* fact_table,
    loom_template_provider_summary_t* providers,
    loom_template_provider_catalog_bucket_t* buckets) {
  for (iree_host_size_t i = module->symbols.count; i-- > 0;) {
    const loom_symbol_t* symbol = &module->symbols.entries[i];
    loom_template_provider_kind_t kind = LOOM_TEMPLATE_PROVIDER_KIND_NONE;
    if (!loom_template_provider_symbol_is_live_provider(symbol, &kind)) {
      continue;
    }

    const loom_func_symbol_facts_t* facts = NULL;
    IREE_RETURN_IF_ERROR(loom_template_provider_catalog_lookup_facts(
        module, fact_table, (loom_symbol_id_t)i, &facts));
    loom_func_like_t function =
        loom_func_like_cast(module, symbol->defining_op);
    if (!loom_func_like_isa(function)) {
      IREE_ASSERT_UNREACHABLE("template provider symbol is not function-like");
      IREE_BUILTIN_UNREACHABLE();
    }

    loom_template_provider_catalog_bucket_t* bucket =
        &buckets[facts->template_family.symbol_id];
    IREE_ASSERT(bucket->first_provider_index !=
                LOOM_TEMPLATE_PROVIDER_INDEX_INVALID);
    IREE_ASSERT(bucket->first_provider_index > 0);
    loom_template_provider_summary_t* provider =
        &providers[--bucket->first_provider_index];
    *provider = (loom_template_provider_summary_t){
        .module = module,
        .kind = kind,
        .has_body = facts->has_body,
        .symbol =
            (loom_symbol_ref_t){
                .module_id = 0,
                .symbol_id = (loom_symbol_id_t)i,
            },
        .target_symbol = facts->target_symbol,
        .function = function,
        .func_facts = facts,
        .origin_ordinal = IREE_HOST_SIZE_MAX,
        .family = facts->template_family,
        .family_name = facts->template_family_name,
        .name = facts->name,
        .priority = facts->priority,
        .predicates = facts->predicates,
        .predicate_count = facts->predicate_count,
        .target_conditions = facts->target_conditions,
        .target_condition_count = facts->target_condition_count,
    };
    IREE_RETURN_IF_ERROR(loom_template_provider_catalog_populate_signature(
        catalog, module, facts, provider));
    IREE_RETURN_IF_ERROR(loom_template_provider_catalog_target_facts(
        module, fact_table, facts->target_symbol, &provider->target_facts));
  }
  return iree_ok_status();
}

static void loom_template_provider_catalog_initialize_buckets(
    loom_template_provider_catalog_bucket_t* buckets,
    iree_host_size_t bucket_count) {
  for (iree_host_size_t i = 0; i < bucket_count; ++i) {
    buckets[i] = (loom_template_provider_catalog_bucket_t){
        .first_provider_index = LOOM_TEMPLATE_PROVIDER_INDEX_INVALID,
    };
  }
}

static iree_status_t loom_template_provider_catalog_prepare_buckets(
    loom_template_provider_catalog_t* catalog,
    loom_template_provider_catalog_bucket_t* buckets) {
  uint32_t provider_end = 0;
  for (iree_host_size_t i = 0; i < catalog->bucket_count; ++i) {
    loom_template_provider_catalog_bucket_t* bucket = &buckets[i];
    if (bucket->provider_count == 0) {
      continue;
    }
    if (bucket->provider_count > UINT32_MAX - provider_end) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "template provider catalog exceeds uint32_t "
                              "index range");
    }
    provider_end += bucket->provider_count;
    // Reverse population decrements this family end into its stable start.
    bucket->first_provider_index = provider_end;
  }
  IREE_ASSERT(provider_end == catalog->provider_count);
  return iree_ok_status();
}

void loom_template_provider_catalog_initialize(
    loom_template_provider_catalog_t* catalog, iree_arena_allocator_t* arena) {
  IREE_ASSERT_ARGUMENT(catalog);
  memset(catalog, 0, sizeof(*catalog));
  catalog->arena = arena;
}

void loom_template_provider_catalog_reset(
    loom_template_provider_catalog_t* catalog) {
  IREE_ASSERT_ARGUMENT(catalog);
  iree_arena_allocator_t* arena = catalog->arena;
  memset(catalog, 0, sizeof(*catalog));
  catalog->arena = arena;
}

iree_status_t loom_template_provider_catalog_build(
    loom_template_provider_catalog_t* catalog, const loom_module_t* module,
    loom_symbol_fact_table_t* fact_table,
    const loom_template_provider_summary_t* external_providers,
    iree_host_size_t external_provider_count) {
  IREE_ASSERT_ARGUMENT(catalog);
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(fact_table);
  if (!catalog->arena) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "template provider catalog has no arena");
  }

  loom_template_provider_catalog_reset(catalog);
  catalog->module = module;
  catalog->bucket_count = module->symbols.count;

  if (external_provider_count != 0 && external_providers == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "external provider count is non-zero but values is NULL");
  }

  loom_template_provider_catalog_bucket_t* buckets = NULL;
  if (catalog->bucket_count > 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(catalog->arena, catalog->bucket_count,
                                  sizeof(*buckets), (void**)&buckets));
    loom_template_provider_catalog_initialize_buckets(buckets,
                                                      catalog->bucket_count);
  }
  catalog->buckets_by_symbol_id = buckets;

  iree_host_size_t local_provider_count = 0;
  IREE_RETURN_IF_ERROR(loom_template_provider_catalog_count_local(
      module, buckets, &local_provider_count));
  for (iree_host_size_t i = 0; i < external_provider_count; ++i) {
    IREE_ASSERT(external_providers[i].module == module ||
                external_providers[i].signature_is_module_independent);
    const loom_symbol_ref_t family = external_providers[i].family;
    IREE_ASSERT(loom_symbol_ref_is_valid(family));
    IREE_ASSERT(family.module_id == 0);
    IREE_ASSERT(family.symbol_id < catalog->bucket_count);
    loom_template_provider_catalog_bucket_t* bucket =
        &buckets[family.symbol_id];
    if (bucket->provider_count == UINT32_MAX) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "template family provider count overflow");
    }
    ++bucket->provider_count;
  }

  iree_host_size_t provider_count = 0;
  if (!iree_host_size_checked_add(local_provider_count, external_provider_count,
                                  &provider_count) ||
      provider_count > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "template provider count overflow");
  }
  catalog->provider_count = provider_count;
  IREE_RETURN_IF_ERROR(
      loom_template_provider_catalog_prepare_buckets(catalog, buckets));

  loom_template_provider_summary_t* providers = NULL;
  if (provider_count > 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(catalog->arena, provider_count,
                                  sizeof(*providers), (void**)&providers));
  }
  catalog->providers = providers;

  for (iree_host_size_t i = external_provider_count; i-- > 0;) {
    const loom_template_provider_summary_t* external = &external_providers[i];
    loom_template_provider_catalog_bucket_t* bucket =
        &buckets[external->family.symbol_id];
    IREE_ASSERT(bucket->first_provider_index > 0);
    providers[--bucket->first_provider_index] = *external;
  }
  IREE_RETURN_IF_ERROR(loom_template_provider_catalog_populate_local(
      catalog, module, fact_table, providers, buckets));
  return iree_ok_status();
}

iree_status_t loom_template_provider_catalog_build_local(
    loom_template_provider_catalog_t* catalog, const loom_module_t* module,
    loom_symbol_fact_table_t* fact_table) {
  return loom_template_provider_catalog_build(catalog, module, fact_table,
                                              /*external_providers=*/NULL,
                                              /*external_provider_count=*/0);
}

iree_status_t loom_template_provider_summary_project(
    const loom_template_provider_summary_t* source,
    loom_module_t* target_module, loom_symbol_ref_t target_family,
    iree_host_size_t origin_ordinal,
    loom_ir_remap_symbol_callback_t symbol_remap, iree_arena_allocator_t* arena,
    loom_template_provider_summary_t* out_provider) {
  IREE_ASSERT_ARGUMENT(source);
  IREE_ASSERT_ARGUMENT(source->module);
  IREE_ASSERT_ARGUMENT(target_module);
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_provider);
  *out_provider = (loom_template_provider_summary_t){0};
  if (!loom_symbol_ref_is_valid(target_family) ||
      target_family.module_id != 0 ||
      target_family.symbol_id >= target_module->symbols.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "projected template family is invalid");
  }

  loom_ir_remap_t remap = {0};
  IREE_RETURN_IF_ERROR(loom_ir_remap_initialize(
      source->module, target_module, arena,
      &(loom_ir_remap_options_t){
          .remap_symbol = symbol_remap,
          .value_map_kind = LOOM_IR_REMAP_VALUE_MAP_SPARSE,
      },
      &remap));

  const iree_host_size_t value_count =
      (iree_host_size_t)source->argument_count + source->result_count;
  loom_value_id_t* value_ids = NULL;
  loom_type_t* types = NULL;
  loom_value_id_t base_value_id = LOOM_VALUE_ID_INVALID;
  if (value_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, value_count, sizeof(*value_ids), (void**)&value_ids));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, value_count, sizeof(*types), (void**)&types));
    IREE_RETURN_IF_ERROR(loom_module_define_untyped_values(
        target_module, value_count, &base_value_id));
    for (iree_host_size_t i = 0; i < value_count; ++i) {
      value_ids[i] = (loom_value_id_t)(base_value_id + i);
    }
    IREE_RETURN_IF_ERROR(loom_ir_remap_map_values(
        &remap, source->argument_ids, value_ids, source->argument_count));
    IREE_RETURN_IF_ERROR(loom_ir_remap_map_values(
        &remap, source->result_ids, value_ids + source->argument_count,
        source->result_count));
    for (uint16_t i = 0; i < source->argument_count; ++i) {
      IREE_RETURN_IF_ERROR(
          loom_ir_remap_type(&remap, source->argument_types[i], &types[i]));
    }
    for (uint16_t i = 0; i < source->result_count; ++i) {
      IREE_RETURN_IF_ERROR(loom_ir_remap_type(
          &remap, source->result_types[i], &types[source->argument_count + i]));
    }
    for (iree_host_size_t i = 0; i < value_count; ++i) {
      IREE_RETURN_IF_ERROR(
          loom_module_set_value_type(target_module, value_ids[i], types[i]));
    }
  }

  loom_predicate_t* predicates = NULL;
  IREE_RETURN_IF_ERROR(loom_ir_remap_predicate_list(
      &remap, source->predicates, source->predicate_count, &predicates));
  loom_target_condition_t* target_conditions = NULL;
  if (source->target_condition_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, source->target_condition_count, sizeof(*target_conditions),
        (void**)&target_conditions));
    for (uint16_t i = 0; i < source->target_condition_count; ++i) {
      target_conditions[i].descriptor = source->target_conditions[i].descriptor;
      IREE_RETURN_IF_ERROR(
          loom_ir_remap_attribute(&remap, source->target_conditions[i].value,
                                  &target_conditions[i].value));
    }
  }

  loom_symbol_ref_t target_symbol = loom_symbol_ref_null();
  IREE_RETURN_IF_ERROR(
      loom_ir_remap_symbol_ref(&remap, source->target_symbol, &target_symbol));
  const loom_symbol_t* family_symbol =
      &target_module->symbols.entries[target_family.symbol_id];
  *out_provider = *source;
  out_provider->module = target_module;
  out_provider->symbol = loom_symbol_ref_null();
  out_provider->target_symbol = target_symbol;
  out_provider->function = (loom_func_like_t){0};
  out_provider->func_facts = NULL;
  out_provider->origin_ordinal = origin_ordinal;
  out_provider->family = target_family;
  out_provider->family_name =
      target_module->strings.entries[family_symbol->name_id];
  out_provider->argument_types = source->argument_count > 0 ? types : NULL;
  out_provider->result_types =
      source->result_count > 0 ? types + source->argument_count : NULL;
  out_provider->argument_ids = source->argument_count > 0 ? value_ids : NULL;
  out_provider->result_ids =
      source->result_count > 0 ? value_ids + source->argument_count : NULL;
  out_provider->predicates = predicates;
  out_provider->target_conditions = target_conditions;
  return iree_ok_status();
}

loom_template_provider_slice_t loom_template_provider_catalog_lookup(
    const loom_template_provider_catalog_t* catalog, loom_symbol_ref_t family) {
  IREE_ASSERT_ARGUMENT(catalog);
  if (!catalog->providers || !catalog->buckets_by_symbol_id ||
      !loom_symbol_ref_is_valid(family) || family.module_id != 0 ||
      family.symbol_id >= catalog->bucket_count) {
    return loom_template_provider_slice_empty();
  }
  const loom_template_provider_catalog_bucket_t* bucket =
      &catalog->buckets_by_symbol_id[family.symbol_id];
  if (bucket->first_provider_index == LOOM_TEMPLATE_PROVIDER_INDEX_INVALID ||
      bucket->provider_count == 0) {
    return loom_template_provider_slice_empty();
  }
  return (loom_template_provider_slice_t){
      .providers = catalog->providers + bucket->first_provider_index,
      .count = bucket->provider_count,
  };
}

// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/kernel/kernel_request_producer.h"

#include <string.h>

#include "loom/analysis/symbol_facts.h"
#include "loom/analysis/symbol_references.h"
#include "loom/analysis/symbolic_expr.h"
#include "loom/analysis/template_provider_catalog.h"
#include "loom/ir/module.h"
#include "loom/link/index_materializer.h"
#include "loom/link/template_candidate_loader.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/target/facts.h"
#include "loom/ops/template/ops.h"
#include "loom/ops/type_registry.h"
#include "loom/transforms/symbol/template_decision_model.h"
#include "loom/util/adaptive_sort.h"
#include "loom/util/fact_table.h"

typedef struct loom_kernel_request_access_entry_t {
  // Accepted implementation trace; shape refinements do not change effects.
  loom_kernel_class_trace_id_t trace_id;
  // Source-independent facts, or NULL until this trace is queried.
  const loom_storage_access_summary_t* summary;
} loom_kernel_request_access_entry_t;

typedef struct loom_kernel_request_access_cache_t {
  // Publication scratch retaining compact summaries and their sorted index.
  iree_arena_allocator_t* arena;
  // Pool for one temporary reached-source graph at a time.
  iree_arena_block_pool_t* block_pool;
  // Lazily constructed unique trace index.
  loom_kernel_request_access_entry_t* entries;
  // Number of unique trace entries.
  iree_host_size_t count;
} loom_kernel_request_access_cache_t;

struct loom_kernel_request_source_t {
  // Callback-live boundary classifier over the materialized source module.
  loom_kernel_class_classifier_t classifier;
  // Existing source provider catalog, borrowed by canonical effect analysis.
  loom_template_provider_catalog_t providers;
  // Callback-live demand cache; its graphs never survive an individual query.
  loom_kernel_request_access_cache_t* storage_access;
};

static bool loom_kernel_request_access_entry_less(
    const loom_kernel_request_access_entry_t* lhs,
    const loom_kernel_request_access_entry_t* rhs) {
  return lhs->trace_id < rhs->trace_id;
}
LOOM_DEFINE_ADAPTIVE_SORT(loom_kernel_request_sort_access_entries,
                          loom_kernel_request_access_entry_t,
                          loom_kernel_request_access_entry_less)

static bool loom_kernel_request_binding_less(
    const loom_storage_access_binding_t* lhs,
    const loom_storage_access_binding_t* rhs) {
  return (uintptr_t)lhs->operation < (uintptr_t)rhs->operation;
}
LOOM_DEFINE_ADAPTIVE_SORT(loom_kernel_request_sort_bindings,
                          loom_storage_access_binding_t,
                          loom_kernel_request_binding_less)

iree_status_t loom_kernel_request_require_storage_access(
    const loom_kernel_request_t* request,
    const loom_storage_access_summary_t** out_summary) {
  *out_summary = NULL;
  loom_kernel_request_access_cache_t* cache = request->source->storage_access;
  const loom_kernel_class_collection_t* collection = request->collection;
  if (!cache->entries) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        cache->arena, collection->class_count, sizeof(*cache->entries),
        (void**)&cache->entries));
    for (uint32_t i = 0; i < collection->class_count; ++i) {
      cache->entries[i] = (loom_kernel_request_access_entry_t){
          .trace_id = collection->classes[i].trace_id};
    }
    loom_kernel_request_sort_access_entries(cache->entries,
                                            collection->class_count);
    for (uint32_t i = 0; i < collection->class_count; ++i) {
      if (cache->count == 0 || cache->entries[cache->count - 1].trace_id !=
                                   cache->entries[i].trace_id) {
        cache->entries[cache->count++] = cache->entries[i];
      }
    }
  }
  const loom_kernel_class_trace_id_t trace_id =
      collection->classes[request->class_ordinal].trace_id;
  iree_host_size_t begin = 0, end = cache->count;
  while (begin < end) {
    const iree_host_size_t middle = begin + (end - begin) / 2;
    if (cache->entries[middle].trace_id < trace_id) {
      begin = middle + 1;
    } else {
      end = middle;
    }
  }
  loom_kernel_request_access_entry_t* entry = &cache->entries[begin];
  if (entry->summary) {
    *out_summary = entry->summary;
    return iree_ok_status();
  }

  const loom_kernel_class_classifier_t* classifier =
      &request->source->classifier;
  loom_module_t* module = (loom_module_t*)classifier->module;
  iree_arena_allocator_t arena;
  iree_arena_initialize(cache->block_pool, &arena);
  loom_storage_access_scope_t scope;
  loom_storage_access_scope_initialize(module, &arena, &scope);
  scope.providers = &request->source->providers;
  loom_storage_access_binding_t* bindings = NULL;
  iree_status_t status =
      iree_arena_allocate_array(&arena, collection->accepted_decision_count,
                                sizeof(*bindings), (void**)&bindings);
  for (loom_kernel_class_trace_id_t current = trace_id;
       iree_status_is_ok(status) &&
       current != LOOM_KERNEL_CLASS_TRACE_ID_INVALID;
       current = collection->traces[current].parent_trace_id) {
    const loom_kernel_class_trace_t* trace = &collection->traces[current];
    const loom_kernel_class_decision_t* decision =
        &classifier->decisions[trace->decision_ordinal];
    const loom_template_provider_summary_t* provider =
        loom_template_decision_model_provider(decision->model,
                                              trace->action_ordinal);
    if (provider->module != module || !loom_func_like_isa(provider->function) ||
        loom_func_like_is_kernel(provider->function) ||
        loom_func_like_repr_contract(provider->function) !=
            LOOM_STRING_ID_INVALID ||
        !loom_func_like_body(provider->function)) {
      continue;
    }
    const loom_op_t* operation = decision->demand->apply_op;
    const loom_value_slice_t arguments =
        loom_template_apply_operands(operation);
    bindings[scope.binding_count++] = (loom_storage_access_binding_t){
        .operation = operation,
        .function = provider->function,
        .argument_offset =
            (uint16_t)(arguments.values - loom_op_const_operands(operation)),
        .argument_count = arguments.count,
    };
  }
  loom_local_value_domain_t domain = {0};
  if (iree_status_is_ok(status)) {
    loom_kernel_request_sort_bindings(bindings, scope.binding_count);
    scope.bindings = bindings;
    const loom_func_like_t kernel = loom_func_like_cast(
        module,
        module->symbols.entries[classifier->kernel_symbol_id].defining_op);
    status = loom_local_value_domain_acquire_for_region_tree(
        module, loom_func_like_body(kernel), &arena, &domain);
    if (iree_status_is_ok(status)) {
      status = loom_storage_access_require_summary(
          &scope, kernel, &domain, cache->arena, &entry->summary);
    }
    loom_local_value_domain_release(&domain);
  }
  iree_arena_deinitialize(&arena);
  if (iree_status_is_ok(status)) {
    *out_summary = entry->summary;
  }
  return status;
}

struct loom_kernel_request_producer_t {
  // Immutable provider-backed source universe.
  const loom_link_module_index_t* index;

  // Host allocator used for producer storage.
  iree_allocator_t allocator;

  // Lazy provider-header cache shared by every kernel publication.
  loom_link_template_candidate_loader_t* candidate_loader;
};

iree_status_t loom_kernel_request_producer_allocate(
    const loom_link_module_index_t* index,
    const loom_link_plan_materialization_environment_t* environment,
    loom_kernel_request_producer_t** out_producer) {
  IREE_ASSERT_ARGUMENT(index);
  IREE_ASSERT_ARGUMENT(environment);
  IREE_ASSERT_ARGUMENT(environment->context);
  IREE_ASSERT_ARGUMENT(environment->block_pool);
  IREE_ASSERT_ARGUMENT(out_producer);
  *out_producer = NULL;

  loom_kernel_request_producer_t* producer = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      environment->allocator, sizeof(*producer), (void**)&producer));
  *producer = (loom_kernel_request_producer_t){
      .index = index,
      .allocator = environment->allocator,
  };
  iree_status_t status = loom_link_template_candidate_loader_allocate(
      index, environment, &producer->candidate_loader);
  if (iree_status_is_ok(status)) {
    *out_producer = producer;
  } else {
    loom_kernel_request_producer_free(producer);
  }
  return status;
}

void loom_kernel_request_producer_free(
    loom_kernel_request_producer_t* producer) {
  if (producer == NULL) {
    return;
  }
  const iree_allocator_t allocator = producer->allocator;
  loom_link_template_candidate_loader_free(producer->candidate_loader);
  iree_allocator_free(allocator, producer);
}

static iree_status_t loom_kernel_request_resolve_target(
    const loom_module_t* module, loom_func_like_t kernel,
    loom_symbol_fact_table_t* symbol_facts,
    loom_template_applicability_target_t* out_target) {
  *out_target = (loom_template_applicability_target_t){
      .witness = loom_func_like_target(kernel),
  };
  if (!loom_symbol_ref_is_valid(out_target->witness)) {
    return iree_ok_status();
  }
  const loom_symbol_facts_base_t* base_facts = NULL;
  IREE_RETURN_IF_ERROR(loom_symbol_fact_table_lookup_ref(
      symbol_facts, module, out_target->witness, &base_facts));
  const loom_target_symbol_facts_t* target_facts =
      loom_target_symbol_facts_cast(base_facts);
  // A target declaration carries identity but no device facts until the
  // consumer specializes the published request. Keep its witness so template
  // classification can preserve unresolved target-dependent alternatives.
  out_target->facts = target_facts ? target_facts->projection : NULL;
  return iree_ok_status();
}

static iree_status_t loom_kernel_request_source_prepare(
    loom_kernel_request_producer_t* producer,
    const loom_link_plan_materialization_environment_t* environment,
    iree_host_size_t source_symbol_ordinal,
    iree_arena_allocator_t* scratch_arena,
    loom_kernel_request_source_t* out_source,
    loom_link_index_materialization_t* out_materialization) {
  *out_source = (loom_kernel_request_source_t){0};
  *out_materialization = (loom_link_index_materialization_t){0};
  const loom_link_module_index_symbol_t* source_symbol =
      loom_link_module_index_symbol_at(producer->index, source_symbol_ordinal);
  const uint8_t facet_count = source_symbol->facets.schema.facet_count;
  loom_link_plan_root_facet_t* root_facets = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, facet_count, sizeof(*root_facets), (void**)&root_facets));
  for (uint8_t i = 0; i < facet_count; ++i) {
    root_facets[i] = (loom_link_plan_root_facet_t){
        .symbol_ordinal = source_symbol_ordinal,
        .kind = loom_link_module_index_symbol_facet_kind_at(source_symbol, i),
    };
  }
  const loom_link_plan_options_t link_options = {
      .mode = LOOM_LINK_PLAN_LINK,
      .unresolved_policy = LOOM_LINK_PLAN_UNRESOLVED_ALLOW,
      .root_facets =
          {
              .count = facet_count,
              .values = root_facets,
          },
      .dependency_policy = LOOM_LINK_PLAN_DEPENDENCY_REQUESTED_FACETS,
  };
  loom_link_index_materialization_t materialization = {0};
  iree_status_t status = loom_link_index_materialize(
      producer->index, &link_options, environment,
      IREE_SV("kernel_request_source"), &materialization);

  loom_symbol_ref_t kernel_ref = loom_symbol_ref_null();
  loom_func_like_t kernel = {0};
  if (iree_status_is_ok(status)) {
    IREE_ASSERT_LT(source_symbol_ordinal,
                   materialization.product.target_symbols.count);
    kernel_ref =
        materialization.product.target_symbols.values[source_symbol_ordinal];
    IREE_ASSERT(loom_symbol_ref_is_valid(kernel_ref));
    IREE_ASSERT_EQ(kernel_ref.module_id, 0u);
    IREE_ASSERT_LT(kernel_ref.symbol_id,
                   materialization.product.module->symbols.count);
    kernel = loom_func_like_cast(
        materialization.product.module,
        materialization.product.module->symbols.entries[kernel_ref.symbol_id]
            .defining_op);
    IREE_ASSERT(loom_func_like_body(kernel) != NULL);
  }

  loom_template_provider_slice_t external_candidates =
      loom_template_provider_slice_empty();
  if (iree_status_is_ok(status)) {
    status = loom_link_template_candidate_loader_load(
        producer->candidate_loader, materialization.plan,
        &materialization.product, scratch_arena, &external_candidates);
  }

  loom_symbol_reference_table_t references = {0};
  if (iree_status_is_ok(status)) {
    status = loom_symbol_reference_table_build(materialization.product.module,
                                               scratch_arena, &references);
  }
  loom_symbol_fact_table_t symbol_facts = {0};
  loom_symbol_fact_table_initialize(&symbol_facts, scratch_arena);
  loom_template_provider_catalog_t* providers = &out_source->providers;
  loom_template_provider_catalog_initialize(providers, scratch_arena);
  if (iree_status_is_ok(status)) {
    status = loom_template_provider_catalog_build(
        providers, materialization.product.module, &symbol_facts,
        external_candidates.providers, external_candidates.count);
  }
  loom_template_decision_model_catalog_t decision_models = {0};
  if (iree_status_is_ok(status)) {
    status = loom_template_decision_model_catalog_build(
        materialization.product.module, &symbol_facts, &references, providers,
        scratch_arena, &decision_models);
  }

  loom_template_applicability_target_t kernel_target = {0};
  if (iree_status_is_ok(status)) {
    status = loom_kernel_request_resolve_target(
        materialization.product.module, kernel, &symbol_facts, &kernel_target);
  }
  loom_value_fact_table_t kernel_facts = {0};
  if (iree_status_is_ok(status)) {
    status = loom_value_fact_table_initialize(
        &kernel_facts, scratch_arena,
        materialization.product.module->values.count);
  }
  if (iree_status_is_ok(status)) {
    loom_type_registry_configure_fact_context(&kernel_facts.context);
    // Query operands and direct target requirements share the bound target.
    kernel_facts.context.target_facts = kernel_target.facts;
    status = loom_value_fact_table_compute(
        &kernel_facts, materialization.product.module, kernel);
  }
  loom_symbolic_expr_context_t expression_context = {0};
  if (iree_status_is_ok(status)) {
    loom_symbolic_expr_context_initialize(materialization.product.module,
                                          /*value_domain=*/NULL, &kernel_facts,
                                          scratch_arena, &expression_context);
  }
  if (iree_status_is_ok(status)) {
    status = loom_kernel_class_classifier_build(
        materialization.product.module, kernel_ref.symbol_id, &references,
        &decision_models, &kernel_facts, &expression_context, &kernel_target,
        scratch_arena, &out_source->classifier);
  }

  if (iree_status_is_ok(status)) {
    *out_materialization = materialization;
    materialization = (loom_link_index_materialization_t){0};
  }
  loom_link_index_materialization_deinitialize(&materialization);
  return status;
}

iree_status_t loom_kernel_request_materialize(
    const loom_kernel_request_t* request, iree_arena_block_pool_t* block_pool,
    iree_allocator_t allocator, loom_kernel_class_product_t* out_product) {
  IREE_ASSERT_ARGUMENT(request);
  IREE_ASSERT_ARGUMENT(request->source);
  IREE_ASSERT_ARGUMENT(request->collection);
  IREE_ASSERT_LT(request->class_ordinal, request->collection->class_count);
  IREE_ASSERT_ARGUMENT(block_pool);
  IREE_ASSERT_ARGUMENT(out_product);
  return loom_kernel_class_materialize(
      &request->source->classifier, request->collection, request->class_ordinal,
      block_pool, allocator, out_product);
}

iree_status_t loom_kernel_request_producer_publish(
    loom_kernel_request_producer_t* producer,
    const loom_link_plan_materialization_environment_t* environment,
    iree_host_size_t source_symbol_ordinal, const loom_module_t* site_module,
    const loom_kernel_class_site_t* sites, iree_host_size_t site_count,
    const loom_kernel_class_collection_options_t* collection_options,
    loom_kernel_request_sink_t sink, iree_arena_allocator_t* scratch_arena,
    loom_kernel_class_collection_t* out_collection) {
  IREE_ASSERT_ARGUMENT(producer);
  IREE_ASSERT_ARGUMENT(environment);
  IREE_ASSERT_ARGUMENT(sites);
  IREE_ASSERT_GT(site_count, 0u);
  IREE_ASSERT_ARGUMENT(collection_options);
  IREE_ASSERT_ARGUMENT(sink.publish);
  IREE_ASSERT_ARGUMENT(scratch_arena);
  IREE_ASSERT_ARGUMENT(out_collection);
  *out_collection = (loom_kernel_class_collection_t){0};

  loom_kernel_request_source_t source = {0};
  loom_link_index_materialization_t materialization = {0};
  iree_status_t status = loom_kernel_request_source_prepare(
      producer, environment, source_symbol_ordinal, scratch_arena, &source,
      &materialization);
  loom_kernel_class_collection_t collection = {0};
  loom_kernel_request_access_cache_t storage_access = {
      .arena = scratch_arena,
      .block_pool = environment->block_pool,
  };
  source.storage_access = &storage_access;
  if (iree_status_is_ok(status)) {
    status = loom_kernel_class_classifier_collect(
        &source.classifier, site_module, sites, site_count, collection_options,
        scratch_arena, &collection);
  }

  for (loom_decision_class_ordinal_t class_ordinal = 0;
       class_ordinal < collection.class_count && iree_status_is_ok(status);
       ++class_ordinal) {
    const loom_kernel_request_t request = {
        .source_symbol_ordinal = source_symbol_ordinal,
        .class_ordinal = class_ordinal,
        .member_count = collection.classes[class_ordinal].member_count,
        .source = &source,
        .collection = &collection,
    };
    status = sink.publish(sink.user_data, &request);
  }

  if (iree_status_is_ok(status)) {
    *out_collection = collection;
  }
  loom_link_index_materialization_deinitialize(&materialization);
  return status;
}

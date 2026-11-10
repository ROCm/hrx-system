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
#include "loom/link/planner.h"
#include "loom/link/template_candidate_loader.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/target/facts.h"
#include "loom/ops/type_registry.h"
#include "loom/transforms/symbol/template_decision_model.h"
#include "loom/util/fact_table.h"

struct loom_kernel_request_source_t {
  // Callback-live boundary classifier over the materialized source module.
  const loom_kernel_class_classifier_t* classifier;

  // Invocation-local classes and accepted decisions, borrowed during
  // publication.
  const loom_kernel_class_collection_t* collection;
};

struct loom_kernel_request_producer_t {
  // Immutable provider-backed source universe.
  const loom_link_module_index_t* index;

  // Host allocator used for producer storage.
  iree_allocator_t allocator;

  // Lazy provider-header cache shared by every kernel publication.
  loom_link_template_candidate_loader_t* candidate_loader;
};

struct loom_kernel_request_preparation_t {
  // Host allocator owning this preparation and its module metadata.
  iree_allocator_t allocator;

  // Source identity within the immutable provider index.
  iree_host_size_t source_symbol_ordinal;

  // Owned header loader whose immutable summaries back the classifier.
  loom_kernel_request_producer_t* producer;

  // Owned linked module, detached from its temporary materialization plan.
  loom_module_t* module;

  // Classifier analyses and configuration-key array backing the source view.
  iree_arena_allocator_t arena;

  // Read-only boundary classifier shared by concurrent class collections.
  loom_kernel_class_classifier_t classifier;

  // Unique names of config symbols selected by the successful source closure.
  iree_string_view_list_t config_keys;
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
    loom_kernel_class_classifier_t* out_classifier,
    loom_link_index_materialization_t* out_materialization) {
  *out_classifier = (loom_kernel_class_classifier_t){0};
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
  iree_arena_allocator_t construction_arena;
  iree_arena_initialize(scratch_arena->block_pool, &construction_arena);
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
    status = loom_symbol_reference_table_build(
        materialization.product.module, &construction_arena, &references);
  }
  loom_symbol_fact_table_t symbol_facts = {0};
  loom_symbol_fact_table_initialize(&symbol_facts, scratch_arena);
  loom_template_provider_catalog_t providers = {0};
  loom_template_provider_catalog_initialize(&providers, scratch_arena);
  if (iree_status_is_ok(status)) {
    status = loom_template_provider_catalog_build(
        &providers, materialization.product.module, &symbol_facts,
        external_candidates.providers, external_candidates.count);
  }
  loom_template_decision_model_catalog_t decision_models = {0};
  if (iree_status_is_ok(status)) {
    status = loom_template_decision_model_catalog_build(
        materialization.product.module, &symbol_facts, &references, &providers,
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
        &kernel_facts, &construction_arena,
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
                                          &construction_arena,
                                          &expression_context);
  }
  if (iree_status_is_ok(status)) {
    status = loom_kernel_class_classifier_build(
        materialization.product.module, kernel_ref.symbol_id, &references,
        &decision_models, &kernel_facts, &expression_context, &kernel_target,
        scratch_arena, out_classifier);
  }

  if (iree_status_is_ok(status)) {
    *out_materialization = materialization;
    materialization = (loom_link_index_materialization_t){0};
  }
  loom_link_index_materialization_deinitialize(&materialization);
  iree_arena_deinitialize(&construction_arena);
  return status;
}

iree_status_t loom_kernel_request_materialize(
    const loom_kernel_request_t* request, iree_arena_block_pool_t* block_pool,
    iree_allocator_t allocator, loom_kernel_class_product_t* out_product) {
  IREE_ASSERT_ARGUMENT(request);
  IREE_ASSERT_ARGUMENT(request->source);
  IREE_ASSERT_LT(request->class_ordinal,
                 request->source->collection->class_count);
  IREE_ASSERT_ARGUMENT(block_pool);
  IREE_ASSERT_ARGUMENT(out_product);
  return loom_kernel_class_materialize(
      request->source->classifier, request->source->collection,
      request->class_ordinal, block_pool, allocator, out_product);
}

static iree_status_t loom_kernel_request_source_publish(
    const loom_kernel_class_classifier_t* classifier,
    iree_host_size_t source_symbol_ordinal, const loom_module_t* site_module,
    const loom_kernel_class_site_t* sites, iree_host_size_t site_count,
    const loom_kernel_class_collection_options_t* collection_options,
    loom_kernel_request_sink_t sink, iree_arena_allocator_t* scratch_arena,
    const loom_decision_class_ordinal_t** out_site_classes) {
  IREE_ASSERT_ARGUMENT(classifier);
  IREE_ASSERT_ARGUMENT(sites);
  IREE_ASSERT_GT(site_count, 0u);
  IREE_ASSERT_ARGUMENT(collection_options);
  IREE_ASSERT_ARGUMENT(sink.publish);
  IREE_ASSERT_ARGUMENT(scratch_arena);
  IREE_ASSERT_ARGUMENT(out_site_classes);
  *out_site_classes = NULL;

  loom_kernel_class_collection_t collection = {0};
  iree_status_t status = loom_kernel_class_classifier_collect(
      classifier, site_module, sites, site_count, collection_options,
      scratch_arena, &collection);
  const loom_kernel_request_source_t source = {
      .classifier = classifier,
      .collection = &collection,
  };

  for (loom_decision_class_ordinal_t class_ordinal = 0;
       class_ordinal < collection.class_count && iree_status_is_ok(status);
       ++class_ordinal) {
    const loom_kernel_request_t request = {
        .source_symbol_ordinal = source_symbol_ordinal,
        .class_ordinal = class_ordinal,
        .member_count = collection.classes[class_ordinal].member_count,
        .source = &source,
    };
    status = sink.publish(sink.user_data, &request);
  }

  if (iree_status_is_ok(status)) {
    *out_site_classes = collection.site_classes;
  }
  return status;
}

iree_status_t loom_kernel_request_producer_publish(
    loom_kernel_request_producer_t* producer,
    const loom_link_plan_materialization_environment_t* environment,
    iree_host_size_t source_symbol_ordinal, const loom_module_t* site_module,
    const loom_kernel_class_site_t* sites, iree_host_size_t site_count,
    const loom_kernel_class_collection_options_t* collection_options,
    loom_kernel_request_sink_t sink, iree_arena_allocator_t* scratch_arena,
    const loom_decision_class_ordinal_t** out_site_classes) {
  *out_site_classes = NULL;
  loom_kernel_class_classifier_t classifier = {0};
  loom_link_index_materialization_t materialization = {0};
  iree_status_t status = loom_kernel_request_source_prepare(
      producer, environment, source_symbol_ordinal, scratch_arena, &classifier,
      &materialization);
  if (iree_status_is_ok(status)) {
    status = loom_kernel_request_source_publish(
        &classifier, source_symbol_ordinal, site_module, sites, site_count,
        collection_options, sink, scratch_arena, out_site_classes);
  }
  loom_link_index_materialization_deinitialize(&materialization);
  return status;
}

static iree_status_t loom_kernel_request_preparation_capture_config_keys(
    const loom_link_module_index_t* index,
    const loom_link_index_materialization_t* materialization,
    loom_kernel_request_preparation_t* preparation) {
  const loom_module_t* module = materialization->product.module;
  const iree_host_size_t selection_count =
      loom_link_plan_symbol_count(materialization->plan);
  iree_host_size_t count = 0;
  iree_host_size_t capacity = 0;
  loom_symbol_id_t* selected_ids = NULL;
  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(preparation->arena.block_pool, &scratch_arena);
  uint8_t* selected = NULL;
  iree_status_t status =
      iree_arena_allocate_array(&scratch_arena, module->symbols.count,
                                sizeof(*selected), (void**)&selected);
  if (iree_status_is_ok(status)) {
    memset(selected, 0, module->symbols.count * sizeof(*selected));
    for (iree_host_size_t i = 0;
         i < selection_count && iree_status_is_ok(status); ++i) {
      const loom_link_plan_symbol_t* selection =
          loom_link_plan_symbol_at(materialization->plan, i);
      const loom_link_module_index_symbol_t* symbol =
          loom_link_module_index_symbol_at(index, selection->symbol_ordinal);
      if (!iree_any_bit_set(symbol->flags, LOOM_LINK_SYMBOL_FLAG_CONFIG)) {
        continue;
      }
      const loom_symbol_ref_t target = materialization->product.target_symbols
                                           .values[selection->symbol_ordinal];
      if (selected[target.symbol_id]) {
        continue;
      }
      selected[target.symbol_id] = 1;
      status = iree_arena_grow_array(&scratch_arena, count, count + 1,
                                     sizeof(*selected_ids), &capacity,
                                     (void**)&selected_ids);
      if (iree_status_is_ok(status)) {
        selected_ids[count++] = target.symbol_id;
      }
    }
  }
  iree_string_view_t* keys = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_arena_allocate_array(&preparation->arena, count,
                                       sizeof(*keys), (void**)&keys);
  }
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < count; ++i) {
      keys[i] = loom_string_table_get(
          &module->strings, module->symbols.entries[selected_ids[i]].name_id);
    }
    preparation->config_keys =
        (iree_string_view_list_t){.count = count, .values = keys};
  }
  iree_arena_deinitialize(&scratch_arena);
  return status;
}

iree_status_t loom_kernel_request_preparation_allocate(
    const loom_link_module_index_t* index,
    const loom_link_plan_materialization_environment_t* environment,
    iree_host_size_t source_symbol_ordinal,
    loom_kernel_request_preparation_t** out_preparation) {
  *out_preparation = NULL;
  loom_kernel_request_preparation_t* preparation = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      environment->allocator, sizeof(*preparation), (void**)&preparation));
  *preparation = (loom_kernel_request_preparation_t){
      .allocator = environment->allocator,
      .source_symbol_ordinal = source_symbol_ordinal,
  };
  iree_arena_initialize(environment->block_pool, &preparation->arena);
  iree_status_t status = loom_kernel_request_producer_allocate(
      index, environment, &preparation->producer);
  loom_link_index_materialization_t materialization = {0};
  if (iree_status_is_ok(status)) {
    status = loom_kernel_request_source_prepare(
        preparation->producer, environment, source_symbol_ordinal,
        &preparation->arena, &preparation->classifier, &materialization);
  }
  if (iree_status_is_ok(status)) {
    status = loom_kernel_request_preparation_capture_config_keys(
        index, &materialization, preparation);
  }
  if (iree_status_is_ok(status)) {
    preparation->module = materialization.product.module;
    materialization.product.module = NULL;
    *out_preparation = preparation;
    preparation = NULL;
  }
  loom_link_index_materialization_deinitialize(&materialization);
  loom_kernel_request_preparation_free(preparation);
  return status;
}

void loom_kernel_request_preparation_free(
    loom_kernel_request_preparation_t* preparation) {
  if (preparation == NULL) {
    return;
  }
  const iree_allocator_t allocator = preparation->allocator;
  loom_module_free(preparation->module);
  iree_arena_deinitialize(&preparation->arena);
  loom_kernel_request_producer_free(preparation->producer);
  iree_allocator_free(allocator, preparation);
}

iree_string_view_list_t loom_kernel_request_preparation_config_keys(
    const loom_kernel_request_preparation_t* preparation) {
  return preparation->config_keys;
}

iree_status_t loom_kernel_request_preparation_publish(
    const loom_kernel_request_preparation_t* preparation,
    const loom_module_t* site_module, const loom_kernel_class_site_t* sites,
    iree_host_size_t site_count,
    const loom_kernel_class_collection_options_t* collection_options,
    loom_kernel_request_sink_t sink, iree_arena_allocator_t* scratch_arena,
    const loom_decision_class_ordinal_t** out_site_classes) {
  return loom_kernel_request_source_publish(
      &preparation->classifier, preparation->source_symbol_ordinal, site_module,
      sites, site_count, collection_options, sink, scratch_arena,
      out_site_classes);
}

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
#include "loom/ops/type_registry.h"
#include "loom/transforms/symbol/template_decision_model.h"
#include "loom/util/fact_table.h"

struct loom_kernel_request_producer_t {
  // Immutable provider-backed source universe.
  const loom_link_module_index_t* index;

  // Copied environment borrowing all referenced materialization resources.
  loom_link_plan_materialization_environment_t environment;

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
      .environment = *environment,
  };
  iree_status_t status = loom_link_template_candidate_loader_allocate(
      index, &producer->environment, &producer->candidate_loader);
  if (iree_status_is_ok(status)) {
    *out_producer = producer;
  } else {
    iree_allocator_free(environment->allocator, producer);
  }
  return status;
}

void loom_kernel_request_producer_free(
    loom_kernel_request_producer_t* producer) {
  if (producer == NULL) return;
  const iree_allocator_t allocator = producer->environment.allocator;
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
  IREE_ASSERT(target_facts != NULL);
  out_target->facts = target_facts->projection;
  return iree_ok_status();
}

iree_status_t loom_kernel_request_producer_publish(
    loom_kernel_request_producer_t* producer,
    iree_host_size_t source_symbol_ordinal,
    const loom_kernel_class_site_t* sites, iree_host_size_t site_count,
    const loom_kernel_class_collection_options_t* collection_options,
    loom_kernel_request_sink_t sink, iree_arena_allocator_t* scratch_arena,
    loom_kernel_class_collection_t* out_collection) {
  IREE_ASSERT_ARGUMENT(producer);
  IREE_ASSERT_ARGUMENT(sites);
  IREE_ASSERT_GT(site_count, 0u);
  IREE_ASSERT_ARGUMENT(collection_options);
  IREE_ASSERT_ARGUMENT(sink.publish);
  IREE_ASSERT_ARGUMENT(scratch_arena);
  IREE_ASSERT_ARGUMENT(out_collection);
  *out_collection = (loom_kernel_class_collection_t){0};

  const loom_link_plan_root_facet_t root_facets[] = {
      {
          .symbol_ordinal = source_symbol_ordinal,
          .kind = LOOM_LINK_SYMBOL_FACET_KERNEL_CONTRACT,
      },
      {
          .symbol_ordinal = source_symbol_ordinal,
          .kind = LOOM_LINK_SYMBOL_FACET_KERNEL_CONFIGURATION,
      },
      {
          .symbol_ordinal = source_symbol_ordinal,
          .kind = LOOM_LINK_SYMBOL_FACET_KERNEL_IMPLEMENTATION,
      },
  };
  const loom_link_plan_options_t link_options = {
      .mode = LOOM_LINK_PLAN_LINK,
      .unresolved_policy = LOOM_LINK_PLAN_UNRESOLVED_ALLOW,
      .root_facets =
          {
              .count = IREE_ARRAYSIZE(root_facets),
              .values = root_facets,
          },
      .dependency_policy = LOOM_LINK_PLAN_DEPENDENCY_REQUESTED_FACETS,
  };
  loom_link_index_materialization_t materialization = {0};
  iree_status_t status = loom_link_index_materialize(
      producer->index, &link_options, &producer->environment,
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
    IREE_ASSERT(loom_func_like_is_kernel_entry(kernel));
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

  loom_value_fact_table_t kernel_facts = {0};
  if (iree_status_is_ok(status)) {
    status = loom_value_fact_table_initialize(
        &kernel_facts, scratch_arena,
        materialization.product.module->values.count);
  }
  if (iree_status_is_ok(status)) {
    loom_type_registry_configure_fact_context(&kernel_facts.context);
    status = loom_value_fact_table_compute(
        &kernel_facts, materialization.product.module, kernel);
  }
  loom_symbolic_expr_context_t expression_context = {0};
  if (iree_status_is_ok(status)) {
    loom_symbolic_expr_context_initialize(materialization.product.module,
                                          &kernel_facts, scratch_arena,
                                          &expression_context);
  }
  loom_template_applicability_target_t kernel_target = {0};
  if (iree_status_is_ok(status)) {
    status = loom_kernel_request_resolve_target(
        materialization.product.module, kernel, &symbol_facts, &kernel_target);
  }

  loom_kernel_class_classifier_t classifier = {0};
  if (iree_status_is_ok(status)) {
    status = loom_kernel_class_classifier_build(
        materialization.product.module, kernel_ref.symbol_id, &references,
        &decision_models, &kernel_facts, &expression_context, &kernel_target,
        scratch_arena, &classifier);
  }
  loom_kernel_class_collection_t collection = {0};
  if (iree_status_is_ok(status)) {
    status = loom_kernel_class_classifier_collect(
        &classifier, sites, site_count, collection_options, scratch_arena,
        &collection);
  }

  for (loom_decision_class_ordinal_t class_ordinal = 0;
       class_ordinal < collection.class_count && iree_status_is_ok(status);
       ++class_ordinal) {
    loom_kernel_class_product_t product = {0};
    status = loom_kernel_class_materialize(
        &classifier, &collection, class_ordinal,
        producer->environment.block_pool, producer->environment.allocator,
        &product);
    if (iree_status_is_ok(status)) {
      const loom_kernel_request_t request = {
          .source_symbol_ordinal = source_symbol_ordinal,
          .class_ordinal = class_ordinal,
          .member_count = collection.classes[class_ordinal].member_count,
          .product = product,
      };
      product = (loom_kernel_class_product_t){0};
      status = sink.publish(sink.user_data, request);
    }
    loom_kernel_class_product_deinitialize(&product);
  }

  if (iree_status_is_ok(status)) {
    *out_collection = collection;
  }
  loom_link_index_materialization_deinitialize(&materialization);
  return status;
}

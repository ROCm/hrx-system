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
#include "loom/rewrite/module_projection.h"
#include "loom/transforms/symbol/template_decision_model.h"
#include "loom/util/fact_table.h"
#include "loom/util/segmented_storage.h"

struct loom_kernel_request_source_t {
  // Callback-live boundary classifier over the materialized source module.
  const loom_kernel_class_classifier_t* classifier;

  // Invocation-local classes and accepted decisions, borrowed during
  // publication.
  const loom_kernel_class_collection_t* collection;

  // Publication-wide ordinal of this tuple group's first class.
  uint32_t class_base;
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

  // Complete provider models retained before staged inputs establish
  // applicability.
  loom_template_decision_model_catalog_t models;

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
    loom_template_decision_model_catalog_t* out_models,
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
  bool accepted = materialization.product.module != NULL;

  loom_symbol_ref_t kernel_ref = loom_symbol_ref_null();
  loom_func_like_t kernel = {0};
  if (iree_status_is_ok(status) && accepted) {
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
  if (iree_status_is_ok(status) && accepted) {
    status = loom_link_template_candidate_loader_load(
        producer->candidate_loader, materialization.plan,
        &materialization.product, environment->diagnostics, scratch_arena,
        &external_candidates, &accepted);
  }

  loom_symbol_reference_table_t references = {0};
  if (iree_status_is_ok(status) && accepted) {
    status = loom_symbol_reference_table_build(
        materialization.product.module, &construction_arena, &references);
  }
  loom_symbol_fact_table_t symbol_facts = {0};
  loom_symbol_fact_table_initialize(&symbol_facts, scratch_arena);
  loom_template_provider_catalog_t providers = {0};
  loom_template_provider_catalog_initialize(&providers, scratch_arena);
  if (iree_status_is_ok(status) && accepted) {
    status = loom_template_provider_catalog_build(
        &providers, materialization.product.module, &symbol_facts,
        external_candidates.providers, external_candidates.count);
  }
  loom_template_decision_model_catalog_t decision_models = {0};
  if (iree_status_is_ok(status) && accepted) {
    status = loom_template_decision_model_catalog_build(
        materialization.product.module, &symbol_facts, &references, &providers,
        scratch_arena, &decision_models);
  }

  loom_template_applicability_target_t kernel_target = {0};
  if (iree_status_is_ok(status) && accepted) {
    status = loom_kernel_request_resolve_target(
        materialization.product.module, kernel, &symbol_facts, &kernel_target);
  }
  loom_value_fact_table_t kernel_facts = {0};
  if (iree_status_is_ok(status) && accepted) {
    status = loom_value_fact_table_initialize(
        &kernel_facts, &construction_arena,
        materialization.product.module->values.count);
  }
  if (iree_status_is_ok(status) && accepted) {
    loom_type_registry_configure_fact_context(&kernel_facts.context);
    // Query operands and direct target requirements share the bound target.
    kernel_facts.context.target_facts = kernel_target.facts;
    status = loom_value_fact_table_compute(
        &kernel_facts, materialization.product.module, kernel);
  }
  loom_symbolic_expr_context_t expression_context = {0};
  if (iree_status_is_ok(status) && accepted) {
    loom_symbolic_expr_context_initialize(materialization.product.module,
                                          /*value_domain=*/NULL, &kernel_facts,
                                          &construction_arena,
                                          &expression_context);
  }
  if (iree_status_is_ok(status) && accepted) {
    status = loom_kernel_class_classifier_build(
        materialization.product.module, kernel_ref.symbol_id, &references,
        &decision_models, &kernel_facts, &expression_context, &kernel_target,
        scratch_arena, out_classifier);
  }

  if (iree_status_is_ok(status) && accepted) {
    *out_models = decision_models;
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
  IREE_ASSERT_LT(request->class_ordinal - request->source->class_base,
                 request->source->collection->class_count);
  IREE_ASSERT_ARGUMENT(block_pool);
  IREE_ASSERT_ARGUMENT(out_product);
  return loom_kernel_class_materialize(
      request->source->classifier, request->source->collection,
      request->class_ordinal - request->source->class_base, block_pool,
      allocator, out_product);
}

// Invocation-local translation of immutable models through an exact module
// clone. Compiled predicates and ranking are shared; no provider is
// re-discovered.
static iree_status_t loom_kernel_request_project_models(
    const loom_template_decision_model_catalog_t* source,
    loom_ir_module_projection_t* projection, iree_arena_allocator_t* arena,
    loom_template_decision_model_catalog_t* target) {
  *target = *source;
  target->module = projection->target_module;
  loom_template_decision_model_t* models = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, source->model_count, sizeof(*models), (void**)&models));
  target->models = models;
  for (iree_host_size_t i = 0; i < source->model_count; ++i) {
    const loom_template_decision_model_t* original = &source->models[i];
    loom_template_decision_model_t* model = &models[i];
    *model = *original;
    model->module = target->module;
    model->family = loom_ir_module_projection_target_symbol(
        projection, original->family.symbol_id);
    loom_template_provider_summary_t* providers = NULL;
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(arena, original->providers.count,
                                  sizeof(*providers), (void**)&providers));
    model->providers.providers = providers;
    for (iree_host_size_t j = 0; j < original->providers.count; ++j) {
      const loom_template_provider_summary_t* provider =
          &original->providers.providers[j];
      providers[j] = *provider;
      providers[j].module = target->module;
      providers[j].family = model->family;
      if (loom_symbol_ref_is_valid(provider->symbol) &&
          provider->symbol.module_id == 0) {
        providers[j].symbol = loom_ir_module_projection_target_symbol(
            projection, provider->symbol.symbol_id);
      }
      if (loom_symbol_ref_is_valid(provider->target_symbol) &&
          provider->target_symbol.module_id == 0) {
        providers[j].target_symbol = loom_ir_module_projection_target_symbol(
            projection, provider->target_symbol.symbol_id);
      }
    }
    if (original->program.feature_count) {
      loom_template_decision_feature_t* features = NULL;
      IREE_RETURN_IF_ERROR(
          iree_arena_allocate_array(arena, original->program.feature_count,
                                    sizeof(*features), (void**)&features));
      model->features = features;
      for (uint32_t j = 0; j < original->program.feature_count; ++j) {
        features[j] = original->features[j];
        if (features[j].kind == LOOM_TEMPLATE_DECISION_FEATURE_INPUT &&
            features[j].value.input.axis == UINT8_MAX &&
            !features[j].value.input.expected_is_argument &&
            features[j].value.input.expected) {
          uint16_t encoding;
          IREE_RETURN_IF_ERROR(loom_ir_remap_encoding_id(
              &projection->remap, (uint16_t)features[j].value.input.expected,
              &encoding));
          features[j].value.input.expected = encoding;
        }

        if (features[j].kind ==
                LOOM_TEMPLATE_DECISION_FEATURE_TARGET_IDENTITY &&
            features[j].value.target_identity.module == source->module) {
          features[j].value.target_identity.module = target->module;
          const loom_symbol_ref_t symbol =
              features[j].value.target_identity.target_symbol;
          if (loom_symbol_ref_is_valid(symbol) && symbol.module_id == 0) {
            features[j].value.target_identity.target_symbol =
                loom_ir_module_projection_target_symbol(projection,
                                                        symbol.symbol_id);
          }
        }
      }
    }
  }
  return iree_ok_status();
}

// Clone-produced correspondence returns bound projections to the shared source.
typedef struct {
  // Source identities in pooled segments indexed by dense cloned value ID.
  // Erased source IDs do not reserve holes in this clone-owned domain.
  loom_segmented_storage_t source_values;
  // Invocation-owned backing for observed correspondence only.
  iree_arena_allocator_t* arena;
} loom_kernel_request_correspondence_t;

enum { LOOM_KERNEL_REQUEST_CORRESPONDENCE_SEGMENT_COUNT = 256 };

static iree_status_t loom_kernel_request_record_value(
    loom_kernel_request_correspondence_t* correspondence,
    loom_value_id_t source, loom_value_id_t target) {
  const uint32_t segment_index =
      target / LOOM_KERNEL_REQUEST_CORRESPONDENCE_SEGMENT_COUNT;
  while (correspondence->source_values.segment_count <= segment_index) {
    void* segment = NULL;
    IREE_RETURN_IF_ERROR(loom_segmented_storage_append(
        &correspondence->source_values, correspondence->arena, &segment));
    memset(segment, 0xFF, correspondence->source_values.segment_size);
  }
  loom_value_id_t* segment = loom_segmented_storage_segment(
      &correspondence->source_values, segment_index);
  segment[target % LOOM_KERNEL_REQUEST_CORRESPONDENCE_SEGMENT_COUNT] = source;
  return iree_ok_status();
}

static iree_status_t loom_kernel_request_record_values(void* user_data,
                                                       const loom_op_t* source,
                                                       loom_op_t* target) {
  loom_kernel_request_correspondence_t* correspondence = user_data;
  for (uint16_t i = 0; i < source->region_count; ++i) {
    const loom_region_t* source_region = loom_op_regions(source)[i];
    const loom_region_t* target_region = loom_op_regions(target)[i];
    for (uint16_t j = 0; j < source_region->block_count; ++j) {
      const loom_block_t* source_block =
          loom_region_const_block(source_region, j);
      const loom_block_t* target_block =
          loom_region_const_block(target_region, j);
      for (uint16_t k = 0; k < source_block->arg_count; ++k) {
        IREE_RETURN_IF_ERROR(loom_kernel_request_record_value(
            correspondence, source_block->arg_ids[k],
            target_block->arg_ids[k]));
      }
    }
  }
  for (uint16_t i = 0; i < source->result_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_kernel_request_record_value(
        correspondence, loom_op_const_results(source)[i],
        loom_op_const_results(target)[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_kernel_request_bind_classifier(
    const loom_kernel_class_classifier_t* common,
    const loom_template_decision_model_catalog_t* common_models,
    const loom_module_t* site_module,
    const loom_kernel_specialization_value_t* tuple,
    iree_arena_allocator_t* arena, iree_arena_allocator_t* temporary,
    loom_module_t** out_bound, loom_kernel_class_classifier_t* out_classifier) {
  const uint16_t prefix_count = common->specialization_count;
  loom_kernel_request_correspondence_t correspondence = {.arena = temporary};
  loom_segmented_storage_initialize(
      LOOM_KERNEL_REQUEST_CORRESPONDENCE_SEGMENT_COUNT *
          sizeof(loom_value_id_t),
      iree_alignof(loom_value_id_t), &correspondence.source_values);
  loom_ir_remap_op_projection_t* applications = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(temporary, common->decision_count,
                                sizeof(*applications), (void**)&applications));
  for (uint32_t i = 0; i < common->decision_count; ++i) {
    applications[i] = (loom_ir_remap_op_projection_t){
        .source_op = common->decisions[i].application};
  }
  const loom_ir_module_clone_options_t clone_options = {
      .clone_observer = {.fn = loom_kernel_request_record_values,
                         .user_data = &correspondence},
      .operations = {.entries = applications, .count = common->decision_count},
  };
  loom_ir_module_projection_t projection = {0};
  IREE_RETURN_IF_ERROR(loom_ir_module_clone(
      common->module, &clone_options, temporary->block_pool, temporary,
      temporary->block_pool->block_allocator, &projection, out_bound));
  loom_module_t* bound = *out_bound;
  const iree_host_size_t cloned_value_count = bound->values.count;
  const loom_func_like_t kernel = loom_func_like_cast(
      bound, bound->symbols.entries[common->kernel_symbol_id].defining_op);
  loom_rewriter_t rewriter = {0};
  loom_rewriter_initialize(&rewriter, bound, temporary);
  iree_status_t status = loom_kernel_specializations_apply(
      site_module, tuple, prefix_count, kernel, &rewriter, temporary);
  loom_rewriter_deinitialize(&rewriter);
  IREE_RETURN_IF_ERROR(status);
  loom_symbol_reference_table_t references = {0};
  IREE_RETURN_IF_ERROR(
      loom_symbol_reference_table_build(bound, temporary, &references));
  loom_symbol_fact_table_t symbol_facts = {0};
  loom_symbol_fact_table_initialize(&symbol_facts, temporary);
  loom_template_decision_model_catalog_t models = {0};
  IREE_RETURN_IF_ERROR(loom_kernel_request_project_models(
      common_models, &projection, temporary, &models));
  loom_template_applicability_target_t target = {0};
  IREE_RETURN_IF_ERROR(loom_kernel_request_resolve_target(
      bound, kernel, &symbol_facts, &target));
  loom_value_fact_table_t facts = {0};
  IREE_RETURN_IF_ERROR(loom_value_fact_table_initialize(&facts, temporary, 0));
  loom_type_registry_configure_fact_context(&facts.context);
  facts.context.target_facts = target.facts;
  IREE_RETURN_IF_ERROR(loom_value_fact_table_compute(&facts, bound, kernel));
  loom_symbolic_expr_context_t expressions = {0};
  loom_symbolic_expr_context_initialize(bound, NULL, &facts, temporary,
                                        &expressions);
  IREE_RETURN_IF_ERROR(loom_kernel_class_classifier_build(
      bound, common->kernel_symbol_id, &references, &models, &facts,
      &expressions, &target, arena, out_classifier));
  IREE_ASSERT(out_classifier->decision_count == common->decision_count);
  for (uint32_t i = 0; i < common->decision_count; ++i) {
    loom_kernel_class_decision_t* decision =
        (loom_kernel_class_decision_t*)&out_classifier->decisions[i];
    IREE_ASSERT(decision->application == applications[i].target_op);
    IREE_ASSERT(decision->model->providers.count ==
                common->decisions[i].model->providers.count);
    decision->application = common->decisions[i].application;
    decision->model = common->decisions[i].model;
  }
  for (uint32_t i = 0; i < out_classifier->projection_count; ++i) {
    loom_kernel_class_projection_t* value =
        (loom_kernel_class_projection_t*)&out_classifier->projections[i];
    loom_value_id_t original = LOOM_VALUE_ID_INVALID;
    if (value->source_value_id < cloned_value_count) {
      const loom_value_id_t* segment = loom_segmented_storage_const_segment(
          &correspondence.source_values,
          value->source_value_id /
              LOOM_KERNEL_REQUEST_CORRESPONDENCE_SEGMENT_COUNT);
      original = segment[value->source_value_id %
                         LOOM_KERNEL_REQUEST_CORRESPONDENCE_SEGMENT_COUNT];
    }
    // Binding creates constants and identity views, never a new affine source.
    // Every materialized affine projection must refer back to a cloned value.
    IREE_ASSERT(value->kind != LOOM_KERNEL_CLASS_PROJECTION_AFFINE ||
                original != LOOM_VALUE_ID_INVALID);
    value->source_value_id = original;
    for (uint16_t j = 0; j < value->term_count; ++j) {
      ((loom_kernel_class_projection_term_t*)value->terms)[j]
          .argument_ordinal += prefix_count;
    }
  }
  loom_value_facts_t* argument_facts = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, common->kernel_argument_count, sizeof(*argument_facts),
      (void**)&argument_facts));
  memcpy(argument_facts, common->argument_facts,
         prefix_count * sizeof(*argument_facts));
  memcpy(argument_facts + prefix_count, out_classifier->argument_facts,
         out_classifier->kernel_argument_count * sizeof(*argument_facts));
  out_classifier->argument_facts = argument_facts;
  out_classifier->module = common->module;
  out_classifier->kernel_symbol_id = common->kernel_symbol_id;
  out_classifier->kernel_argument_count = common->kernel_argument_count;
  out_classifier->specialization_count = prefix_count;
  return iree_ok_status();
}

static iree_status_t loom_kernel_request_bind(
    const loom_kernel_class_classifier_t* common,
    const loom_template_decision_model_catalog_t* common_models,
    const loom_module_t* site_module,
    const loom_kernel_specialization_value_t* tuple,
    iree_arena_allocator_t* arena, iree_host_size_t construction_block_size,
    loom_kernel_class_classifier_t* out_classifier) {
  iree_arena_block_pool_t pool;
  iree_arena_block_pool_initialize(construction_block_size,
                                   arena->block_pool->block_allocator, &pool);
  iree_arena_allocator_t temporary;
  iree_arena_initialize(&pool, &temporary);
  loom_module_t* bound = NULL;
  iree_status_t status = loom_kernel_request_bind_classifier(
      common, common_models, site_module, tuple, arena, &temporary, &bound,
      out_classifier);
  loom_module_free(bound);
  iree_arena_deinitialize(&temporary);
  iree_arena_block_pool_deinitialize(&pool);
  return status;
}

static iree_status_t loom_kernel_request_source_publish_direct(
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

static iree_status_t loom_kernel_request_source_publish(
    const loom_kernel_class_classifier_t* classifier,
    const loom_template_decision_model_catalog_t* models,
    iree_host_size_t source_symbol_ordinal, const loom_module_t* site_module,
    const loom_kernel_class_site_t* sites, iree_host_size_t site_count,
    const loom_kernel_class_collection_options_t* options,
    loom_kernel_request_plan_provider_t plans, loom_kernel_request_sink_t sink,
    iree_arena_allocator_t* arena,
    const loom_decision_class_ordinal_t** out_site_classes) {
  // Without decisions or structural requirements, binding cannot change class
  // applicability. The ordinary collector already partitions exact tuples.
  if (!classifier->specialization_count ||
      (!classifier->decision_count && !classifier->has_shape_requirements)) {
    return loom_kernel_request_source_publish_direct(
        classifier, source_symbol_ordinal, site_module, sites, site_count,
        options, sink, arena, out_site_classes);
  }
  *out_site_classes = NULL;
  loom_decision_class_partition_t partition = {0};
  IREE_RETURN_IF_ERROR(loom_decision_class_partition_initialize(
      site_count, options->class_limit, 1, arena, &partition));
  const loom_kernel_specialization_value_t** tuples = NULL;
  IREE_RETURN_IF_ERROR(loom_kernel_specializations_collect(
      site_module, sites, classifier->specialization_count, &partition, arena,
      &tuples));
  const uint32_t group_count = partition.class_count;
  loom_kernel_class_group_t* groups = NULL;
  loom_kernel_class_classifier_t* classifiers = NULL;
  loom_kernel_class_collection_t* collections = NULL;
  iree_host_size_t* offsets = NULL;
  iree_host_size_t* positions = NULL;
  iree_host_size_t* original_sites = NULL;
  loom_kernel_class_site_t* grouped_sites = NULL;
  loom_decision_class_ordinal_t* site_classes = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, group_count, sizeof(*groups), (void**)&groups));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, group_count, sizeof(*classifiers), (void**)&classifiers));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, group_count, sizeof(*collections), (void**)&collections));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, group_count + 1, sizeof(*offsets), (void**)&offsets));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, group_count, sizeof(*positions), (void**)&positions));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, site_count, sizeof(*original_sites), (void**)&original_sites));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, site_count, sizeof(*grouped_sites), (void**)&grouped_sites));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, site_count, sizeof(*site_classes), (void**)&site_classes));
  memset(offsets, 0, (group_count + 1) * sizeof(*offsets));
  for (iree_host_size_t site = 0; site < site_count; ++site) {
    ++offsets[partition.site_classes[site] + 1];
  }
  for (uint32_t group = 0; group < group_count; ++group) {
    offsets[group + 1] += offsets[group];
  }
  memcpy(positions, offsets, group_count * sizeof(*positions));
  for (iree_host_size_t site = 0; site < site_count; ++site) {
    const iree_host_size_t position = positions[partition.site_classes[site]]++;
    grouped_sites[position] = sites[site];
    original_sites[position] = site;
  }
  void** leases = NULL;
  if (plans.acquire) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, group_count, sizeof(*leases), (void**)&leases));
  }
  uint32_t acquired_count = 0;
  iree_status_t status = iree_ok_status();
  for (uint32_t group = 0; group < group_count && iree_status_is_ok(status);
       ++group) {
    const loom_kernel_specialization_value_t* tuple =
        tuples[original_sites[offsets[group]]];
    if (plans.acquire) {
      const loom_kernel_class_classifier_t* retained = NULL;
      status = plans.acquire(plans.user_data, site_module, tuple,
                             classifier->specialization_count, arena, &retained,
                             &leases[group]);
      if (iree_status_is_ok(status)) {
        classifiers[group] = *retained;
        ++acquired_count;
      }
    } else {
      status = loom_kernel_request_bind(
          classifier, models, site_module, tuple, arena,
          arena->block_pool->total_block_size, &classifiers[group]);
    }
    if (iree_status_is_ok(status)) {
      // Tuple grouping is invocation state. This shallow view borrows immutable
      // retained payload; the shared classifier itself is never modified.
      classifiers[group].specialization_count = 0;
      groups[group] = (loom_kernel_class_group_t){
          .classifier = &classifiers[group],
          .sites = grouped_sites + offsets[group],
          .site_count = offsets[group + 1] - offsets[group],
      };
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_kernel_class_collect_groups(groups, group_count, site_module,
                                              options, arena, collections);
  }
  uint32_t class_base = 0;
  for (uint32_t group = 0; group < group_count && iree_status_is_ok(status);
       ++group) {
    classifiers[group].specialization_count = classifier->specialization_count;
    loom_kernel_class_collection_t* collection = &collections[group];
    const loom_kernel_request_source_t source = {
        .classifier = &classifiers[group],
        .collection = collection,
        .class_base = class_base,
    };
    const loom_kernel_specialization_value_t* tuple =
        tuples[original_sites[offsets[group]]];
    for (uint32_t ordinal = 0;
         ordinal < collection->class_count && iree_status_is_ok(status);
         ++ordinal) {
      ((loom_kernel_class_t*)collection->classes)[ordinal].specializations =
          tuple;
      const loom_kernel_request_t request = {
          .source_symbol_ordinal = source_symbol_ordinal,
          .class_ordinal = class_base + ordinal,
          .member_count = collection->classes[ordinal].member_count,
          .source = &source,
      };
      status = sink.publish(sink.user_data, &request);
    }
    for (iree_host_size_t site = 0; site < groups[group].site_count; ++site) {
      site_classes[original_sites[offsets[group] + site]] =
          class_base + collection->site_classes[site];
    }
    class_base += collection->class_count;
  }
  for (uint32_t group = 0; group < acquired_count; ++group) {
    plans.release(plans.user_data, leases[group]);
  }
  if (iree_status_is_ok(status)) {
    *out_site_classes = site_classes;
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
  loom_template_decision_model_catalog_t models = {0};
  loom_link_index_materialization_t materialization = {0};
  iree_status_t status = loom_kernel_request_source_prepare(
      producer, environment, source_symbol_ordinal, scratch_arena, &classifier,
      &models, &materialization);
  if (iree_status_is_ok(status) && materialization.product.module != NULL) {
    status = loom_kernel_request_source_publish(
        &classifier, &models, source_symbol_ordinal, site_module, sites,
        site_count, collection_options,
        (loom_kernel_request_plan_provider_t){0}, sink, scratch_arena,
        out_site_classes);
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
        &preparation->arena, &preparation->classifier, &preparation->models,
        &materialization);
  }
  if (iree_status_is_ok(status) && materialization.product.module != NULL) {
    status = loom_kernel_request_preparation_capture_config_keys(
        index, &materialization, preparation);
  }
  if (iree_status_is_ok(status) && materialization.product.module != NULL) {
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

iree_status_t loom_kernel_request_preparation_bind(
    const loom_kernel_request_preparation_t* preparation,
    const loom_module_t* site_module,
    const loom_kernel_specialization_value_t* values,
    iree_arena_allocator_t* arena,
    loom_kernel_class_classifier_t* out_classifier) {
  return loom_kernel_request_bind(
      &preparation->classifier, &preparation->models, site_module, values,
      arena, preparation->arena.block_pool->total_block_size, out_classifier);
}

iree_status_t loom_kernel_request_preparation_publish(
    const loom_kernel_request_preparation_t* preparation,
    const loom_module_t* site_module, const loom_kernel_class_site_t* sites,
    iree_host_size_t site_count,
    const loom_kernel_class_collection_options_t* collection_options,
    loom_kernel_request_plan_provider_t plans, loom_kernel_request_sink_t sink,
    iree_arena_allocator_t* scratch_arena,
    const loom_decision_class_ordinal_t** out_site_classes) {
  return loom_kernel_request_source_publish(
      &preparation->classifier, &preparation->models,
      preparation->source_symbol_ordinal, site_module, sites, site_count,
      collection_options, plans, sink, scratch_arena, out_site_classes);
}

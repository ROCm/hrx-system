// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/cmd/lower/program_plan_index.h"

#include "loom/error/error_catalog.h"
#include "loom/error/parameter.h"
#include "loom/ir/module.h"
#include "loom/link/index_materializer.h"
#include "loom/link/planner.h"

static iree_status_t loom_cmd_program_plan_diagnose_unresolved_definitions(
    const loom_link_module_index_t* index,
    const loom_link_index_materialization_t* materialization,
    iree_diagnostic_emitter_t diagnostic_emitter, bool* out_valid) {
  *out_valid = true;
  const iree_host_size_t plan_symbol_count =
      loom_link_plan_symbol_count(materialization->plan);
  for (iree_host_size_t i = 0; i < plan_symbol_count; ++i) {
    const loom_link_plan_symbol_t* selection =
        loom_link_plan_symbol_at(materialization->plan, i);
    if (!selection->has_unresolved_definition_dependency) {
      continue;
    }

    const loom_link_module_index_symbol_t* source_symbol =
        loom_link_module_index_symbol_at(index, selection->symbol_ordinal);
    IREE_ASSERT(source_symbol);
    IREE_ASSERT_LT(selection->symbol_ordinal,
                   materialization->product.target_symbols.count);
    const loom_symbol_ref_t target_ref = materialization->product.target_symbols
                                             .values[selection->symbol_ordinal];
    IREE_ASSERT(loom_symbol_ref_is_valid(target_ref));
    IREE_ASSERT_EQ(target_ref.module_id, 0u);
    IREE_ASSERT_LT(target_ref.symbol_id,
                   materialization->product.module->symbols.count);
    const loom_op_t* target_op =
        materialization->product.module->symbols.entries[target_ref.symbol_id]
            .defining_op;
    IREE_ASSERT(target_op);

    const loom_diagnostic_param_t params[] = {
        loom_param_string(source_symbol->name),
    };
    const loom_diagnostic_emission_t emission = {
        .module = materialization->product.module,
        .op = target_op,
        .error = LOOM_ERR_SYMBOL_002,
        .params = params,
        .param_count = IREE_ARRAYSIZE(params),
    };
    *out_valid = false;
    IREE_RETURN_IF_ERROR(iree_diagnostic_emit(diagnostic_emitter, &emission));
  }
  return iree_ok_status();
}

static iree_status_t loom_cmd_program_plan_produce_kernel_requests(
    void* user_data,
    const loom_link_plan_materialization_environment_t* environment,
    iree_host_size_t source_symbol_ordinal, const loom_module_t* site_module,
    const loom_kernel_class_site_t* sites, iree_host_size_t site_count,
    const loom_kernel_class_collection_options_t* collection_options,
    loom_kernel_request_sink_t sink, iree_arena_allocator_t* scratch_arena,
    const loom_decision_class_ordinal_t** out_site_classes) {
  return loom_kernel_request_producer_publish(
      (loom_kernel_request_producer_t*)user_data, environment,
      source_symbol_ordinal, site_module, sites, site_count, collection_options,
      sink, scratch_arena, out_site_classes);
}

static bool loom_cmd_program_plan_has_source_kernels(
    const loom_link_module_index_t* index,
    const loom_link_plan_materialization_t* materialization) {
  for (iree_host_size_t i = 0;
       i < materialization->target_source_definitions.count; ++i) {
    const iree_host_size_t source_ordinal =
        materialization->target_source_definitions.values[i];
    if (source_ordinal == LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL) {
      continue;
    }
    const loom_link_module_index_symbol_t* source =
        loom_link_module_index_symbol_at(index, source_ordinal);
    if (iree_any_bit_set(
            source->facets.schema.interfaces,
            LOOM_SYMBOL_INTERFACE_KERNEL | LOOM_SYMBOL_INTERFACE_PIPELINE)) {
      return true;
    }
  }
  return false;
}

iree_status_t loom_cmd_program_plan_build_from_index(
    const loom_link_module_index_t* index,
    const iree_host_size_t* program_symbol_ordinals,
    iree_host_size_t program_count,
    const loom_cmd_program_plan_index_options_t* options,
    const loom_pass_registry_t* pass_registry,
    const loom_cleanup_pattern_provider_set_t* cleanup_pattern_provider_set,
    iree_diagnostic_emitter_t diagnostic_emitter,
    const loom_link_plan_materialization_environment_t*
        materialization_environment,
    iree_arena_allocator_t* scratch_arena, bool* out_valid,
    loom_cmd_program_plan_t* out_plan) {
  if (index == NULL || program_symbol_ordinals == NULL || program_count == 0 ||
      pass_registry == NULL || cleanup_pattern_provider_set == NULL ||
      materialization_environment == NULL ||
      materialization_environment->context == NULL ||
      materialization_environment->block_pool == NULL ||
      scratch_arena == NULL || out_valid == NULL || out_plan == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "command program plan inputs must be present");
  }
  *out_valid = false;
  *out_plan = (loom_cmd_program_plan_t){0};

  loom_link_plan_root_facet_t* root_facets = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(scratch_arena, program_count,
                                                 sizeof(*root_facets),
                                                 (void**)&root_facets));
  for (iree_host_size_t i = 0; i < program_count; ++i) {
    const loom_link_module_index_symbol_t* symbol =
        loom_link_module_index_symbol_at(index, program_symbol_ordinals[i]);
    if (symbol == NULL ||
        loom_link_module_index_symbol_facet_ordinal(
            symbol, LOOM_LINK_SYMBOL_FACET_COMMAND_IMPLEMENTATION) ==
            LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "command program root ordinal %" PRIhsz
                              " does not expose a command implementation",
                              program_symbol_ordinals[i]);
    }
    root_facets[i] = (loom_link_plan_root_facet_t){
        .symbol_ordinal = program_symbol_ordinals[i],
        .kind = LOOM_LINK_SYMBOL_FACET_COMMAND_IMPLEMENTATION,
    };
  }

  const loom_link_plan_options_t link_options = {
      .mode = LOOM_LINK_PLAN_LINK,
      .unresolved_policy = LOOM_LINK_PLAN_UNRESOLVED_ALLOW,
      .root_facets =
          {
              .count = program_count,
              .values = root_facets,
          },
      .dependency_policy = LOOM_LINK_PLAN_DEPENDENCY_REQUESTED_FACETS,
  };
  loom_link_index_materialization_t materialization = {0};
  iree_status_t status = loom_link_index_materialize(
      index, &link_options, materialization_environment,
      IREE_SV("command_program_roots"), &materialization);
  bool source_valid = true;
  if (iree_status_is_ok(status)) {
    status = loom_cmd_program_plan_diagnose_unresolved_definitions(
        index, &materialization, diagnostic_emitter, &source_valid);
  }
  if (iree_status_is_ok(status) && source_valid && options != NULL &&
      options->verification.fn != NULL) {
    status =
        options->verification.fn(options->verification.user_data,
                                 materialization.product.module, &source_valid);
  }

  loom_symbol_ref_t* target_root_refs = NULL;
  if (iree_status_is_ok(status) && source_valid) {
    status = iree_arena_allocate_array(scratch_arena, program_count,
                                       sizeof(*target_root_refs),
                                       (void**)&target_root_refs);
  }
  for (iree_host_size_t i = 0;
       i < program_count && iree_status_is_ok(status) && source_valid; ++i) {
    const iree_host_size_t source_ordinal = program_symbol_ordinals[i];
    IREE_ASSERT_LT(source_ordinal,
                   materialization.product.target_symbols.count);
    target_root_refs[i] =
        materialization.product.target_symbols.values[source_ordinal];
    IREE_ASSERT(loom_symbol_ref_is_valid(target_root_refs[i]));
  }

  loom_kernel_request_producer_t* kernel_request_producer = NULL;
  loom_cmd_program_kernel_source_t kernel_source = {0};
  if (iree_status_is_ok(status) && source_valid && options != NULL &&
      options->kernel_request_sink.publish != NULL) {
    kernel_source = (loom_cmd_program_kernel_source_t){
        .environment = materialization_environment,
        .collection_options = options->kernel_class_collection,
        .sink = options->kernel_request_sink,
    };
    if (loom_cmd_program_plan_has_source_kernels(index,
                                                 &materialization.product)) {
      kernel_source.provider = options->kernel_request_provider;
      if (kernel_source.provider.publish == NULL) {
        status = loom_kernel_request_producer_allocate(
            index, materialization_environment, &kernel_request_producer);
        kernel_source.provider = (loom_kernel_request_provider_t){
            .publish = loom_cmd_program_plan_produce_kernel_requests,
            .user_data = kernel_request_producer,
        };
      }
      if (iree_status_is_ok(status)) {
        kernel_source.source_definitions.values =
            materialization.product.target_source_definitions.values;
        kernel_source.source_definitions.count =
            materialization.product.target_source_definitions.count;
      }
    }
  }

  if (iree_status_is_ok(status) && source_valid) {
    status = loom_cmd_program_plan_build_from_materialization(
        &materialization.product, target_root_refs, program_count,
        kernel_source.sink.publish != NULL ? &kernel_source : NULL,
        options != NULL && options->live_kernel_entry_validator.fn != NULL
            ? &options->live_kernel_entry_validator
            : NULL,
        pass_registry, cleanup_pattern_provider_set, diagnostic_emitter,
        materialization_environment->block_pool,
        materialization_environment->allocator, out_valid, out_plan);
  }

  loom_kernel_request_producer_free(kernel_request_producer);
  loom_link_index_materialization_deinitialize(&materialization);
  return status;
}

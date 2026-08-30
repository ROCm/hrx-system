// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/cmd/lower/program_plan_index.h"

#include "loom/ir/module.h"
#include "loom/link/index_materializer.h"
#include "loom/link/planner.h"

static bool loom_cmd_program_plan_has_source_kernels(
    const loom_link_plan_materialization_t* materialization) {
  if (materialization->target_kernel_configurations.count == 0) return false;
  IREE_ASSERT_EQ(materialization->target_kernel_configurations.count,
                 materialization->target_source_definitions.count);
  for (iree_host_size_t i = 0;
       i < materialization->target_kernel_configurations.count; ++i) {
    if (loom_symbol_ref_is_valid(
            materialization->target_kernel_configurations.values[i]) &&
        materialization->target_source_definitions.values[i] !=
            LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL) {
      return true;
    }
  }
  return false;
}

iree_status_t loom_cmd_program_plan_prepare_index(
    const loom_link_module_index_t* index,
    const iree_host_size_t* program_symbol_ordinals,
    iree_host_size_t program_count,
    const loom_cmd_program_plan_index_options_t* options,
    const loom_pass_registry_t* pass_registry,
    iree_diagnostic_emitter_t diagnostic_emitter,
    const loom_link_plan_materialization_environment_t*
        materialization_environment,
    iree_arena_allocator_t* scratch_arena, bool* out_valid,
    loom_cmd_program_plan_t* out_plan) {
  if (index == NULL || program_symbol_ordinals == NULL || program_count == 0 ||
      pass_registry == NULL || materialization_environment == NULL ||
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

  loom_symbol_ref_t* target_root_refs = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_arena_allocate_array(scratch_arena, program_count,
                                       sizeof(*target_root_refs),
                                       (void**)&target_root_refs);
  }
  for (iree_host_size_t i = 0; i < program_count && iree_status_is_ok(status);
       ++i) {
    const iree_host_size_t source_ordinal = program_symbol_ordinals[i];
    IREE_ASSERT_LT(source_ordinal,
                   materialization.product.target_symbols.count);
    target_root_refs[i] =
        materialization.product.target_symbols.values[source_ordinal];
    IREE_ASSERT(loom_symbol_ref_is_valid(target_root_refs[i]));
  }

  loom_kernel_request_producer_t* kernel_request_producer = NULL;
  loom_cmd_program_kernel_source_t kernel_source = {0};
  if (iree_status_is_ok(status) && options != NULL &&
      options->kernel_request_sink.publish != NULL &&
      loom_cmd_program_plan_has_source_kernels(&materialization.product)) {
    status = loom_kernel_request_producer_allocate(
        index, materialization_environment, &kernel_request_producer);
    if (iree_status_is_ok(status)) {
      kernel_source = (loom_cmd_program_kernel_source_t){
          .producer = kernel_request_producer,
          .environment = materialization_environment,
          .source_definitions =
              {
                  .values =
                      materialization.product.target_source_definitions.values,
                  .count =
                      materialization.product.target_source_definitions.count,
              },
          .collection_options = options->kernel_class_collection,
          .sink = options->kernel_request_sink,
      };
    }
  }

  if (iree_status_is_ok(status)) {
    status = loom_cmd_program_plan_prepare_materialization(
        &materialization.product, target_root_refs, program_count,
        kernel_source.producer != NULL ? &kernel_source : NULL, pass_registry,
        diagnostic_emitter, materialization_environment->block_pool, out_valid,
        out_plan, materialization_environment->allocator);
  }

  loom_kernel_request_producer_free(kernel_request_producer);
  loom_link_index_materialization_deinitialize(&materialization);
  return status;
}

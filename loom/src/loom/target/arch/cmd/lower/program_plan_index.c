// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/cmd/lower/program_plan_index.h"

#include "loom/ir/module.h"
#include "loom/link/plan_materializer.h"
#include "loom/link/planner.h"

iree_status_t loom_cmd_program_plan_prepare_index(
    const loom_link_module_index_t* index,
    const iree_host_size_t* program_symbol_ordinals,
    iree_host_size_t program_count, const loom_pass_registry_t* pass_registry,
    iree_diagnostic_emitter_t diagnostic_emitter,
    iree_arena_block_pool_t* block_pool, iree_arena_allocator_t* scratch_arena,
    bool* out_valid, loom_cmd_program_plan_t* out_plan,
    iree_allocator_t host_allocator) {
  if (index == NULL || program_symbol_ordinals == NULL || program_count == 0 ||
      pass_registry == NULL || block_pool == NULL || scratch_arena == NULL ||
      out_valid == NULL || out_plan == NULL) {
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

  loom_link_plan_t* link_plan = NULL;
  iree_status_t status = loom_link_plan_build(
      index,
      &(loom_link_plan_options_t){
          .mode = LOOM_LINK_PLAN_LINK,
          .unresolved_policy = LOOM_LINK_PLAN_UNRESOLVED_ALLOW,
          .root_facets =
              {
                  .count = program_count,
                  .values = root_facets,
              },
          .dependency_policy = LOOM_LINK_PLAN_DEPENDENCY_REQUESTED_FACETS,
      },
      host_allocator, &link_plan);

  loom_link_plan_materialization_t materialization = {0};
  if (iree_status_is_ok(status)) {
    status = loom_link_plan_materialize(
        link_plan,
        &(loom_link_plan_materialization_environment_t){
            .context = loom_link_module_index_context(index),
            .block_pool = block_pool,
            .allocator = host_allocator,
        },
        IREE_SV("command_program_roots"), scratch_arena, &materialization);
  }

  loom_symbol_ref_t* target_root_refs = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_arena_allocate_array(scratch_arena, program_count,
                                       sizeof(*target_root_refs),
                                       (void**)&target_root_refs);
  }
  for (iree_host_size_t i = 0; i < program_count && iree_status_is_ok(status);
       ++i) {
    const iree_host_size_t source_ordinal = program_symbol_ordinals[i];
    IREE_ASSERT_LT(source_ordinal, materialization.target_symbols.count);
    target_root_refs[i] = materialization.target_symbols.values[source_ordinal];
    IREE_ASSERT(loom_symbol_ref_is_valid(target_root_refs[i]));
  }

  if (iree_status_is_ok(status)) {
    status = loom_cmd_program_plan_prepare_materialization(
        &materialization, target_root_refs, program_count, pass_registry,
        diagnostic_emitter, block_pool, out_valid, out_plan, host_allocator);
  }

  if (materialization.module != NULL) {
    loom_module_free(materialization.module);
  }
  loom_link_plan_free(link_plan);
  return status;
}

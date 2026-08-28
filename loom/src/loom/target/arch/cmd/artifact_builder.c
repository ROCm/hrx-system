// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/cmd/artifact_builder.h"

iree_status_t loom_cmd_program_artifact_set_build_from_index(
    const loom_link_module_index_t* index,
    const iree_host_size_t* root_symbol_ordinals,
    iree_host_size_t root_symbol_count,
    const loom_cmd_program_artifact_builder_options_t* options,
    iree_arena_allocator_t* scratch_arena, bool* out_valid,
    loom_cmd_program_artifact_set_t* out_artifact_set,
    iree_allocator_t host_allocator) {
  *out_valid = false;
  *out_artifact_set = (loom_cmd_program_artifact_set_t){0};

  loom_cmd_program_plan_t plan = {0};
  iree_status_t status = loom_cmd_program_plan_prepare_index(
      index, root_symbol_ordinals, root_symbol_count, options->plan_options,
      options->pass_registry, options->diagnostic_emitter,
      options->materialization_environment, scratch_arena, out_valid, &plan);
  if (iree_status_is_ok(status) && *out_valid) {
    status = loom_cmd_program_artifact_set_build(&plan, out_artifact_set,
                                                 host_allocator);
  }
  loom_cmd_program_plan_deinitialize(&plan);
  return status;
}

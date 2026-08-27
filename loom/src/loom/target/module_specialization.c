// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/module_specialization.h"

#include "loom/ir/module.h"
#include "loom/target/function_version_projection.h"

iree_status_t loom_target_specialize_module(
    const loom_target_environment_t* environment,
    loom_target_specialization_request_list_t requests,
    loom_target_declaration_binding_list_t bindings,
    iree_diagnostic_emitter_t diagnostic_emitter,
    iree_arena_block_pool_t* block_pool, iree_allocator_t allocator,
    loom_module_t** inout_module, uint32_t* out_error_count) {
  IREE_ASSERT_ARGUMENT(environment);
  IREE_ASSERT_ARGUMENT(block_pool);
  IREE_ASSERT_ARGUMENT(inout_module);
  IREE_ASSERT_ARGUMENT(*inout_module);
  IREE_ASSERT_ARGUMENT(out_error_count);
  *out_error_count = 0;

  iree_arena_allocator_t arena;
  iree_arena_initialize(block_pool, &arena);
  loom_target_specialization_result_t specialization = {0};
  iree_status_t status = loom_target_specialize_functions(
      environment, *inout_module, requests, bindings, diagnostic_emitter,
      &arena, &specialization);
  if (iree_status_is_ok(status)) {
    *out_error_count = specialization.error_count;
  }

  loom_module_t* projected_module = NULL;
  if (iree_status_is_ok(status) && specialization.error_count == 0 &&
      specialization.function_versions.list.count != 0) {
    status = loom_target_function_versions_project_module(
        *inout_module, &specialization.function_versions.list, block_pool,
        allocator, &projected_module);
  }
  if (iree_status_is_ok(status) && projected_module != NULL) {
    loom_module_free(*inout_module);
    *inout_module = projected_module;
    projected_module = NULL;
  }

  loom_module_free(projected_module);
  iree_arena_deinitialize(&arena);
  return status;
}

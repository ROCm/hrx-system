// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/module_specialization.h"

#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"
#include "loom/target/function_version_projection.h"

static iree_status_t loom_target_specialize_module_with_arena(
    const loom_target_environment_t* environment,
    loom_target_specialization_request_list_t requests,
    loom_target_declaration_binding_list_t bindings,
    iree_diagnostic_emitter_t diagnostic_emitter, iree_arena_allocator_t* arena,
    iree_arena_block_pool_t* block_pool, iree_allocator_t allocator,
    loom_module_t** inout_module, uint32_t* out_error_count) {
  IREE_ASSERT_ARGUMENT(environment);
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(block_pool);
  IREE_ASSERT_ARGUMENT(inout_module);
  IREE_ASSERT_ARGUMENT(*inout_module);
  IREE_ASSERT_ARGUMENT(out_error_count);
  *out_error_count = 0;

  loom_target_specialization_result_t specialization = {0};
  iree_status_t status = loom_target_specialize_functions(
      environment, *inout_module, requests, bindings, diagnostic_emitter, arena,
      &specialization);
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
  return status;
}

iree_status_t loom_target_specialize_module(
    const loom_target_environment_t* environment,
    loom_target_specialization_request_list_t requests,
    loom_target_declaration_binding_list_t bindings,
    iree_diagnostic_emitter_t diagnostic_emitter,
    iree_arena_block_pool_t* block_pool, iree_allocator_t allocator,
    loom_module_t** inout_module, uint32_t* out_error_count) {
  iree_arena_allocator_t arena;
  iree_arena_initialize(block_pool, &arena);
  iree_status_t status = loom_target_specialize_module_with_arena(
      environment, requests, bindings, diagnostic_emitter, &arena, block_pool,
      allocator, inout_module, out_error_count);
  iree_arena_deinitialize(&arena);
  return status;
}

iree_status_t loom_target_specialize_module_kernel_entries(
    const loom_target_environment_t* environment,
    const loom_target_profile_t* target_profile,
    iree_diagnostic_emitter_t diagnostic_emitter,
    iree_arena_block_pool_t* block_pool, iree_allocator_t allocator,
    loom_module_t** inout_module, uint32_t* out_error_count) {
  IREE_ASSERT_ARGUMENT(environment);
  IREE_ASSERT_ARGUMENT(target_profile);
  IREE_ASSERT_ARGUMENT(block_pool);
  IREE_ASSERT_ARGUMENT(inout_module);
  IREE_ASSERT_ARGUMENT(*inout_module);
  IREE_ASSERT_ARGUMENT(out_error_count);
  *out_error_count = 0;

  iree_host_size_t request_count = 0;
  loom_op_t* op = NULL;
  loom_block_for_each_op(loom_module_block(*inout_module), op) {
    if (loom_func_like_is_kernel_entry(
            loom_func_like_cast(*inout_module, op))) {
      ++request_count;
    }
  }
  if (request_count == 0) {
    return iree_ok_status();
  }

  iree_arena_allocator_t arena;
  iree_arena_initialize(block_pool, &arena);
  loom_target_specialization_request_t* requests = NULL;
  iree_status_t status = iree_arena_allocate_array(
      &arena, request_count, sizeof(*requests), (void**)&requests);
  if (iree_status_is_ok(status)) {
    iree_host_size_t request_ordinal = 0;
    loom_block_for_each_op(loom_module_block(*inout_module), op) {
      const loom_func_like_t function = loom_func_like_cast(*inout_module, op);
      if (!loom_func_like_is_kernel_entry(function)) {
        continue;
      }
      const loom_symbol_ref_t function_ref = loom_func_like_callee(function);
      const loom_symbol_t* function_symbol =
          &(*inout_module)->symbols.entries[function_ref.symbol_id];
      requests[request_ordinal++] = (loom_target_specialization_request_t){
          .function_name =
              (*inout_module)->strings.entries[function_symbol->name_id],
          .target_profile = target_profile,
      };
    }
    status = loom_target_specialize_module_with_arena(
        environment,
        (loom_target_specialization_request_list_t){
            .values = requests,
            .count = request_count,
        },
        (loom_target_declaration_binding_list_t){0}, diagnostic_emitter, &arena,
        block_pool, allocator, inout_module, out_error_count);
  }
  iree_arena_deinitialize(&arena);
  return status;
}

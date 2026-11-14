// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/loom-benchmark/case_request.h"

#include <string.h>

#include "iree/base/internal/arena.h"
#include "loom/tooling/testbench/case_command_program.h"
#include "loom/tools/loom-benchmark/output.h"
#include "loomc/interop.h"
#include "loomc/iree.h"
#include "loomc/target/cmd/program.h"

#define LOOM_BENCHMARK_CASE_REQUEST_BLOCK_SIZE (32 * 1024)

static loomc_status_t LOOMC_API_PTR
loom_benchmark_case_request_publish(void* user_data, loomc_request_t* request) {
  loom_benchmark_case_request_capture_t* capture = user_data;
  if (capture->request_count == capture->request_capacity) {
    iree_host_size_t new_capacity =
        capture->request_capacity == 0 ? 4 : capture->request_capacity * 2;
    if (new_capacity < capture->request_capacity) {
      loomc_request_release(request);
      return loomc_status_from_iree(iree_make_status(
          IREE_STATUS_RESOURCE_EXHAUSTED,
          "case request publication count exceeds host limits"));
    }
    iree_status_t status = iree_allocator_realloc_array(
        capture->host_allocator, new_capacity, sizeof(*capture->requests),
        (void**)&capture->requests);
    if (!iree_status_is_ok(status)) {
      loomc_request_release(request);
      return loomc_status_from_iree(status);
    }
    capture->request_capacity = new_capacity;
  }
  capture->requests[capture->request_count++] = request;
  return loomc_ok_status();
}

static iree_status_t loom_benchmark_case_request_copy_program_metadata(
    const loom_module_t* module,
    const loom_testbench_case_command_program_t* program,
    loom_benchmark_case_request_capture_t* capture) {
  const loom_symbol_t* symbol =
      &module->symbols.entries[program->root_ref.symbol_id];
  const iree_string_view_t root_symbol =
      loom_string_table_get(&module->strings, symbol->name_id);
  IREE_RETURN_IF_ERROR(loom_benchmark_dup_string_view(
      root_symbol, capture->host_allocator, &capture->root_symbol_storage));
  capture->root_symbol =
      iree_make_string_view(capture->root_symbol_storage, root_symbol.size);

  if (program->binding_count != 0) {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
        capture->host_allocator, program->binding_count,
        sizeof(*capture->binding_value_ids),
        (void**)&capture->binding_value_ids));
    memcpy(capture->binding_value_ids, program->binding_value_ids,
           program->binding_count * sizeof(*capture->binding_value_ids));
    capture->binding_count = program->binding_count;
  }
  return iree_ok_status();
}

static iree_status_t loom_benchmark_case_request_build(
    const loom_benchmark_case_request_options_t* options,
    loom_benchmark_case_request_capture_t* capture,
    loomc_result_t** out_result) {
  const loomc_allocator_t loom_allocator =
      loomc_allocator_from_iree(capture->host_allocator);
  loomc_module_t* adapted_module = NULL;
  IREE_RETURN_IF_ERROR(iree_status_from_loomc(loomc_module_clone(
      options->module, options->workspace, loom_allocator, &adapted_module)));

  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(LOOM_BENCHMARK_CASE_REQUEST_BLOCK_SIZE,
                                   capture->host_allocator, &block_pool);
  iree_arena_allocator_t plan_arena;
  iree_arena_initialize(&block_pool, &plan_arena);

  const loomc_module_mutable_interop_view_t view =
      loomc_module_get_mutable_interop_view(adapted_module);
  loom_testbench_module_plan_t module_plan = {0};
  loom_testbench_plan_options_t plan_options = {0};
  loom_testbench_plan_options_initialize(&plan_options);
  iree_status_t status = loom_testbench_plan_module(view.module, &plan_options,
                                                    &plan_arena, &module_plan);
  const loom_testbench_case_plan_t* case_plan = NULL;
  if (iree_status_is_ok(status)) {
    // Cloning preserves the source-order case ordinals from benchmark planning.
    IREE_ASSERT_LT(options->case_index, module_plan.case_count);
    capture->case_index = options->case_index;
    case_plan = &module_plan.cases[capture->case_index];
  }

  loom_testbench_case_command_program_t program = {0};
  if (iree_status_is_ok(status)) {
    status = loom_testbench_case_command_program_build(
        view.module, &module_plan, case_plan, options->case_sample, &program);
  }
  if (iree_status_is_ok(status)) {
    status = loom_benchmark_case_request_copy_program_metadata(
        view.module, &program, capture);
  }

  if (iree_status_is_ok(status)) {
    const loomc_string_view_t root_symbol =
        loomc_string_view_from_iree(capture->root_symbol);
    const loomc_cmd_program_module_options_t product_options = {
        .type = LOOMC_STRUCTURE_TYPE_CMD_PROGRAM_MODULE_OPTIONS,
        .structure_size = sizeof(product_options),
        .provider_name = loomc_make_cstring_view("loom-benchmark"),
        .root_symbols = &root_symbol,
        .root_symbol_count = 1,
        .config =
            options->config ? *options->config : (loomc_config_options_t){0},
        .target_profile = options->target_profile,
        .request_sink =
            {
                .publish = loom_benchmark_case_request_publish,
                .user_data = capture,
            },
    };
    status = iree_status_from_loomc(loomc_cmd_program_product_build_module(
        options->workspace, adapted_module, &product_options, loom_allocator,
        &capture->command_product, out_result));
  }

  iree_arena_deinitialize(&plan_arena);
  iree_arena_block_pool_deinitialize(&block_pool);
  loomc_module_release(adapted_module);
  return status;
}

iree_status_t loom_benchmark_case_request_capture(
    const loom_benchmark_case_request_options_t* options,
    iree_allocator_t host_allocator,
    loom_benchmark_case_request_capture_t* out_capture,
    loomc_result_t** out_result) {
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(options->workspace);
  IREE_ASSERT_ARGUMENT(options->module);
  IREE_ASSERT_ARGUMENT(out_capture);
  IREE_ASSERT_ARGUMENT(out_result);
  *out_capture = (loom_benchmark_case_request_capture_t){
      .host_allocator = host_allocator,
  };
  *out_result = NULL;

  iree_status_t status =
      loom_benchmark_case_request_build(options, out_capture, out_result);
  if (!iree_status_is_ok(status) || !loomc_result_succeeded(*out_result)) {
    loom_benchmark_case_request_capture_deinitialize(out_capture);
    out_capture->host_allocator = host_allocator;
  }
  return status;
}

void loom_benchmark_case_request_capture_deinitialize(
    loom_benchmark_case_request_capture_t* capture) {
  if (capture == NULL) {
    return;
  }
  for (iree_host_size_t i = capture->request_count; i > 0; --i) {
    loomc_request_release(capture->requests[i - 1]);
  }
  loomc_product_release(capture->command_product);
  iree_allocator_free(capture->host_allocator, capture->root_symbol_storage);
  iree_allocator_free(capture->host_allocator, capture->binding_value_ids);
  iree_allocator_free(capture->host_allocator, capture->requests);
  *capture = (loom_benchmark_case_request_capture_t){0};
}

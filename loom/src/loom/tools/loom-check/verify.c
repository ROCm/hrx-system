// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/loom-check/compile.h"
#include "loom/tools/loom-check/compile_diagnostics.h"
#include "loom/tools/loom-check/diagnostics.h"
#include "loom/tools/loom-check/execute.h"
#include "loomc/iree.h"

//===----------------------------------------------------------------------===//
// Verify execution
//===----------------------------------------------------------------------===//

iree_status_t loom_check_execute_verify(
    const loom_test_case_t* test_case, iree_host_size_t case_index,
    loom_check_file_report_t* report, iree_string_view_t filename,
    const loom_check_input_t* input_request,
    const loom_check_environment_t* environment,
    iree_arena_block_pool_t* block_pool, iree_allocator_t allocator,
    loom_check_result_t* result) {
  // Arena for collector storage. Entries and message copies are released
  // together at the end of this case.
  iree_arena_allocator_t collector_arena;
  iree_arena_initialize(block_pool, &collector_arena);

  loom_check_diagnostic_collector_t collector = {
      .arena = &collector_arena,
      .host_allocator = allocator,
      .filename = filename,
      .result = result,
  };

  loomc_module_t* module = NULL;
  iree_status_t status = loom_check_compile_admit_module(
      test_case, filename, input_request, environment->compile_session,
      environment, /*max_errors=*/100, &collector, block_pool, allocator,
      &module);
  if (iree_status_is_ok(status) && module != NULL) {
    const loomc_module_verify_options_t options = {
        .type = LOOMC_STRUCTURE_TYPE_MODULE_VERIFY_OPTIONS,
        .structure_size = sizeof(options),
        .max_errors = 100,
    };
    loomc_result_t* verification = NULL;
    status = iree_status_from_loomc(loomc_module_verify(
        module, &options, loomc_allocator_from_iree(allocator), &verification));
    if (iree_status_is_ok(status)) {
      status = loom_check_compile_append_result_diagnostics(&collector,
                                                            verification);
    }
    loomc_result_release(verification);
  }
  loomc_module_release(module);

  if (iree_status_is_ok(status)) {
    status = loom_check_diagnostic_collector_finish(
        &collector, test_case, case_index, report, allocator, result);
  }

  iree_arena_deinitialize(&collector_arena);
  return status;
}

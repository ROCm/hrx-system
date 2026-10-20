// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/iree-benchmark-loom/testbench.h"

#include <string.h>

#include "loom/ops/check/ops.h"
#include "loomc/iree.h"

iree_status_t iree_benchmark_loom_add_runtime_requirements(
    loom_run_hal_testbench_context_t* hal_context, const loomc_module_t* module,
    const loomc_sanitizer_options_t* sanitizer_options,
    const loom_testbench_module_plan_t* module_plan,
    const iree_benchmark_loom_work_plan_t* work_plan,
    iree_arena_block_pool_t* block_pool) {
  iree_arena_allocator_t arena;
  iree_arena_initialize(block_pool, &arena);
  const iree_host_size_t record_count =
      module_plan->case_count + module_plan->scenario_count;
  uint8_t* selected_records = NULL;
  iree_status_t status =
      iree_arena_allocate_array(&arena, record_count, sizeof(*selected_records),
                                (void**)&selected_records);
  if (iree_status_is_ok(status)) {
    memset(selected_records, 0, record_count);
  }
  iree_host_size_t capacity = 0;
  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) && i < work_plan->selected_benchmark_count;
       ++i) {
    const iree_benchmark_loom_selected_benchmark_t* selected =
        &work_plan->selected_benchmarks[i];
    const iree_host_size_t record_index =
        selected->case_plan != NULL
            ? selected->benchmark_plan->case_index
            : module_plan->case_count +
                  selected->benchmark_plan->scenario_index;
    if (selected_records[record_index]) {
      continue;
    }
    selected_records[record_index] = 1;
    capacity += selected->case_plan != NULL
                    ? selected->case_plan->invocation_count
                    : selected->scenario_plan->trial_count;
  }
  loomc_string_view_t* roots = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_arena_allocate_array(&arena, capacity, sizeof(*roots),
                                       (void**)&roots);
  }
  iree_host_size_t count = 0;
  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) && i < work_plan->selected_benchmark_count;
       ++i) {
    const iree_benchmark_loom_selected_benchmark_t* selected =
        &work_plan->selected_benchmarks[i];
    const iree_host_size_t record_index =
        selected->case_plan != NULL
            ? selected->benchmark_plan->case_index
            : module_plan->case_count +
                  selected->benchmark_plan->scenario_index;
    if (!selected_records[record_index]) {
      continue;
    }
    selected_records[record_index] = 0;
    const loom_testbench_case_plan_t* case_plan = selected->case_plan;
    const loom_testbench_scenario_plan_t* scenario = selected->scenario_plan;
    for (iree_host_size_t j = 0;
         j < (case_plan != NULL ? case_plan->invocation_count
                                : scenario->trial_count);
         ++j) {
      const loom_testbench_invocation_plan_t* invocation =
          case_plan != NULL ? &case_plan->invocations[j]
                            : &scenario->trials[j].action.target;
      // Case function calls and oracle calls run through host providers.
      if (case_plan != NULL &&
          invocation->kind != LOOM_TESTBENCH_INVOCATION_KERNEL_LAUNCH &&
          invocation->kind != LOOM_TESTBENCH_INVOCATION_PIPELINE &&
          invocation->kind != LOOM_TESTBENCH_INVOCATION_COMMAND_PROGRAM) {
        continue;
      }
      const loom_symbol_t* symbol =
          &invocation->module->symbols
               .entries[invocation->callee_ref.symbol_id];
      roots[count++] = loomc_string_view_from_iree(
          loom_string_table_get(&invocation->module->strings, symbol->name_id));
    }
  }
  if (iree_status_is_ok(status) && count != 0) {
    status = loom_run_hal_testbench_context_add_module_runtime_requirements(
        hal_context, module, sanitizer_options, roots, count);
  }
  iree_arena_deinitialize(&arena);
  return status;
}

enum {
  IREE_BENCHMARK_LOOM_MAX_REQUIREMENT_PROVIDERS = 8,
};

iree_status_t iree_benchmark_loom_evaluate_requirements(
    const iree_benchmark_loom_configuration_t* configuration,
    loom_run_hal_testbench_context_t* hal_context,
    const loom_testbench_module_plan_t* module_plan,
    iree_arena_allocator_t* arena, iree_benchmark_loom_work_plan_t* work_plan) {
  loom_testbench_requirement_provider_t
      requirement_providers[IREE_BENCHMARK_LOOM_MAX_REQUIREMENT_PROVIDERS] = {
          0};
  iree_host_size_t requirement_provider_count = 0;
  if (configuration->populate_requirement_providers.fn != NULL) {
    IREE_RETURN_IF_ERROR(configuration->populate_requirement_providers.fn(
        configuration->populate_requirement_providers.user_data, hal_context,
        IREE_ARRAYSIZE(requirement_providers), requirement_providers,
        &requirement_provider_count));
  }
  if (requirement_provider_count > IREE_ARRAYSIZE(requirement_providers)) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "iree-benchmark-loom requirement provider capacity exceeded");
  }
  loom_testbench_requirement_provider_registry_t requirement_registry = {0};
  loom_testbench_requirement_provider_registry_initialize(
      requirement_providers, requirement_provider_count, &requirement_registry);
  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(arena->block_pool, &scratch_arena);
  const iree_host_size_t record_count =
      module_plan->case_count + module_plan->scenario_count;
  const iree_benchmark_loom_selected_benchmark_t** first_selections = NULL;
  iree_status_t status = iree_arena_allocate_array(&scratch_arena, record_count,
                                                   sizeof(*first_selections),
                                                   (void**)&first_selections);
  if (iree_status_is_ok(status)) {
    memset(first_selections, 0, record_count * sizeof(*first_selections));
  }
  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) && i < work_plan->selected_benchmark_count;
       ++i) {
    iree_benchmark_loom_selected_benchmark_t* selection =
        &work_plan->selected_benchmarks[i];
    const iree_host_size_t record_index =
        selection->case_plan != NULL
            ? selection->case_plan - module_plan->cases
            : module_plan->case_count +
                  (selection->scenario_plan - module_plan->scenarios);
    if (first_selections[record_index] != NULL) {
      selection->skip_reason = first_selections[record_index]->skip_reason;
    } else {
      const loom_region_t* body =
          selection->case_plan != NULL
              ? loom_check_case_body(selection->case_plan->op)
              : loom_check_scenario_body(selection->scenario_plan->op);
      loom_testbench_requirement_result_t result = {0};
      status = loom_testbench_evaluate_requirements(
          module_plan->module, body, &requirement_registry, arena, &result);
      if (iree_status_is_ok(status) && result.skipped) {
        loom_testbench_requirement_result_t* retained_result = NULL;
        status = iree_arena_allocate(arena, sizeof(*retained_result),
                                     (void**)&retained_result);
        if (iree_status_is_ok(status)) {
          *retained_result = result;
          selection->skip_reason = retained_result;
        }
      }
      first_selections[record_index] = selection;
    }
  }
  iree_arena_deinitialize(&scratch_arena);
  return status;
}

void iree_benchmark_loom_configure_reference_oracles(
    loom_run_hal_testbench_context_t* context, iree_allocator_t host_allocator,
    iree_benchmark_loom_reference_oracles_t* out_oracles,
    loom_testbench_case_execution_options_t* inout_execution_options) {
  *out_oracles = (iree_benchmark_loom_reference_oracles_t){0};
  out_oracles->options = (loom_testbench_reference_matmul_oracle_options_t){
      .result_pool = context->runtime.staging_pool,
      .host_allocator = host_allocator,
  };
  loom_testbench_reference_matmul_oracle_provider_initialize(
      &out_oracles->options, &out_oracles->providers[0]);
  loom_testbench_reference_tiled_matmul_oracle_provider_initialize(
      &out_oracles->options, &out_oracles->providers[1]);
  inout_execution_options->invocation.oracle_providers =
      loom_make_testbench_oracle_provider_list(
          out_oracles->providers,
          IREE_BENCHMARK_LOOM_REFERENCE_ORACLE_PROVIDER_COUNT);
}

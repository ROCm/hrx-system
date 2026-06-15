// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/iree-benchmark-loom/testbench.h"

enum {
  IREE_BENCHMARK_LOOM_MAX_REQUIREMENT_PROVIDERS = 8,
};

iree_status_t iree_benchmark_loom_evaluate_case_requirements(
    const iree_benchmark_loom_configuration_t* configuration,
    loom_run_hal_testbench_context_t* hal_context,
    const loom_testbench_module_plan_t* module_plan,
    const loom_testbench_case_plan_t* case_plan,
    loom_testbench_requirement_result_t* out_result) {
  loom_testbench_requirement_provider_t
      requirement_providers[IREE_BENCHMARK_LOOM_MAX_REQUIREMENT_PROVIDERS] = {
          0};
  iree_host_size_t requirement_provider_count = 0;
  loom_run_hal_testbench_requirement_provider_initialize(
      hal_context, &requirement_providers[requirement_provider_count++]);
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
  return loom_testbench_evaluate_case_requirements(
      module_plan->module, case_plan, &requirement_registry, out_result);
}

void iree_benchmark_loom_configure_reference_oracles(
    loom_run_hal_testbench_context_t* context, iree_allocator_t host_allocator,
    iree_benchmark_loom_reference_oracles_t* out_oracles,
    loom_testbench_case_execution_options_t* inout_execution_options) {
  *out_oracles = (iree_benchmark_loom_reference_oracles_t){0};
  out_oracles->options = (loom_testbench_reference_matmul_oracle_options_t){
      .device = context->runtime.device,
      .device_allocator = iree_hal_device_allocator(context->runtime.device),
      .result_buffer_params =
          loom_run_hal_testbench_host_visible_buffer_params(),
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

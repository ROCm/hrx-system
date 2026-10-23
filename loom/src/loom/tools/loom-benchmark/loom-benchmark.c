// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// loom-benchmark binary with build-selected compiler and device providers.

#include <stddef.h>
#include <stdio.h>

#include "loom/tooling/execution/hal/testbench_actual.h"
#include "loom/tooling/input/loomc_configured.h"
#include "loom/tools/loom-benchmark/main.h"
#include "loomc/iree.h"
#include "loomc/target/configured.h"

#ifndef LOOM_BENCHMARK_HAVE_AMDGPU
#define LOOM_BENCHMARK_HAVE_AMDGPU 0
#endif  // LOOM_BENCHMARK_HAVE_AMDGPU
#ifndef LOOM_BENCHMARK_HAVE_SPIRV
#define LOOM_BENCHMARK_HAVE_SPIRV 0
#endif  // LOOM_BENCHMARK_HAVE_SPIRV
#ifndef LOOM_BENCHMARK_HAVE_XDNA
#define LOOM_BENCHMARK_HAVE_XDNA 0
#endif  // LOOM_BENCHMARK_HAVE_XDNA
#ifndef LOOM_BENCHMARK_HAVE_TASK
#define LOOM_BENCHMARK_HAVE_TASK 0
#endif  // LOOM_BENCHMARK_HAVE_TASK
#ifndef LOOM_BENCHMARK_HAVE_VM
#define LOOM_BENCHMARK_HAVE_VM 0
#endif  // LOOM_BENCHMARK_HAVE_VM

#define LOOM_BENCHMARK_HAVE_ANY_DEVICE_PROVIDER               \
  (LOOM_BENCHMARK_HAVE_AMDGPU || LOOM_BENCHMARK_HAVE_SPIRV || \
   LOOM_BENCHMARK_HAVE_TASK || LOOM_BENCHMARK_HAVE_XDNA)

#if LOOM_BENCHMARK_HAVE_AMDGPU
#include "loom/tooling/target/amdgpu/testbench_requirements.h"
#include "loomc/target/amdgpu/iree_hal.h"
#endif  // LOOM_BENCHMARK_HAVE_AMDGPU
#if LOOM_BENCHMARK_HAVE_SPIRV
#include "loom/tooling/target/spirv/testbench_requirements.h"
#include "loomc/target/spirv/iree_hal.h"
#endif  // LOOM_BENCHMARK_HAVE_SPIRV
#if LOOM_BENCHMARK_HAVE_XDNA
#include "loomc/target/amd/xdna/iree_hal.h"
#endif  // LOOM_BENCHMARK_HAVE_XDNA
#if LOOM_BENCHMARK_HAVE_TASK
#include "loomc/target/cpu/iree_hal.h"
#endif  // LOOM_BENCHMARK_HAVE_TASK
#if LOOM_BENCHMARK_HAVE_VM
#include "loom/tooling/target/vm/testbench.h"
#endif  // LOOM_BENCHMARK_HAVE_VM

#if LOOM_BENCHMARK_HAVE_AMDGPU || LOOM_BENCHMARK_HAVE_SPIRV
static iree_status_t loom_benchmark_append_requirement_provider(
    iree_host_size_t provider_capacity,
    loom_testbench_requirement_provider_t* providers,
    iree_host_size_t* inout_provider_count,
    loom_testbench_requirement_provider_t provider) {
  if (*inout_provider_count >= provider_capacity) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "loom-benchmark requirement provider capacity exceeded");
  }
  providers[(*inout_provider_count)++] = provider;
  return iree_ok_status();
}
#endif  // LOOM_BENCHMARK_HAVE_AMDGPU || LOOM_BENCHMARK_HAVE_SPIRV

static iree_status_t loom_benchmark_populate_requirement_providers(
    void* user_data, loom_run_hal_testbench_context_t* hal_context,
    iree_host_size_t provider_capacity,
    loom_testbench_requirement_provider_t* providers,
    iree_host_size_t* inout_provider_count) {
  (void)user_data;
#if LOOM_BENCHMARK_HAVE_AMDGPU
  loom_testbench_requirement_provider_t descriptor_set_provider = {0};
  loom_amdgpu_hal_descriptor_set_testbench_requirement_provider_initialize(
      hal_context, &descriptor_set_provider);
  IREE_RETURN_IF_ERROR(loom_benchmark_append_requirement_provider(
      provider_capacity, providers, inout_provider_count,
      descriptor_set_provider));
  loom_testbench_requirement_provider_t target_provider = {0};
  loom_amdgpu_hal_target_testbench_requirement_provider_initialize(
      hal_context, &target_provider);
  IREE_RETURN_IF_ERROR(loom_benchmark_append_requirement_provider(
      provider_capacity, providers, inout_provider_count, target_provider));
#endif  // LOOM_BENCHMARK_HAVE_AMDGPU
#if LOOM_BENCHMARK_HAVE_SPIRV
  loom_testbench_requirement_provider_t vulkan_feature_provider = {0};
  loom_spirv_vulkan_feature_testbench_requirement_provider_initialize(
      hal_context, &vulkan_feature_provider);
  IREE_RETURN_IF_ERROR(loom_benchmark_append_requirement_provider(
      provider_capacity, providers, inout_provider_count,
      vulkan_feature_provider));
  loom_testbench_requirement_provider_t cooperative_matrix_provider = {0};
  loom_spirv_vulkan_cooperative_matrix_testbench_requirement_provider_initialize(
      hal_context, &cooperative_matrix_provider);
  IREE_RETURN_IF_ERROR(loom_benchmark_append_requirement_provider(
      provider_capacity, providers, inout_provider_count,
      cooperative_matrix_provider));
#endif  // LOOM_BENCHMARK_HAVE_SPIRV
#if !LOOM_BENCHMARK_HAVE_AMDGPU && !LOOM_BENCHMARK_HAVE_SPIRV
  (void)hal_context;
  (void)provider_capacity;
  (void)providers;
  (void)inout_provider_count;
#endif  // !LOOM_BENCHMARK_HAVE_AMDGPU && !LOOM_BENCHMARK_HAVE_SPIRV
  return iree_ok_status();
}

int main(int argc, char** argv) {
  loomc_target_environment_t* target_environment = NULL;
  iree_status_t status =
      iree_status_from_loomc(loomc_target_environment_create_configured(
          loomc_allocator_system(), &target_environment));
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    return 1;
  }
#if LOOM_BENCHMARK_HAVE_ANY_DEVICE_PROVIDER
  const loom_run_hal_target_route_t hal_target_routes[] = {
#if LOOM_BENCHMARK_HAVE_AMDGPU
      {
          .driver_name = IREE_SV("legacy-amdgpu"),
          .provider = loomc_amdgpu_iree_hal_target_provider(),
      },
      {
          .driver_name = IREE_SV("amdgpu"),
          .provider = loomc_amdgpu_iree_hal_target_provider(),
      },
      {
          .driver_name = IREE_SV("rdna"),
          .provider = loomc_amdgpu_iree_hal_target_provider(),
      },
      {
          .driver_name = IREE_SV("cdna"),
          .provider = loomc_amdgpu_iree_hal_target_provider(),
      },
#endif  // LOOM_BENCHMARK_HAVE_AMDGPU
#if LOOM_BENCHMARK_HAVE_SPIRV
      {
          .driver_name = IREE_SV("vulkan"),
          .provider = loomc_spirv_iree_hal_target_provider(),
      },
#endif  // LOOM_BENCHMARK_HAVE_SPIRV
#if LOOM_BENCHMARK_HAVE_XDNA
      {
          .driver_name = IREE_SV("xdna"),
          .provider = loomc_xdna_iree_hal_target_provider(),
      },
#endif  // LOOM_BENCHMARK_HAVE_XDNA
#if LOOM_BENCHMARK_HAVE_TASK
      {
          .driver_name = IREE_SV("task"),
          .provider = loomc_cpu_iree_hal_target_provider(),
      },
#endif  // LOOM_BENCHMARK_HAVE_TASK
  };
#endif  // LOOM_BENCHMARK_HAVE_ANY_DEVICE_PROVIDER

  loom_benchmark_configuration_t configuration = {
      .input_formats = loom_configured_input_loomc_formats(),
      .tool_name = "loom-benchmark",
      .target_environment = target_environment,
      .import = loom_configured_input_loomc_importer(),
#if LOOM_BENCHMARK_HAVE_ANY_DEVICE_PROVIDER
      .hal_target_routes = hal_target_routes,
      .hal_target_route_count = IREE_ARRAYSIZE(hal_target_routes),
#endif  // LOOM_BENCHMARK_HAVE_ANY_DEVICE_PROVIDER
      .populate_requirement_providers =
          {
              .fn = loom_benchmark_populate_requirement_providers,
          },
  };
#if LOOM_BENCHMARK_HAVE_VM
  loom_vm_testbench_t vm_testbench;
  status = loom_vm_testbench_initialize(target_environment,
                                        iree_allocator_system(), &vm_testbench);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    loom_vm_testbench_deinitialize(&vm_testbench);
    loomc_target_environment_release(target_environment);
    return 1;
  }
  configuration.function_call_provider =
      (loom_testbench_function_call_provider_callback_t){
          .fn = loom_vm_testbench_invocation_provider,
          .user_data = &vm_testbench,
      };
  configuration.scenario_oracle_profile =
      (loom_testbench_bind_scenario_profile_callback_t){
          .fn = loom_vm_testbench_execution_profile,
          .user_data = &vm_testbench,
      };
#endif  // LOOM_BENCHMARK_HAVE_VM
  int exit_code = loom_benchmark_main(argc, argv, &configuration);
#if LOOM_BENCHMARK_HAVE_VM
  loom_vm_testbench_deinitialize(&vm_testbench);
#endif  // LOOM_BENCHMARK_HAVE_VM
  loomc_target_environment_release(target_environment);
  return exit_code;
}

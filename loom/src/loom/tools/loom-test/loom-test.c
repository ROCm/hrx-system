// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// loom-test binary with build-selected execution providers.

#include <stddef.h>
#include <stdio.h>

#include "loom/tooling/execution/hal/testbench_actual.h"
#include "loom/tooling/input/loomc_configured.h"
#include "loom/tools/loom-test/main.h"
#include "loomc/iree.h"
#include "loomc/target/configured.h"

#ifndef LOOM_TEST_HAVE_AMDGPU
#define LOOM_TEST_HAVE_AMDGPU 0
#endif  // LOOM_TEST_HAVE_AMDGPU
#ifndef LOOM_TEST_HAVE_SPIRV
#define LOOM_TEST_HAVE_SPIRV 0
#endif  // LOOM_TEST_HAVE_SPIRV
#ifndef LOOM_TEST_HAVE_XDNA
#define LOOM_TEST_HAVE_XDNA 0
#endif  // LOOM_TEST_HAVE_XDNA
#ifndef LOOM_TEST_HAVE_VM
#define LOOM_TEST_HAVE_VM 0
#endif  // LOOM_TEST_HAVE_VM
#ifndef LOOM_TEST_HAVE_WASM
#define LOOM_TEST_HAVE_WASM 0
#endif  // LOOM_TEST_HAVE_WASM

#ifndef LOOM_TEST_HAVE_TASK
#define LOOM_TEST_HAVE_TASK 0
#endif  // LOOM_TEST_HAVE_TASK

#define LOOM_TEST_HAVE_ANY_DEVICE_PROVIDER                                 \
  (LOOM_TEST_HAVE_AMDGPU || LOOM_TEST_HAVE_SPIRV || LOOM_TEST_HAVE_XDNA || \
   LOOM_TEST_HAVE_TASK)

#if LOOM_TEST_HAVE_AMDGPU
#include "loom/tooling/target/amdgpu/testbench_requirements.h"
#include "loomc/target/amdgpu/iree_hal.h"
#endif  // LOOM_TEST_HAVE_AMDGPU
#if LOOM_TEST_HAVE_SPIRV
#include "loom/tooling/target/spirv/testbench_requirements.h"
#include "loomc/target/spirv/iree_hal.h"
#endif  // LOOM_TEST_HAVE_SPIRV
#if LOOM_TEST_HAVE_XDNA
#include "loomc/target/amd/xdna/iree_hal.h"
#endif  // LOOM_TEST_HAVE_XDNA
#if LOOM_TEST_HAVE_TASK
#include "loomc/target/cpu/iree_hal.h"
#endif  // LOOM_TEST_HAVE_TASK
#if LOOM_TEST_HAVE_VM
#include "loom/tooling/target/vm/testbench.h"
#endif  // LOOM_TEST_HAVE_VM
#if LOOM_TEST_HAVE_WASM && defined(IREE_PLATFORM_WASM)
#include "loom/tooling/target/wasm/testbench.h"
#endif  // LOOM_TEST_HAVE_WASM && IREE_PLATFORM_WASM

#if LOOM_TEST_HAVE_AMDGPU || LOOM_TEST_HAVE_SPIRV
static iree_status_t loom_test_append_requirement_provider(
    iree_host_size_t provider_capacity,
    loom_testbench_requirement_provider_t* providers,
    iree_host_size_t* inout_provider_count,
    loom_testbench_requirement_provider_t provider) {
  if (*inout_provider_count >= provider_capacity) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "loom-test requirement provider capacity exceeded");
  }
  providers[(*inout_provider_count)++] = provider;
  return iree_ok_status();
}
#endif  // LOOM_TEST_HAVE_AMDGPU || LOOM_TEST_HAVE_SPIRV

static iree_status_t loom_test_populate_requirement_providers(
    void* user_data, loom_run_hal_testbench_context_t* hal_context,
    iree_host_size_t provider_capacity,
    loom_testbench_requirement_provider_t* providers,
    iree_host_size_t* inout_provider_count) {
  (void)user_data;
#if LOOM_TEST_HAVE_AMDGPU
  loom_testbench_requirement_provider_t descriptor_set_provider = {0};
  loom_amdgpu_hal_descriptor_set_testbench_requirement_provider_initialize(
      hal_context, &descriptor_set_provider);
  IREE_RETURN_IF_ERROR(loom_test_append_requirement_provider(
      provider_capacity, providers, inout_provider_count,
      descriptor_set_provider));
  loom_testbench_requirement_provider_t target_provider = {0};
  loom_amdgpu_hal_target_testbench_requirement_provider_initialize(
      hal_context, &target_provider);
  IREE_RETURN_IF_ERROR(loom_test_append_requirement_provider(
      provider_capacity, providers, inout_provider_count, target_provider));
#endif  // LOOM_TEST_HAVE_AMDGPU
#if LOOM_TEST_HAVE_SPIRV
  loom_testbench_requirement_provider_t vulkan_feature_provider = {0};
  loom_spirv_vulkan_feature_testbench_requirement_provider_initialize(
      hal_context, &vulkan_feature_provider);
  IREE_RETURN_IF_ERROR(loom_test_append_requirement_provider(
      provider_capacity, providers, inout_provider_count,
      vulkan_feature_provider));
  loom_testbench_requirement_provider_t cooperative_matrix_provider = {0};
  loom_spirv_vulkan_cooperative_matrix_testbench_requirement_provider_initialize(
      hal_context, &cooperative_matrix_provider);
  IREE_RETURN_IF_ERROR(loom_test_append_requirement_provider(
      provider_capacity, providers, inout_provider_count,
      cooperative_matrix_provider));
#endif  // LOOM_TEST_HAVE_SPIRV
#if !LOOM_TEST_HAVE_AMDGPU && !LOOM_TEST_HAVE_SPIRV
  (void)hal_context;
  (void)provider_capacity;
  (void)providers;
  (void)inout_provider_count;
#endif  // !LOOM_TEST_HAVE_AMDGPU && !LOOM_TEST_HAVE_SPIRV
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
#if LOOM_TEST_HAVE_ANY_DEVICE_PROVIDER
  const loom_run_hal_target_route_t hal_target_routes[] = {
#if LOOM_TEST_HAVE_AMDGPU
      {
          .driver_name = IREE_SV("legacy-amdgpu"),
          .provider = loomc_amdgpu_iree_hal_target_provider(),
      },
      {
          .driver_name = IREE_SV("amdgpu2"),
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
#endif  // LOOM_TEST_HAVE_AMDGPU
#if LOOM_TEST_HAVE_SPIRV
      {
          .driver_name = IREE_SV("vulkan"),
          .provider = loomc_spirv_iree_hal_target_provider(),
      },
#endif  // LOOM_TEST_HAVE_SPIRV
#if LOOM_TEST_HAVE_XDNA
      {
          .driver_name = IREE_SV("xdna"),
          .provider = loomc_xdna_iree_hal_target_provider(),
      },
#endif  // LOOM_TEST_HAVE_XDNA
#if LOOM_TEST_HAVE_TASK
      {
          .driver_name = IREE_SV("task"),
          .provider = loomc_cpu_iree_hal_target_provider(),
      },
#endif  // LOOM_TEST_HAVE_TASK
  };
#endif  // LOOM_TEST_HAVE_ANY_DEVICE_PROVIDER
  loom_test_configuration_t configuration = {
      .input_formats = loom_configured_input_loomc_formats(),
      .tool_name = "loom-test",
      .target_environment = target_environment,
      .import = loom_configured_input_loomc_importer(),
#if LOOM_TEST_HAVE_ANY_DEVICE_PROVIDER
      .hal_target_routes = hal_target_routes,
      .hal_target_route_count = IREE_ARRAYSIZE(hal_target_routes),
#endif  // LOOM_TEST_HAVE_ANY_DEVICE_PROVIDER
      .populate_requirement_providers =
          {
              .fn = loom_test_populate_requirement_providers,
          },
  };
#if LOOM_TEST_HAVE_VM
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
  configuration.function_call_provider.fn =
      loom_vm_testbench_invocation_provider;
  configuration.function_call_provider.user_data = &vm_testbench;
  configuration.scenario_target_profile.fn =
      loom_vm_testbench_execution_profile;
  configuration.scenario_target_profile.user_data = &vm_testbench;
  configuration.scenario_oracle_profile.fn =
      loom_vm_testbench_execution_profile;
  configuration.scenario_oracle_profile.user_data = &vm_testbench;
#endif  // LOOM_TEST_HAVE_VM
#if LOOM_TEST_HAVE_WASM && defined(IREE_PLATFORM_WASM)
  loom_wasm_testbench_t wasm_testbench;
  status = loom_wasm_testbench_initialize(
      target_environment, iree_allocator_system(), &wasm_testbench);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    loom_wasm_testbench_deinitialize(&wasm_testbench);
#if LOOM_TEST_HAVE_VM
    loom_vm_testbench_deinitialize(&vm_testbench);
#endif  // LOOM_TEST_HAVE_VM
    loomc_target_environment_release(target_environment);
    return 1;
  }
  configuration.scenario_target_profile.fn =
      loom_wasm_testbench_execution_profile;
  configuration.scenario_target_profile.user_data = &wasm_testbench;
#endif  // LOOM_TEST_HAVE_WASM && IREE_PLATFORM_WASM
  int exit_code = loom_test_main(argc, argv, &configuration);
#if LOOM_TEST_HAVE_VM
  loom_vm_testbench_deinitialize(&vm_testbench);
#endif  // LOOM_TEST_HAVE_VM
#if LOOM_TEST_HAVE_WASM && defined(IREE_PLATFORM_WASM)
  loom_wasm_testbench_deinitialize(&wasm_testbench);
#endif  // LOOM_TEST_HAVE_WASM && IREE_PLATFORM_WASM
  loomc_target_environment_release(target_environment);
  return exit_code;
}

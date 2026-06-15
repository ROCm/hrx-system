// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// iree-test-loom binary with build-selected execution providers.

#include <stddef.h>
#include <stdio.h>

#include "loom/tooling/execution/execution_provider.h"
#include "loom/tools/iree-test-loom/main.h"

#ifndef IREE_TEST_LOOM_HAVE_AMDGPU
#define IREE_TEST_LOOM_HAVE_AMDGPU 0
#endif  // IREE_TEST_LOOM_HAVE_AMDGPU
#ifndef IREE_TEST_LOOM_HAVE_IREE_VM
#define IREE_TEST_LOOM_HAVE_IREE_VM 0
#endif  // IREE_TEST_LOOM_HAVE_IREE_VM
#ifndef IREE_TEST_LOOM_HAVE_SPIRV
#define IREE_TEST_LOOM_HAVE_SPIRV 0
#endif  // IREE_TEST_LOOM_HAVE_SPIRV

#define IREE_TEST_LOOM_HAVE_ANY_PROVIDER                        \
  (IREE_TEST_LOOM_HAVE_AMDGPU || IREE_TEST_LOOM_HAVE_IREE_VM || \
   IREE_TEST_LOOM_HAVE_SPIRV)
#define IREE_TEST_LOOM_HAVE_ANY_HAL_ARTIFACT_PROVIDER \
  (IREE_TEST_LOOM_HAVE_AMDGPU || IREE_TEST_LOOM_HAVE_SPIRV)

#if IREE_TEST_LOOM_HAVE_AMDGPU
#include "loom/tooling/target/amdgpu/artifact_provider.h"
#include "loom/tooling/target/amdgpu/execution/provider.h"
#include "loom/tooling/target/amdgpu/testbench_requirements.h"
#endif  // IREE_TEST_LOOM_HAVE_AMDGPU
#if IREE_TEST_LOOM_HAVE_IREE_VM
#include "loom/tooling/execution/ireevm/provider.h"
#endif  // IREE_TEST_LOOM_HAVE_IREE_VM
#if IREE_TEST_LOOM_HAVE_SPIRV
#include "loom/tooling/target/spirv/artifact_provider.h"
#include "loom/tooling/target/spirv/execution/provider.h"
#include "loom/tooling/target/spirv/testbench_requirements.h"
#endif  // IREE_TEST_LOOM_HAVE_SPIRV

#if IREE_TEST_LOOM_HAVE_ANY_PROVIDER
static const loom_run_execution_provider_t* const kIreeTestLoomProviders[] = {
#if IREE_TEST_LOOM_HAVE_AMDGPU
    &loom_amdgpu_hal_execution_provider,
#endif  // IREE_TEST_LOOM_HAVE_AMDGPU
#if IREE_TEST_LOOM_HAVE_IREE_VM
    &loom_ireevm_execution_provider,
#endif  // IREE_TEST_LOOM_HAVE_IREE_VM
#if IREE_TEST_LOOM_HAVE_SPIRV
    &loom_spirv_vulkan_hal_execution_provider,
#endif  // IREE_TEST_LOOM_HAVE_SPIRV
};
#endif  // IREE_TEST_LOOM_HAVE_ANY_PROVIDER

static const loom_run_execution_provider_set_t kIreeTestLoomProviderSet = {
#if IREE_TEST_LOOM_HAVE_ANY_PROVIDER
    .providers = kIreeTestLoomProviders,
    .provider_count = IREE_ARRAYSIZE(kIreeTestLoomProviders),
#else
    .providers = NULL,
    .provider_count = 0,
#endif  // IREE_TEST_LOOM_HAVE_ANY_PROVIDER
};

#if IREE_TEST_LOOM_HAVE_ANY_HAL_ARTIFACT_PROVIDER
static const loom_run_hal_artifact_provider_t* const
    kIreeTestLoomHalArtifactProviders[] = {
#if IREE_TEST_LOOM_HAVE_AMDGPU
        &loom_amdgpu_hal_artifact_provider,
#endif  // IREE_TEST_LOOM_HAVE_AMDGPU
#if IREE_TEST_LOOM_HAVE_SPIRV
        &loom_spirv_vulkan_hal_artifact_provider,
#endif  // IREE_TEST_LOOM_HAVE_SPIRV
};
#endif  // IREE_TEST_LOOM_HAVE_ANY_HAL_ARTIFACT_PROVIDER

static const loom_run_hal_artifact_provider_registry_t
    kIreeTestLoomHalArtifactProviderRegistry = {
#if IREE_TEST_LOOM_HAVE_ANY_HAL_ARTIFACT_PROVIDER
        .providers = kIreeTestLoomHalArtifactProviders,
        .provider_count = IREE_ARRAYSIZE(kIreeTestLoomHalArtifactProviders),
#else
        .providers = NULL,
        .provider_count = 0,
#endif  // IREE_TEST_LOOM_HAVE_ANY_HAL_ARTIFACT_PROVIDER
};

#if IREE_TEST_LOOM_HAVE_AMDGPU || IREE_TEST_LOOM_HAVE_SPIRV
static iree_status_t iree_test_loom_append_requirement_provider(
    iree_host_size_t provider_capacity,
    loom_testbench_requirement_provider_t* providers,
    iree_host_size_t* inout_provider_count,
    loom_testbench_requirement_provider_t provider) {
  if (*inout_provider_count >= provider_capacity) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "iree-test-loom requirement provider capacity exceeded");
  }
  providers[(*inout_provider_count)++] = provider;
  return iree_ok_status();
}
#endif  // IREE_TEST_LOOM_HAVE_AMDGPU || IREE_TEST_LOOM_HAVE_SPIRV

static iree_status_t iree_test_loom_populate_requirement_providers(
    void* user_data, loom_run_hal_testbench_context_t* hal_context,
    iree_host_size_t provider_capacity,
    loom_testbench_requirement_provider_t* providers,
    iree_host_size_t* inout_provider_count) {
  (void)user_data;
#if IREE_TEST_LOOM_HAVE_AMDGPU
  loom_testbench_requirement_provider_t amdgpu_provider = {0};
  loom_amdgpu_hal_testbench_requirement_provider_initialize(hal_context,
                                                            &amdgpu_provider);
  IREE_RETURN_IF_ERROR(iree_test_loom_append_requirement_provider(
      provider_capacity, providers, inout_provider_count, amdgpu_provider));
#endif  // IREE_TEST_LOOM_HAVE_AMDGPU
#if IREE_TEST_LOOM_HAVE_SPIRV
  loom_testbench_requirement_provider_t spirv_provider = {0};
  loom_spirv_vulkan_hal_testbench_requirement_provider_initialize(
      hal_context, &spirv_provider);
  IREE_RETURN_IF_ERROR(iree_test_loom_append_requirement_provider(
      provider_capacity, providers, inout_provider_count, spirv_provider));
#endif  // IREE_TEST_LOOM_HAVE_SPIRV
#if !IREE_TEST_LOOM_HAVE_AMDGPU && !IREE_TEST_LOOM_HAVE_SPIRV
  (void)hal_context;
  (void)provider_capacity;
  (void)providers;
  (void)inout_provider_count;
#endif  // !IREE_TEST_LOOM_HAVE_AMDGPU && !IREE_TEST_LOOM_HAVE_SPIRV
  return iree_ok_status();
}

int main(int argc, char** argv) {
  loom_run_execution_environment_t environment;
  iree_status_t status = loom_run_execution_environment_initialize(
      &kIreeTestLoomProviderSet, &environment);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    return 1;
  }

  const iree_test_loom_configuration_t configuration = {
      .tool_name = "iree-test-loom",
      .register_context =
          loom_run_execution_environment_register_context_callback(
              &environment),
      .target_environment =
          loom_run_execution_environment_target_environment(&environment),
      .hal_artifact_provider_registry =
          &kIreeTestLoomHalArtifactProviderRegistry,
      .populate_requirement_providers =
          {
              .fn = iree_test_loom_populate_requirement_providers,
          },
      .initialize_low_descriptor_registry =
          loom_run_execution_environment_low_descriptor_registry_callback(
              &environment),
  };
  int exit_code = iree_test_loom_main(argc, argv, &configuration);
  loom_run_execution_environment_deinitialize(&environment);
  return exit_code;
}

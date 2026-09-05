// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/execution/configured_testbench.h"

#ifndef LOOM_CONFIG_EXECUTION_HAVE_AMDGPU
#define LOOM_CONFIG_EXECUTION_HAVE_AMDGPU 0
#endif  // LOOM_CONFIG_EXECUTION_HAVE_AMDGPU
#ifndef LOOM_CONFIG_EXECUTION_HAVE_SPIRV
#define LOOM_CONFIG_EXECUTION_HAVE_SPIRV 0
#endif  // LOOM_CONFIG_EXECUTION_HAVE_SPIRV

#define LOOM_CONFIG_EXECUTION_HAVE_ANY_REQUIREMENT_PROVIDER \
  (LOOM_CONFIG_EXECUTION_HAVE_AMDGPU || LOOM_CONFIG_EXECUTION_HAVE_SPIRV)

#if LOOM_CONFIG_EXECUTION_HAVE_AMDGPU
#include "loom/tooling/target/amdgpu/testbench_requirements.h"
#endif  // LOOM_CONFIG_EXECUTION_HAVE_AMDGPU
#if LOOM_CONFIG_EXECUTION_HAVE_SPIRV
#include "loom/tooling/target/spirv/testbench_requirements.h"
#endif  // LOOM_CONFIG_EXECUTION_HAVE_SPIRV

#if LOOM_CONFIG_EXECUTION_HAVE_ANY_REQUIREMENT_PROVIDER
static const loom_run_hal_testbench_requirement_provider_initializer_t
    kConfiguredRequirementProviderInitializers[] = {
#if LOOM_CONFIG_EXECUTION_HAVE_AMDGPU
        loom_amdgpu_hal_testbench_requirement_provider_initialize,
#endif  // LOOM_CONFIG_EXECUTION_HAVE_AMDGPU
#if LOOM_CONFIG_EXECUTION_HAVE_SPIRV
        loom_spirv_vulkan_feature_testbench_requirement_provider_initialize,
        loom_spirv_vulkan_cooperative_matrix_testbench_requirement_provider_initialize,
#endif  // LOOM_CONFIG_EXECUTION_HAVE_SPIRV
};
#endif  // LOOM_CONFIG_EXECUTION_HAVE_ANY_REQUIREMENT_PROVIDER

static const loom_run_hal_testbench_requirement_initializer_set_t
    kConfiguredRequirementProviderInitializerSet = {
#if LOOM_CONFIG_EXECUTION_HAVE_ANY_REQUIREMENT_PROVIDER
        .initializers = kConfiguredRequirementProviderInitializers,
        .initializer_count =
            IREE_ARRAYSIZE(kConfiguredRequirementProviderInitializers),
#else
        .initializers = NULL,
        .initializer_count = 0,
#endif  // LOOM_CONFIG_EXECUTION_HAVE_ANY_REQUIREMENT_PROVIDER
};

const loom_run_hal_testbench_requirement_initializer_set_t*
loom_tooling_configured_testbench_requirement_initializers(void) {
  return &kConfiguredRequirementProviderInitializerSet;
}

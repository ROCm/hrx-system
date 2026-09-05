// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/execution/configured.h"

#ifndef LOOM_CONFIG_EXECUTION_HAVE_AMDGPU
#define LOOM_CONFIG_EXECUTION_HAVE_AMDGPU 0
#endif  // LOOM_CONFIG_EXECUTION_HAVE_AMDGPU
#ifndef LOOM_CONFIG_EXECUTION_HAVE_SPIRV
#define LOOM_CONFIG_EXECUTION_HAVE_SPIRV 0
#endif  // LOOM_CONFIG_EXECUTION_HAVE_SPIRV

#define LOOM_CONFIG_EXECUTION_HAVE_ANY_PROVIDER \
  (LOOM_CONFIG_EXECUTION_HAVE_AMDGPU || LOOM_CONFIG_EXECUTION_HAVE_SPIRV)

#if LOOM_CONFIG_EXECUTION_HAVE_AMDGPU
#include "loom/tooling/target/amdgpu/execution_provider.h"
#endif  // LOOM_CONFIG_EXECUTION_HAVE_AMDGPU
#if LOOM_CONFIG_EXECUTION_HAVE_SPIRV
#include "loom/tooling/target/spirv/execution_provider.h"
#endif  // LOOM_CONFIG_EXECUTION_HAVE_SPIRV

#if LOOM_CONFIG_EXECUTION_HAVE_ANY_PROVIDER
static const loom_run_execution_provider_t* const
    kConfiguredExecutionProviderEntries[] = {
#if LOOM_CONFIG_EXECUTION_HAVE_AMDGPU
        &loom_amdgpu_execution_provider,
#endif  // LOOM_CONFIG_EXECUTION_HAVE_AMDGPU
#if LOOM_CONFIG_EXECUTION_HAVE_SPIRV
        &loom_spirv_vulkan_execution_provider,
#endif  // LOOM_CONFIG_EXECUTION_HAVE_SPIRV
};
#endif  // LOOM_CONFIG_EXECUTION_HAVE_ANY_PROVIDER

static const loom_run_execution_provider_set_t kConfiguredExecutionProviderSet =
    {
#if LOOM_CONFIG_EXECUTION_HAVE_ANY_PROVIDER
        .providers = kConfiguredExecutionProviderEntries,
        .provider_count = IREE_ARRAYSIZE(kConfiguredExecutionProviderEntries),
#else
        .providers = NULL,
        .provider_count = 0,
#endif  // LOOM_CONFIG_EXECUTION_HAVE_ANY_PROVIDER
};

const loom_run_execution_provider_set_t*
loom_tooling_configured_execution_providers(void) {
  return &kConfiguredExecutionProviderSet;
}

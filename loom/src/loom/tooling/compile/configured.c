// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/compile/configured.h"

#include "iree/base/threading/call_once.h"
#include "loom/target/arch/cmd/provider.h"
#include "loom/target/configured/provider_set.h"

#ifndef LOOM_CONFIG_COMPILE_HAVE_AMDGPU_ARTIFACTS
#define LOOM_CONFIG_COMPILE_HAVE_AMDGPU_ARTIFACTS 0
#endif  // LOOM_CONFIG_COMPILE_HAVE_AMDGPU_ARTIFACTS
#ifndef LOOM_CONFIG_COMPILE_HAVE_SPIRV_ARTIFACTS
#define LOOM_CONFIG_COMPILE_HAVE_SPIRV_ARTIFACTS 0
#endif  // LOOM_CONFIG_COMPILE_HAVE_SPIRV_ARTIFACTS
#ifndef LOOM_CONFIG_COMPILE_HAVE_LLVMIR_ARTIFACTS
#define LOOM_CONFIG_COMPILE_HAVE_LLVMIR_ARTIFACTS 0
#endif  // LOOM_CONFIG_COMPILE_HAVE_LLVMIR_ARTIFACTS

#define LOOM_CONFIG_COMPILE_HAVE_ANY_ARTIFACT_PROVIDER \
  (LOOM_CONFIG_COMPILE_HAVE_AMDGPU_ARTIFACTS ||        \
   LOOM_CONFIG_COMPILE_HAVE_SPIRV_ARTIFACTS)

#if LOOM_CONFIG_COMPILE_HAVE_AMDGPU_ARTIFACTS
#include "loom/tooling/target/amdgpu/artifact_provider.h"
#endif  // LOOM_CONFIG_COMPILE_HAVE_AMDGPU_ARTIFACTS
#if LOOM_CONFIG_COMPILE_HAVE_SPIRV_ARTIFACTS
#include "loom/tooling/target/spirv/artifact_provider.h"
#endif  // LOOM_CONFIG_COMPILE_HAVE_SPIRV_ARTIFACTS
#if LOOM_CONFIG_COMPILE_HAVE_LLVMIR_ARTIFACTS
#include "loom/target/emit/llvmir/artifact_emitter.h"
#endif  // LOOM_CONFIG_COMPILE_HAVE_LLVMIR_ARTIFACTS

enum {
  LOOM_TOOLING_CONFIGURED_COMPILE_ADDITIONAL_TARGET_PROVIDER_COUNT =
      1 + LOOM_CONFIG_COMPILE_HAVE_LLVMIR_ARTIFACTS,
  LOOM_TOOLING_CONFIGURED_COMPILE_TARGET_PROVIDER_CAPACITY = 64,
};

typedef struct loom_tooling_configured_compile_storage_t {
  // Configured target providers plus compile-only provider contributions.
  const loom_target_provider_t* target_providers
      [LOOM_TOOLING_CONFIGURED_COMPILE_TARGET_PROVIDER_CAPACITY];
  // Number of entries in |target_providers|.
  iree_host_size_t target_provider_count;
  // Provider-set view over |target_providers|.
  loom_target_provider_set_t target_provider_set;
  // Composed compiler target environment.
  loom_target_environment_t target_environment;
  // Public borrowed view over the configured compiler providers.
  loom_tooling_compile_environment_t environment;
} loom_tooling_configured_compile_storage_t;

#if LOOM_CONFIG_COMPILE_HAVE_ANY_ARTIFACT_PROVIDER
static const loom_artifact_provider_t* const kConfiguredArtifactProviders[] = {
#if LOOM_CONFIG_COMPILE_HAVE_AMDGPU_ARTIFACTS
    &loom_amdgpu_artifact_provider,
#endif  // LOOM_CONFIG_COMPILE_HAVE_AMDGPU_ARTIFACTS
#if LOOM_CONFIG_COMPILE_HAVE_SPIRV_ARTIFACTS
    &loom_spirv_vulkan_artifact_provider,
#endif  // LOOM_CONFIG_COMPILE_HAVE_SPIRV_ARTIFACTS
};
#endif  // LOOM_CONFIG_COMPILE_HAVE_ANY_ARTIFACT_PROVIDER

static const loom_artifact_provider_registry_t
    kConfiguredArtifactProviderRegistry = {
#if LOOM_CONFIG_COMPILE_HAVE_ANY_ARTIFACT_PROVIDER
        .providers = kConfiguredArtifactProviders,
        .provider_count = IREE_ARRAYSIZE(kConfiguredArtifactProviders),
#else
        .providers = NULL,
        .provider_count = 0,
#endif  // LOOM_CONFIG_COMPILE_HAVE_ANY_ARTIFACT_PROVIDER
};

static loom_tooling_configured_compile_storage_t configured_compile_storage;
static iree_once_flag configured_compile_once = IREE_ONCE_FLAG_INIT;

static iree_status_t loom_tooling_configured_compile_initialize_storage(void) {
  const loom_target_provider_set_t* configured_target_providers =
      loom_configured_target_provider_set();
  if (configured_target_providers->provider_count >
      IREE_ARRAYSIZE(configured_compile_storage.target_providers) -
          LOOM_TOOLING_CONFIGURED_COMPILE_ADDITIONAL_TARGET_PROVIDER_COUNT) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "configured compile target provider capacity "
                            "exceeded");
  }
  for (iree_host_size_t i = 0; i < configured_target_providers->provider_count;
       ++i) {
    configured_compile_storage
        .target_providers[configured_compile_storage.target_provider_count++] =
        configured_target_providers->providers[i];
  }
  configured_compile_storage
      .target_providers[configured_compile_storage.target_provider_count++] =
      &loom_cmd_target_provider;
#if LOOM_CONFIG_COMPILE_HAVE_LLVMIR_ARTIFACTS
  configured_compile_storage
      .target_providers[configured_compile_storage.target_provider_count++] =
      &loom_llvmir_artifact_emitter_provider;
#endif  // LOOM_CONFIG_COMPILE_HAVE_LLVMIR_ARTIFACTS
  configured_compile_storage.target_provider_set =
      loom_target_provider_set_make(
          configured_compile_storage.target_providers,
          configured_compile_storage.target_provider_count);
  IREE_RETURN_IF_ERROR(loom_target_environment_initialize(
      &configured_compile_storage.target_provider_set,
      &configured_compile_storage.target_environment));
  configured_compile_storage.environment = (loom_tooling_compile_environment_t){
      .target_environment = &configured_compile_storage.target_environment,
      .artifact_provider_registry = &kConfiguredArtifactProviderRegistry,
  };
  return iree_ok_status();
}

static void loom_tooling_configured_compile_initialize_once(void) {
  IREE_CHECK_OK(loom_tooling_configured_compile_initialize_storage());
}

const loom_tooling_compile_environment_t*
loom_tooling_configured_compile_environment(void) {
  iree_call_once(&configured_compile_once,
                 loom_tooling_configured_compile_initialize_once);
  return &configured_compile_storage.environment;
}

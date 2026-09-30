// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/compile/configured.h"

#include "iree/base/threading/call_once.h"
#include "loom/target/configured/provider_set.h"
#include "loom/transforms/cleanup/configured.h"

#ifndef LOOM_CONFIG_COMPILE_HAVE_AMDGPU_ARTIFACTS
#define LOOM_CONFIG_COMPILE_HAVE_AMDGPU_ARTIFACTS 0
#endif  // LOOM_CONFIG_COMPILE_HAVE_AMDGPU_ARTIFACTS
#ifndef LOOM_CONFIG_COMPILE_HAVE_SPIRV_ARTIFACTS
#define LOOM_CONFIG_COMPILE_HAVE_SPIRV_ARTIFACTS 0
#endif  // LOOM_CONFIG_COMPILE_HAVE_SPIRV_ARTIFACTS
#ifndef LOOM_CONFIG_COMPILE_HAVE_VM_ARTIFACTS
#define LOOM_CONFIG_COMPILE_HAVE_VM_ARTIFACTS 0
#endif  // LOOM_CONFIG_COMPILE_HAVE_VM_ARTIFACTS
#ifndef LOOM_CONFIG_COMPILE_HAVE_WASM_ARTIFACTS
#define LOOM_CONFIG_COMPILE_HAVE_WASM_ARTIFACTS 0
#endif  // LOOM_CONFIG_COMPILE_HAVE_WASM_ARTIFACTS
#ifndef LOOM_CONFIG_COMPILE_HAVE_XDNA_ARTIFACTS
#define LOOM_CONFIG_COMPILE_HAVE_XDNA_ARTIFACTS 0
#endif  // LOOM_CONFIG_COMPILE_HAVE_XDNA_ARTIFACTS

#if LOOM_CONFIG_COMPILE_HAVE_AMDGPU_ARTIFACTS
#include "loom/target/emit/native/amdgpu/hal_kernel_library.h"
#endif  // LOOM_CONFIG_COMPILE_HAVE_AMDGPU_ARTIFACTS
#if LOOM_CONFIG_COMPILE_HAVE_SPIRV_ARTIFACTS
#include "loom/tooling/target/spirv/artifact_provider.h"
#endif  // LOOM_CONFIG_COMPILE_HAVE_SPIRV_ARTIFACTS
#if LOOM_CONFIG_COMPILE_HAVE_XDNA_ARTIFACTS
#include "loom/target/arch/amd/xdna/aie2p/emit/artifact.h"
#endif  // LOOM_CONFIG_COMPILE_HAVE_XDNA_ARTIFACTS
#if LOOM_CONFIG_COMPILE_HAVE_VM_ARTIFACTS
#include "loom/target/emit/vm/module_compiler.h"
#endif  // LOOM_CONFIG_COMPILE_HAVE_VM_ARTIFACTS
#if LOOM_CONFIG_COMPILE_HAVE_WASM_ARTIFACTS
#include "loom/target/emit/wasm/module_compiler.h"
#endif  // LOOM_CONFIG_COMPILE_HAVE_WASM_ARTIFACTS

typedef struct loom_tooling_configured_compile_storage_t {
  // Configured target providers plus compile-only provider contributions.
  loom_target_provider_set_storage_t target_provider_storage;
  // Composed compiler target environment.
  loom_target_environment_t target_environment;
  // Public borrowed view over the configured compiler providers.
  loom_tooling_compile_environment_t environment;
} loom_tooling_configured_compile_storage_t;

#if LOOM_CONFIG_COMPILE_HAVE_SPIRV_ARTIFACTS
static const loom_artifact_provider_t* const kConfiguredArtifactProviders[] = {
    &loom_spirv_vulkan_artifact_provider,
};
#endif  // LOOM_CONFIG_COMPILE_HAVE_SPIRV_ARTIFACTS

static const loom_artifact_provider_registry_t
    kConfiguredArtifactProviderRegistry = {
#if LOOM_CONFIG_COMPILE_HAVE_SPIRV_ARTIFACTS
        .providers = kConfiguredArtifactProviders,
        .provider_count = IREE_ARRAYSIZE(kConfiguredArtifactProviders),
#else
        .providers = NULL,
        .provider_count = 0,
#endif  // LOOM_CONFIG_COMPILE_HAVE_SPIRV_ARTIFACTS
};

static loom_tooling_configured_compile_storage_t configured_compile_storage;
static iree_once_flag configured_compile_once = IREE_ONCE_FLAG_INIT;

static iree_status_t loom_tooling_configured_compile_initialize_storage(void) {
  loom_target_provider_set_storage_initialize(
      &configured_compile_storage.target_provider_storage);
  IREE_RETURN_IF_ERROR(loom_target_provider_set_storage_append_set(
      &configured_compile_storage.target_provider_storage,
      loom_configured_target_provider_set()));
#if LOOM_CONFIG_COMPILE_HAVE_AMDGPU_ARTIFACTS
  IREE_RETURN_IF_ERROR(loom_target_provider_set_storage_append(
      &configured_compile_storage.target_provider_storage,
      &loom_amdgpu_hal_kernel_library_provider));
#endif  // LOOM_CONFIG_COMPILE_HAVE_AMDGPU_ARTIFACTS
#if LOOM_CONFIG_COMPILE_HAVE_VM_ARTIFACTS
  IREE_RETURN_IF_ERROR(loom_target_provider_set_storage_append(
      &configured_compile_storage.target_provider_storage,
      &loom_vm_module_provider));
#endif  // LOOM_CONFIG_COMPILE_HAVE_VM_ARTIFACTS
#if LOOM_CONFIG_COMPILE_HAVE_WASM_ARTIFACTS
  IREE_RETURN_IF_ERROR(loom_target_provider_set_storage_append(
      &configured_compile_storage.target_provider_storage,
      &loom_wasm_module_provider));
#endif  // LOOM_CONFIG_COMPILE_HAVE_WASM_ARTIFACTS
#if LOOM_CONFIG_COMPILE_HAVE_XDNA_ARTIFACTS
  IREE_RETURN_IF_ERROR(loom_target_provider_set_storage_append(
      &configured_compile_storage.target_provider_storage,
      &loom_aie2p_xdna_artifact_provider));
#endif  // LOOM_CONFIG_COMPILE_HAVE_XDNA_ARTIFACTS
  IREE_RETURN_IF_ERROR(loom_target_environment_initialize(
      &configured_compile_storage.target_provider_storage.provider_set,
      &configured_compile_storage.target_environment));
  configured_compile_storage.environment = (loom_tooling_compile_environment_t){
      .target_environment = &configured_compile_storage.target_environment,
      .artifact_provider_registry = &kConfiguredArtifactProviderRegistry,
      .cleanup_pattern_provider_set =
          loom_cleanup_configured_pattern_provider_set(),
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

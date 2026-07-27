// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/configured/provider.h"

#include "iree/base/threading/call_once.h"

#ifndef LOOM_CONFIG_TARGET_HAVE_AMDGPU
#define LOOM_CONFIG_TARGET_HAVE_AMDGPU 0
#endif  // LOOM_CONFIG_TARGET_HAVE_AMDGPU
#ifndef LOOM_CONFIG_TARGET_HAVE_IREE_VM
#define LOOM_CONFIG_TARGET_HAVE_IREE_VM 0
#endif  // LOOM_CONFIG_TARGET_HAVE_IREE_VM
#ifndef LOOM_CONFIG_TARGET_HAVE_LLVMIR
#define LOOM_CONFIG_TARGET_HAVE_LLVMIR 0
#endif  // LOOM_CONFIG_TARGET_HAVE_LLVMIR
#ifndef LOOM_CONFIG_TARGET_HAVE_SPIRV
#define LOOM_CONFIG_TARGET_HAVE_SPIRV 0
#endif  // LOOM_CONFIG_TARGET_HAVE_SPIRV
#ifndef LOOM_CONFIG_TARGET_HAVE_WASM
#define LOOM_CONFIG_TARGET_HAVE_WASM 0
#endif  // LOOM_CONFIG_TARGET_HAVE_WASM
#ifndef LOOM_CONFIG_TARGET_HAVE_X86
#define LOOM_CONFIG_TARGET_HAVE_X86 0
#endif  // LOOM_CONFIG_TARGET_HAVE_X86

#define LOOM_CONFIG_TARGET_HAVE_ANY_PROVIDER                            \
  (LOOM_CONFIG_TARGET_HAVE_AMDGPU || LOOM_CONFIG_TARGET_HAVE_IREE_VM || \
   LOOM_CONFIG_TARGET_HAVE_LLVMIR || LOOM_CONFIG_TARGET_HAVE_SPIRV ||   \
   LOOM_CONFIG_TARGET_HAVE_WASM || LOOM_CONFIG_TARGET_HAVE_X86)

#if LOOM_CONFIG_TARGET_HAVE_AMDGPU
#include "loom/target/arch/amdgpu/provider.h"
#endif  // LOOM_CONFIG_TARGET_HAVE_AMDGPU
#if LOOM_CONFIG_TARGET_HAVE_IREE_VM
#include "loom/target/arch/ireevm/provider.h"
#endif  // LOOM_CONFIG_TARGET_HAVE_IREE_VM
#if LOOM_CONFIG_TARGET_HAVE_LLVMIR
#include "loom/target/arch/llvmir/provider.h"
#endif  // LOOM_CONFIG_TARGET_HAVE_LLVMIR
#if LOOM_CONFIG_TARGET_HAVE_SPIRV
#include "loom/target/arch/spirv/provider.h"
#endif  // LOOM_CONFIG_TARGET_HAVE_SPIRV
#if LOOM_CONFIG_TARGET_HAVE_WASM
#include "loom/target/arch/wasm/provider.h"
#endif  // LOOM_CONFIG_TARGET_HAVE_WASM
#if LOOM_CONFIG_TARGET_HAVE_X86
#include "loom/target/arch/x86/provider.h"
#endif  // LOOM_CONFIG_TARGET_HAVE_X86

#if LOOM_CONFIG_TARGET_HAVE_ANY_PROVIDER
static const loom_target_provider_t* const kConfiguredTargetProviders[] = {
#if LOOM_CONFIG_TARGET_HAVE_AMDGPU
    &loom_amdgpu_target_provider,
#endif  // LOOM_CONFIG_TARGET_HAVE_AMDGPU
#if LOOM_CONFIG_TARGET_HAVE_IREE_VM
    &loom_ireevm_target_provider,
#endif  // LOOM_CONFIG_TARGET_HAVE_IREE_VM
#if LOOM_CONFIG_TARGET_HAVE_LLVMIR
    &loom_llvmir_target_provider,
#endif  // LOOM_CONFIG_TARGET_HAVE_LLVMIR
#if LOOM_CONFIG_TARGET_HAVE_SPIRV
    &loom_spirv_target_provider,
#endif  // LOOM_CONFIG_TARGET_HAVE_SPIRV
#if LOOM_CONFIG_TARGET_HAVE_WASM
    &loom_wasm_target_provider,
#endif  // LOOM_CONFIG_TARGET_HAVE_WASM
#if LOOM_CONFIG_TARGET_HAVE_X86
    &loom_x86_target_provider,
#endif  // LOOM_CONFIG_TARGET_HAVE_X86
};
#endif  // LOOM_CONFIG_TARGET_HAVE_ANY_PROVIDER

static const loom_target_provider_set_t kConfiguredTargetProviderSet = {
#if LOOM_CONFIG_TARGET_HAVE_ANY_PROVIDER
    .providers = kConfiguredTargetProviders,
    .provider_count = IREE_ARRAYSIZE(kConfiguredTargetProviders),
#else
    .providers = NULL,
    .provider_count = 0,
#endif  // LOOM_CONFIG_TARGET_HAVE_ANY_PROVIDER
};

static loom_target_environment_t configured_target_environment;
static iree_once_flag configured_target_environment_once = IREE_ONCE_FLAG_INIT;

static void loom_configured_target_environment_initialize_once(void) {
  IREE_CHECK_OK(loom_target_environment_initialize(
      &kConfiguredTargetProviderSet, &configured_target_environment));
}

const loom_target_provider_set_t* loom_configured_target_provider_set(void) {
  return &kConfiguredTargetProviderSet;
}

const loom_target_environment_t* loom_configured_target_environment(void) {
  iree_call_once(&configured_target_environment_once,
                 loom_configured_target_environment_initialize_once);
  return &configured_target_environment;
}

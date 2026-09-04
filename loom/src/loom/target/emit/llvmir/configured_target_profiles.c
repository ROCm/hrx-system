// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/llvmir/configured_target_profiles.h"

#ifndef LOOM_CONFIG_LLVMIR_HAVE_AMDGPU_TARGET_PROFILES
#define LOOM_CONFIG_LLVMIR_HAVE_AMDGPU_TARGET_PROFILES 0
#endif  // LOOM_CONFIG_LLVMIR_HAVE_AMDGPU_TARGET_PROFILES
#ifndef LOOM_CONFIG_LLVMIR_HAVE_X86_TARGET_PROFILES
#define LOOM_CONFIG_LLVMIR_HAVE_X86_TARGET_PROFILES 0
#endif  // LOOM_CONFIG_LLVMIR_HAVE_X86_TARGET_PROFILES

#define LOOM_CONFIG_LLVMIR_HAVE_ANY_TARGET_PROFILES  \
  (LOOM_CONFIG_LLVMIR_HAVE_AMDGPU_TARGET_PROFILES || \
   LOOM_CONFIG_LLVMIR_HAVE_X86_TARGET_PROFILES)

#if LOOM_CONFIG_LLVMIR_HAVE_AMDGPU_TARGET_PROFILES
#include "loom/target/emit/llvmir/amdgpu/target_env.h"
#endif  // LOOM_CONFIG_LLVMIR_HAVE_AMDGPU_TARGET_PROFILES
#if LOOM_CONFIG_LLVMIR_HAVE_X86_TARGET_PROFILES
#include "loom/target/emit/llvmir/x86/target_env.h"
#endif  // LOOM_CONFIG_LLVMIR_HAVE_X86_TARGET_PROFILES

#if LOOM_CONFIG_LLVMIR_HAVE_ANY_TARGET_PROFILES
static const loom_llvmir_target_profile_provider_t* const
    kConfiguredTargetProfileProviders[] = {
#if LOOM_CONFIG_LLVMIR_HAVE_X86_TARGET_PROFILES
        &loom_llvmir_x86_target_profile_provider,
#endif  // LOOM_CONFIG_LLVMIR_HAVE_X86_TARGET_PROFILES
#if LOOM_CONFIG_LLVMIR_HAVE_AMDGPU_TARGET_PROFILES
        &loom_llvmir_amdgpu_target_profile_provider,
#endif  // LOOM_CONFIG_LLVMIR_HAVE_AMDGPU_TARGET_PROFILES
};
#endif  // LOOM_CONFIG_LLVMIR_HAVE_ANY_TARGET_PROFILES

const loom_llvmir_target_profile_registry_t
    loom_llvmir_configured_target_profile_registry = {
#if LOOM_CONFIG_LLVMIR_HAVE_ANY_TARGET_PROFILES
        .providers = kConfiguredTargetProfileProviders,
        .provider_count = IREE_ARRAYSIZE(kConfiguredTargetProfileProviders),
#else
        .providers = NULL,
        .provider_count = 0,
#endif  // LOOM_CONFIG_LLVMIR_HAVE_ANY_TARGET_PROFILES
};

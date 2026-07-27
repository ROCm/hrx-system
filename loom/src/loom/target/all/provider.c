// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/all/provider.h"

#include "iree/base/threading/call_once.h"
#include "loom/target/arch/amdgpu/provider.h"
#include "loom/target/arch/ireevm/provider.h"
#include "loom/target/arch/spirv/provider.h"
#include "loom/target/arch/wasm/provider.h"
#include "loom/target/arch/x86/provider.h"

static const loom_target_provider_t* const kAllTargetProviders[] = {
    &loom_ireevm_target_provider, &loom_spirv_target_provider,
    &loom_wasm_target_provider,   &loom_x86_target_provider,
    &loom_amdgpu_target_provider,
};

static const loom_target_provider_set_t kAllTargetProviderSet = {
    .providers = kAllTargetProviders,
    .provider_count = IREE_ARRAYSIZE(kAllTargetProviders),
};

static loom_target_environment_t kAllTargetEnvironment;
static iree_once_flag kAllTargetEnvironmentOnce = IREE_ONCE_FLAG_INIT;

static void loom_all_target_environment_initialize_once(void) {
  IREE_CHECK_OK(loom_target_environment_initialize(&kAllTargetProviderSet,
                                                   &kAllTargetEnvironment));
}

const loom_target_provider_set_t* loom_all_target_provider_set(void) {
  return &kAllTargetProviderSet;
}

const loom_target_environment_t* loom_all_target_environment(void) {
  iree_call_once(&kAllTargetEnvironmentOnce,
                 loom_all_target_environment_initialize_once);
  return &kAllTargetEnvironment;
}

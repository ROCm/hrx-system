// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/llvmir/provider.h"
#include "loom/target/emit/llvmir/artifact_emitter.h"
#include "loomc/target/llvmir/base.h"
#include "target.h"

static const loom_target_provider_t* const kLoomcLlvmirTargetProviders[] = {
    &loom_llvmir_target_provider,
    &loom_llvmir_artifact_emitter_provider,
};

static const loom_target_provider_set_t loomc_llvmir_target_provider_set = {
    .providers = kLoomcLlvmirTargetProviders,
    .provider_count = IREE_ARRAYSIZE(kLoomcLlvmirTargetProviders),
};

loomc_status_t loomc_target_environment_create_llvmir(
    loomc_allocator_t allocator,
    loomc_target_environment_t** out_target_environment) {
  return loomc_target_environment_create_from_provider_set(
      &loomc_llvmir_target_provider_set, allocator, out_target_environment);
}

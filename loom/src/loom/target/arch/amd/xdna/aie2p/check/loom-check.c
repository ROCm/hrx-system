// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// loom-check binary with the independently enabled AIE2P provider.

#include <stddef.h>

#include "loom/target/arch/amd/xdna/aie2p/check/configuration.h"
#include "loom/target/arch/amd/xdna/aie2p/check/provider.h"
#include "loom/target/arch/amd/xdna/aie2p/emit/artifact.h"
#include "loom/tools/loom-check/provider.h"
#include "loom/tools/loom-check/test_provider.h"

static const loom_check_provider_t* const kAie2pCheckProviders[] = {
    &loom_check_test_provider,
    &loom_aie2p_check_provider,
};

static const loom_target_provider_t* const kAie2pEmitterProviders[] = {
    &loom_aie2p_xdna_artifact_provider,
    &loom_aie2p_configuration_check_provider,
};

static const loom_target_provider_set_t kAie2pEmitterProviderSet = {
    .providers = kAie2pEmitterProviders,
    .provider_count = IREE_ARRAYSIZE(kAie2pEmitterProviders),
};

static const loom_check_provider_set_t kAie2pCheckProviderSet = {
    .providers = kAie2pCheckProviders,
    .provider_count = IREE_ARRAYSIZE(kAie2pCheckProviders),
    .target_provider_set = &kAie2pEmitterProviderSet,
};

int main(int argc, char** argv) {
  IREE_TRACE_APP_ENTER();
  IREE_TRACE_ZONE_BEGIN(z0);
  const int exit_code = loom_check_provider_main(
      argc, argv, &kAie2pCheckProviderSet, (loom_input_provider_list_t){0});
  IREE_TRACE_ZONE_END(z0);
  IREE_TRACE_APP_EXIT(exit_code);
  return exit_code;
}

// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/testing/target_environment.h"

#include "loom/target/provider.h"
#include "loom/target/test/ir_provider.h"
#include "target.h"

loomc_status_t loomc_target_environment_create_test_ir(
    loomc_allocator_t allocator,
    loomc_target_environment_t** out_target_environment) {
  static const loom_target_provider_t* const providers[] = {
      &loom_test_ir_target_provider,
  };
  static const loom_target_provider_set_t provider_set = {
      .providers = providers,
      .provider_count = IREE_ARRAYSIZE(providers),
  };
  return loomc_target_environment_create_from_provider_set_internal(
      &provider_set, allocator, out_target_environment);
}

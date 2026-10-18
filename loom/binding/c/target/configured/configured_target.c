// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/target/configured_target.h"

#include "loom/target/configured/provider_set.h"
#include "target.h"

loomc_status_t loomc_target_environment_create_configured_target(
    loomc_allocator_t allocator,
    loomc_target_environment_t** out_target_environment) {
  return loomc_target_environment_create_from_provider_set_internal(
      loom_configured_target_provider_set(), allocator, out_target_environment);
}

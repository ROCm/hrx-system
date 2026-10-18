// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/loom-serve/activity_test.h"
#include "loomc/target/cpu/iree_hal.h"
#include "loomc/target/x86.h"

int main(int argc, char** argv) {
  const loomc_iree_hal_target_provider_t* providers[] = {
      loomc_cpu_iree_hal_target_provider(),
  };
  const loom_serve_compiler_options_t compiler = {
      .target_environment_create = loomc_target_environment_create_x86,
      .target_providers = providers,
      .target_provider_count = IREE_ARRAYSIZE(providers),
  };
  return loom_serve_activity_test_main(argc, argv, &compiler, NULL);
}

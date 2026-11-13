// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/runtime/testing/compiler.h"
#include "loomc/target/spirv/base.h"
#include "loomc/target/spirv/iree_hal.h"

const loom_runtime_test_compiler_t* loom_runtime_test_compiler(void) {
  static const loom_runtime_test_compiler_t compiler = {
      .target_environment_create = loomc_target_environment_create_spirv,
      .target_provider = loomc_spirv_iree_hal_target_provider,
  };
  return &compiler;
}

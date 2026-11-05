// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/runtime/testing/compiler.h"

iree_status_t loom_runtime_test_host_create(iree_host_size_t worker_count,
                                            iree_allocator_t host_allocator,
                                            loom_runtime_host_t** out_host) {
  const loom_runtime_test_compiler_t* compiler = loom_runtime_test_compiler();
  const loomc_iree_hal_target_provider_t* target_provider =
      compiler->target_provider();
  const loom_runtime_compiler_options_t compiler_options = {
      .target_environment_create = compiler->target_environment_create,
      .target_providers = &target_provider,
      .target_provider_count = 1,
  };
  return loom_runtime_host_create(worker_count, &compiler_options,
                                  host_allocator, out_host);
}

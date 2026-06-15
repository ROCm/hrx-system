// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/iree-benchmark-loom/context.h"

void iree_benchmark_loom_hal_context_initialize(
    const iree_benchmark_loom_configuration_t* configuration,
    iree_allocator_t host_allocator,
    iree_benchmark_loom_hal_context_t* out_context) {
  IREE_ASSERT_ARGUMENT(configuration);
  IREE_ASSERT_ARGUMENT(out_context);
  *out_context = (iree_benchmark_loom_hal_context_t){
      .configuration = configuration,
  };
  loom_run_hal_testbench_context_initialize(
      configuration->hal_artifact_provider_registry, host_allocator,
      &out_context->execution);
}

void iree_benchmark_loom_hal_context_deinitialize(
    iree_benchmark_loom_hal_context_t* context) {
  if (context == NULL) {
    return;
  }
  loom_run_hal_testbench_context_deinitialize(&context->execution);
  *context = (iree_benchmark_loom_hal_context_t){0};
}

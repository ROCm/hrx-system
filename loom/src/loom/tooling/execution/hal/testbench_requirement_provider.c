// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/execution/hal/testbench_requirement_provider.h"

iree_status_t loom_run_hal_testbench_requirement_providers_populate(
    const loom_run_hal_testbench_requirement_initializer_set_t* initializer_set,
    loom_run_hal_testbench_context_t* context,
    iree_host_size_t provider_capacity,
    loom_testbench_requirement_provider_t* providers,
    iree_host_size_t* out_provider_count) {
  IREE_ASSERT_ARGUMENT(initializer_set);
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(providers || initializer_set->initializer_count == 0);
  IREE_ASSERT_ARGUMENT(out_provider_count);
  *out_provider_count = 0;

  if (initializer_set->initializer_count > provider_capacity) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "HAL testbench requirement provider capacity exceeded");
  }
  for (iree_host_size_t i = 0; i < initializer_set->initializer_count; ++i) {
    initializer_set->initializers[i](context, &providers[i]);
  }
  *out_provider_count = initializer_set->initializer_count;
  return iree_ok_status();
}

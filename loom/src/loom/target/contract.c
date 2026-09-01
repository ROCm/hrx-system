// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/contract.h"

iree_status_t loom_target_contract_query_get_or_allocate_target_state(
    const loom_target_contract_query_environment_t* environment,
    const void* key, iree_host_size_t data_length, void** out_data) {
  IREE_ASSERT_ARGUMENT(environment);
  IREE_ASSERT_ARGUMENT(key);
  IREE_ASSERT_GT(data_length, 0);
  IREE_ASSERT_ARGUMENT(out_data);
  *out_data = NULL;
  if (environment->target_state_allocator.fn == NULL) {
    return iree_ok_status();
  }
  return environment->target_state_allocator.fn(
      environment->target_state_allocator.user_data, key, data_length,
      out_data);
}

// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/planning/storage_lease.h"

#include "loom/target/arch/amdgpu/target_info_defs.h"

static iree_status_t loom_amdgpu_storage_lease_query(
    void* user_data, const loom_low_schedule_table_t* schedule,
    const loom_low_schedule_node_t* node, loom_low_storage_lease_emit_fn_t emit,
    void* emit_user_data) {
  if (schedule == NULL || node == NULL || node->descriptor == NULL ||
      schedule->target.descriptor_set == NULL) {
    return iree_ok_status();
  }
  const loom_low_descriptor_set_t* descriptor_set =
      schedule->target.descriptor_set;
  if (descriptor_set->target_stable_id != LOOM_AMDGPU_TARGET_STABLE_ID) {
    return iree_ok_status();
  }
  return loom_low_storage_lease_query_descriptor_rows(user_data, schedule, node,
                                                      emit, emit_user_data);
}

void loom_amdgpu_storage_lease_provider(
    loom_low_storage_lease_provider_t* out_provider) {
  IREE_ASSERT_ARGUMENT(out_provider);
  *out_provider = (loom_low_storage_lease_provider_t){
      .query = loom_amdgpu_storage_lease_query,
  };
}

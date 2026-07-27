// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/abi/feedback.h"
#include "iree/hal/drivers/amdgpu/abi/tsan.h"
#include "iree/hal/drivers/amdgpu/device/support/kernel.h"
#include "iree/hal/drivers/amdgpu/device/support/tsan.h"

[[gnu::visibility("protected"), gnu::used]]
volatile iree_hal_amdgpu_feedback_config_t iree_feedback_config = {0};

[[gnu::visibility("protected"), gnu::used]]
volatile iree_hal_amdgpu_tsan_config_t iree_tsan_config = {0};

IREE_AMDGPU_ATTRIBUTE_KERNEL void export0(uint64_t* lhs, uint64_t* rhs,
                                          uint32_t c0, uint32_t c1) {
  if (c0 == 0x5453414Eu && c1 == 0x43464721u) {
    lhs[0] = iree_tsan_config.record_length;
    lhs[1] = iree_tsan_config.flags;
    lhs[2] = iree_tsan_config.shadow_base;
    lhs[3] = iree_tsan_config.shadow_size;
    lhs[4] = iree_tsan_config.workgroup_shadow_stride;
    lhs[5] = iree_tsan_config.workgroup_capacity;
    lhs[6] = iree_tsan_config.shadow_entry_size;
    lhs[7] = iree_tsan_config.memory_granule_shift;
    lhs[8] = iree_tsan_config.queue_aql_base;
    lhs[9] = iree_tsan_config.queue_aql_slot_mask;
    lhs[10] = iree_tsan_config.queue_state_base;
    lhs[11] = iree_tsan_config.shadow_slot_count;
  } else if (c0 == 0x4644424Bu && c1 == 0x43464721u) {
    lhs[0] = iree_feedback_config.record_length;
    lhs[1] = iree_feedback_config.flags;
    lhs[2] = iree_feedback_config.channel_base;
    lhs[3] = iree_feedback_config.notify_signal.handle;
    lhs[4] = iree_feedback_config.source_context;
  } else if (c0 == 0x5453414Eu && c1 == 0x52505421u) {
    lhs[0] = iree_hal_amdgpu_tsan_report_data_race(
                 &iree_feedback_config,
                 IREE_HAL_AMDGPU_TSAN_REPORT_FLAG_CURRENT_WORKITEM_LINEAR |
                     IREE_HAL_AMDGPU_TSAN_REPORT_FLAG_PRIOR_WORKITEM_LINEAR,
                 IREE_HAL_AMDGPU_TSAN_MEMORY_SPACE_WORKGROUP,
                 IREE_HAL_AMDGPU_TSAN_ACCESS_KIND_WRITE,
                 IREE_HAL_AMDGPU_TSAN_ACCESS_KIND_READ, 4,
                 0x5453414E00000002ull, 0x5453414E00000001ull, 0x20, 0xABC0,
                 0x12345678, 0, 0, 0, 5, 0, 0, 0, 0, 0, 3, 0, 0)
                 ? 1
                 : 0;
  } else {
    lhs[0] = rhs[0];
  }
}

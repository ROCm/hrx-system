// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/abi/feedback.h"
#include "iree/hal/drivers/amdgpu/device/support/kernel.h"

[[gnu::visibility("protected"), gnu::used]]
volatile iree_hal_amdgpu_feedback_config_t iree_feedback_config = {0};

IREE_AMDGPU_ATTRIBUTE_KERNEL void export0(uint64_t* lhs, uint64_t* rhs,
                                          uint32_t c0, uint32_t c1) {
  if (c0 == 0x4644424Bu && c1 == 0x43464721u) {
    lhs[0] = iree_feedback_config.record_length;
    lhs[1] = iree_feedback_config.flags;
    lhs[2] = iree_feedback_config.channel_base;
    lhs[3] = iree_feedback_config.notify_signal.handle;
    lhs[4] = iree_feedback_config.source_context;
  } else {
    lhs[0] = rhs[0];
  }
}

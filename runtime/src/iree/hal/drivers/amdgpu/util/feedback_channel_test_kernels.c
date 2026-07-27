// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/device/support/feedback.h"

typedef struct iree_hal_amdgpu_feedback_channel_test_payload_t {
  // Magic value supplied by the host.
  uint64_t magic;
  // X dimension workgroup id captured by the kernel.
  uint32_t workgroup_id_x;
  // X dimension workitem id captured by the kernel.
  uint32_t workitem_id_x;
  // Dispatch packet pointer captured by the kernel.
  uint64_t dispatch_ptr;
} iree_hal_amdgpu_feedback_channel_test_payload_t;

IREE_AMDGPU_ATTRIBUTE_KERNEL void iree_hal_amdgpu_feedback_channel_test_emit(
    const iree_hal_amdgpu_feedback_config_t* config, uint64_t magic) {
  iree_hal_amdgpu_feedback_packet_t* packet = NULL;
  if (!iree_hal_amdgpu_feedback_try_reserve(
          config, IREE_HAL_AMDGPU_FEEDBACK_PACKET_KIND_USER,
          IREE_HAL_AMDGPU_FEEDBACK_PACKET_FLAG_ASYNC,
          sizeof(iree_hal_amdgpu_feedback_channel_test_payload_t), &packet)) {
    return;
  }

  iree_hal_amdgpu_feedback_channel_test_payload_t* payload =
      (iree_hal_amdgpu_feedback_channel_test_payload_t*)
          iree_hal_amdgpu_feedback_packet_payload(packet);
  payload->magic = magic;
  payload->workgroup_id_x = iree_hal_amdgpu_device_group_id_x();
  payload->workitem_id_x = iree_hal_amdgpu_device_local_id_x();
  payload->dispatch_ptr = packet->source_dispatch_ptr;
  iree_hal_amdgpu_feedback_publish(config, packet);
}

// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/device/support/feedback.h"

#define IREE_HAL_AMDGPU_FEEDBACK_CHANNEL_TEST_MARKER_COUNT 12

typedef struct iree_hal_amdgpu_feedback_channel_test_payload_t {
  // Magic value supplied by the host.
  uint64_t magic;
  // X dimension workgroup id captured by the kernel.
  uint32_t workgroup_id_x;
  // X dimension workitem id captured by the kernel.
  uint32_t workitem_id_x;
  // Dispatch packet pointer captured by the kernel.
  uint64_t dispatch_ptr;
  // Values spanning the remaining cache lines of a TSAN-sized payload.
  uint64_t markers[IREE_HAL_AMDGPU_FEEDBACK_CHANNEL_TEST_MARKER_COUNT];
} iree_hal_amdgpu_feedback_channel_test_payload_t;
IREE_AMDGPU_STATIC_ASSERT(
    sizeof(iree_hal_amdgpu_feedback_channel_test_payload_t) == 120,
    "test payload must match the TSAN report payload size");

static inline IREE_AMDGPU_ATTRIBUTE_ALWAYS_INLINE void
iree_hal_amdgpu_feedback_channel_test_fill_packet(
    iree_hal_amdgpu_feedback_packet_t* packet, uint64_t magic,
    uint32_t workgroup_id_x, uint32_t workitem_id_x) {
  iree_hal_amdgpu_feedback_channel_test_payload_t* payload =
      (iree_hal_amdgpu_feedback_channel_test_payload_t*)
          iree_hal_amdgpu_feedback_packet_payload(packet);
  payload->magic = magic;
  payload->workgroup_id_x = workgroup_id_x;
  payload->workitem_id_x = workitem_id_x;
  payload->dispatch_ptr = packet->source_dispatch_ptr;
  for (uint32_t i = 0; i < IREE_HAL_AMDGPU_FEEDBACK_CHANNEL_TEST_MARKER_COUNT;
       ++i) {
    payload->markers[i] = magic + ((uint64_t)workitem_id_x << 32) + i;
  }
}

IREE_AMDGPU_ATTRIBUTE_KERNEL void iree_hal_amdgpu_feedback_channel_test_emit(
    const iree_hal_amdgpu_feedback_config_t* config, uint64_t magic,
    iree_amdgpu_scoped_atomic_uint32_t* release_flag) {
  const uint32_t workgroup_id_x = iree_hal_amdgpu_device_group_id_x();
  const uint32_t workitem_id_x = iree_hal_amdgpu_device_local_id_x();

  if (workitem_id_x == 0) {
    iree_hal_amdgpu_feedback_packet_t* packet = NULL;
    if (iree_hal_amdgpu_feedback_try_reserve(
            config, IREE_HAL_AMDGPU_FEEDBACK_PACKET_KIND_USER,
            IREE_HAL_AMDGPU_FEEDBACK_PACKET_FLAG_ASYNC,
            sizeof(iree_hal_amdgpu_feedback_channel_test_payload_t), &packet)) {
      iree_hal_amdgpu_feedback_channel_test_fill_packet(
          packet, magic, workgroup_id_x, workitem_id_x);
      // Publish packet zero without notifying to reproduce the legal MPSC
      // interleaving where a later producer wakes the host first.
      iree_amdgpu_scoped_atomic_store(
          (iree_amdgpu_scoped_atomic_uint32_t*)&packet->state,
          IREE_HAL_AMDGPU_FEEDBACK_PACKET_STATE_READY,
          iree_amdgpu_memory_order_release, iree_amdgpu_memory_scope_system);
    }
  }
  __builtin_amdgcn_s_barrier();

  if (workitem_id_x == 64) {
    iree_hal_amdgpu_feedback_packet_t* packet = NULL;
    if (iree_hal_amdgpu_feedback_try_reserve(
            config, IREE_HAL_AMDGPU_FEEDBACK_PACKET_KIND_USER,
            IREE_HAL_AMDGPU_FEEDBACK_PACKET_FLAG_ASYNC,
            sizeof(iree_hal_amdgpu_feedback_channel_test_payload_t), &packet)) {
      iree_hal_amdgpu_feedback_channel_test_fill_packet(
          packet, magic, workgroup_id_x, workitem_id_x);
      iree_hal_amdgpu_feedback_publish(config, packet);
    }
  }
  __builtin_amdgcn_s_barrier();

  // Keep the dispatch active while the host drains both packets. A completion
  // fence would hide the concurrent visibility contract under test.
  while (iree_amdgpu_scoped_atomic_load(release_flag,
                                        iree_amdgpu_memory_order_acquire,
                                        iree_amdgpu_memory_scope_system) == 0) {
    iree_amdgpu_yield();
  }
}

// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Device-side ASAN helpers built on top of the AMDGPU feedback channel.

#ifndef IREE_HAL_DRIVERS_AMDGPU_DEVICE_SUPPORT_ASAN_H_
#define IREE_HAL_DRIVERS_AMDGPU_DEVICE_SUPPORT_ASAN_H_

#include "iree/hal/drivers/amdgpu/abi/asan.h"  // IWYU pragma: export
#include "iree/hal/drivers/amdgpu/device/support/feedback.h"

// Attempts to publish an ASAN access report to |feedback_config|.
//
// Returns false if feedback is disabled, the feedback ring is full, or the
// runtime feedback ABI is unavailable. Fatal ASAN paths should trap after this
// helper returns regardless of whether the report was published.
static inline IREE_AMDGPU_ATTRIBUTE_ALWAYS_INLINE bool
iree_hal_amdgpu_asan_report_access(
    const volatile iree_hal_amdgpu_feedback_config_t* IREE_AMDGPU_RESTRICT
        feedback_config,
    iree_hal_amdgpu_asan_access_kind_t access_kind, uint64_t fault_address,
    uint64_t access_size, uint64_t site_id, uint64_t shadow_address,
    uint64_t shadow_value) {
  if (!feedback_config) return false;

  iree_hal_amdgpu_feedback_config_t feedback_config_snapshot;
  feedback_config_snapshot.record_length = feedback_config->record_length;
  feedback_config_snapshot.abi_version = feedback_config->abi_version;
  feedback_config_snapshot.flags = feedback_config->flags;
  feedback_config_snapshot.reserved0 = feedback_config->reserved0;
  feedback_config_snapshot.channel_base = feedback_config->channel_base;
  feedback_config_snapshot.notify_signal.handle =
      feedback_config->notify_signal.handle;
  feedback_config_snapshot.source_context = feedback_config->source_context;
  feedback_config_snapshot.reserved[0] = feedback_config->reserved[0];
  feedback_config_snapshot.reserved[1] = feedback_config->reserved[1];
  feedback_config_snapshot.reserved[2] = feedback_config->reserved[2];

  iree_hal_amdgpu_feedback_packet_t* packet = NULL;
  if (!iree_hal_amdgpu_feedback_try_reserve(
          &feedback_config_snapshot, IREE_HAL_AMDGPU_FEEDBACK_PACKET_KIND_ASAN,
          IREE_HAL_AMDGPU_FEEDBACK_PACKET_FLAG_ASYNC,
          sizeof(iree_hal_amdgpu_asan_report_t), &packet)) {
    return false;
  }

  iree_hal_amdgpu_asan_report_t* report =
      (iree_hal_amdgpu_asan_report_t*)iree_hal_amdgpu_feedback_packet_payload(
          packet);
  report->record_length = sizeof(*report);
  report->abi_version = IREE_HAL_AMDGPU_ASAN_REPORT_ABI_VERSION_0;
  report->access_kind = access_kind;
  report->flags = IREE_HAL_AMDGPU_ASAN_REPORT_FLAG_NONE;
  report->fault_address = fault_address;
  report->access_size = access_size;
  report->site_id = site_id;
  report->shadow_address = shadow_address;
  report->shadow_value = shadow_value;
  report->reserved[0] = 0;

  iree_hal_amdgpu_feedback_publish(&feedback_config_snapshot, packet);
  return true;
}

#endif  // IREE_HAL_DRIVERS_AMDGPU_DEVICE_SUPPORT_ASAN_H_

// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Device-side TSAN helpers built on top of the AMDGPU feedback channel.

#ifndef IREE_HAL_DRIVERS_AMDGPU_DEVICE_SUPPORT_TSAN_H_
#define IREE_HAL_DRIVERS_AMDGPU_DEVICE_SUPPORT_TSAN_H_

#include "iree/hal/drivers/amdgpu/abi/tsan.h"  // IWYU pragma: export
#include "iree/hal/drivers/amdgpu/device/support/feedback.h"

// Attempts to publish a TSAN data-race report to |feedback_config|.
//
// Returns false if feedback is disabled, the feedback ring is full, or the
// runtime feedback ABI is unavailable. Fatal sanitizer paths should trap after
// this helper returns regardless of whether the report was published.
static inline IREE_AMDGPU_ATTRIBUTE_ALWAYS_INLINE bool
iree_hal_amdgpu_tsan_report_data_race(
    const volatile iree_hal_amdgpu_feedback_config_t* IREE_AMDGPU_RESTRICT
        feedback_config,
    iree_hal_amdgpu_tsan_report_flags_t flags,
    iree_hal_amdgpu_tsan_memory_space_t memory_space,
    iree_hal_amdgpu_tsan_access_kind_t current_access_kind,
    iree_hal_amdgpu_tsan_access_kind_t prior_access_kind, uint32_t access_size,
    uint64_t current_site_id, uint64_t prior_site_id, uint64_t memory_address,
    uint64_t shadow_address, uint64_t shadow_value,
    uint32_t current_workgroup_id_x, uint32_t current_workgroup_id_y,
    uint32_t current_workgroup_id_z, uint32_t current_workitem_id_x,
    uint32_t current_workitem_id_y, uint32_t current_workitem_id_z,
    uint32_t prior_workgroup_id_x, uint32_t prior_workgroup_id_y,
    uint32_t prior_workgroup_id_z, uint32_t prior_workitem_id_x,
    uint32_t prior_workitem_id_y, uint32_t prior_workitem_id_z) {
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
          &feedback_config_snapshot, IREE_HAL_AMDGPU_FEEDBACK_PACKET_KIND_TSAN,
          IREE_HAL_AMDGPU_FEEDBACK_PACKET_FLAG_ASYNC,
          sizeof(iree_hal_amdgpu_tsan_report_t), &packet)) {
    return false;
  }

  iree_hal_amdgpu_tsan_report_t* report =
      (iree_hal_amdgpu_tsan_report_t*)iree_hal_amdgpu_feedback_packet_payload(
          packet);
  report->record_length = sizeof(*report);
  report->abi_version = IREE_HAL_AMDGPU_TSAN_REPORT_ABI_VERSION_0;
  report->check_kind = IREE_HAL_AMDGPU_TSAN_CHECK_KIND_DATA_RACE;
  report->flags = flags;
  report->memory_space = memory_space;
  report->current_access_kind = current_access_kind;
  report->prior_access_kind = prior_access_kind;
  report->access_size = access_size;
  report->current_site_id = current_site_id;
  report->prior_site_id = prior_site_id;
  report->memory_address = memory_address;
  report->shadow_address = shadow_address;
  report->shadow_value = shadow_value;
  report->current_workgroup_id[0] = current_workgroup_id_x;
  report->current_workgroup_id[1] = current_workgroup_id_y;
  report->current_workgroup_id[2] = current_workgroup_id_z;
  report->current_workitem_id[0] = current_workitem_id_x;
  report->current_workitem_id[1] = current_workitem_id_y;
  report->current_workitem_id[2] = current_workitem_id_z;
  report->prior_workgroup_id[0] = prior_workgroup_id_x;
  report->prior_workgroup_id[1] = prior_workgroup_id_y;
  report->prior_workgroup_id[2] = prior_workgroup_id_z;
  report->prior_workitem_id[0] = prior_workitem_id_x;
  report->prior_workitem_id[1] = prior_workitem_id_y;
  report->prior_workitem_id[2] = prior_workitem_id_z;

  iree_hal_amdgpu_feedback_publish(&feedback_config_snapshot, packet);
  return true;
}

#endif  // IREE_HAL_DRIVERS_AMDGPU_DEVICE_SUPPORT_TSAN_H_

// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/feedback_state_test_util.h"

#include <array>
#include <cstdint>

#include "iree/base/internal/atomics.h"
#include "iree/hal/device_event.h"
#include "iree/hal/drivers/amdgpu/abi/feedback.h"
#include "iree/hal/drivers/amdgpu/abi/tsan.h"
#include "iree/hal/drivers/amdgpu/feedback_state.h"
#include "iree/hal/drivers/amdgpu/logical_device.h"

iree_status_t iree_hal_amdgpu_feedback_state_test_make_tsan_fail_device_status(
    void) {
  constexpr size_t kPacketLength = sizeof(iree_hal_amdgpu_feedback_packet_t) +
                                   sizeof(iree_hal_amdgpu_tsan_report_t);
  alignas(IREE_HAL_AMDGPU_FEEDBACK_PACKET_ALIGNMENT)
      std::array<uint8_t, kPacketLength>
          storage = {};

  auto* packet =
      reinterpret_cast<iree_hal_amdgpu_feedback_packet_t*>(storage.data());
  packet->record_length = kPacketLength;
  packet->header_length = sizeof(*packet);
  packet->kind = IREE_HAL_AMDGPU_FEEDBACK_PACKET_KIND_TSAN;
  packet->source_workgroup_id_x = 3;
  packet->source_workitem_id_x = 7;

  auto* report = reinterpret_cast<iree_hal_amdgpu_tsan_report_t*>(
      storage.data() + packet->header_length);
  report->record_length = sizeof(*report);
  report->abi_version = IREE_HAL_AMDGPU_TSAN_REPORT_ABI_VERSION_0;
  report->check_kind = IREE_HAL_AMDGPU_TSAN_CHECK_KIND_DATA_RACE;
  report->memory_space = IREE_HAL_AMDGPU_TSAN_MEMORY_SPACE_GLOBAL;
  report->current_access_kind = IREE_HAL_AMDGPU_TSAN_ACCESS_KIND_WRITE;
  report->prior_access_kind = IREE_HAL_AMDGPU_TSAN_ACCESS_KIND_READ;
  report->access_size = 4;
  report->current_site_id = 0x1234;
  report->prior_site_id = 0x5678;
  report->memory_address = 0xDEAD0000;

  iree_hal_amdgpu_feedback_state_t feedback_state = {};
  feedback_state.event_sink = iree_hal_device_event_sink_discard();
  feedback_state.tsan_report_policy =
      IREE_HAL_AMDGPU_TSAN_REPORT_POLICY_FAIL_DEVICE;
  iree_status_t producer_status = iree_hal_amdgpu_feedback_state_handle_packet(
      &feedback_state, /*physical_device_ordinal=*/0, packet);
  if (iree_status_is_ok(producer_status)) return producer_status;

  iree_hal_amdgpu_logical_device_t logical_device = {};
  iree_hal_amdgpu_logical_device_error_handler(&logical_device,
                                               producer_status);
  iree_status_t observed_status =
      iree_hal_amdgpu_logical_device_check_failure(&logical_device);
  iree_status_t stored_status = (iree_status_t)iree_atomic_exchange(
      &logical_device.failure_status, 0, iree_memory_order_acq_rel);
  iree_status_free(stored_status);
  return observed_status;
}

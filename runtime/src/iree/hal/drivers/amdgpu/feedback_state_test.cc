// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/feedback_state.h"

#include <array>
#include <cstdint>
#include <cstring>

#include "iree/hal/drivers/amdgpu/abi/tsan.h"
#include "iree/hal/drivers/amdgpu/device/support/feedback.h"
#include "iree/testing/gtest.h"

namespace iree::hal::amdgpu {
namespace {

class FeedbackStateTest : public ::testing::Test {
 protected:
  void SetUp() override {
    control_.record_length = sizeof(control_);
    control_.abi_version = IREE_HAL_AMDGPU_FEEDBACK_CHANNEL_ABI_VERSION_0;
    control_.flags = IREE_HAL_AMDGPU_FEEDBACK_CONFIG_FLAG_ENABLED;
    control_.ring_base = reinterpret_cast<uintptr_t>(ring_.data());
    control_.ring_capacity = ring_.size();

    device_state_.parent = &state_;
    device_state_.physical_device_ordinal = 0;
    iree_slim_mutex_initialize(&device_state_.drain_mutex);
    device_state_.channel.control = &control_;
    device_state_.channel.config.record_length =
        sizeof(device_state_.channel.config);
    device_state_.channel.config.abi_version =
        IREE_HAL_AMDGPU_FEEDBACK_CONFIG_ABI_VERSION_0;
    device_state_.channel.config.flags =
        IREE_HAL_AMDGPU_FEEDBACK_CONFIG_FLAG_ENABLED;
    device_state_.channel.config.channel_base =
        reinterpret_cast<uintptr_t>(&control_);

    state_.is_enabled = true;
    state_.event_sink = iree_hal_device_event_sink_discard();
    state_.tsan_report_policy = IREE_HAL_AMDGPU_TSAN_REPORT_POLICY_FAIL_DEVICE;
    state_.device_state_count = 1;
    state_.device_states = &device_state_;
    state_.error_handler = CaptureError;
    state_.error_handler_user_data = this;
  }

  void TearDown() override {
    iree_status_free(error_status_);
    iree_slim_mutex_deinitialize(&device_state_.drain_mutex);
  }

  static void CaptureError(void* user_data, iree_status_t status) {
    auto* test = static_cast<FeedbackStateTest*>(user_data);
    iree_status_free(test->error_status_);
    test->error_status_ = status;
  }

  void PublishTsanDataRace() {
    iree_hal_amdgpu_feedback_packet_t* packet = nullptr;
    ASSERT_TRUE(iree_hal_amdgpu_feedback_try_reserve(
        &device_state_.channel.config,
        IREE_HAL_AMDGPU_FEEDBACK_PACKET_KIND_TSAN,
        IREE_HAL_AMDGPU_FEEDBACK_PACKET_FLAG_ASYNC,
        sizeof(iree_hal_amdgpu_tsan_report_t), &packet));
    auto* report = reinterpret_cast<iree_hal_amdgpu_tsan_report_t*>(
        iree_hal_amdgpu_feedback_packet_payload(packet));
    std::memset(report, 0, sizeof(*report));
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
    iree_hal_amdgpu_feedback_publish(&device_state_.channel.config, packet);
  }

  static constexpr size_t kRingCapacity = 1024;
  alignas(IREE_HAL_AMDGPU_FEEDBACK_PACKET_ALIGNMENT)
      std::array<uint8_t, kRingCapacity> ring_ = {};
  iree_hal_amdgpu_feedback_channel_header_t control_ = {};
  iree_hal_amdgpu_feedback_device_state_t device_state_ = {};
  iree_hal_amdgpu_feedback_state_t state_ = {};
  iree_status_t error_status_ = iree_ok_status();
};

TEST_F(FeedbackStateTest, TsanFailDeviceReportsDataLossThroughChannelDrain) {
  PublishTsanDataRace();

  iree_hal_amdgpu_feedback_state_drain_physical_device(&state_, 0);

  EXPECT_EQ(IREE_STATUS_DATA_LOSS, iree_status_code(error_status_));
  EXPECT_EQ(control_.read_tail, control_.reservation_head);
}

}  // namespace
}  // namespace iree::hal::amdgpu

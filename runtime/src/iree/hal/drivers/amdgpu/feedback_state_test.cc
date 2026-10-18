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
#include "iree/hal/drivers/amdgpu/abi/ubsan.h"
#include "iree/hal/drivers/amdgpu/device/support/feedback.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

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
    state_.event_sink = {CaptureEvent, this};
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

  static void CaptureEvent(void* user_data,
                           const iree_hal_device_event_t* event) {
    auto* test = static_cast<FeedbackStateTest*>(user_data);
    ++test->event_count_;
    test->last_event_type_ = event->type;
    if (event->type == IREE_HAL_DEVICE_EVENT_TYPE_UBSAN_REPORT &&
        event->payload.data_length >= sizeof(iree_hal_device_ubsan_report_t)) {
      std::memcpy(&test->last_ubsan_report_, event->payload.data,
                  sizeof(test->last_ubsan_report_));
    } else if (event->type == IREE_HAL_DEVICE_EVENT_TYPE_TSAN_REPORT &&
               event->payload.data_length >=
                   sizeof(iree_hal_device_tsan_report_t)) {
      std::memcpy(&test->last_tsan_report_, event->payload.data,
                  sizeof(test->last_tsan_report_));
    }
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
    auto& data_race = report->detail.data_race;
    data_race.memory_space = IREE_HAL_AMDGPU_TSAN_MEMORY_SPACE_GLOBAL;
    data_race.current_access_kind = IREE_HAL_AMDGPU_TSAN_ACCESS_KIND_WRITE;
    data_race.prior_access_kind = IREE_HAL_AMDGPU_TSAN_ACCESS_KIND_READ;
    data_race.access_size = 4;
    data_race.current_site_id = 0x1234;
    data_race.prior_site_id = 0x5678;
    data_race.memory_address = 0xDEAD0000;
    iree_hal_amdgpu_feedback_publish(&device_state_.channel.config, packet);
  }

  void PublishTsanShadowCapacity() {
    iree_hal_amdgpu_feedback_packet_t* packet = nullptr;
    ASSERT_TRUE(iree_hal_amdgpu_feedback_try_reserve(
        &device_state_.channel.config,
        IREE_HAL_AMDGPU_FEEDBACK_PACKET_KIND_TSAN,
        IREE_HAL_AMDGPU_FEEDBACK_PACKET_FLAG_ASYNC,
        sizeof(iree_hal_amdgpu_tsan_report_t), &packet));
    packet->source_dispatch_ptr = 0x12340000u;
    auto* report = reinterpret_cast<iree_hal_amdgpu_tsan_report_t*>(
        iree_hal_amdgpu_feedback_packet_payload(packet));
    std::memset(report, 0, sizeof(*report));
    report->record_length = sizeof(*report);
    report->abi_version = IREE_HAL_AMDGPU_TSAN_REPORT_ABI_VERSION_0;
    report->check_kind = IREE_HAL_AMDGPU_TSAN_CHECK_KIND_SHADOW_CAPACITY;
    auto& capacity = report->detail.shadow_capacity;
    capacity.shadow_size = 4096;
    capacity.required_shadow_end = 4112;
    capacity.workgroup_shadow_stride = 16;
    capacity.workgroup_ordinal = 256;
    capacity.workgroup_local_memory_size = 2048;
    capacity.shadow_cell_size = 16;
    capacity.shadow_history_count = 2;
    capacity.current_workgroup_id[0] = 16;
    capacity.current_workgroup_id[1] = 4;
    capacity.current_workgroup_id[2] = 1;
    iree_hal_amdgpu_feedback_publish(&device_state_.channel.config, packet);
  }

  void PublishUbsanAssertion(bool valid_report) {
    iree_hal_amdgpu_feedback_packet_t* packet = nullptr;
    ASSERT_TRUE(iree_hal_amdgpu_feedback_try_reserve(
        &device_state_.channel.config,
        IREE_HAL_AMDGPU_FEEDBACK_PACKET_KIND_UBSAN,
        IREE_HAL_AMDGPU_FEEDBACK_PACKET_FLAG_ASYNC,
        sizeof(iree_hal_amdgpu_ubsan_report_t), &packet));
    packet->source_dispatch_ptr = 0x12340000u;
    packet->source_workgroup_id_x = 7;
    packet->source_workitem_id_x = 11;
    auto* report = reinterpret_cast<iree_hal_amdgpu_ubsan_report_t*>(
        iree_hal_amdgpu_feedback_packet_payload(packet));
    std::memset(report, 0, sizeof(*report));
    report->record_length = valid_report ? sizeof(*report) : 0;
    report->abi_version = IREE_HAL_AMDGPU_UBSAN_REPORT_ABI_VERSION_0;
    report->check_kind = IREE_HAL_AMDGPU_UBSAN_CHECK_KIND_ASSERTION;
    report->site_id = 0x4567;
    report->operand0 = 0x89AB;
    report->operand1 = 0xCDEF;
    iree_hal_amdgpu_feedback_publish(&device_state_.channel.config, packet);
  }

  static constexpr size_t kRingCapacity = 1024;
  alignas(IREE_HAL_AMDGPU_FEEDBACK_PACKET_ALIGNMENT)
      std::array<uint8_t, kRingCapacity> ring_ = {};
  iree_hal_amdgpu_feedback_channel_header_t control_ = {};
  iree_hal_amdgpu_feedback_device_state_t device_state_ = {};
  iree_hal_amdgpu_feedback_state_t state_ = {};
  iree_status_t error_status_ = iree_ok_status();
  iree_host_size_t event_count_ = 0;
  iree_hal_device_event_type_t last_event_type_ =
      IREE_HAL_DEVICE_EVENT_TYPE_NONE;
  iree_hal_device_ubsan_report_t last_ubsan_report_ = {};
  iree_hal_device_tsan_report_t last_tsan_report_ = {};
};

TEST_F(FeedbackStateTest, UbsanAssertionPublishesEvent) {
  PublishUbsanAssertion(/*valid_report=*/true);

  iree_hal_amdgpu_feedback_state_drain_physical_device(&state_, 0);

  IREE_EXPECT_OK(iree_status_clone(error_status_));
  EXPECT_EQ(event_count_, 1u);
  EXPECT_EQ(last_event_type_, IREE_HAL_DEVICE_EVENT_TYPE_UBSAN_REPORT);
  EXPECT_EQ(last_ubsan_report_.record_length, sizeof(last_ubsan_report_));
  EXPECT_EQ(last_ubsan_report_.abi_version,
            IREE_HAL_DEVICE_UBSAN_REPORT_ABI_VERSION_0);
  EXPECT_EQ(last_ubsan_report_.check_kind,
            IREE_HAL_DEVICE_UBSAN_CHECK_KIND_ASSERTION);
  EXPECT_EQ(last_ubsan_report_.site_id, 0x4567u);
  EXPECT_EQ(last_ubsan_report_.operand0, 0x89ABu);
  EXPECT_EQ(last_ubsan_report_.operand1, 0xCDEFu);
  EXPECT_EQ(last_ubsan_report_.workgroup_id[0], 7u);
  EXPECT_EQ(last_ubsan_report_.workitem_id[0], 11u);
  EXPECT_EQ(last_ubsan_report_.source_dispatch_ptr, 0x12340000u);
  EXPECT_EQ(control_.read_tail, control_.reservation_head);
}

TEST_F(FeedbackStateTest, MalformedUbsanReportFailsDevice) {
  PublishUbsanAssertion(/*valid_report=*/false);

  iree_hal_amdgpu_feedback_state_drain_physical_device(&state_, 0);

  EXPECT_EQ(IREE_STATUS_DATA_LOSS, iree_status_code(error_status_));
  EXPECT_EQ(event_count_, 0u);
  EXPECT_EQ(control_.read_tail, control_.reservation_head);
}

TEST_F(FeedbackStateTest, TsanFailDeviceReportsDataLossThroughChannelDrain) {
  PublishTsanDataRace();

  iree_hal_amdgpu_feedback_state_drain_physical_device(&state_, 0);

  EXPECT_EQ(IREE_STATUS_DATA_LOSS, iree_status_code(error_status_));
  EXPECT_EQ(control_.read_tail, control_.reservation_head);
}

TEST_F(FeedbackStateTest, TsanShadowCapacityPublishesTypedResourceError) {
  PublishTsanShadowCapacity();

  iree_hal_amdgpu_feedback_state_drain_physical_device(&state_, 0);

  EXPECT_EQ(IREE_STATUS_RESOURCE_EXHAUSTED, iree_status_code(error_status_));
  EXPECT_EQ(event_count_, 1u);
  EXPECT_EQ(last_event_type_, IREE_HAL_DEVICE_EVENT_TYPE_TSAN_REPORT);
  EXPECT_EQ(last_tsan_report_.record_length, sizeof(last_tsan_report_));
  EXPECT_EQ(last_tsan_report_.abi_version,
            IREE_HAL_DEVICE_TSAN_REPORT_ABI_VERSION_0);
  EXPECT_EQ(last_tsan_report_.check_kind,
            IREE_HAL_DEVICE_TSAN_CHECK_KIND_SHADOW_CAPACITY);
  const auto& capacity = last_tsan_report_.detail.shadow_capacity;
  EXPECT_EQ(capacity.shadow_size, 4096u);
  EXPECT_EQ(capacity.required_shadow_end, 4112u);
  EXPECT_EQ(capacity.workgroup_shadow_stride, 16u);
  EXPECT_EQ(capacity.workgroup_ordinal, 256u);
  EXPECT_EQ(capacity.workgroup_local_memory_size, 2048u);
  EXPECT_EQ(capacity.shadow_cell_size, 16u);
  EXPECT_EQ(capacity.shadow_history_count, 2u);
  EXPECT_EQ(capacity.current_workgroup_id[0], 16u);
  EXPECT_EQ(capacity.current_workgroup_id[1], 4u);
  EXPECT_EQ(capacity.current_workgroup_id[2], 1u);
  EXPECT_EQ(last_tsan_report_.source_dispatch_ptr, 0x12340000u);
  EXPECT_EQ(control_.read_tail, control_.reservation_head);
}

}  // namespace
}  // namespace iree::hal::amdgpu

// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/feedback.h"

#include <array>
#include <cstring>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amd::gpu {
namespace {

class FeedbackTest : public ::testing::Test {
 protected:
  void SetUp() override {
    context_.event_sink = {Publish, this};
    feedback_.context = &context_;
    feedback_.device_id = IREE_SV("rdna");
    feedback_.control = &control_;
    feedback_.ring = ring_.data();
    control_.ring_capacity = ring_.size();
    iree_hal_amd_gpu_source_context_registry_initialize(iree_allocator_system(),
                                                        &feedback_.sources);
    IREE_ASSERT_OK(iree_hal_amd_gpu_source_context_registry_register(
        &feedback_.sources, iree_const_byte_span_empty(), &source_));
  }
  void TearDown() override {
    iree_status_free(feedback_.failure);
    iree_hal_amd_gpu_source_context_registry_deinitialize(&feedback_.sources);
  }
  static void Publish(void* user_data, const iree_hal_device_event_t* event) {
    auto& self = *static_cast<FeedbackTest*>(user_data);
    if (event->type == IREE_HAL_DEVICE_EVENT_TYPE_UBSAN_REPORT) {
      EXPECT_EQ(event->source.executable_id,
                iree_hal_amd_gpu_source_context_executable_id(self.source_));
      EXPECT_TRUE(
          iree_string_view_equal(event->source.device_id, IREE_SV("rdna")));
      EXPECT_EQ(event->implementation_payload.data_length, 128u);
      self.sequences_.push_back(event->sequence);
      self.reports_.push_back(
          *reinterpret_cast<const iree_hal_device_ubsan_report_t*>(
              event->payload.data));
    } else if (event->type == IREE_HAL_DEVICE_EVENT_TYPE_ASAN_REPORT) {
      self.asan_reports_.push_back(
          *reinterpret_cast<const iree_hal_device_asan_report_t*>(
              event->payload.data));
    } else if (event->type == IREE_HAL_DEVICE_EVENT_TYPE_TSAN_REPORT) {
      EXPECT_EQ(event->source.executable_id,
                iree_hal_amd_gpu_source_context_executable_id(self.source_));
      self.tsan_reports_.push_back(
          *reinterpret_cast<const iree_hal_device_tsan_report_t*>(
              event->payload.data));
    } else {
      EXPECT_EQ(event->type, IREE_HAL_DEVICE_EVENT_TYPE_DRIVER_FAILURE);
      self.failures_.push_back(
          reinterpret_cast<const iree_hal_device_driver_failure_event_t*>(
              event->payload.data)
              ->status_code);
    }
  }
  iree_hal_amd_gpu_feedback_packet_t* Packet(uint64_t position) {
    auto* packet = reinterpret_cast<iree_hal_amd_gpu_feedback_packet_t*>(
        ring_.data() + (position & (ring_.size() - 1)));
    std::memset(packet, 0, 128);
    packet->record_length = 128;
    packet->header_length = 64;
    packet->kind = 5;
    packet->state = 1;
    packet->sequence = position;
    packet->source_context = reinterpret_cast<uintptr_t>(source_);
    packet->source_dispatch_ptr = 0x1200000040;
    packet->source_workgroup_id_x = 7;
    packet->source_workitem_id_x = 11;
    auto* payload = reinterpret_cast<uint8_t*>(packet + 1);
    iree_unaligned_store_le_u32(payload, 64);
    iree_unaligned_store_le_u32(payload + 8,
                                IREE_HAL_DEVICE_UBSAN_CHECK_KIND_ASSERTION);
    iree_unaligned_store_le_u64(payload + 16, 19);
    iree_unaligned_store_le_u64(payload + 24, 23);
    iree_unaligned_store_le_u64(payload + 32, 29);
    return packet;
  }
  iree_hal_amd_gpu_feedback_packet_t* TsanPacket(uint64_t position) {
    auto* packet = Packet(position);
    packet->kind = 4;
    packet->record_length = 192;
    auto* payload = reinterpret_cast<uint8_t*>(packet + 1);
    std::memset(payload, 0, 128);
    iree_unaligned_store_le_u32(payload, 120);
    return packet;
  }
  // Public event destination supplied by the owning device.
  iree_hal_amd_gpu_context_t context_ = {};
  // Production consumer; the test supplies bytes at its producer boundary.
  iree_hal_amd_gpu_feedback_t feedback_ = {};
  // Native shared control layout, exercised synchronously by this fixture.
  iree_hal_amd_gpu_feedback_control_t control_ = {};
  // Report storage with enough records to cross the live-drain budget.
  alignas(64) std::array<uint8_t, 16384> ring_ = {};
  // Device-owned metadata independent of packet storage and executable
  // lifetime.
  const iree_hal_amd_gpu_source_context_t* source_ = nullptr;
  // Delivered packet generations, copied while the callback owns the view.
  std::vector<uint64_t> sequences_;
  // Delivered public report payloads.
  std::vector<iree_hal_device_ubsan_report_t> reports_;
  // Access reports copied from the shared consumer.
  std::vector<iree_hal_device_asan_report_t> asan_reports_;
  // Race reports copied before the native packet storage is reclaimed.
  std::vector<iree_hal_device_tsan_report_t> tsan_reports_;
  // Channel errors, distinct from authored diagnostic events.
  std::vector<iree_status_code_t> failures_;
};

TEST_F(FeedbackTest, TsanReportsPreserveBothAccessesAndWorkitemCoordinates) {
  auto* packet = TsanPacket(0);
  auto* payload = reinterpret_cast<uint8_t*>(packet + 1);
  iree_unaligned_store_le_u32(payload + 8,
                              IREE_HAL_DEVICE_TSAN_CHECK_KIND_DATA_RACE);
  iree_unaligned_store_le_u32(payload + 12,
                              IREE_HAL_DEVICE_TSAN_REPORT_FLAG_PRIOR_ATOMIC);
  iree_unaligned_store_le_u32(payload + 16,
                              IREE_HAL_DEVICE_TSAN_MEMORY_SPACE_WORKGROUP);
  iree_unaligned_store_le_u32(payload + 20,
                              IREE_HAL_DEVICE_TSAN_ACCESS_KIND_WRITE);
  iree_unaligned_store_le_u32(payload + 24,
                              IREE_HAL_DEVICE_TSAN_ACCESS_KIND_ATOMIC);
  iree_unaligned_store_le_u32(payload + 28, 8);
  for (uint32_t i = 0; i < 5; ++i) {
    iree_unaligned_store_le_u64(payload + 32 + i * 8,
                                UINT64_C(0x123400000000) + i);
  }
  for (uint32_t i = 0; i < 12; ++i) {
    iree_unaligned_store_le_u32(payload + 72 + i * 4, 17 + i);
  }
  control_.reservation_head = 192;
  iree_hal_amd_gpu_feedback_drain(&feedback_);
  ASSERT_EQ(tsan_reports_.size(), 1u);
  const auto& report = tsan_reports_[0];
  EXPECT_EQ(report.record_length, 128u);
  EXPECT_EQ(report.abi_version, 0u);
  EXPECT_EQ(report.check_kind, IREE_HAL_DEVICE_TSAN_CHECK_KIND_DATA_RACE);
  EXPECT_EQ(report.flags, IREE_HAL_DEVICE_TSAN_REPORT_FLAG_PRIOR_ATOMIC);
  EXPECT_EQ(report.memory_space, IREE_HAL_DEVICE_TSAN_MEMORY_SPACE_WORKGROUP);
  EXPECT_EQ(report.current_access_kind, IREE_HAL_DEVICE_TSAN_ACCESS_KIND_WRITE);
  EXPECT_EQ(report.prior_access_kind, IREE_HAL_DEVICE_TSAN_ACCESS_KIND_ATOMIC);
  EXPECT_EQ(report.access_length, 8u);
  EXPECT_EQ(report.current_site_id, UINT64_C(0x123400000000));
  EXPECT_EQ(report.prior_site_id, UINT64_C(0x123400000001));
  EXPECT_EQ(report.memory_address, UINT64_C(0x123400000002));
  EXPECT_EQ(report.shadow_address, UINT64_C(0x123400000003));
  EXPECT_EQ(report.shadow_value, UINT64_C(0x123400000004));
  for (uint32_t i = 0; i < 3; ++i) {
    EXPECT_EQ(report.current_workgroup_id[i], 17 + i);
    EXPECT_EQ(report.current_workitem_id[i], 20 + i);
    EXPECT_EQ(report.prior_workgroup_id[i], 23 + i);
    EXPECT_EQ(report.prior_workitem_id[i], 26 + i);
  }
  EXPECT_EQ(report.source_dispatch_ptr, 0x1200000040u);
  EXPECT_EQ(control_.read_tail, 192u);
  EXPECT_EQ(
      iree_atomic_load(&feedback_.failure_code, iree_memory_order_acquire),
      IREE_STATUS_OK);
}

TEST_F(FeedbackTest, TsanFailDeviceRetainsAllReadyReports) {
  feedback_.tsan_report_policy =
      IREE_HAL_AMD_GPU_SANITIZER_REPORT_POLICY_FAIL_DEVICE;
  TsanPacket(0);
  TsanPacket(192);
  control_.reservation_head = 384;
  iree_hal_amd_gpu_feedback_drain(&feedback_);
  EXPECT_EQ(tsan_reports_.size(), 2u);
  EXPECT_EQ(control_.read_tail, 384u);
  EXPECT_EQ(
      iree_atomic_load(&feedback_.failure_code, iree_memory_order_acquire),
      IREE_STATUS_DATA_LOSS);
}

TEST_F(FeedbackTest, RejectsTruncatedTsanPayloadBeforeReadingCoordinates) {
  Packet(0)->kind = 4;
  control_.reservation_head = 128;
  iree_hal_amd_gpu_feedback_drain(&feedback_);
  EXPECT_TRUE(tsan_reports_.empty());
  ASSERT_EQ(failures_.size(), 1u);
  EXPECT_EQ(failures_[0], IREE_STATUS_DATA_LOSS);
  EXPECT_EQ(control_.read_tail, 0u);
}

TEST_F(FeedbackTest, AsanReportsPreserveAccessAndSourceFields) {
  auto* packet = Packet(0);
  packet->kind = 1;
  auto* payload = reinterpret_cast<uint8_t*>(packet + 1);
  iree_unaligned_store_le_u32(payload + 8,
                              IREE_HAL_DEVICE_ASAN_ACCESS_KIND_WRITE);
  iree_unaligned_store_le_u64(payload + 16, 0x1234);
  iree_unaligned_store_le_u64(payload + 24, 4);
  iree_unaligned_store_le_u64(payload + 32, 19);
  iree_unaligned_store_le_u64(payload + 40, 0x2345);
  iree_unaligned_store_le_u64(payload + 48, 0xfa);
  control_.reservation_head = 128;
  iree_hal_amd_gpu_feedback_drain(&feedback_);
  ASSERT_EQ(asan_reports_.size(), 1u);
  const auto& report = asan_reports_[0];
  EXPECT_EQ(report.access_kind, IREE_HAL_DEVICE_ASAN_ACCESS_KIND_WRITE);
  EXPECT_EQ(report.access_length, 4u);
  EXPECT_EQ(report.fault_address, 0x1234u);
  EXPECT_EQ(report.shadow_address, 0x2345u);
  EXPECT_EQ(report.shadow_value, 0xfau);
  EXPECT_EQ(report.site_id, 19u);
  EXPECT_EQ(report.workgroup_id[0], 7u);
  EXPECT_EQ(report.workitem_id[0], 11u);
  EXPECT_EQ(report.source_dispatch_ptr, 0x1200000040u);
  EXPECT_EQ(
      iree_atomic_load(&feedback_.failure_code, iree_memory_order_acquire),
      IREE_STATUS_OK);
}

TEST_F(FeedbackTest, FailDeviceNotifiesOnceAndStillDrainsEveryReport) {
  feedback_.asan_report_policy =
      IREE_HAL_AMD_GPU_SANITIZER_REPORT_POLICY_FAIL_DEVICE;
  uint32_t notifications = 0;
  feedback_.failure_callback = [](void* user_data) {
    ++*static_cast<uint32_t*>(user_data);
  };
  feedback_.failure_user_data = &notifications;
  Packet(0)->kind = 1;
  Packet(128)->kind = 1;
  control_.reservation_head = 256;
  iree_hal_amd_gpu_feedback_drain(&feedback_);
  EXPECT_EQ(notifications, 1u);
  EXPECT_EQ(asan_reports_.size(), 2u);
  EXPECT_EQ(control_.read_tail, 256u);
  EXPECT_EQ(
      iree_atomic_load(&feedback_.failure_code, iree_memory_order_acquire),
      IREE_STATUS_ABORTED);
}

TEST_F(FeedbackTest, DrainsOnlyReadyPrefixWithoutWaitingForReservedHole) {
  Packet(0)->state = 0;
  Packet(128);
  control_.reservation_head = 256;
  EXPECT_FALSE(iree_hal_amd_gpu_feedback_poll(&feedback_, AMDF_STATUS_OK));
  EXPECT_EQ(control_.read_tail, 0u);
  EXPECT_TRUE(reports_.empty());
  Packet(0);
  iree_hal_amd_gpu_feedback_drain(&feedback_);
  EXPECT_EQ(control_.read_tail, 256u);
  ASSERT_EQ(reports_.size(), 2u);
  EXPECT_EQ(reports_[0].site_id, 19u);
  EXPECT_EQ(reports_[0].operand0, 23u);
  EXPECT_EQ(reports_[0].operand1, 29u);
  EXPECT_EQ(reports_[0].workgroup_id[0], 7u);
  EXPECT_EQ(reports_[0].workitem_id[0], 11u);
  EXPECT_EQ(reports_[0].source_dispatch_ptr, 0x1200000040u);
}

TEST_F(FeedbackTest, LiveDrainIsBoundedAndRetirementDrainsObservedPrefix) {
  for (uint64_t i = 0; i < 100; ++i) {
    Packet(i * 128);
  }
  control_.reservation_head = 12800;
  EXPECT_TRUE(iree_hal_amd_gpu_feedback_poll(&feedback_, AMDF_STATUS_OK));
  EXPECT_EQ(reports_.size(), 64u);
  EXPECT_EQ(control_.read_tail, 8192u);
  iree_hal_amd_gpu_feedback_drain(&feedback_);
  EXPECT_EQ(reports_.size(), 100u);
  EXPECT_EQ(control_.read_tail, 12800u);
}

TEST_F(FeedbackTest, StaleReadyGenerationCannotBeReplayedAfterWrap) {
  const uint64_t start = UINT64_C(0x100000000) - 128;
  control_.read_tail = start;
  control_.reservation_head = start + 256;
  Packet(start)->sequence = start - ring_.size();
  Packet(start + 128);
  iree_hal_amd_gpu_feedback_drain(&feedback_);
  EXPECT_EQ(control_.read_tail, start);
  EXPECT_TRUE(reports_.empty());
  Packet(start);
  iree_hal_amd_gpu_feedback_drain(&feedback_);
  EXPECT_EQ(control_.read_tail, start + 256);
  ASSERT_EQ(sequences_.size(), 2u);
  EXPECT_EQ(sequences_[0], start);
  EXPECT_EQ(sequences_[1], start + 128);
}

TEST_F(FeedbackTest, ReportsDropsOncePerObservedChange) {
  control_.dropped_packet_count = 3;
  iree_hal_amd_gpu_feedback_drain(&feedback_);
  iree_hal_amd_gpu_feedback_drain(&feedback_);
  ASSERT_EQ(failures_.size(), 1u);
  EXPECT_EQ(failures_[0], IREE_STATUS_RESOURCE_EXHAUSTED);
  control_.dropped_packet_count = 5;
  iree_hal_amd_gpu_feedback_drain(&feedback_);
  EXPECT_EQ(failures_.size(), 2u);
  EXPECT_EQ(feedback_.reported_drops, 5u);
}

TEST_F(FeedbackTest, ChannelFailureIsReportedOnceWithoutReusingMalformedData) {
  Packet(0)->record_length = 0;
  control_.reservation_head = 128;
  iree_hal_amd_gpu_feedback_drain(&feedback_);
  Packet(0);
  iree_hal_amd_gpu_feedback_drain(&feedback_);
  EXPECT_EQ(control_.read_tail, 0u);
  EXPECT_TRUE(reports_.empty());
  ASSERT_EQ(failures_.size(), 1u);
  EXPECT_EQ(failures_[0], IREE_STATUS_DATA_LOSS);
}

}  // namespace
}  // namespace iree::hal::amd::gpu

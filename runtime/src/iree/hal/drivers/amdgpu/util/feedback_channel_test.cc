// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/util/feedback_channel.h"

#include <cstdint>
#include <cstring>
#include <vector>

#include "iree/hal/drivers/amdgpu/device/support/feedback.h"
#include "iree/hal/drivers/amdgpu/device/support/tsan.h"
#include "iree/hal/drivers/amdgpu/util/topology.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amdgpu {
namespace {

struct TestPayload {
  // Caller-defined payload value.
  uint64_t value;
  // Caller-defined payload ordinal.
  uint32_t ordinal;
  // Reserved padding for stable test payload layout.
  uint32_t reserved;
};

struct DrainedPacket {
  // Packet sequence observed by the host drain.
  uint64_t sequence;
  // Packet kind observed by the host drain.
  uint16_t kind;
  // Packet payload copied by the host drain.
  TestPayload payload;
};

struct DrainState {
  // Captured packets in drain order.
  std::vector<DrainedPacket> packets;
};

struct TsanDrainState {
  // Packet sequence observed by the host drain.
  uint64_t sequence = 0;
  // Packet kind observed by the host drain.
  uint16_t kind = 0;
  // Packet header source dispatch pointer observed by the host drain.
  uint64_t source_dispatch_ptr = 0;
  // Captured TSAN report payload.
  iree_hal_amdgpu_tsan_report_t report = {};
};

static iree_status_t CapturePacket(
    const iree_hal_amdgpu_feedback_packet_t* packet, void* user_data) {
  DrainState* state = reinterpret_cast<DrainState*>(user_data);
  const TestPayload* payload = reinterpret_cast<const TestPayload*>(
      iree_hal_amdgpu_feedback_packet_const_payload(packet));
  DrainedPacket drained = {};
  drained.sequence = packet->sequence;
  drained.kind = packet->kind;
  drained.payload = *payload;
  state->packets.push_back(drained);
  return iree_ok_status();
}

static iree_status_t CaptureTsanPacket(
    const iree_hal_amdgpu_feedback_packet_t* packet, void* user_data) {
  TsanDrainState* state = reinterpret_cast<TsanDrainState*>(user_data);
  const auto* report = reinterpret_cast<const iree_hal_amdgpu_tsan_report_t*>(
      iree_hal_amdgpu_feedback_packet_const_payload(packet));
  state->sequence = packet->sequence;
  state->kind = packet->kind;
  state->source_dispatch_ptr = packet->source_dispatch_ptr;
  state->report = *report;
  return iree_ok_status();
}

class FeedbackChannelTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    host_allocator = iree_allocator_system();
    iree_status_t status = iree_hal_amdgpu_libhsa_initialize(
        IREE_HAL_AMDGPU_LIBHSA_FLAG_NONE, iree_string_view_list_empty(),
        host_allocator, &libhsa);
    if (!iree_status_is_ok(status)) {
      iree_status_fprint(stderr, status);
      iree_status_free(status);
      GTEST_SKIP() << "HSA not available, skipping tests";
    }
    IREE_ASSERT_OK(
        iree_hal_amdgpu_topology_initialize_with_defaults(&libhsa, &topology));
    if (topology.gpu_agent_count == 0 || topology.cpu_agent_count == 0) {
      GTEST_SKIP() << "CPU and GPU agents are required, skipping tests";
    }
  }

  static void TearDownTestSuite() {
    iree_hal_amdgpu_topology_deinitialize(&topology);
    iree_hal_amdgpu_libhsa_deinitialize(&libhsa);
  }

  void SetUp() override {
    IREE_ASSERT_OK(iree_hal_amdgpu_find_fine_global_memory_pool(
        &libhsa, topology.cpu_agents[0], &control_memory_pool_));
    IREE_ASSERT_OK(iree_hal_amdgpu_find_coarse_global_memory_pool(
        &libhsa, topology.gpu_agents[0], &ring_memory_pool_));

    iree_hal_amdgpu_feedback_channel_params_t params = {};
    params.libhsa = &libhsa;
    params.device_agent = topology.gpu_agents[0];
    params.control_memory_pool = control_memory_pool_;
    params.ring_memory_pool = ring_memory_pool_;
    params.topology = &topology;
    params.minimum_capacity = 4 * 1024;
    IREE_ASSERT_OK(
        iree_hal_amdgpu_feedback_channel_initialize(&params, &channel_));
  }

  void TearDown() override {
    iree_hal_amdgpu_feedback_channel_deinitialize(&channel_);
  }

  static iree_allocator_t host_allocator;
  static iree_hal_amdgpu_libhsa_t libhsa;
  static iree_hal_amdgpu_topology_t topology;
  // CPU fine-grained pool used for the channel control block.
  hsa_amd_memory_pool_t control_memory_pool_ = {};
  // GPU global memory pool used to create the pinned VMM packet ring.
  hsa_amd_memory_pool_t ring_memory_pool_ = {};
  // Feedback channel under test.
  iree_hal_amdgpu_feedback_channel_t channel_ = {};
};

iree_allocator_t FeedbackChannelTest::host_allocator;
iree_hal_amdgpu_libhsa_t FeedbackChannelTest::libhsa;
iree_hal_amdgpu_topology_t FeedbackChannelTest::topology;

TEST_F(FeedbackChannelTest, HostProducerDrain) {
  constexpr uint32_t kPacketCount = 4;
  for (uint32_t i = 0; i < kPacketCount; ++i) {
    iree_hal_amdgpu_feedback_packet_t* packet = nullptr;
    ASSERT_TRUE(iree_hal_amdgpu_feedback_try_reserve(
        &channel_.config, IREE_HAL_AMDGPU_FEEDBACK_PACKET_KIND_USER,
        IREE_HAL_AMDGPU_FEEDBACK_PACKET_FLAG_ASYNC, sizeof(TestPayload),
        &packet));
    TestPayload* payload = reinterpret_cast<TestPayload*>(
        iree_hal_amdgpu_feedback_packet_payload(packet));
    payload->value = 0xABCDEF0000000000ull + i;
    payload->ordinal = i;
    payload->reserved = 0;
    iree_hal_amdgpu_feedback_publish(&channel_.config, packet);
  }

  DrainState state;
  iree_host_size_t packet_count = 0;
  IREE_ASSERT_OK(iree_hal_amdgpu_feedback_channel_drain(
      &channel_, kPacketCount, CapturePacket, &state, &packet_count));
  ASSERT_EQ(packet_count, kPacketCount);
  ASSERT_EQ(state.packets.size(), kPacketCount);
  for (uint32_t i = 0; i < kPacketCount; ++i) {
    EXPECT_EQ(state.packets[i].kind, IREE_HAL_AMDGPU_FEEDBACK_PACKET_KIND_USER);
    EXPECT_EQ(state.packets[i].payload.value, 0xABCDEF0000000000ull + i);
    EXPECT_EQ(state.packets[i].payload.ordinal, i);
    EXPECT_EQ(state.packets[i].payload.reserved, 0u);
  }
  EXPECT_EQ(channel_.control->read_tail, channel_.control->reservation_head);
  EXPECT_EQ(channel_.control->dropped_packet_count, 0u);
}

TEST_F(FeedbackChannelTest, WrapsPacketStorage) {
  const uint64_t start_position = channel_.control->ring_capacity -
                                  IREE_HAL_AMDGPU_FEEDBACK_PACKET_ALIGNMENT;
  channel_.control->read_tail = start_position;
  channel_.control->reservation_head = start_position;

  iree_hal_amdgpu_feedback_packet_t* packet = nullptr;
  ASSERT_TRUE(iree_hal_amdgpu_feedback_try_reserve(
      &channel_.config, IREE_HAL_AMDGPU_FEEDBACK_PACKET_KIND_USER,
      IREE_HAL_AMDGPU_FEEDBACK_PACKET_FLAG_ASYNC, sizeof(TestPayload),
      &packet));
  EXPECT_EQ(packet->sequence, start_position);

  TestPayload* payload = reinterpret_cast<TestPayload*>(
      iree_hal_amdgpu_feedback_packet_payload(packet));
  payload->value = 0xFEEDFACECAFEBEEFull;
  payload->ordinal = 42;
  payload->reserved = 0;
  iree_hal_amdgpu_feedback_publish(&channel_.config, packet);

  DrainState state;
  iree_host_size_t packet_count = 0;
  IREE_ASSERT_OK(iree_hal_amdgpu_feedback_channel_drain(
      &channel_, 1, CapturePacket, &state, &packet_count));
  ASSERT_EQ(packet_count, 1u);
  ASSERT_EQ(state.packets.size(), 1u);
  EXPECT_EQ(state.packets[0].sequence, start_position);
  EXPECT_EQ(state.packets[0].payload.value, 0xFEEDFACECAFEBEEFull);
  EXPECT_EQ(state.packets[0].payload.ordinal, 42u);
  EXPECT_EQ(channel_.control->read_tail, channel_.control->reservation_head);
}

TEST_F(FeedbackChannelTest, HostTsanProducerDrain) {
  ASSERT_TRUE(iree_hal_amdgpu_tsan_report_data_race(
      &channel_.config, IREE_HAL_AMDGPU_TSAN_REPORT_FLAG_NONE,
      IREE_HAL_AMDGPU_TSAN_MEMORY_SPACE_WORKGROUP,
      IREE_HAL_AMDGPU_TSAN_ACCESS_KIND_WRITE,
      IREE_HAL_AMDGPU_TSAN_ACCESS_KIND_READ,
      /*access_size=*/4, /*current_site_id=*/17, /*prior_site_id=*/11,
      /*memory_address=*/0x20, /*shadow_address=*/0xABC0,
      /*shadow_value=*/0x12345678, /*current_workgroup_id_x=*/2,
      /*current_workgroup_id_y=*/3, /*current_workgroup_id_z=*/4,
      /*current_workitem_id_x=*/5, /*current_workitem_id_y=*/6,
      /*current_workitem_id_z=*/7, /*prior_workgroup_id_x=*/8,
      /*prior_workgroup_id_y=*/9, /*prior_workgroup_id_z=*/10,
      /*prior_workitem_id_x=*/11, /*prior_workitem_id_y=*/12,
      /*prior_workitem_id_z=*/13));

  TsanDrainState state;
  iree_host_size_t packet_count = 0;
  IREE_ASSERT_OK(iree_hal_amdgpu_feedback_channel_drain(
      &channel_, 1, CaptureTsanPacket, &state, &packet_count));
  ASSERT_EQ(packet_count, 1u);
  EXPECT_EQ(state.kind, IREE_HAL_AMDGPU_FEEDBACK_PACKET_KIND_TSAN);
  EXPECT_EQ(state.source_dispatch_ptr, 0u);
  EXPECT_EQ(state.report.record_length, sizeof(iree_hal_amdgpu_tsan_report_t));
  EXPECT_EQ(state.report.abi_version,
            IREE_HAL_AMDGPU_TSAN_REPORT_ABI_VERSION_0);
  EXPECT_EQ(state.report.check_kind, IREE_HAL_AMDGPU_TSAN_CHECK_KIND_DATA_RACE);
  EXPECT_EQ(state.report.flags, IREE_HAL_AMDGPU_TSAN_REPORT_FLAG_NONE);
  EXPECT_EQ(state.report.memory_space,
            IREE_HAL_AMDGPU_TSAN_MEMORY_SPACE_WORKGROUP);
  EXPECT_EQ(state.report.current_access_kind,
            IREE_HAL_AMDGPU_TSAN_ACCESS_KIND_WRITE);
  EXPECT_EQ(state.report.prior_access_kind,
            IREE_HAL_AMDGPU_TSAN_ACCESS_KIND_READ);
  EXPECT_EQ(state.report.access_size, 4u);
  EXPECT_EQ(state.report.current_site_id, 17u);
  EXPECT_EQ(state.report.prior_site_id, 11u);
  EXPECT_EQ(state.report.memory_address, 0x20u);
  EXPECT_EQ(state.report.shadow_address, 0xABC0u);
  EXPECT_EQ(state.report.shadow_value, 0x12345678u);
  EXPECT_EQ(state.report.current_workgroup_id[0], 2u);
  EXPECT_EQ(state.report.current_workgroup_id[1], 3u);
  EXPECT_EQ(state.report.current_workgroup_id[2], 4u);
  EXPECT_EQ(state.report.current_workitem_id[0], 5u);
  EXPECT_EQ(state.report.current_workitem_id[1], 6u);
  EXPECT_EQ(state.report.current_workitem_id[2], 7u);
  EXPECT_EQ(state.report.prior_workgroup_id[0], 8u);
  EXPECT_EQ(state.report.prior_workgroup_id[1], 9u);
  EXPECT_EQ(state.report.prior_workgroup_id[2], 10u);
  EXPECT_EQ(state.report.prior_workitem_id[0], 11u);
  EXPECT_EQ(state.report.prior_workitem_id[1], 12u);
  EXPECT_EQ(state.report.prior_workitem_id[2], 13u);
  EXPECT_EQ(channel_.control->read_tail, channel_.control->reservation_head);
}

TEST_F(FeedbackChannelTest, DropWhenFull) {
  channel_.control->read_tail = 0;
  channel_.control->reservation_head = channel_.control->ring_capacity;

  iree_hal_amdgpu_feedback_packet_t* packet = nullptr;
  EXPECT_FALSE(iree_hal_amdgpu_feedback_try_reserve(
      &channel_.config, IREE_HAL_AMDGPU_FEEDBACK_PACKET_KIND_USER,
      IREE_HAL_AMDGPU_FEEDBACK_PACKET_FLAG_ASYNC, sizeof(TestPayload),
      &packet));
  EXPECT_EQ(packet, nullptr);
  EXPECT_EQ(channel_.control->dropped_packet_count, 1u);
}

}  // namespace
}  // namespace iree::hal::amdgpu

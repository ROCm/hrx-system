// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>

#include "iree/base/api.h"
#include "iree/hal/device_event.h"
#include "iree/hal/driver/amd/fabric/asan.h"
#include "iree/hal/driver/amd/gpu/context.h"
#include "iree/hal/driver/amd/gpu/feedback.h"
#include "iree/hal/driver/amd/gpu/source_context.h"
#include "iree/hal/driver/amd/gpu/tsan.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/target/arch/amdgpu/abi/asan.h"
#include "loom/target/arch/amdgpu/abi/feedback.h"
#include "loom/target/arch/amdgpu/abi/tsan.h"
#include "loom/target/arch/amdgpu/abi/ubsan.h"

namespace {

TEST(AmdgpuRuntimeAbiTest, ReadsNativeAsanConfiguration) {
  EXPECT_EQ(LOOM_AMDGPU_ASAN_CONFIG_BYTE_LENGTH,
            sizeof(iree_hal_amd_fabric_asan_config_t));
  EXPECT_EQ(LOOM_AMDGPU_ASAN_CONFIG_RECORD_LENGTH_OFFSET,
            offsetof(iree_hal_amd_fabric_asan_config_t, record_length));
  EXPECT_EQ(LOOM_AMDGPU_ASAN_CONFIG_ABI_VERSION_OFFSET,
            offsetof(iree_hal_amd_fabric_asan_config_t, abi_version));
  EXPECT_EQ(LOOM_AMDGPU_ASAN_CONFIG_FLAGS_OFFSET,
            offsetof(iree_hal_amd_fabric_asan_config_t, flags));
  EXPECT_EQ(LOOM_AMDGPU_ASAN_CONFIG_SHADOW_SCALE_SHIFT_OFFSET,
            offsetof(iree_hal_amd_fabric_asan_config_t, shadow_scale_shift));
  EXPECT_EQ(LOOM_AMDGPU_ASAN_CONFIG_SHADOW_BASE_OFFSET,
            offsetof(iree_hal_amd_fabric_asan_config_t, shadow_base));
  EXPECT_EQ(
      LOOM_AMDGPU_ASAN_CONFIG_APPLICATION_WINDOW_BASE_OFFSET,
      offsetof(iree_hal_amd_fabric_asan_config_t, application_window_base));
  EXPECT_EQ(
      LOOM_AMDGPU_ASAN_CONFIG_APPLICATION_WINDOW_SIZE_OFFSET,
      offsetof(iree_hal_amd_fabric_asan_config_t, application_window_size));
  EXPECT_EQ(LOOM_AMDGPU_ASAN_CONFIG_SHADOW_SIZE_OFFSET,
            offsetof(iree_hal_amd_fabric_asan_config_t, shadow_size));
  EXPECT_EQ(LOOM_AMDGPU_ASAN_CONFIG_SHADOW_SLAB_SIZE_OFFSET,
            offsetof(iree_hal_amd_fabric_asan_config_t, shadow_slab_size));
  EXPECT_EQ(LOOM_AMDGPU_ASAN_CONFIG_RESERVED_ARRAY_0_OFFSET,
            offsetof(iree_hal_amd_fabric_asan_config_t, reserved[0]));
  EXPECT_EQ(LOOM_AMDGPU_ASAN_CONFIG_RESERVED_ARRAY_1_OFFSET,
            offsetof(iree_hal_amd_fabric_asan_config_t, reserved[1]));
  EXPECT_EQ(LOOM_AMDGPU_ASAN_CONFIG_RESERVED_ARRAY_2_OFFSET,
            offsetof(iree_hal_amd_fabric_asan_config_t, reserved[2]));
  EXPECT_EQ(LOOM_AMDGPU_ASAN_CONFIG_RESERVED_ARRAY_3_OFFSET,
            offsetof(iree_hal_amd_fabric_asan_config_t, reserved[3]));
  EXPECT_EQ(LOOM_AMDGPU_ASAN_CONFIG_RESERVED_ARRAY_4_OFFSET,
            offsetof(iree_hal_amd_fabric_asan_config_t, reserved[4]));
}

TEST(AmdgpuRuntimeAbiTest, ReadsNativeFeedbackConfiguration) {
  EXPECT_EQ(LOOM_AMDGPU_FEEDBACK_CONFIG_BYTE_LENGTH,
            sizeof(iree_hal_amd_gpu_feedback_config_t));
  EXPECT_EQ(LOOM_AMDGPU_FEEDBACK_CONFIG_CHANNEL_BASE_OFFSET,
            offsetof(iree_hal_amd_gpu_feedback_config_t, channel_base));
  EXPECT_EQ(LOOM_AMDGPU_FEEDBACK_CONFIG_NOTIFY_SIGNAL_OFFSET,
            offsetof(iree_hal_amd_gpu_feedback_config_t, notify_signal));
  EXPECT_EQ(LOOM_AMDGPU_FEEDBACK_CONFIG_SOURCE_CONTEXT_OFFSET,
            offsetof(iree_hal_amd_gpu_feedback_config_t, source_context));

  EXPECT_EQ(LOOM_AMDGPU_FEEDBACK_CHANNEL_BYTE_LENGTH,
            sizeof(iree_hal_amd_gpu_feedback_control_t));
  EXPECT_EQ(LOOM_AMDGPU_FEEDBACK_CHANNEL_RING_BASE_OFFSET,
            offsetof(iree_hal_amd_gpu_feedback_control_t, ring_base));
  EXPECT_EQ(LOOM_AMDGPU_FEEDBACK_CHANNEL_RING_CAPACITY_OFFSET,
            offsetof(iree_hal_amd_gpu_feedback_control_t, ring_capacity));
  EXPECT_EQ(LOOM_AMDGPU_FEEDBACK_CHANNEL_READ_TAIL_OFFSET,
            offsetof(iree_hal_amd_gpu_feedback_control_t, read_tail));
  EXPECT_EQ(LOOM_AMDGPU_FEEDBACK_CHANNEL_RESERVATION_HEAD_OFFSET,
            offsetof(iree_hal_amd_gpu_feedback_control_t, reservation_head));
  EXPECT_EQ(
      LOOM_AMDGPU_FEEDBACK_CHANNEL_DROPPED_PACKET_COUNT_OFFSET,
      offsetof(iree_hal_amd_gpu_feedback_control_t, dropped_packet_count));
}

TEST(AmdgpuRuntimeAbiTest, ReadsNativeTsanConfiguration) {
  iree_hal_amd_gpu_tsan_options_t options = {};
  iree_hal_amd_gpu_tsan_options_initialize(&options);
  options.shadow_size = 1440;
  iree_hal_amd_gpu_tsan_config_t config = {};
  IREE_ASSERT_OK(iree_hal_amd_gpu_tsan_calculate_config(&options, &config));
  EXPECT_EQ(LOOM_AMDGPU_TSAN_CONFIG_BYTE_LENGTH, sizeof(config));
  const auto* bytes = reinterpret_cast<const uint8_t*>(&config);
  EXPECT_EQ(
      iree_unaligned_load_le_u32(bytes + LOOM_AMDGPU_TSAN_CONFIG_FLAGS_OFFSET),
      LOOM_AMDGPU_TSAN_CONFIG_FLAG_ENABLED |
          LOOM_AMDGPU_TSAN_CONFIG_FLAG_QUEUE_SHADOW);
  EXPECT_EQ(iree_unaligned_load_le_u64(
                bytes + LOOM_AMDGPU_TSAN_CONFIG_SHADOW_SIZE_OFFSET),
            options.shadow_size);
  EXPECT_EQ(LOOM_AMDGPU_TSAN_CONFIG_SHADOW_BASE_OFFSET,
            offsetof(iree_hal_amd_gpu_tsan_config_t, shadow_base));
}

// Exercise the device wire representation through the native HAL consumer. The
// producer uses Loom offsets, while decoding and public events belong to the
// HAL.
class AmdgpuReportAbiTest : public ::testing::Test {
 protected:
  void SetUp() override {
    context_.event_sink = {Publish, this};
    feedback_.context = &context_;
    feedback_.device_id = IREE_SV("amdgpu");
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

  void BeginPacket(loom_amdgpu_feedback_packet_kind_t kind,
                   uint32_t payload_length) {
    // Each witness starts with an empty channel at a nonzero generation.
    ring_.fill(0);
    control_.read_tail = LOOM_AMDGPU_FEEDBACK_PACKET_ALIGNMENT;
    control_.reservation_head = control_.read_tail;
    event_count_ = 0;
    packet_ = ring_.data() + control_.read_tail;
    iree_unaligned_store_le_u32(
        packet_ + LOOM_AMDGPU_FEEDBACK_PACKET_RECORD_LENGTH_OFFSET,
        loom_amdgpu_feedback_packet_length(payload_length));
    iree_unaligned_store_le_u16(
        packet_ + LOOM_AMDGPU_FEEDBACK_PACKET_HEADER_LENGTH_OFFSET,
        LOOM_AMDGPU_FEEDBACK_PACKET_BYTE_LENGTH);
    iree_unaligned_store_le_u16(
        packet_ + LOOM_AMDGPU_FEEDBACK_PACKET_KIND_OFFSET, kind);
    iree_unaligned_store_le_u32(
        packet_ + LOOM_AMDGPU_FEEDBACK_PACKET_FLAGS_OFFSET,
        LOOM_AMDGPU_FEEDBACK_PACKET_FLAG_ASYNC);
    iree_unaligned_store_le_u32(
        packet_ + LOOM_AMDGPU_FEEDBACK_PACKET_STATE_OFFSET,
        LOOM_AMDGPU_FEEDBACK_PACKET_STATE_RESERVED);
    iree_unaligned_store_le_u64(
        packet_ + LOOM_AMDGPU_FEEDBACK_PACKET_SEQUENCE_OFFSET,
        control_.read_tail);
    iree_unaligned_store_le_u64(
        packet_ + LOOM_AMDGPU_FEEDBACK_PACKET_SOURCE_DISPATCH_PTR_OFFSET,
        UINT64_C(0x123456780000));
    iree_unaligned_store_le_u32(
        packet_ + LOOM_AMDGPU_FEEDBACK_PACKET_SOURCE_WORKGROUP_ID_X_OFFSET, 7);
    iree_unaligned_store_le_u32(
        packet_ + LOOM_AMDGPU_FEEDBACK_PACKET_SOURCE_WORKITEM_ID_X_OFFSET, 11);
    iree_unaligned_store_le_u64(
        packet_ + LOOM_AMDGPU_FEEDBACK_PACKET_SOURCE_CONTEXT_OFFSET,
        reinterpret_cast<uintptr_t>(source_));
  }

  void Write32(uint32_t offset, uint32_t value) {
    iree_unaligned_store_le_u32(
        packet_ + LOOM_AMDGPU_FEEDBACK_PACKET_BYTE_LENGTH + offset, value);
  }
  void Write64(uint32_t offset, uint64_t value) {
    iree_unaligned_store_le_u64(
        packet_ + LOOM_AMDGPU_FEEDBACK_PACKET_BYTE_LENGTH + offset, value);
  }

  void Drain(iree_hal_device_event_type_t expected_type) {
    const uint64_t sequence = control_.read_tail;
    control_.reservation_head += iree_unaligned_load_le_u32(
        packet_ + LOOM_AMDGPU_FEEDBACK_PACKET_RECORD_LENGTH_OFFSET);
    iree_hal_amd_gpu_feedback_drain(&feedback_);
    EXPECT_EQ(event_count_, 0u);
    EXPECT_EQ(control_.read_tail, sequence);
    iree_unaligned_store_le_u32(
        packet_ + LOOM_AMDGPU_FEEDBACK_PACKET_STATE_OFFSET,
        LOOM_AMDGPU_FEEDBACK_PACKET_STATE_READY);
    iree_hal_amd_gpu_feedback_drain(&feedback_);
    IREE_ASSERT_OK(iree_status_clone(feedback_.failure));
    ASSERT_EQ(event_count_, 1u);
    EXPECT_EQ(event_type_, expected_type);
    EXPECT_EQ(control_.read_tail, control_.reservation_head);
  }

  static void Publish(void* user_data, const iree_hal_device_event_t* event) {
    auto& self = *static_cast<AmdgpuReportAbiTest*>(user_data);
    ++self.event_count_;
    self.event_type_ = event->type;
    EXPECT_EQ(event->sequence, LOOM_AMDGPU_FEEDBACK_PACKET_ALIGNMENT);
    EXPECT_EQ(event->severity, IREE_HAL_DEVICE_EVENT_SEVERITY_ERROR);
    EXPECT_TRUE(
        iree_string_view_equal(event->source.driver_id, IREE_SV("amd")));
    EXPECT_EQ(event->source.executable_id,
              iree_hal_amd_gpu_source_context_executable_id(self.source_));
    switch (event->type) {
      case IREE_HAL_DEVICE_EVENT_TYPE_ASAN_REPORT:
        self.asan_ = *reinterpret_cast<const iree_hal_device_asan_report_t*>(
            event->payload.data);
        break;
      case IREE_HAL_DEVICE_EVENT_TYPE_TSAN_REPORT:
        self.tsan_ = *reinterpret_cast<const iree_hal_device_tsan_report_t*>(
            event->payload.data);
        break;
      case IREE_HAL_DEVICE_EVENT_TYPE_UBSAN_REPORT:
        self.ubsan_ = *reinterpret_cast<const iree_hal_device_ubsan_report_t*>(
            event->payload.data);
        break;
      default:
        ADD_FAILURE() << "Unexpected device event " << event->type;
        break;
    }
  }

  // Event destination supplied by the physical device owner.
  iree_hal_amd_gpu_context_t context_ = {};
  // Production decoder with host storage supplied at its wire boundary.
  iree_hal_amd_gpu_feedback_t feedback_ = {};
  // Native control block shared with the simulated device producer.
  iree_hal_amd_gpu_feedback_control_t control_ = {};
  // One bounded wire witness; no device or memory mapping is required.
  alignas(64) std::array<uint8_t, 512> ring_ = {};
  // Opaque source identity retained by the production registry.
  const iree_hal_amd_gpu_source_context_t* source_ = nullptr;
  // Current packet within the shared ring.
  uint8_t* packet_ = nullptr;
  // Number of public events delivered for the current packet.
  uint32_t event_count_ = 0;
  // Public event kind delivered by the production decoder.
  iree_hal_device_event_type_t event_type_ = IREE_HAL_DEVICE_EVENT_TYPE_NONE;
  // ASAN payload copied while the callback owns the event view.
  iree_hal_device_asan_report_t asan_ = {};
  // TSAN payload copied while the callback owns the event view.
  iree_hal_device_tsan_report_t tsan_ = {};
  // UBSAN payload copied while the callback owns the event view.
  iree_hal_device_ubsan_report_t ubsan_ = {};
};

TEST_F(AmdgpuReportAbiTest, DecodesAsanAccesses) {
  const std::pair<uint32_t, uint32_t> access_kinds[] = {
      {LOOM_AMDGPU_ASAN_ACCESS_KIND_UNKNOWN,
       IREE_HAL_DEVICE_ASAN_ACCESS_KIND_UNKNOWN},
      {LOOM_AMDGPU_ASAN_ACCESS_KIND_READ,
       IREE_HAL_DEVICE_ASAN_ACCESS_KIND_READ},
      {LOOM_AMDGPU_ASAN_ACCESS_KIND_WRITE,
       IREE_HAL_DEVICE_ASAN_ACCESS_KIND_WRITE},
      {LOOM_AMDGPU_ASAN_ACCESS_KIND_ATOMIC,
       IREE_HAL_DEVICE_ASAN_ACCESS_KIND_ATOMIC},
  };
  for (auto [wire_kind, public_kind] : access_kinds) {
    SCOPED_TRACE(wire_kind);
    BeginPacket(LOOM_AMDGPU_FEEDBACK_PACKET_KIND_ASAN,
                LOOM_AMDGPU_ASAN_REPORT_BYTE_LENGTH);
    Write32(LOOM_AMDGPU_ASAN_REPORT_RECORD_LENGTH_OFFSET,
            LOOM_AMDGPU_ASAN_REPORT_BYTE_LENGTH);
    Write32(LOOM_AMDGPU_ASAN_REPORT_ABI_VERSION_OFFSET,
            LOOM_AMDGPU_ASAN_REPORT_ABI_VERSION);
    Write32(LOOM_AMDGPU_ASAN_REPORT_ACCESS_KIND_OFFSET, wire_kind);
    Write32(LOOM_AMDGPU_ASAN_REPORT_FLAGS_OFFSET,
            LOOM_AMDGPU_ASAN_REPORT_FLAG_NONE);
    Write64(LOOM_AMDGPU_ASAN_REPORT_FAULT_ADDRESS_OFFSET,
            UINT64_C(0x100000001));
    Write64(LOOM_AMDGPU_ASAN_REPORT_ACCESS_SIZE_OFFSET, 8);
    Write64(LOOM_AMDGPU_ASAN_REPORT_SITE_ID_OFFSET, 19);
    Write64(LOOM_AMDGPU_ASAN_REPORT_SHADOW_ADDRESS_OFFSET,
            UINT64_C(0x200000003));
    Write64(LOOM_AMDGPU_ASAN_REPORT_SHADOW_VALUE_OFFSET, 0xfd);
    Drain(IREE_HAL_DEVICE_EVENT_TYPE_ASAN_REPORT);
    EXPECT_EQ(asan_.access_kind, public_kind);
    EXPECT_EQ(asan_.flags, IREE_HAL_DEVICE_ASAN_REPORT_FLAG_NONE);
    EXPECT_EQ(asan_.fault_address, UINT64_C(0x100000001));
    EXPECT_EQ(asan_.access_length, 8u);
    EXPECT_EQ(asan_.site_id, 19u);
    EXPECT_EQ(asan_.shadow_address, UINT64_C(0x200000003));
    EXPECT_EQ(asan_.shadow_value, 0xfdu);
    EXPECT_EQ(asan_.workgroup_id[0], 7u);
    EXPECT_EQ(asan_.workitem_id[0], 11u);
    EXPECT_EQ(asan_.source_dispatch_ptr, UINT64_C(0x123456780000));
  }
}

TEST_F(AmdgpuReportAbiTest, DecodesUbsanChecks) {
  const std::pair<uint32_t, uint32_t> check_kinds[] = {
      {LOOM_AMDGPU_UBSAN_CHECK_KIND_UNKNOWN,
       IREE_HAL_DEVICE_UBSAN_CHECK_KIND_UNKNOWN},
      {LOOM_AMDGPU_UBSAN_CHECK_KIND_INTEGER_OVERFLOW,
       IREE_HAL_DEVICE_UBSAN_CHECK_KIND_INTEGER_OVERFLOW},
      {LOOM_AMDGPU_UBSAN_CHECK_KIND_DIVIDE_BY_ZERO,
       IREE_HAL_DEVICE_UBSAN_CHECK_KIND_DIVIDE_BY_ZERO},
      {LOOM_AMDGPU_UBSAN_CHECK_KIND_ALIGNMENT,
       IREE_HAL_DEVICE_UBSAN_CHECK_KIND_ALIGNMENT},
      {LOOM_AMDGPU_UBSAN_CHECK_KIND_FLOAT_NAN_CONTRACT,
       IREE_HAL_DEVICE_UBSAN_CHECK_KIND_FLOAT_NAN_CONTRACT},
      {LOOM_AMDGPU_UBSAN_CHECK_KIND_UNREACHABLE,
       IREE_HAL_DEVICE_UBSAN_CHECK_KIND_UNREACHABLE},
      {LOOM_AMDGPU_UBSAN_CHECK_KIND_ASSERTION,
       IREE_HAL_DEVICE_UBSAN_CHECK_KIND_ASSERTION},
  };
  for (auto [wire_kind, public_kind] : check_kinds) {
    SCOPED_TRACE(wire_kind);
    BeginPacket(LOOM_AMDGPU_FEEDBACK_PACKET_KIND_UBSAN,
                LOOM_AMDGPU_UBSAN_REPORT_BYTE_LENGTH);
    Write32(LOOM_AMDGPU_UBSAN_REPORT_RECORD_LENGTH_OFFSET,
            LOOM_AMDGPU_UBSAN_REPORT_BYTE_LENGTH);
    Write32(LOOM_AMDGPU_UBSAN_REPORT_ABI_VERSION_OFFSET,
            LOOM_AMDGPU_UBSAN_REPORT_ABI_VERSION);
    Write32(LOOM_AMDGPU_UBSAN_REPORT_CHECK_KIND_OFFSET, wire_kind);
    Write32(LOOM_AMDGPU_UBSAN_REPORT_FLAGS_OFFSET,
            LOOM_AMDGPU_UBSAN_REPORT_FLAG_NONE);
    Write64(LOOM_AMDGPU_UBSAN_REPORT_SITE_ID_OFFSET, 23);
    Write64(LOOM_AMDGPU_UBSAN_REPORT_OPERAND0_OFFSET, UINT64_C(0x300000005));
    Write64(LOOM_AMDGPU_UBSAN_REPORT_OPERAND1_OFFSET, UINT64_C(0x400000007));
    Drain(IREE_HAL_DEVICE_EVENT_TYPE_UBSAN_REPORT);
    EXPECT_EQ(ubsan_.check_kind, public_kind);
    EXPECT_EQ(ubsan_.flags, IREE_HAL_DEVICE_UBSAN_REPORT_FLAG_NONE);
    EXPECT_EQ(ubsan_.site_id, 23u);
    EXPECT_EQ(ubsan_.operand0, UINT64_C(0x300000005));
    EXPECT_EQ(ubsan_.operand1, UINT64_C(0x400000007));
    EXPECT_EQ(ubsan_.workgroup_id[0], 7u);
    EXPECT_EQ(ubsan_.workitem_id[0], 11u);
    EXPECT_EQ(ubsan_.source_dispatch_ptr, UINT64_C(0x123456780000));
  }
}

TEST_F(AmdgpuReportAbiTest, DecodesTsanRace) {
  BeginPacket(LOOM_AMDGPU_FEEDBACK_PACKET_KIND_TSAN,
              LOOM_AMDGPU_TSAN_REPORT_BYTE_LENGTH);
  Write32(LOOM_AMDGPU_TSAN_REPORT_RECORD_LENGTH_OFFSET,
          LOOM_AMDGPU_TSAN_REPORT_BYTE_LENGTH);
  Write32(LOOM_AMDGPU_TSAN_REPORT_ABI_VERSION_OFFSET,
          LOOM_AMDGPU_TSAN_REPORT_ABI_VERSION);
  Write32(LOOM_AMDGPU_TSAN_REPORT_CHECK_KIND_OFFSET,
          LOOM_AMDGPU_TSAN_CHECK_KIND_DATA_RACE);
  Write32(LOOM_AMDGPU_TSAN_REPORT_FLAGS_OFFSET,
          LOOM_AMDGPU_TSAN_REPORT_FLAG_PRIOR_ATOMIC);
  Write32(LOOM_AMDGPU_TSAN_REPORT_MEMORY_SPACE_OFFSET,
          LOOM_AMDGPU_TSAN_MEMORY_SPACE_WORKGROUP);
  Write32(LOOM_AMDGPU_TSAN_REPORT_CURRENT_ACCESS_KIND_OFFSET,
          LOOM_AMDGPU_TSAN_ACCESS_KIND_WRITE);
  Write32(LOOM_AMDGPU_TSAN_REPORT_PRIOR_ACCESS_KIND_OFFSET,
          LOOM_AMDGPU_TSAN_ACCESS_KIND_ATOMIC);
  Write32(LOOM_AMDGPU_TSAN_REPORT_ACCESS_SIZE_OFFSET, 8);
  Write64(LOOM_AMDGPU_TSAN_REPORT_CURRENT_SITE_ID_OFFSET, 29);
  Write64(LOOM_AMDGPU_TSAN_REPORT_PRIOR_SITE_ID_OFFSET, 31);
  Write64(LOOM_AMDGPU_TSAN_REPORT_MEMORY_ADDRESS_OFFSET, UINT64_C(0x500000009));
  Write64(LOOM_AMDGPU_TSAN_REPORT_SHADOW_ADDRESS_OFFSET, UINT64_C(0x60000000b));
  Write64(LOOM_AMDGPU_TSAN_REPORT_SHADOW_VALUE_OFFSET, UINT64_C(0x70000000d));
  const uint32_t coordinate_offsets[] = {
      LOOM_AMDGPU_TSAN_REPORT_CURRENT_WORKGROUP_ID_X_OFFSET,
      LOOM_AMDGPU_TSAN_REPORT_CURRENT_WORKGROUP_ID_Y_OFFSET,
      LOOM_AMDGPU_TSAN_REPORT_CURRENT_WORKGROUP_ID_Z_OFFSET,
      LOOM_AMDGPU_TSAN_REPORT_CURRENT_WORKITEM_ID_X_OFFSET,
      LOOM_AMDGPU_TSAN_REPORT_CURRENT_WORKITEM_ID_Y_OFFSET,
      LOOM_AMDGPU_TSAN_REPORT_CURRENT_WORKITEM_ID_Z_OFFSET,
      LOOM_AMDGPU_TSAN_REPORT_PRIOR_WORKGROUP_ID_X_OFFSET,
      LOOM_AMDGPU_TSAN_REPORT_PRIOR_WORKGROUP_ID_Y_OFFSET,
      LOOM_AMDGPU_TSAN_REPORT_PRIOR_WORKGROUP_ID_Z_OFFSET,
      LOOM_AMDGPU_TSAN_REPORT_PRIOR_WORKITEM_ID_X_OFFSET,
      LOOM_AMDGPU_TSAN_REPORT_PRIOR_WORKITEM_ID_Y_OFFSET,
      LOOM_AMDGPU_TSAN_REPORT_PRIOR_WORKITEM_ID_Z_OFFSET,
  };
  for (uint32_t i = 0; i < IREE_ARRAYSIZE(coordinate_offsets); ++i) {
    Write32(coordinate_offsets[i], 17 + i);
  }
  Drain(IREE_HAL_DEVICE_EVENT_TYPE_TSAN_REPORT);
  EXPECT_EQ(tsan_.check_kind, IREE_HAL_DEVICE_TSAN_CHECK_KIND_DATA_RACE);
  EXPECT_EQ(tsan_.flags, IREE_HAL_DEVICE_TSAN_REPORT_FLAG_PRIOR_ATOMIC);
  EXPECT_EQ(tsan_.source_dispatch_ptr, UINT64_C(0x123456780000));
  const auto& race = tsan_.detail.data_race;
  EXPECT_EQ(race.memory_space, IREE_HAL_DEVICE_TSAN_MEMORY_SPACE_WORKGROUP);
  EXPECT_EQ(race.current_access_kind, IREE_HAL_DEVICE_TSAN_ACCESS_KIND_WRITE);
  EXPECT_EQ(race.prior_access_kind, IREE_HAL_DEVICE_TSAN_ACCESS_KIND_ATOMIC);
  EXPECT_EQ(race.access_length, 8u);
  EXPECT_EQ(race.current_site_id, 29u);
  EXPECT_EQ(race.prior_site_id, 31u);
  EXPECT_EQ(race.memory_address, UINT64_C(0x500000009));
  EXPECT_EQ(race.shadow_address, UINT64_C(0x60000000b));
  EXPECT_EQ(race.shadow_value, UINT64_C(0x70000000d));
  for (uint32_t i = 0; i < 3; ++i) {
    EXPECT_EQ(race.current_workgroup_id[i], 17 + i);
    EXPECT_EQ(race.current_workitem_id[i], 20 + i);
    EXPECT_EQ(race.prior_workgroup_id[i], 23 + i);
    EXPECT_EQ(race.prior_workitem_id[i], 26 + i);
  }
}

TEST_F(AmdgpuReportAbiTest, DecodesTsanShadowCapacity) {
  BeginPacket(LOOM_AMDGPU_FEEDBACK_PACKET_KIND_TSAN,
              LOOM_AMDGPU_TSAN_REPORT_BYTE_LENGTH);
  Write32(LOOM_AMDGPU_TSAN_REPORT_RECORD_LENGTH_OFFSET,
          LOOM_AMDGPU_TSAN_REPORT_BYTE_LENGTH);
  Write32(LOOM_AMDGPU_TSAN_REPORT_ABI_VERSION_OFFSET,
          LOOM_AMDGPU_TSAN_REPORT_ABI_VERSION);
  Write32(LOOM_AMDGPU_TSAN_REPORT_CHECK_KIND_OFFSET,
          LOOM_AMDGPU_TSAN_CHECK_KIND_SHADOW_CAPACITY);
  Write64(LOOM_AMDGPU_TSAN_REPORT_CAPACITY_SHADOW_SIZE_OFFSET,
          UINT64_C(0x100000000));
  Write64(LOOM_AMDGPU_TSAN_REPORT_CAPACITY_REQUIRED_SHADOW_END_OFFSET,
          UINT64_C(0x100000040));
  Write64(LOOM_AMDGPU_TSAN_REPORT_CAPACITY_WORKGROUP_SHADOW_STRIDE_OFFSET, 192);
  Write64(LOOM_AMDGPU_TSAN_REPORT_CAPACITY_WORKGROUP_ORDINAL_OFFSET, 41);
  Write32(LOOM_AMDGPU_TSAN_REPORT_CAPACITY_LOCAL_MEMORY_SIZE_OFFSET, 4096);
  Write32(LOOM_AMDGPU_TSAN_REPORT_CAPACITY_SHADOW_CELL_SIZE_OFFSET, 16);
  Write32(LOOM_AMDGPU_TSAN_REPORT_CAPACITY_SHADOW_HISTORY_COUNT_OFFSET, 8);
  const uint32_t coordinate_offsets[] = {
      LOOM_AMDGPU_TSAN_REPORT_CAPACITY_WORKGROUP_ID_X_OFFSET,
      LOOM_AMDGPU_TSAN_REPORT_CAPACITY_WORKGROUP_ID_Y_OFFSET,
      LOOM_AMDGPU_TSAN_REPORT_CAPACITY_WORKGROUP_ID_Z_OFFSET,
      LOOM_AMDGPU_TSAN_REPORT_CAPACITY_WORKITEM_ID_X_OFFSET,
      LOOM_AMDGPU_TSAN_REPORT_CAPACITY_WORKITEM_ID_Y_OFFSET,
      LOOM_AMDGPU_TSAN_REPORT_CAPACITY_WORKITEM_ID_Z_OFFSET,
  };
  for (uint32_t i = 0; i < IREE_ARRAYSIZE(coordinate_offsets); ++i) {
    Write32(coordinate_offsets[i], 43 + i);
  }
  Drain(IREE_HAL_DEVICE_EVENT_TYPE_TSAN_REPORT);
  EXPECT_EQ(tsan_.check_kind, IREE_HAL_DEVICE_TSAN_CHECK_KIND_SHADOW_CAPACITY);
  const auto& capacity = tsan_.detail.shadow_capacity;
  EXPECT_EQ(capacity.shadow_size, UINT64_C(0x100000000));
  EXPECT_EQ(capacity.required_shadow_end, UINT64_C(0x100000040));
  EXPECT_EQ(capacity.workgroup_shadow_stride, 192u);
  EXPECT_EQ(capacity.workgroup_ordinal, 41u);
  EXPECT_EQ(capacity.workgroup_local_memory_size, 4096u);
  EXPECT_EQ(capacity.shadow_cell_size, 16u);
  EXPECT_EQ(capacity.shadow_history_count, 8u);
  for (uint32_t i = 0; i < 3; ++i) {
    EXPECT_EQ(capacity.current_workgroup_id[i], 43 + i);
    EXPECT_EQ(capacity.current_workitem_id[i], 46 + i);
  }
}

TEST_F(AmdgpuReportAbiTest, DecodesTsanSynchronizationCapacity) {
  BeginPacket(LOOM_AMDGPU_FEEDBACK_PACKET_KIND_TSAN,
              LOOM_AMDGPU_TSAN_REPORT_BYTE_LENGTH);
  Write32(LOOM_AMDGPU_TSAN_REPORT_RECORD_LENGTH_OFFSET,
          LOOM_AMDGPU_TSAN_REPORT_BYTE_LENGTH);
  Write32(LOOM_AMDGPU_TSAN_REPORT_ABI_VERSION_OFFSET,
          LOOM_AMDGPU_TSAN_REPORT_ABI_VERSION);
  Write32(LOOM_AMDGPU_TSAN_REPORT_CHECK_KIND_OFFSET,
          LOOM_AMDGPU_TSAN_CHECK_KIND_SYNCHRONIZATION_CAPACITY);
  Write64(LOOM_AMDGPU_TSAN_REPORT_SYNC_CAPACITY_CURRENT_SITE_ID_OFFSET, 53);
  Write32(LOOM_AMDGPU_TSAN_REPORT_SYNC_CAPACITY_SUBGROUP_EPOCH_OFFSET, 255);
  Write32(LOOM_AMDGPU_TSAN_REPORT_SYNC_CAPACITY_SUBGROUP_EPOCH_CAPACITY_OFFSET,
          256);
  Write32(LOOM_AMDGPU_TSAN_REPORT_SYNC_CAPACITY_WAVEFRONT_SIZE_OFFSET, 32);
  Write32(LOOM_AMDGPU_TSAN_REPORT_SYNC_CAPACITY_SUBGROUP_ID_OFFSET, 2);
  Write32(LOOM_AMDGPU_TSAN_REPORT_SYNC_CAPACITY_WORKGROUP_ID_X_OFFSET, 59);
  Write32(LOOM_AMDGPU_TSAN_REPORT_SYNC_CAPACITY_WORKGROUP_ID_Y_OFFSET, 61);
  Write32(LOOM_AMDGPU_TSAN_REPORT_SYNC_CAPACITY_WORKGROUP_ID_Z_OFFSET, 67);
  Write32(LOOM_AMDGPU_TSAN_REPORT_SYNC_CAPACITY_WORKITEM_LINEAR_ID_OFFSET, 71);
  Write64(LOOM_AMDGPU_TSAN_REPORT_SYNC_CAPACITY_SUBGROUP_CLOCK_ADDRESS_OFFSET,
          UINT64_C(0x80000000f));
  Drain(IREE_HAL_DEVICE_EVENT_TYPE_TSAN_REPORT);
  EXPECT_EQ(tsan_.check_kind,
            IREE_HAL_DEVICE_TSAN_CHECK_KIND_SYNCHRONIZATION_CAPACITY);
  const auto& capacity = tsan_.detail.synchronization_capacity;
  EXPECT_EQ(capacity.current_site_id, 53u);
  EXPECT_EQ(capacity.subgroup_epoch, 255u);
  EXPECT_EQ(capacity.subgroup_epoch_capacity, 256u);
  EXPECT_EQ(capacity.wavefront_size, 32u);
  EXPECT_EQ(capacity.subgroup_id, 2u);
  EXPECT_EQ(capacity.current_workgroup_id[0], 59u);
  EXPECT_EQ(capacity.current_workgroup_id[1], 61u);
  EXPECT_EQ(capacity.current_workgroup_id[2], 67u);
  EXPECT_EQ(capacity.current_workitem_linear_id, 71u);
  EXPECT_EQ(capacity.subgroup_clock_address, UINT64_C(0x80000000f));
}

}  // namespace

// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <thread>
#include <utility>

#include "amdf/amdf.h"
#include "amdf/gpu.h"
#include "gpu_device_fixture.h"
#include "gtest/gtest.h"
#include "util/native_event.h"
#include "util/wait.h"

namespace {

constexpr uint64_t kCommandByteOffset = 0;
constexpr uint64_t kStagedCommandByteOffset = 512;
constexpr uint64_t kVerifiedCommandByteOffset = 640;
constexpr uint64_t kSourceByteOffset = 1024;
constexpr uint64_t kTargetByteOffset = 1088;
constexpr uint64_t kLocalByteOffset = 256;
constexpr uint64_t kMemoryByteLength = 4096;
constexpr uint32_t kCopyDataDwordCount = 6;
constexpr uint32_t kSdmaCopyDwordCount = 7;

uint32_t MakePm4Header(uint32_t opcode, uint32_t dword_count) {
  return (3u << 30) | (opcode << 8) | ((dword_count - 2u) << 16);
}

std::array<uint32_t, kCopyDataDwordCount> MakeCopyData32(
    uint64_t source_address, uint64_t target_address) {
  constexpr uint32_t kCopyDataOpcode = 0x40;
  constexpr uint32_t kSourceTcL2 = 2u << 0;
  constexpr uint32_t kTargetTcL2 = 2u << 8;
  constexpr uint32_t kWaitForWriteConfirmation = 1u << 20;
  return {
      MakePm4Header(kCopyDataOpcode, kCopyDataDwordCount),
      kSourceTcL2 | kTargetTcL2 | kWaitForWriteConfirmation,
      static_cast<uint32_t>(source_address & UINT64_C(0xFFFFFFFC)),
      static_cast<uint32_t>(source_address >> 32),
      static_cast<uint32_t>(target_address & UINT64_C(0xFFFFFFFC)),
      static_cast<uint32_t>(target_address >> 32),
  };
}

std::array<uint32_t, kSdmaCopyDwordCount> MakeSdmaCopy32(
    amdf_queue_format_features_t features, uint64_t source_address,
    uint64_t target_address) {
  constexpr uint32_t kSdmaCopyLinearOpcode = 1;
  constexpr uint32_t kSystemScope = 3;
  const bool has_scope_fields =
      (features & AMDF_GPU_SDMA_FORMAT_FEATURE_MEMORY_SCOPE) != 0;
  const uint32_t scope_fields =
      has_scope_fields ? (kSystemScope << 18) | (kSystemScope << 26) : 0;
  return {
      kSdmaCopyLinearOpcode | (has_scope_fields ? 1u << 28 : 0),
      sizeof(uint32_t) - 1,
      scope_fields,
      static_cast<uint32_t>(source_address),
      static_cast<uint32_t>(source_address >> 32),
      static_cast<uint32_t>(target_address),
      static_cast<uint32_t>(target_address >> 32),
  };
}

class GpuKernelQueueTest : public GpuDeviceFixture {
 protected:
  explicit GpuKernelQueueTest(amdf_queue_command_type_t command_type)
      : command_type_(command_type) {}

  void TearDown() override {
    if (mapping_ != nullptr && !indirect_memory_may_be_in_use_) {
      EXPECT_TRUE(amdf_status_is_ok(api_->host_mapping_destroy(mapping_)));
    }
    if (queue_ != nullptr) {
      const auto status = DestroyQueue();
      EXPECT_EQ(status, AMDF_STATUS_OK);
      if (!amdf_status_is_ok(status)) {
        return;
      }
    }
    if (local_memory_ != nullptr && !indirect_memory_may_be_in_use_) {
      EXPECT_TRUE(
          amdf_status_is_ok(api_->memory_destroy(local_memory_, nullptr)));
    }
    if (memory_ != nullptr && !indirect_memory_may_be_in_use_) {
      EXPECT_TRUE(amdf_status_is_ok(api_->memory_destroy(memory_, nullptr)));
    }
    GpuDeviceFixture::TearDown();
  }

  amdf_status_t DestroyQueue() {
    const auto status = api_->kernel_queue_destroy(queue_);
    if (status != amdf_make_api_status(AMDF_STATUS_CODE_BUSY)) {
      queue_ = nullptr;
    }
    return status;
  }

  amdf_status_t MatchGpuEndpoint(amdf_endpoint_t* endpoint,
                                 bool* out_matches) override {
    amdf_endpoint_info_t endpoint_info = {
        .type = AMDF_STRUCTURE_TYPE_ENDPOINT_INFO,
        .structure_size = sizeof(endpoint_info),
    };
    amdf_status_t status = api_->endpoint_query_info(endpoint, &endpoint_info);
    if (!amdf_status_is_ok(status)) {
      return status;
    }
    for (uint32_t ordinal = 0; ordinal < endpoint_info.queue_family_count;
         ++ordinal) {
      amdf_queue_family_info_t family_info = {
          .type = AMDF_STRUCTURE_TYPE_QUEUE_FAMILY_INFO,
          .structure_size = sizeof(family_info),
      };
      status = api_->endpoint_query_queue_family_info(endpoint, ordinal,
                                                      &family_info);
      if (!amdf_status_is_ok(status)) {
        return status;
      }
      if (family_info.command_type == command_type_ &&
          family_info.format_version ==
              (command_type_ == AMDF_QUEUE_COMMAND_TYPE_GPU_PM4
                   ? AMDF_GPU_PM4_QUEUE_FORMAT_VERSION_1
                   : AMDF_GPU_SDMA_QUEUE_FORMAT_VERSION_1) &&
          (family_info.roles & AMDF_QUEUE_ROLE_TRANSFER) != 0 &&
          (family_info.publication_modes &
           AMDF_QUEUE_PUBLICATION_MODE_KERNEL) != 0) {
        family_ = family_info;
        *out_matches = true;
        return AMDF_STATUS_OK;
      }
    }
    *out_matches = false;
    return AMDF_STATUS_OK;
  }

  amdf_status_t CreateQueue(uint32_t family_ordinal,
                            uint32_t pending_capacity = 0) {
    amdf_gpu_kernel_queue_create_info_t create_info = {
        .type = AMDF_STRUCTURE_TYPE_GPU_KERNEL_QUEUE_CREATE_INFO,
        .structure_size = sizeof(create_info),
        .queue_family_ordinal = family_ordinal,
        .maximum_pending_submission_count = pending_capacity,
    };
    return gpu_api_->kernel_queue_create(device_, &create_info, &queue_);
  }

  uint64_t CreateCommandMemory(uint64_t byte_length = kMemoryByteLength) {
    const amdf_memory_device_access_t access = {
        device_,
        {
            .access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE |
                      AMDF_MEMORY_ACCESS_EXECUTE,
            .flags = AMDF_MEMORY_FLAG_DEVICE_ADDRESS,
        }};
    amdf_memory_create_info_t create_info = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO,
        .structure_size = sizeof(create_info),
        .access_count = 1,
        .required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE,
        .accesses = &access,
    };
    create_info.memory_profile_ordinal = FindMemoryProfileOrdinal(
        system_scope_,
        AMDF_MEMORY_PROFILE_ROLE_CREATE | AMDF_MEMORY_PROFILE_ROLE_HOST_MAP,
        create_info.required_flags, access.requirements);
    create_info.byte_length = byte_length;
    amdf_memory_profile_t profile = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE,
        .structure_size = sizeof(profile),
    };
    amdf_memory_access_capabilities_t capabilities = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES,
        .structure_size = sizeof(capabilities),
    };
    const amdf_status_t profile_status =
        QueryMemoryProfile(system_scope_, create_info.memory_profile_ordinal,
                           access.requirements, &profile, &capabilities);
    EXPECT_EQ(profile_status, AMDF_STATUS_OK);
    if (!amdf_status_is_ok(profile_status)) {
      return 0;
    }
    const auto& geometry = profile.allocation;
    const uint64_t granularity = geometry.native_byte_length_granularity;
    const uint64_t native_byte_length =
        ((create_info.byte_length + geometry.native_byte_length_prefix +
          granularity - 1) /
         granularity) *
        granularity;
    EXPECT_EQ(
        api_->memory_create(system_scope_, &create_info, &memory_, nullptr),
        AMDF_STATUS_OK);
    amdf_memory_info_t memory_info = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_INFO,
        .structure_size = sizeof(memory_info),
    };
    EXPECT_EQ(api_->memory_query_info(memory_, &memory_info), AMDF_STATUS_OK);
    EXPECT_EQ(memory_info.native_allocation_byte_length, native_byte_length);
    EXPECT_EQ(memory_info.source_byte_offset,
              geometry.native_byte_length_prefix);
    EXPECT_EQ(memory_info.native_allocation_granularity, granularity);

    uint64_t address = 0;
    EXPECT_EQ(api_->memory_query_address(memory_, 0, AMDF_MEMORY_ADDRESS_GPU,
                                         &address),
              AMDF_STATUS_OK);
    return address;
  }

  amdf_host_mapping_info_t MapCommandMemory(
      uint64_t byte_length = kMemoryByteLength) {
    amdf_memory_map_info_t map_info = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_MAP_INFO,
        .structure_size = sizeof(map_info),
        .byte_length = byte_length,
        .flags = AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE,
    };
    EXPECT_TRUE(
        amdf_status_is_ok(api_->memory_map(memory_, &map_info, &mapping_)));

    amdf_host_mapping_info_t mapping_info = {
        .type = AMDF_STRUCTURE_TYPE_HOST_MAPPING_INFO,
        .structure_size = sizeof(mapping_info),
    };
    EXPECT_TRUE(amdf_status_is_ok(
        api_->host_mapping_query_info(mapping_, &mapping_info)));
    return mapping_info;
  }

  uint64_t CreateLocalExecutableMemory() {
    const amdf_memory_device_access_t access = {
        device_,
        {
            .access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE |
                      AMDF_MEMORY_ACCESS_EXECUTE,
            .flags = AMDF_MEMORY_FLAG_DEVICE_ADDRESS,
        }};
    amdf_memory_create_info_t create_info = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO,
        .structure_size = sizeof(create_info),
        .access_count = 1,
        .required_flags = AMDF_MEMORY_FLAG_DEVICE_LOCAL,
        .accesses = &access,
    };
    create_info.memory_profile_ordinal = FindMemoryProfileOrdinal(
        local_scope_, AMDF_MEMORY_PROFILE_ROLE_CREATE,
        create_info.required_flags, access.requirements);
    create_info.byte_length = kMemoryByteLength;
    amdf_memory_profile_t profile = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE,
        .structure_size = sizeof(profile),
    };
    amdf_memory_access_capabilities_t capabilities = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES,
        .structure_size = sizeof(capabilities),
    };
    const amdf_status_t profile_status =
        QueryMemoryProfile(local_scope_, create_info.memory_profile_ordinal,
                           access.requirements, &profile, &capabilities);
    EXPECT_EQ(profile_status, AMDF_STATUS_OK);
    if (!amdf_status_is_ok(profile_status)) {
      return 0;
    }
    const auto& geometry = profile.allocation;
    const uint64_t granularity = geometry.native_byte_length_granularity;
    const uint64_t native_byte_length =
        ((create_info.byte_length + geometry.native_byte_length_prefix +
          granularity - 1) /
         granularity) *
        granularity;
    EXPECT_EQ(api_->memory_create(local_scope_, &create_info, &local_memory_,
                                  nullptr),
              AMDF_STATUS_OK);

    amdf_memory_info_t memory_info = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_INFO,
        .structure_size = sizeof(memory_info),
    };
    EXPECT_TRUE(amdf_status_is_ok(
        api_->memory_query_info(local_memory_, &memory_info)));
    amdf_memory_access_info_t access_info = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_INFO,
        .structure_size = sizeof(access_info),
    };
    EXPECT_EQ(api_->memory_query_access_info(local_memory_, 0, &access_info),
              AMDF_STATUS_OK);
    EXPECT_EQ(
        (memory_info.flags | access_info.flags) & create_info.required_flags,
        create_info.required_flags);
    EXPECT_EQ(access_info.access, access.requirements.access);
    EXPECT_EQ(access_info.flags & access.requirements.flags,
              access.requirements.flags);
    EXPECT_EQ(memory_info.byte_length, kMemoryByteLength);
    EXPECT_EQ(memory_info.native_allocation_byte_length, native_byte_length);
    EXPECT_EQ(memory_info.source_byte_offset,
              geometry.native_byte_length_prefix);
    EXPECT_EQ(memory_info.native_allocation_granularity, granularity);
    uint64_t address = 0;
    EXPECT_EQ(api_->memory_query_address(local_memory_, 0,
                                         AMDF_MEMORY_ADDRESS_GPU, &address),
              AMDF_STATUS_OK);
    return address;
  }

  void RunPipelinedCopies(uint32_t pending_capacity) {
    ASSERT_EQ(CreateQueue(family_.ordinal, pending_capacity), AMDF_STATUS_OK);
    amdf_kernel_queue_info_t info = {
        .type = AMDF_STRUCTURE_TYPE_KERNEL_QUEUE_INFO,
        .structure_size = sizeof(info),
    };
    ASSERT_EQ(api_->kernel_queue_query_info(queue_, &info), AMDF_STATUS_OK);
    ASSERT_NE(info.maximum_pending_submission_count, 0u);
    // Bound the workload independently of a provider's advertised capacity.
    const uint32_t command_count =
        std::min(info.maximum_pending_submission_count, uint32_t{4096});
    constexpr uint64_t kCommandStride = 32;
    const uint64_t data_offset = command_count * kCommandStride;
    const uint64_t data_byte_length = (uint64_t{command_count} + 1) * 4;
    const uint64_t memory_byte_length =
        std::max(kMemoryByteLength, data_offset + data_byte_length);
    const uint64_t address = CreateCommandMemory(memory_byte_length);
    ASSERT_NE(memory_, nullptr);
    const auto mapping_info = MapCommandMemory(memory_byte_length);
    ASSERT_NE(mapping_, nullptr);
    ASSERT_NE(mapping_info.pointer, nullptr);
    auto* bytes = static_cast<uint8_t*>(mapping_info.pointer);
    const uint64_t command_byte_length =
        (command_type_ == AMDF_QUEUE_COMMAND_TYPE_GPU_PM4
             ? kCopyDataDwordCount
             : kSdmaCopyDwordCount) *
        sizeof(uint32_t);
    for (uint32_t i = 0; i < command_count; ++i) {
      const uint64_t source = address + data_offset + i * 4;
      const uint64_t target = source + 4;
      if (command_type_ == AMDF_QUEUE_COMMAND_TYPE_GPU_PM4) {
        const auto command = MakeCopyData32(source, target);
        std::memcpy(bytes + i * kCommandStride, command.data(),
                    sizeof(command));
      } else {
        const auto command =
            MakeSdmaCopy32(family_.format_features, source, target);
        std::memcpy(bytes + i * kCommandStride, command.data(),
                    sizeof(command));
      }
    }
    amdf_gpu_kernel_command_t command = {
        .memory = memory_,
        .byte_length = command_byte_length,
    };
    amdf_gpu_kernel_queue_submission_info_t submit = {
        .type = AMDF_STRUCTURE_TYPE_GPU_KERNEL_QUEUE_SUBMISSION_INFO,
        .structure_size = sizeof(submit),
        .command_count = 1,
        .commands = &command,
    };
    uint64_t previous_submission = 0;
    for (uint32_t round = 0; round < 3; ++round) {
      const uint32_t expected = 0x13579bdfu + round * 0x1020304u;
      std::memset(bytes + data_offset, 0xa5, data_byte_length);
      std::memcpy(bytes + data_offset, &expected, sizeof(expected));
      ASSERT_EQ(
          api_->host_mapping_cache_control(
              mapping_, AMDF_HOST_CACHE_OPERATION_FLUSH, 0, memory_byte_length),
          AMDF_STATUS_OK);
      amdf_status_t submit_status = AMDF_STATUS_OK;
      uint64_t last_submission = 0;
      for (uint32_t i = 0;
           i < command_count && amdf_status_is_ok(submit_status); ++i) {
        command.byte_offset = i * kCommandStride;
        uint64_t submission = 0;
        submit_status =
            gpu_api_->kernel_queue_submit(queue_, &submit, &submission);
        if (amdf_status_is_ok(submit_status)) {
          indirect_memory_may_be_in_use_ = true;
          EXPECT_GT(submission, previous_submission);
          previous_submission = last_submission = submission;
        }
      }
      // Even a later rejection leaves earlier commands accepted. Drain that
      // exact prefix before reporting rejection or reusing any backing.
      if (last_submission != 0) {
        const auto wait_status = api_->kernel_queue_wait(
            queue_, last_submission, amdf::cts::kWaitTimeoutNs,
            amdf::cts::kWaitPollIntervalNs);
        if (amdf_status_is_ok(wait_status)) {
          indirect_memory_may_be_in_use_ = false;
        }
        ASSERT_EQ(wait_status, AMDF_STATUS_OK);
      }
      ASSERT_EQ(submit_status, AMDF_STATUS_OK);
      ASSERT_EQ(api_->host_mapping_cache_control(
                    mapping_, AMDF_HOST_CACHE_OPERATION_INVALIDATE, data_offset,
                    data_byte_length),
                AMDF_STATUS_OK);
      for (uint32_t i = 1; i <= command_count; ++i) {
        uint32_t actual = 0;
        std::memcpy(&actual, bytes + data_offset + i * 4, sizeof(actual));
        EXPECT_EQ(actual, expected) << "round=" << round << " command=" << i;
      }
    }
  }

  // Complete encoding facts retained from queue-family selection.
  amdf_queue_family_info_t family_ = {};
  // Command representation implemented by this scenario's encoder.
  amdf_queue_command_type_t command_type_;
  // System backing for commands and host-visible results.
  amdf_memory_t* memory_ = nullptr;
  // Local backing reached indirectly by submitted GPU commands.
  amdf_memory_t* local_memory_ = nullptr;
  // Host view of system backing.
  amdf_host_mapping_t* mapping_ = nullptr;
  // Case-owned kernel-mediated queue.
  amdf_kernel_queue_t* queue_ = nullptr;
  // Failed completion cannot establish that indirect resources are idle.
  bool indirect_memory_may_be_in_use_ = false;
};

class Pm4KernelQueueTest : public GpuKernelQueueTest {
 protected:
  Pm4KernelQueueTest() : GpuKernelQueueTest(AMDF_QUEUE_COMMAND_TYPE_GPU_PM4) {}
};

class SdmaKernelQueueTest : public GpuKernelQueueTest {
 protected:
  SdmaKernelQueueTest()
      : GpuKernelQueueTest(AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA) {}
};

class GpuKernelQueueNotificationTest
    : public GpuKernelQueueTest,
      public ::testing::WithParamInterface<amdf_queue_command_type_t> {
 protected:
  GpuKernelQueueNotificationTest() : GpuKernelQueueTest(GetParam()) {}

  void SetUp() override {
    GpuKernelQueueTest::SetUp();
    if (HasFatalFailure() || IsSkipped()) {
      return;
    }
    ASSERT_EQ(CreateQueue(family_.ordinal, 2), AMDF_STATUS_OK);
    amdf_kernel_queue_info_t info = {
        .type = AMDF_STRUCTURE_TYPE_KERNEL_QUEUE_INFO,
        .structure_size = sizeof(info),
    };
    ASSERT_EQ(api_->kernel_queue_query_info(queue_, &info), AMDF_STATUS_OK);
    if ((info.notification_types &
         (UINT64_C(1) << amdf::cts::NativeEventType())) == 0) {
      GTEST_SKIP() << "kernel queue has no native event notification transport";
    }
    for (auto& event : events_) {
      ASSERT_TRUE(amdf::cts::CreateNativeEvent(&event));
    }
    const uint64_t address = CreateCommandMemory();
    ASSERT_NE(memory_, nullptr);
    const auto mapping = MapCommandMemory();
    ASSERT_NE(mapping.pointer, nullptr);
    bytes_ = static_cast<uint8_t*>(mapping.pointer);
    std::memset(bytes_, 0, kMemoryByteLength);
    for (uint32_t i = 0; i < 2; ++i) {
      const uint64_t source = address + kSourceByteOffset + i * 4;
      const uint64_t target = address + kTargetByteOffset + i * 4;
      if (command_type_ == AMDF_QUEUE_COMMAND_TYPE_GPU_PM4) {
        const auto command = MakeCopyData32(source, target);
        std::memcpy(bytes_ + i * 32, command.data(), sizeof(command));
      } else {
        const auto command =
            MakeSdmaCopy32(family_.format_features, source, target);
        std::memcpy(bytes_ + i * 32, command.data(), sizeof(command));
      }
    }
  }

  void TearDown() override {
    // Accepted finite work and native event destinations stay live through
    // queue teardown, including when an assertion interrupted observation.
    for (uint32_t i = 0; i < 2; ++i) {
      if (points_[i] != 0) {
        const auto status = api_->kernel_queue_wait(
            Queue(i), points_[i], amdf::cts::kWaitTimeoutNs, 0);
        EXPECT_EQ(status, AMDF_STATUS_OK);
        if (!amdf_status_is_ok(status)) {
          return;
        }
      }
    }
    indirect_memory_may_be_in_use_ = false;
    if (sibling_ != nullptr) {
      const auto status = api_->kernel_queue_destroy(sibling_);
      if (status != amdf_make_api_status(AMDF_STATUS_CODE_BUSY)) {
        sibling_ = nullptr;
      }
      EXPECT_EQ(status, AMDF_STATUS_OK);
      if (!amdf_status_is_ok(status)) {
        return;
      }
    }
    if (queue_ != nullptr) {
      const auto status = DestroyQueue();
      EXPECT_EQ(status, AMDF_STATUS_OK);
      if (!amdf_status_is_ok(status)) {
        return;
      }
    }
    GpuKernelQueueTest::TearDown();
    if (queue_ == nullptr) {
      for (auto& event : events_) {
        if (event.type != AMDF_NATIVE_EVENT_TYPE_NONE) {
          EXPECT_TRUE(amdf::cts::DestroyNativeEvent(&event));
        }
      }
    }
  }

  amdf_kernel_queue_t* Queue(uint32_t ordinal) {
    return ordinal == 0 ? queue_ : sibling_;
  }

  void PrepareCopies(uint32_t seed) {
    for (uint32_t i = 0; i < 2; ++i) {
      expected_[i] = seed + i;
      std::memcpy(bytes_ + kSourceByteOffset + i * 4, &expected_[i], 4);
      std::memset(bytes_ + kTargetByteOffset + i * 4, 0xa5, 4);
    }
    ASSERT_EQ(
        api_->host_mapping_cache_control(
            mapping_, AMDF_HOST_CACHE_OPERATION_FLUSH, 0, kMemoryByteLength),
        AMDF_STATUS_OK);
  }

  void SubmitCopy(uint32_t queue_ordinal, uint32_t command_ordinal) {
    const amdf_gpu_kernel_command_t command = {
        .memory = memory_,
        .byte_offset = command_ordinal * 32u,
        .byte_length = (command_type_ == AMDF_QUEUE_COMMAND_TYPE_GPU_PM4
                            ? kCopyDataDwordCount
                            : kSdmaCopyDwordCount) *
                       sizeof(uint32_t),
    };
    const amdf_gpu_kernel_queue_submission_info_t submit = {
        .type = AMDF_STRUCTURE_TYPE_GPU_KERNEL_QUEUE_SUBMISSION_INFO,
        .structure_size = sizeof(submit),
        .command_count = 1,
        .commands = &command,
    };
    uint64_t point = 0;
    ASSERT_EQ(
        gpu_api_->kernel_queue_submit(Queue(queue_ordinal), &submit, &point),
        AMDF_STATUS_OK);
    EXPECT_GT(point, points_[queue_ordinal]);
    points_[queue_ordinal] = point;
    indirect_memory_may_be_in_use_ = true;
  }

  void Request(uint32_t queue_ordinal, uint64_t point, uint32_t event_ordinal) {
    // Only the native object survives this call; its descriptor is borrowed.
    const amdf_native_event_t event = events_[event_ordinal];
    ASSERT_EQ(api_->kernel_queue_request_notification(Queue(queue_ordinal),
                                                      point, &event),
              AMDF_STATUS_OK);
  }

  void RefreshThrough(uint32_t queue_ordinal, uint64_t point) {
    amdf_kernel_queue_status_t status = {
        .type = AMDF_STRUCTURE_TYPE_KERNEL_QUEUE_STATUS,
        .structure_size = sizeof(status),
    };
    ASSERT_EQ(api_->kernel_queue_refresh_status(Queue(queue_ordinal), &status),
              AMDF_STATUS_OK);
    ASSERT_EQ(status.terminal_status, AMDF_STATUS_OK);
    EXPECT_GE(status.retired_submission, point);
  }

  void VerifyCopy(uint32_t ordinal) {
    ASSERT_EQ(api_->host_mapping_cache_control(
                  mapping_, AMDF_HOST_CACHE_OPERATION_INVALIDATE,
                  kTargetByteOffset + ordinal * 4, 4),
              AMDF_STATUS_OK);
    uint32_t actual = 0;
    std::memcpy(&actual, bytes_ + kTargetByteOffset + ordinal * 4, 4);
    EXPECT_EQ(actual, expected_[ordinal]);
  }

  // Optional second queue sharing the same device and caller-owned event.
  amdf_kernel_queue_t* sibling_ = nullptr;
  // Caller-owned destinations retained through final native queue release.
  std::array<amdf_native_event_t, 2> events_ = {};
  // Final accepted point per queue, including work awaiting notification.
  std::array<uint64_t, 2> points_ = {};
  // Expected output for the two disjoint finite copies.
  std::array<uint32_t, 2> expected_ = {};
  // Borrowed host view of command and payload backing owned by the base.
  uint8_t* bytes_ = nullptr;
};

TEST_P(GpuKernelQueueNotificationTest, EventDrivenCopiesAndOldPoints) {
  uint64_t oldest = 0;
  for (uint32_t round = 0; round < 8; ++round) {
    ASSERT_NO_FATAL_FAILURE(PrepareCopies(0x13579bdfu + round));
    ASSERT_NO_FATAL_FAILURE(SubmitCopy(0, 0));
    const uint64_t first = points_[0];
    ASSERT_NO_FATAL_FAILURE(SubmitCopy(0, 1));
    const uint64_t last = points_[0];
    if (oldest == 0) {
      oldest = first;
    }
    // Reverse registration order exercises earlier/later accepted points.
    ASSERT_NO_FATAL_FAILURE(Request(0, last, 1));
    ASSERT_NO_FATAL_FAILURE(Request(0, first, 0));
    ASSERT_TRUE(amdf::cts::WaitNativeEvent(events_[0]));
    ASSERT_TRUE(amdf::cts::WaitNativeEvent(events_[1]));
    amdf_kernel_queue_status_t status = {
        .type = AMDF_STRUCTURE_TYPE_KERNEL_QUEUE_STATUS,
        .structure_size = sizeof(status),
    };
    ASSERT_EQ(api_->kernel_queue_query_status(queue_, &status), AMDF_STATUS_OK);
    EXPECT_LT(status.retired_submission, first);
    ASSERT_NO_FATAL_FAILURE(RefreshThrough(0, last));
    ASSERT_NO_FATAL_FAILURE(VerifyCopy(0));
    ASSERT_NO_FATAL_FAILURE(VerifyCopy(1));
    // The oldest point still produces independent immediate hints after reuse.
    for (uint32_t i = 0; i < 2; ++i) {
      ASSERT_NO_FATAL_FAILURE(Request(0, oldest, 0));
      bool ready = false;
      ASSERT_TRUE(amdf::cts::TryConsumeNativeEvent(events_[0], &ready));
      ASSERT_TRUE(ready);
    }
  }
}

TEST_P(GpuKernelQueueNotificationTest, RepeatedRequestsKeepBothDestinations) {
  ASSERT_NO_FATAL_FAILURE(PrepareCopies(0x10293847u));
  ASSERT_NO_FATAL_FAILURE(SubmitCopy(0, 0));
  ASSERT_NO_FATAL_FAILURE(Request(0, points_[0], 0));
  ASSERT_NO_FATAL_FAILURE(Request(0, points_[0], 1));
  ASSERT_TRUE(amdf::cts::WaitNativeEvent(events_[0]));
  ASSERT_TRUE(amdf::cts::WaitNativeEvent(events_[1]));
  ASSERT_NO_FATAL_FAILURE(RefreshThrough(0, points_[0]));
  ASSERT_NO_FATAL_FAILURE(VerifyCopy(0));
}

TEST_P(GpuKernelQueueNotificationTest, IndependentQueuesShareOneWakeEvent) {
  const amdf_gpu_kernel_queue_create_info_t create = {
      .type = AMDF_STRUCTURE_TYPE_GPU_KERNEL_QUEUE_CREATE_INFO,
      .structure_size = sizeof(create),
      .queue_family_ordinal = family_.ordinal,
      .maximum_pending_submission_count = 2,
  };
  ASSERT_EQ(gpu_api_->kernel_queue_create(device_, &create, &sibling_),
            AMDF_STATUS_OK);
  ASSERT_NO_FATAL_FAILURE(PrepareCopies(0x2468ace0u));
  for (uint32_t i = 0; i < 2; ++i) {
    ASSERT_NO_FATAL_FAILURE(SubmitCopy(i, i));
    ASSERT_NO_FATAL_FAILURE(Request(i, points_[i], 0));
  }
  const amdf::cts::WaitDeadline deadline;
  bool complete = false;
  while (!complete) {
    ASSERT_TRUE(amdf::cts::WaitNativeEvent(events_[0], deadline));
    complete = true;
    for (uint32_t i = 0; i < 2; ++i) {
      amdf_kernel_queue_status_t status = {
          .type = AMDF_STRUCTURE_TYPE_KERNEL_QUEUE_STATUS,
          .structure_size = sizeof(status),
      };
      ASSERT_EQ(api_->kernel_queue_refresh_status(Queue(i), &status),
                AMDF_STATUS_OK);
      ASSERT_EQ(status.terminal_status, AMDF_STATUS_OK);
      complete &= status.retired_submission >= points_[i];
    }
  }
  ASSERT_NO_FATAL_FAILURE(VerifyCopy(0));
  ASSERT_NO_FATAL_FAILURE(VerifyCopy(1));
}

TEST_P(GpuKernelQueueNotificationTest, ConcurrentWaitRefreshAndRegistration) {
  ASSERT_NO_FATAL_FAILURE(PrepareCopies(0x76543210u));
  ASSERT_NO_FATAL_FAILURE(SubmitCopy(0, 0));
  const uint64_t first = points_[0];
  amdf_status_t wait_status = AMDF_STATUS_OK;
  amdf_status_t refresh_status = AMDF_STATUS_OK;
  amdf_kernel_queue_status_t observed = {
      .type = AMDF_STRUCTURE_TYPE_KERNEL_QUEUE_STATUS,
      .structure_size = sizeof(observed),
  };
  std::jthread waiter([&] {
    wait_status =
        api_->kernel_queue_wait(queue_, first, amdf::cts::kWaitTimeoutNs, 0);
  });
  std::jthread observer([&] {
    refresh_status = api_->kernel_queue_refresh_status(queue_, &observed);
  });
  std::jthread registrar([&] { Request(0, first, 0); });
  ASSERT_NO_FATAL_FAILURE(SubmitCopy(0, 1));
  ASSERT_NO_FATAL_FAILURE(Request(0, points_[0], 1));
  registrar.join();
  observer.join();
  waiter.join();
  ASSERT_FALSE(HasFatalFailure());
  EXPECT_EQ(wait_status, AMDF_STATUS_OK);
  EXPECT_EQ(refresh_status, AMDF_STATUS_OK);
  EXPECT_EQ(observed.terminal_status, AMDF_STATUS_OK);
  ASSERT_TRUE(amdf::cts::WaitNativeEvent(events_[0]));
  ASSERT_TRUE(amdf::cts::WaitNativeEvent(events_[1]));
  ASSERT_NO_FATAL_FAILURE(RefreshThrough(0, points_[0]));
  ASSERT_NO_FATAL_FAILURE(VerifyCopy(0));
  ASSERT_NO_FATAL_FAILURE(VerifyCopy(1));
}

INSTANTIATE_TEST_SUITE_P(
    CommandFormats, GpuKernelQueueNotificationTest,
    ::testing::Values(AMDF_QUEUE_COMMAND_TYPE_GPU_PM4,
                      AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA),
    [](const ::testing::TestParamInfo<amdf_queue_command_type_t>& info) {
      return info.param == AMDF_QUEUE_COMMAND_TYPE_GPU_PM4 ? "Pm4" : "Sdma";
    });

TEST_F(GpuDeviceFixture, RejectsOutOfRangeFamilyWithoutPublishingQueue) {
  amdf_device_info_t info = {
      .type = AMDF_STRUCTURE_TYPE_DEVICE_INFO,
      .structure_size = sizeof(info),
  };
  ASSERT_EQ(api_->device_query_info(device_, &info), AMDF_STATUS_OK);
  amdf_gpu_kernel_queue_create_info_t create_info = {
      .type = AMDF_STRUCTURE_TYPE_GPU_KERNEL_QUEUE_CREATE_INFO,
      .structure_size = sizeof(create_info),
      .queue_family_ordinal = info.queue_family_count,
  };
  amdf_kernel_queue_t* output =
      reinterpret_cast<amdf_kernel_queue_t*>(uintptr_t{1});
  EXPECT_EQ(amdf_status_code(
                gpu_api_->kernel_queue_create(device_, &create_info, &output)),
            AMDF_STATUS_CODE_OUT_OF_RANGE);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(output), uintptr_t{1});
}

TEST_F(Pm4KernelQueueTest, ExecutesMaterializedCopyData) {
  const uint32_t family_ordinal = family_.ordinal;
  ASSERT_TRUE(amdf_status_is_ok(CreateQueue(family_ordinal)));
  EXPECT_EQ(amdf_status_code(api_->device_destroy(device_)),
            AMDF_STATUS_CODE_BUSY);

  amdf_kernel_queue_info_t queue_info = {
      .type = AMDF_STRUCTURE_TYPE_KERNEL_QUEUE_INFO,
      .structure_size = sizeof(queue_info),
  };
  ASSERT_TRUE(
      amdf_status_is_ok(api_->kernel_queue_query_info(queue_, &queue_info)));
  EXPECT_EQ(queue_info.queue_family_ordinal, family_ordinal);
  EXPECT_EQ(queue_info.command_type, AMDF_QUEUE_COMMAND_TYPE_GPU_PM4);
  ASSERT_GE(queue_info.maximum_pending_submission_count, 1u);
  ASSERT_GE(queue_info.maximum_command_count, 1u);

  const uint64_t address = CreateCommandMemory();
  ASSERT_NE(memory_, nullptr);
  const amdf_host_mapping_info_t mapping_info = MapCommandMemory();
  ASSERT_NE(mapping_, nullptr);
  ASSERT_NE(mapping_info.pointer, nullptr);
  ASSERT_GE(mapping_info.byte_length, kMemoryByteLength);

  constexpr uint32_t kSourceValue = 0x13579BDFu;
  constexpr uint32_t kTargetSentinel = 0xA5A5A5A5u;
  uint8_t* const bytes = static_cast<uint8_t*>(mapping_info.pointer);
  std::memcpy(bytes + kSourceByteOffset, &kSourceValue, sizeof(kSourceValue));
  std::memcpy(bytes + kTargetByteOffset, &kTargetSentinel,
              sizeof(kTargetSentinel));
  const std::array<uint32_t, kCopyDataDwordCount> command =
      MakeCopyData32(address + kSourceByteOffset, address + kTargetByteOffset);
  std::memcpy(bytes + kCommandByteOffset, command.data(), sizeof(command));
  ASSERT_TRUE(amdf_status_is_ok(api_->host_mapping_cache_control(
      mapping_, AMDF_HOST_CACHE_OPERATION_FLUSH, 0, kMemoryByteLength)));

  amdf_gpu_kernel_command_t command_descriptor = {
      .memory = memory_,
      .byte_offset = kCommandByteOffset,
      .byte_length = sizeof(command),
  };
  amdf_gpu_kernel_queue_submission_info_t submission_info = {
      .type = AMDF_STRUCTURE_TYPE_GPU_KERNEL_QUEUE_SUBMISSION_INFO,
      .structure_size = sizeof(submission_info),
      .command_count = 1,
      .commands = &command_descriptor,
  };

  command_descriptor.byte_length = 0;
  uint64_t invalid_submission = 42;
  EXPECT_EQ(amdf_status_code(gpu_api_->kernel_queue_submit(
                queue_, &submission_info, &invalid_submission)),
            AMDF_STATUS_CODE_INVALID_ARGUMENT);
  EXPECT_EQ(invalid_submission, 42u);
  command_descriptor.byte_length = sizeof(command);

  command_descriptor.access_ordinal = 1;
  EXPECT_EQ(amdf_status_code(gpu_api_->kernel_queue_submit(
                queue_, &submission_info, &invalid_submission)),
            AMDF_STATUS_CODE_OUT_OF_RANGE);
  EXPECT_EQ(invalid_submission, 42u);
  command_descriptor.access_ordinal = 0;

  uint64_t submission = 42;
  ASSERT_TRUE(amdf_status_is_ok(
      gpu_api_->kernel_queue_submit(queue_, &submission_info, &submission)));
  indirect_memory_may_be_in_use_ = true;

  amdf_kernel_queue_status_t queue_status = {
      .type = AMDF_STRUCTURE_TYPE_KERNEL_QUEUE_STATUS,
      .structure_size = sizeof(queue_status),
  };
  ASSERT_EQ(api_->kernel_queue_query_status(queue_, &queue_status),
            AMDF_STATUS_OK);
  EXPECT_EQ(queue_status.retired_submission, 0u);
  EXPECT_EQ(queue_status.terminal_status, AMDF_STATUS_OK);

  ASSERT_TRUE(amdf_status_is_ok(api_->host_mapping_destroy(mapping_)));
  mapping_ = nullptr;

  // Poll checked progress without an exact-point native wait. Successful
  // retirement still gates result access, including after deadline failure.
  const amdf::cts::WaitDeadline deadline;
  while (queue_status.retired_submission < submission) {
    ASSERT_EQ(api_->kernel_queue_refresh_status(queue_, &queue_status),
              AMDF_STATUS_OK);
    ASSERT_EQ(queue_status.terminal_status, AMDF_STATUS_OK);
    if (queue_status.retired_submission < submission) {
      ASSERT_TRUE(deadline.Pause())
          << "retired " << queue_status.retired_submission << ", waiting for "
          << submission;
    }
  }
  indirect_memory_may_be_in_use_ = false;
  ASSERT_TRUE(amdf_status_is_ok(
      api_->kernel_queue_query_status(queue_, &queue_status)));
  EXPECT_EQ(queue_status.retired_submission, submission);
  EXPECT_EQ(queue_status.state, AMDF_QUEUE_STATE_ACTIVE);

  const amdf_host_mapping_info_t result_mapping_info = MapCommandMemory();
  ASSERT_NE(mapping_, nullptr);
  ASSERT_TRUE(amdf_status_is_ok(api_->host_mapping_cache_control(
      mapping_, AMDF_HOST_CACHE_OPERATION_INVALIDATE, kTargetByteOffset,
      sizeof(uint32_t))));
  uint32_t target_value = 0;
  std::memcpy(
      &target_value,
      static_cast<uint8_t*>(result_mapping_info.pointer) + kTargetByteOffset,
      sizeof(target_value));
  EXPECT_EQ(target_value, kSourceValue);

  ASSERT_TRUE(amdf_status_is_ok(api_->host_mapping_destroy(mapping_)));
  mapping_ = nullptr;
  ASSERT_TRUE(amdf_status_is_ok(
      api_->memory_destroy(std::exchange(memory_, nullptr), nullptr)));
  ASSERT_EQ(DestroyQueue(), AMDF_STATUS_OK);
}

TEST_F(Pm4KernelQueueTest, PipelinesCommandsAtConfiguredCapacity) {
  RunPipelinedCopies(8);
}

TEST_F(Pm4KernelQueueTest, PipelinesCommandsAtDefaultCapacity) {
  RunPipelinedCopies(0);
}

TEST_F(SdmaKernelQueueTest, PipelinesCommandsAtConfiguredCapacity) {
  RunPipelinedCopies(8);
}

TEST_F(SdmaKernelQueueTest, PipelinesCommandsAtDefaultCapacity) {
  RunPipelinedCopies(0);
}

TEST_F(SdmaKernelQueueTest, ExecutesMaterializedSdmaCopy) {
  const uint32_t family_ordinal = family_.ordinal;
  const amdf_status_t create_status = CreateQueue(family_ordinal);
  ASSERT_TRUE(amdf_status_is_ok(create_status))
      << "domain=" << amdf_status_domain(create_status)
      << " code=" << amdf_status_code(create_status);

  amdf_kernel_queue_info_t queue_info = {
      .type = AMDF_STRUCTURE_TYPE_KERNEL_QUEUE_INFO,
      .structure_size = sizeof(queue_info),
  };
  ASSERT_TRUE(
      amdf_status_is_ok(api_->kernel_queue_query_info(queue_, &queue_info)));
  EXPECT_EQ(queue_info.queue_family_ordinal, family_ordinal);
  EXPECT_EQ(queue_info.command_type, AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA);

  const uint64_t address = CreateCommandMemory();
  ASSERT_NE(memory_, nullptr);
  const amdf_host_mapping_info_t mapping_info = MapCommandMemory();
  ASSERT_NE(mapping_, nullptr);
  ASSERT_NE(mapping_info.pointer, nullptr);

  constexpr uint32_t kSourceValue = 0x89ABCDEFu;
  constexpr uint32_t kTargetSentinel = 0xA5A5A5A5u;
  uint8_t* const bytes = static_cast<uint8_t*>(mapping_info.pointer);
  std::memcpy(bytes + kSourceByteOffset, &kSourceValue, sizeof(kSourceValue));
  std::memcpy(bytes + kTargetByteOffset, &kTargetSentinel,
              sizeof(kTargetSentinel));
  const std::array<uint32_t, kSdmaCopyDwordCount> command =
      MakeSdmaCopy32(family_.format_features, address + kSourceByteOffset,
                     address + kTargetByteOffset);
  std::memcpy(bytes + kCommandByteOffset, command.data(), sizeof(command));
  ASSERT_TRUE(amdf_status_is_ok(api_->host_mapping_cache_control(
      mapping_, AMDF_HOST_CACHE_OPERATION_FLUSH, 0, kMemoryByteLength)));

  const amdf_gpu_kernel_command_t command_descriptor = {
      .memory = memory_,
      .access_ordinal = 0,
      .byte_offset = kCommandByteOffset,
      .byte_length = sizeof(command),
  };
  amdf_gpu_kernel_queue_submission_info_t submission_info = {
      .type = AMDF_STRUCTURE_TYPE_GPU_KERNEL_QUEUE_SUBMISSION_INFO,
      .structure_size = sizeof(submission_info),
      .command_count = 1,
      .commands = &command_descriptor,
  };
  uint64_t submission = 0;
  ASSERT_TRUE(amdf_status_is_ok(
      gpu_api_->kernel_queue_submit(queue_, &submission_info, &submission)));
  indirect_memory_may_be_in_use_ = true;
  ASSERT_TRUE(amdf_status_is_ok(api_->kernel_queue_wait(
      queue_, submission, amdf::cts::kWaitTimeoutNs, UINT64_C(50000))));
  indirect_memory_may_be_in_use_ = false;

  ASSERT_TRUE(amdf_status_is_ok(api_->host_mapping_cache_control(
      mapping_, AMDF_HOST_CACHE_OPERATION_INVALIDATE, kTargetByteOffset,
      sizeof(uint32_t))));
  uint32_t target_value = 0;
  std::memcpy(&target_value, bytes + kTargetByteOffset, sizeof(target_value));
  EXPECT_EQ(target_value, kSourceValue);

  ASSERT_TRUE(amdf_status_is_ok(api_->host_mapping_destroy(mapping_)));
  mapping_ = nullptr;
  ASSERT_EQ(DestroyQueue(), AMDF_STATUS_OK);
  ASSERT_TRUE(amdf_status_is_ok(
      api_->memory_destroy(std::exchange(memory_, nullptr), nullptr)));
}

TEST_F(Pm4KernelQueueTest, CopiesThroughDeviceLocalExecutableMemory) {
  const uint32_t family_ordinal = family_.ordinal;
  ASSERT_TRUE(amdf_status_is_ok(CreateQueue(family_ordinal)));

  const uint64_t address = CreateCommandMemory();
  ASSERT_NE(memory_, nullptr);
  const amdf_host_mapping_info_t mapping_info = MapCommandMemory();
  ASSERT_NE(mapping_, nullptr);
  ASSERT_NE(mapping_info.pointer, nullptr);
  ASSERT_GE(mapping_info.byte_length, kMemoryByteLength);

  const uint64_t local_address = CreateLocalExecutableMemory();
  ASSERT_NE(local_memory_, nullptr);
  ASSERT_NE(local_address, 0u);

  constexpr uint32_t kSourceValue = 0x2468ACE0u;
  constexpr uint32_t kTargetSentinel = 0xA5A5A5A5u;
  uint8_t* const bytes = static_cast<uint8_t*>(mapping_info.pointer);
  std::memcpy(bytes + kSourceByteOffset, &kSourceValue, sizeof(kSourceValue));
  std::memcpy(bytes + kTargetByteOffset, &kTargetSentinel,
              sizeof(kTargetSentinel));
  const std::array<uint32_t, kCopyDataDwordCount> upload_command =
      MakeCopyData32(address + kSourceByteOffset,
                     local_address + kLocalByteOffset);
  const std::array<uint32_t, kCopyDataDwordCount> download_command =
      MakeCopyData32(local_address + kLocalByteOffset,
                     address + kTargetByteOffset);
  std::array<uint32_t, 2 * kCopyDataDwordCount> command = {};
  std::memcpy(command.data(), upload_command.data(), sizeof(upload_command));
  std::memcpy(command.data() + kCopyDataDwordCount, download_command.data(),
              sizeof(download_command));
  std::memcpy(bytes + kCommandByteOffset, command.data(), sizeof(command));
  ASSERT_TRUE(amdf_status_is_ok(api_->host_mapping_cache_control(
      mapping_, AMDF_HOST_CACHE_OPERATION_FLUSH, 0, kMemoryByteLength)));

  const amdf_gpu_kernel_command_t command_descriptor = {
      .memory = memory_,
      .access_ordinal = 0,
      .byte_offset = kCommandByteOffset,
      .byte_length = sizeof(command),
  };
  amdf_gpu_kernel_queue_submission_info_t submission_info = {
      .type = AMDF_STRUCTURE_TYPE_GPU_KERNEL_QUEUE_SUBMISSION_INFO,
      .structure_size = sizeof(submission_info),
      .command_count = 1,
      .commands = &command_descriptor,
  };
  uint64_t submission = 0;
  ASSERT_TRUE(amdf_status_is_ok(
      gpu_api_->kernel_queue_submit(queue_, &submission_info, &submission)));
  indirect_memory_may_be_in_use_ = true;

  ASSERT_TRUE(amdf_status_is_ok(api_->kernel_queue_wait(
      queue_, submission, amdf::cts::kWaitTimeoutNs, UINT64_C(50000))));
  indirect_memory_may_be_in_use_ = false;
  ASSERT_TRUE(amdf_status_is_ok(api_->host_mapping_cache_control(
      mapping_, AMDF_HOST_CACHE_OPERATION_INVALIDATE, kTargetByteOffset,
      sizeof(uint32_t))));
  uint32_t target_value = 0;
  std::memcpy(&target_value, bytes + kTargetByteOffset, sizeof(target_value));
  EXPECT_EQ(target_value, kSourceValue);

  ASSERT_TRUE(amdf_status_is_ok(api_->host_mapping_destroy(mapping_)));
  mapping_ = nullptr;
  ASSERT_EQ(DestroyQueue(), AMDF_STATUS_OK);
  ASSERT_TRUE(amdf_status_is_ok(
      api_->memory_destroy(std::exchange(local_memory_, nullptr), nullptr)));
  ASSERT_TRUE(amdf_status_is_ok(
      api_->memory_destroy(std::exchange(memory_, nullptr), nullptr)));
}

TEST_F(Pm4KernelQueueTest, ExecutesDeviceLocalCommandStream) {
  const uint32_t family_ordinal = family_.ordinal;
  ASSERT_TRUE(amdf_status_is_ok(CreateQueue(family_ordinal)));

  const uint64_t address = CreateCommandMemory();
  ASSERT_NE(memory_, nullptr);
  const amdf_host_mapping_info_t mapping_info = MapCommandMemory();
  ASSERT_NE(mapping_, nullptr);
  ASSERT_NE(mapping_info.pointer, nullptr);
  ASSERT_GE(mapping_info.byte_length, kMemoryByteLength);
  const uint64_t local_address = CreateLocalExecutableMemory();
  ASSERT_NE(local_memory_, nullptr);
  ASSERT_NE(local_address, 0u);

  constexpr uint32_t kSourceValue = 0x10203040u;
  constexpr uint32_t kTargetSentinel = 0xA5A5A5A5u;
  uint8_t* const bytes = static_cast<uint8_t*>(mapping_info.pointer);
  std::memcpy(bytes + kSourceByteOffset, &kSourceValue, sizeof(kSourceValue));
  std::memcpy(bytes + kTargetByteOffset, &kTargetSentinel,
              sizeof(kTargetSentinel));

  const std::array<uint32_t, kCopyDataDwordCount> local_command =
      MakeCopyData32(address + kSourceByteOffset, address + kTargetByteOffset);
  std::memcpy(bytes + kStagedCommandByteOffset, local_command.data(),
              sizeof(local_command));
  std::array<uint32_t, 2 * kCopyDataDwordCount * kCopyDataDwordCount>
      upload_command = {};
  for (uint32_t i = 0; i < kCopyDataDwordCount; ++i) {
    const std::array<uint32_t, kCopyDataDwordCount> upload_word =
        MakeCopyData32(
            address + kStagedCommandByteOffset + i * sizeof(uint32_t),
            local_address + kLocalByteOffset + i * sizeof(uint32_t));
    const std::array<uint32_t, kCopyDataDwordCount> verify_word =
        MakeCopyData32(
            local_address + kLocalByteOffset + i * sizeof(uint32_t),
            address + kVerifiedCommandByteOffset + i * sizeof(uint32_t));
    std::memcpy(upload_command.data() + 2 * i * kCopyDataDwordCount,
                upload_word.data(), sizeof(upload_word));
    std::memcpy(upload_command.data() + (2 * i + 1) * kCopyDataDwordCount,
                verify_word.data(), sizeof(verify_word));
  }
  std::memcpy(bytes + kCommandByteOffset, upload_command.data(),
              sizeof(upload_command));
  ASSERT_TRUE(amdf_status_is_ok(api_->host_mapping_cache_control(
      mapping_, AMDF_HOST_CACHE_OPERATION_FLUSH, 0, kMemoryByteLength)));

  const amdf_gpu_kernel_command_t upload_descriptor = {
      .memory = memory_,
      .access_ordinal = 0,
      .byte_offset = kCommandByteOffset,
      .byte_length = sizeof(upload_command),
  };
  amdf_gpu_kernel_queue_submission_info_t submission_info = {
      .type = AMDF_STRUCTURE_TYPE_GPU_KERNEL_QUEUE_SUBMISSION_INFO,
      .structure_size = sizeof(submission_info),
      .command_count = 1,
      .commands = &upload_descriptor,
  };
  uint64_t upload_submission = 0;
  ASSERT_TRUE(amdf_status_is_ok(gpu_api_->kernel_queue_submit(
      queue_, &submission_info, &upload_submission)));
  indirect_memory_may_be_in_use_ = true;
  ASSERT_TRUE(amdf_status_is_ok(api_->kernel_queue_wait(
      queue_, upload_submission, amdf::cts::kWaitTimeoutNs, UINT64_C(50000))));
  indirect_memory_may_be_in_use_ = false;
  ASSERT_TRUE(amdf_status_is_ok(api_->host_mapping_cache_control(
      mapping_, AMDF_HOST_CACHE_OPERATION_INVALIDATE,
      kVerifiedCommandByteOffset, sizeof(local_command))));
  EXPECT_EQ(std::memcmp(bytes + kVerifiedCommandByteOffset,
                        local_command.data(), sizeof(local_command)),
            0);

  const amdf_gpu_kernel_command_t local_command_descriptor = {
      .memory = local_memory_,
      .access_ordinal = 0,
      .byte_offset = kLocalByteOffset,
      .byte_length = sizeof(local_command),
  };
  submission_info.commands = &local_command_descriptor;
  uint64_t local_submission = 0;
  ASSERT_TRUE(amdf_status_is_ok(gpu_api_->kernel_queue_submit(
      queue_, &submission_info, &local_submission)));
  indirect_memory_may_be_in_use_ = true;
  ASSERT_TRUE(amdf_status_is_ok(api_->kernel_queue_wait(
      queue_, local_submission, amdf::cts::kWaitTimeoutNs, UINT64_C(50000))));
  indirect_memory_may_be_in_use_ = false;

  ASSERT_TRUE(amdf_status_is_ok(api_->host_mapping_cache_control(
      mapping_, AMDF_HOST_CACHE_OPERATION_INVALIDATE, kTargetByteOffset,
      sizeof(uint32_t))));
  uint32_t target_value = 0;
  std::memcpy(&target_value, bytes + kTargetByteOffset, sizeof(target_value));
  EXPECT_EQ(target_value, kSourceValue);

  ASSERT_TRUE(amdf_status_is_ok(api_->host_mapping_destroy(mapping_)));
  mapping_ = nullptr;
  ASSERT_EQ(DestroyQueue(), AMDF_STATUS_OK);
  ASSERT_TRUE(amdf_status_is_ok(
      api_->memory_destroy(std::exchange(local_memory_, nullptr), nullptr)));
  ASSERT_TRUE(amdf_status_is_ok(
      api_->memory_destroy(std::exchange(memory_, nullptr), nullptr)));
}

}  // namespace

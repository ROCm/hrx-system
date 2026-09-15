// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <cstdint>
#include <cstring>

#include "amdf/amdf.h"
#include "amdf/gpu.h"
#include "gpu_device_fixture.h"
#include "gtest/gtest.h"

namespace {

constexpr uint64_t kMemoryByteLength = 4096;
constexpr uint64_t kCommandStride = 64;
constexpr uint64_t kSourceByteOffset = 1024;
constexpr uint64_t kIntermediateByteOffset = 1088;
constexpr uint64_t kResultByteOffset = 1152;

// One COPY_DATA packet with TC/L2 source and destination and write
// confirmation.
std::array<uint32_t, 6> MakeCopyData32(uint64_t source, uint64_t target) {
  return {
      (3u << 30) | (4u << 16) | (0x40u << 8),
      (2u << 0) | (2u << 8) | (1u << 20),
      static_cast<uint32_t>(source),
      static_cast<uint32_t>(source >> 32),
      static_cast<uint32_t>(target),
      static_cast<uint32_t>(target >> 32),
  };
}

// Each device sees the same host pages through its own attachment and queue.
struct DeviceAccess {
  // Borrowed device, retained by the cache or the explicit peer-lifetime case.
  amdf_device_t* device = nullptr;
  // Device-specific attachment, owning backing only for the first device.
  amdf_memory_t* memory = nullptr;
  // Immutable attachment properties.
  amdf_memory_info_t memory_info = {};
  // GPU address cached before preparing commands.
  uint64_t address = 0;
  // Explicit host access retained through the last use of registered pages.
  amdf_host_mapping_t* mapping = nullptr;
  // Host mapping properties and borrowed base pointer.
  amdf_host_mapping_info_t mapping_info = {};
  // Native PM4 queue belonging to this device.
  amdf_kernel_queue_t* queue = nullptr;
  // Nonzero while accepted work may still reference either attachment.
  uint64_t pending_submission = 0;
};

class GpuMemoryInteropTest : public GpuDeviceFixture {
 protected:
  void TearDown() override {
    for (const DeviceAccess& access : accesses_) {
      if (access.pending_submission != 0) {
        // A failed hardware wait is not proof that indirect resources are idle.
        // Preserve the entire ownership chain until process teardown.
        ADD_FAILURE() << "unretired GPU work: " << access.pending_submission;
        return;
      }
    }
    // Release the registration before its source mapping and backing.
    for (size_t i = accesses_.size(); i > 0; --i) {
      ASSERT_TRUE(amdf_status_is_ok(DestroyAccess(accesses_[i - 1])));
    }
    if (peer_device_ != nullptr) {
      ASSERT_TRUE(amdf_status_is_ok(api_->device_destroy(peer_device_)));
      peer_device_ = nullptr;
    }
    GpuDeviceFixture::TearDown();
  }

  amdf_status_t MatchGpuEndpoint(amdf_endpoint_t* endpoint,
                                 bool* out_matches) override {
    amdf_endpoint_info_t endpoint_info = {};
    endpoint_info.type = AMDF_STRUCTURE_TYPE_ENDPOINT_INFO;
    endpoint_info.structure_size = sizeof(endpoint_info);
    amdf_status_t status = api_->endpoint_query_info(endpoint, &endpoint_info);
    if (!amdf_status_is_ok(status)) return status;
    uint32_t family_ordinal = UINT32_MAX;
    for (uint32_t ordinal = 0; amdf_status_is_ok(status) &&
                               ordinal < endpoint_info.queue_family_count;
         ++ordinal) {
      amdf_queue_family_info_t family = {};
      family.type = AMDF_STRUCTURE_TYPE_QUEUE_FAMILY_INFO;
      family.structure_size = sizeof(family);
      status =
          api_->endpoint_query_queue_family_info(endpoint, ordinal, &family);
      if (amdf_status_is_ok(status) &&
          family.command_type == AMDF_QUEUE_COMMAND_TYPE_GPU_PM4 &&
          family.format_version == AMDF_GPU_PM4_QUEUE_FORMAT_VERSION_1 &&
          (family.roles & AMDF_QUEUE_ROLE_TRANSFER) != 0 &&
          (family.publication_modes & AMDF_QUEUE_PUBLICATION_MODE_KERNEL) !=
              0) {
        family_ordinal = ordinal;
        break;
      }
    }
    if (amdf_status_is_ok(status)) {
      family_ordinal_ = family_ordinal;
      *out_matches = family_ordinal != UINT32_MAX;
    }
    return status;
  }

  amdf_status_t CreateAccess(DeviceAccess& access, uint32_t family_ordinal,
                             void* registered_host_pointer) {
    const amdf_memory_device_access_t device_access = {
        access.device,
        {.access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE |
                   AMDF_MEMORY_ACCESS_EXECUTE,
         .flags = AMDF_MEMORY_FLAG_DEVICE_ADDRESS}};
    amdf_memory_create_info_t memory_info = {};
    memory_info.type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO;
    memory_info.structure_size = sizeof(memory_info);
    memory_info.access_count = 1;
    memory_info.accesses = &device_access;
    memory_info.required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE;
    const bool is_registration = registered_host_pointer != nullptr;
    memory_info.memory_profile_ordinal = FindMemoryProfileOrdinal(
        system_scope_,
        (is_registration ? AMDF_MEMORY_PROFILE_ROLE_REGISTER
                         : AMDF_MEMORY_PROFILE_ROLE_CREATE) |
            AMDF_MEMORY_PROFILE_ROLE_HOST_MAP,
        memory_info.required_flags, device_access.requirements);
    memory_info.byte_length = kMemoryByteLength;
    memory_info.registered_host_pointer = registered_host_pointer;
    amdf_status_t status =
        api_->memory_create(system_scope_, &memory_info, &access.memory);
    if (!amdf_status_is_ok(status)) return status;
    access.memory_info.type = AMDF_STRUCTURE_TYPE_MEMORY_INFO;
    access.memory_info.structure_size = sizeof(access.memory_info);
    status = api_->memory_query_info(access.memory, &access.memory_info);
    if (!amdf_status_is_ok(status)) return status;
    status = api_->memory_query_address(
        access.memory, 0, AMDF_MEMORY_ADDRESS_GPU, &access.address);
    if (!amdf_status_is_ok(status)) return status;

    amdf_memory_map_info_t map_info = {};
    map_info.type = AMDF_STRUCTURE_TYPE_MEMORY_MAP_INFO;
    map_info.structure_size = sizeof(map_info);
    map_info.flags = AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE;
    map_info.byte_length = kMemoryByteLength;
    status = api_->memory_map(access.memory, &map_info, &access.mapping);
    if (!amdf_status_is_ok(status)) return status;
    access.mapping_info.type = AMDF_STRUCTURE_TYPE_HOST_MAPPING_INFO;
    access.mapping_info.structure_size = sizeof(access.mapping_info);
    status =
        api_->host_mapping_query_info(access.mapping, &access.mapping_info);
    if (!amdf_status_is_ok(status)) return status;

    amdf_gpu_kernel_queue_create_info_t queue_info = {};
    queue_info.type = AMDF_STRUCTURE_TYPE_GPU_KERNEL_QUEUE_CREATE_INFO;
    queue_info.structure_size = sizeof(queue_info);
    queue_info.queue_family_ordinal = family_ordinal;
    return gpu_api_->kernel_queue_create(access.device, &queue_info,
                                         &access.queue);
  }

  amdf_status_t DestroyAccess(DeviceAccess& access) {
    if (access.queue != nullptr) {
      const amdf_status_t status = api_->kernel_queue_destroy(access.queue);
      if (!amdf_status_is_ok(status)) return status;
      access.queue = nullptr;
    }
    if (access.mapping != nullptr) {
      const amdf_status_t status = api_->host_mapping_destroy(access.mapping);
      if (!amdf_status_is_ok(status)) return status;
      access.mapping = nullptr;
    }
    if (access.memory != nullptr) {
      const amdf_status_t status = api_->memory_destroy(access.memory);
      if (!amdf_status_is_ok(status)) return status;
      access.memory = nullptr;
    }
    return AMDF_STATUS_OK;
  }

  void WriteCommand(DeviceAccess& access, uint64_t command_offset,
                    uint64_t source_offset, uint64_t target_offset) {
    const auto command = MakeCopyData32(access.address + source_offset,
                                        access.address + target_offset);
    std::memcpy(
        static_cast<uint8_t*>(access.mapping_info.pointer) + command_offset,
        command.data(), sizeof(command));
  }

  amdf_status_t Execute(DeviceAccess& access, uint64_t command_offset) {
    amdf_gpu_kernel_command_t command = {};
    command.memory = access.memory;
    command.byte_offset = command_offset;
    command.byte_length = 6 * sizeof(uint32_t);
    amdf_gpu_kernel_queue_submission_info_t submission = {};
    submission.type = AMDF_STRUCTURE_TYPE_GPU_KERNEL_QUEUE_SUBMISSION_INFO;
    submission.structure_size = sizeof(submission);
    submission.command_count = 1;
    submission.commands = &command;
    amdf_status_t status = gpu_api_->kernel_queue_submit(
        access.queue, &submission, &access.pending_submission);
    if (!amdf_status_is_ok(status)) return status;
    status = api_->kernel_queue_wait(access.queue, access.pending_submission,
                                     AMDF_TIMEOUT_INFINITE, UINT64_C(50000));
    if (amdf_status_is_ok(status)) access.pending_submission = 0;
    return status;
  }

  // Matching family selected before borrowing or creating native devices.
  uint32_t family_ordinal_ = UINT32_MAX;
  // Registration and execution resources, ordered owner before borrower.
  std::array<DeviceAccess, 2> accesses_;
  // Case-owned peer whose destruction must preserve the shared source owner.
  amdf_device_t* peer_device_ = nullptr;
};

TEST_F(GpuMemoryInteropTest, SharedBackingSurvivesIndependentPeerTeardown) {
  const uint32_t family_ordinal = family_ordinal_;
  amdf_gpu_device_create_info_t device_info = {};
  device_info.type = AMDF_STRUCTURE_TYPE_GPU_DEVICE_CREATE_INFO;
  device_info.structure_size = sizeof(device_info);
  ASSERT_TRUE(amdf_status_is_ok(
      gpu_api_->device_create(endpoint_, &device_info, &peer_device_)));
  accesses_[0].device = device_;
  accesses_[1].device = peer_device_;
  ASSERT_NE(device_, peer_device_);

  ASSERT_TRUE(
      amdf_status_is_ok(CreateAccess(accesses_[0], family_ordinal, nullptr)));
  auto* bytes = static_cast<uint8_t*>(accesses_[0].mapping_info.pointer);
  ASSERT_NE(bytes, nullptr);
  const amdf_status_t register_status =
      CreateAccess(accesses_[1], family_ordinal, bytes);
  ASSERT_TRUE(amdf_status_is_ok(register_status))
      << "domain=" << amdf_status_domain(register_status)
      << " code=" << amdf_status_code(register_status);
  ASSERT_EQ(accesses_[1].mapping_info.pointer, bytes);

  constexpr uint32_t kSourceValue = 0x8BADF00Du;
  constexpr uint32_t kSentinel = 0xA5A5A5A5u;
  std::memcpy(bytes + kSourceByteOffset, &kSourceValue, sizeof(kSourceValue));
  std::memcpy(bytes + kIntermediateByteOffset, &kSentinel, sizeof(kSentinel));
  std::memcpy(bytes + kResultByteOffset, &kSentinel, sizeof(kSentinel));
  WriteCommand(accesses_[0], 0, kSourceByteOffset, kIntermediateByteOffset);
  WriteCommand(accesses_[1], kCommandStride, kIntermediateByteOffset,
               kResultByteOffset);
  for (DeviceAccess& access : accesses_) {
    ASSERT_TRUE(amdf_status_is_ok(api_->host_mapping_cache_control(
        access.mapping, AMDF_HOST_CACHE_OPERATION_FLUSH, 0,
        kMemoryByteLength)));
  }

  // Only execution ordering crosses the host between queues. No host read,
  // write, copy, or cache operation touches the intermediate value.
  ASSERT_TRUE(amdf_status_is_ok(Execute(accesses_[0], 0)));
  ASSERT_TRUE(amdf_status_is_ok(Execute(accesses_[1], kCommandStride)));
  ASSERT_TRUE(amdf_status_is_ok(api_->host_mapping_cache_control(
      accesses_[0].mapping, AMDF_HOST_CACHE_OPERATION_INVALIDATE,
      kResultByteOffset, sizeof(uint32_t))));
  uint32_t result = 0;
  std::memcpy(&result, bytes + kResultByteOffset, sizeof(result));
  EXPECT_EQ(result, kSourceValue);

  // Removing a registration and its device must not free the original pages
  // or invalidate the original attachment's address or immutable command.
  ASSERT_TRUE(amdf_status_is_ok(DestroyAccess(accesses_[1])));
  ASSERT_TRUE(amdf_status_is_ok(api_->device_destroy(peer_device_)));
  peer_device_ = nullptr;
  constexpr uint32_t kReplacementValue = 0x12345678u;
  std::memcpy(bytes + kSourceByteOffset, &kReplacementValue,
              sizeof(kReplacementValue));
  ASSERT_TRUE(amdf_status_is_ok(api_->host_mapping_cache_control(
      accesses_[0].mapping, AMDF_HOST_CACHE_OPERATION_FLUSH, kSourceByteOffset,
      sizeof(uint32_t))));
  ASSERT_TRUE(amdf_status_is_ok(Execute(accesses_[0], 0)));
  ASSERT_TRUE(amdf_status_is_ok(api_->host_mapping_cache_control(
      accesses_[0].mapping, AMDF_HOST_CACHE_OPERATION_INVALIDATE,
      kIntermediateByteOffset, sizeof(uint32_t))));
  std::memcpy(&result, bytes + kIntermediateByteOffset, sizeof(result));
  EXPECT_EQ(result, kReplacementValue);
}

}  // namespace

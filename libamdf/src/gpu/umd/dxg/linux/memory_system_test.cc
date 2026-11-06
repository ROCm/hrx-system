// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <errno.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "libamdf/src/allocator.h"
#include "libamdf/src/gpu/umd/dxg/adapter.h"
#include "libamdf/src/gpu/umd/dxg/endpoint_properties.h"
#include "libamdf/src/gpu/umd/dxg/linux/api.h"
#include "libamdf/src/gpu/umd/dxg/linux/device.h"
#include "libamdf/src/gpu/umd/memory.h"
#include "libamdf/src/gpu/umd/memory_reservation.h"
#include "libamdf/src/platform/endpoint.h"
#include "libamdf/src/platform/host_memory.h"
#include "libamdf/src/platform/instance.h"

namespace {

constexpr uint64_t kByteLength = 64 * 1024;
constexpr uint64_t kSourceByteOffset = 4096;
constexpr uint64_t kTargetByteOffset = 8192;

std::array<uint32_t, 6> MakePm4CopyData32(uint64_t source_address,
                                          uint64_t target_address) {
  constexpr uint32_t kCopyDataHeader = (3u << 30) | (4u << 16) | (0x40u << 8);
  constexpr uint32_t kTcL2Source = 2u << 0;
  constexpr uint32_t kTcL2Target = 2u << 8;
  constexpr uint32_t kWaitForWriteConfirmation = 1u << 20;
  return {
      kCopyDataHeader,
      kTcL2Source | kTcL2Target | kWaitForWriteConfirmation,
      static_cast<uint32_t>(source_address),
      static_cast<uint32_t>(source_address >> 32),
      static_cast<uint32_t>(target_address),
      static_cast<uint32_t>(target_address >> 32),
  };
}

std::array<uint32_t, 7> MakeSdmaCopy32(
    amdf_queue_format_features_t format_features, uint64_t source_address,
    uint64_t target_address) {
  constexpr uint32_t kSdmaCopyLinearOpcode = 1;
  constexpr uint32_t kSystemScope = 3;
  const bool has_scope_fields =
      (format_features & AMDF_GPU_SDMA_FORMAT_FEATURE_MEMORY_SCOPE) != 0;
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

template <typename Function>
auto InvokeSilently(Function&& function) {
  ::testing::internal::CaptureStdout();
  ::testing::internal::CaptureStderr();
  auto result = function();
  const std::string stderr_output = ::testing::internal::GetCapturedStderr();
  const std::string stdout_output = ::testing::internal::GetCapturedStdout();
  EXPECT_TRUE(stdout_output.empty()) << stdout_output;
  EXPECT_TRUE(stderr_output.empty()) << stderr_output;
  return result;
}

bool IsDxgUnavailable(amdf_status_t status) {
  return amdf_status_domain(status) == AMDF_STATUS_DOMAIN_ERRNO &&
         (amdf_status_code(status) == ENOENT ||
          amdf_status_code(status) == ENODEV);
}

class LinuxDxgMemorySystemTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const amdf_allocator_t host_allocator = amdf_allocator_system();
    const amdf_status_t instance_status = InvokeSilently([&] {
      return amdf_platform_instance_create(host_allocator, &instance_);
    });
    if (IsDxgUnavailable(instance_status)) {
      GTEST_SKIP() << "/dev/dxg is unavailable on this runner";
    }
    ASSERT_EQ(instance_status, AMDF_STATUS_OK);

    uint32_t endpoint_count = 0;
    ASSERT_EQ(InvokeSilently([&] {
                return amdf_platform_endpoint_enumerate(instance_, 0, nullptr,
                                                        &endpoint_count);
              }),
              AMDF_STATUS_OK);
    if (endpoint_count == 0 && access("/dev/dxg", F_OK) < 0 &&
        errno == ENOENT) {
      GTEST_SKIP() << "/dev/dxg is unavailable on this runner";
    }
    ASSERT_GT(endpoint_count, 0u);
    std::vector<amdf_endpoint_summary_t> summaries(endpoint_count);
    ASSERT_EQ(InvokeSilently([&] {
                return amdf_platform_endpoint_enumerate(
                    instance_, endpoint_count, summaries.data(),
                    &endpoint_count);
              }),
              AMDF_STATUS_OK);

    amdf_endpoint_info_t endpoint_info = {};
    ASSERT_EQ(InvokeSilently([&] {
                return amdf_platform_endpoint_open(instance_,
                                                   &summaries.front().id,
                                                   &endpoint_, &endpoint_info);
              }),
              AMDF_STATUS_OK);
    if (endpoint_info.native_identity.type ==
        AMDF_ENDPOINT_NATIVE_IDENTITY_TYPE_LINUX_DEVICE) {
      GTEST_SKIP() << "/dev/dxg is unavailable on this runner";
    }
    ASSERT_EQ(endpoint_info.engine_kind, AMDF_ENGINE_KIND_GPU);

    amdf_platform_instance_lock_native(instance_);
    const amdf_status_t device_status = InvokeSilently([&] {
      return amdf_gpu_linux_dxg_device_create(
          nullptr, endpoint_, host_allocator, AMDF_NATIVE_LIFETIME_INSTANCE,
          &device_, &device_result_);
    });
    amdf_platform_instance_unlock_native(instance_);
    ASSERT_EQ(device_status, AMDF_STATUS_OK);
    EXPECT_NE(
        device_result_.features & AMDF_GPU_DEVICE_FEATURE_HOST_REGISTRATION,
        0u);
    EXPECT_NE(device_result_.features & AMDF_GPU_DEVICE_FEATURE_LOCAL_MEMORY,
              0u);
  }

  void TearDown() override {
    if (queue_ != nullptr) {
      const amdf_status_t status = InvokeSilently([&] {
        return amdf_gpu_dxg_adapter_destroy_kernel_queue(
            &device_->logical_device.provider_adapter, queue_);
      });
      EXPECT_EQ(status, AMDF_STATUS_OK);
      queue_ = nullptr;
    }
    if (reservation_ != nullptr) {
      const amdf_status_t status = InvokeSilently([&] {
        return amdf_gpu_linux_dxg_memory_reservation_destroy(reservation_);
      });
      EXPECT_EQ(status, AMDF_STATUS_OK);
      reservation_ = nullptr;
    }
    if (mapping_ != nullptr) {
      amdf_gpu_linux_dxg_host_mapping_destroy(mapping_);
      mapping_ = nullptr;
    }
    if (memory_ != nullptr) {
      const amdf_status_t status = InvokeSilently(
          [&] { return amdf_gpu_linux_dxg_memory_destroy(memory_); });
      EXPECT_EQ(status, AMDF_STATUS_OK);
      if (amdf_status_is_ok(status)) {
        memory_ = nullptr;
      }
    }
    if (registered_pointer_ != nullptr && memory_ == nullptr) {
      EXPECT_EQ(
          amdf_platform_host_memory_free(registered_pointer_, kByteLength),
          AMDF_STATUS_OK);
      registered_pointer_ = nullptr;
    }
    if (device_ != nullptr && memory_ == nullptr) {
      const amdf_status_t status = InvokeSilently(
          [&] { return amdf_gpu_linux_dxg_device_destroy(device_); });
      EXPECT_EQ(status, AMDF_STATUS_OK);
      if (amdf_status_is_ok(status)) {
        device_ = nullptr;
      }
    }
    if (endpoint_ != nullptr && device_ == nullptr) {
      const amdf_status_t status = InvokeSilently(
          [&] { return amdf_platform_endpoint_close(endpoint_); });
      EXPECT_EQ(status, AMDF_STATUS_OK);
      if (amdf_status_is_ok(status)) {
        endpoint_ = nullptr;
      }
    }
    if (instance_ != nullptr && endpoint_ == nullptr) {
      const amdf_status_t status = InvokeSilently(
          [&] { return amdf_platform_instance_destroy(instance_); });
      EXPECT_EQ(status, AMDF_STATUS_OK);
      if (amdf_status_is_ok(status)) {
        instance_ = nullptr;
      }
    }
  }

  amdf_memory_native_profile_t QueryProfile(uint32_t ordinal) {
    amdf_memory_native_profile_t profile = {};
    EXPECT_EQ(amdf_gpu_linux_dxg_device_query_memory_profile(device_, ordinal,
                                                             &profile),
              AMDF_STATUS_OK);
    return profile;
  }

  void PrepareMemory(const amdf_memory_native_profile_t& profile,
                     amdf_memory_flags_t required_flags,
                     void* registered_host_pointer,
                     amdf_memory_access_t device_access,
                     amdf_gpu_umd_memory_result_t* out_result) {
    const amdf_memory_native_create_info_t create_info = {
        .device_access = device_access,
        .required_flags = required_flags,
        .byte_length = kByteLength,
        .minimum_alignment = profile.device_address.minimum_alignment,
        .registered_host_pointer = registered_host_pointer,
    };
    ASSERT_EQ(InvokeSilently([&] {
                return amdf_gpu_linux_dxg_memory_prepare(device_, 0, nullptr,
                                                         &profile, &create_info,
                                                         &memory_, out_result);
              }),
              AMDF_STATUS_OK);
    ASSERT_NE(memory_, nullptr);
    EXPECT_EQ(out_result->byte_length, kByteLength);
    EXPECT_EQ(out_result->flags & required_flags, required_flags);
    EXPECT_NE(out_result->device_address, 0u);
  }

  void PrepareMemory(const amdf_memory_native_profile_t& profile,
                     amdf_memory_flags_t required_flags,
                     void* registered_host_pointer,
                     amdf_gpu_umd_memory_result_t* out_result) {
    PrepareMemory(profile, required_flags, registered_host_pointer,
                  AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
                  out_result);
  }

  void MapMemory(const amdf_memory_native_profile_t& profile,
                 amdf_gpu_umd_host_mapping_result_t* out_result) {
    const amdf_memory_map_info_t map_info = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_MAP_INFO,
        .structure_size = sizeof(map_info),
        .byte_offset = 0,
        .byte_length = kByteLength,
        .flags = AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE,
    };
    ASSERT_EQ(InvokeSilently([&] {
                return amdf_gpu_linux_dxg_memory_map(
                    memory_, &profile.host_mapping, &map_info, &mapping_,
                    out_result);
              }),
              AMDF_STATUS_OK);
    ASSERT_NE(mapping_, nullptr);
    ASSERT_NE(out_result->pointer, nullptr);
    EXPECT_EQ(out_result->byte_length, kByteLength);
  }

  void FlushMemory() {
    ASSERT_EQ(InvokeSilently([&] {
                return amdf_gpu_linux_dxg_host_mapping_cache_control(
                    mapping_, AMDF_HOST_CACHE_OPERATION_FLUSH, 0, kByteLength);
              }),
              AMDF_STATUS_OK);
  }

  void InvalidateMemory() {
    ASSERT_EQ(InvokeSilently([&] {
                return amdf_gpu_linux_dxg_host_mapping_cache_control(
                    mapping_, AMDF_HOST_CACHE_OPERATION_INVALIDATE, 0,
                    kByteLength);
              }),
              AMDF_STATUS_OK);
  }

  amdf_dxg_provider_gpu_kernel_queue_info_t CreateKernelQueue(
      amdf_dxg_provider_gpu_queue_command_type_t command_type) {
    const amdf_dxg_provider_gpu_kernel_queue_create_info_t create_info = {
        .device_handle = device_->logical_device.native,
        .command_type = command_type,
    };
    amdf_dxg_provider_gpu_kernel_queue_info_t info = {};
    EXPECT_EQ(InvokeSilently([&] {
                return amdf_gpu_dxg_adapter_create_kernel_queue(
                    &device_->logical_device.provider_adapter, &create_info,
                    &queue_, &info);
              }),
              AMDF_STATUS_OK);
    EXPECT_NE(queue_, nullptr);
    return info;
  }

  template <typename CommandFactory>
  void ExecuteCopyThroughPrivateKernelQueue(
      amdf_dxg_provider_gpu_queue_command_type_t command_type,
      CommandFactory make_commands) {
    const amdf_memory_native_profile_t profile = QueryProfile(0);
    ASSERT_EQ(profile.memory_class, AMDF_MEMORY_CLASS_SYSTEM);
    ASSERT_NE(profile.supported_flags & AMDF_MEMORY_FLAG_QUEUE_STORAGE, 0u);

    amdf_gpu_umd_memory_result_t memory_result = {};
    ASSERT_NO_FATAL_FAILURE(PrepareMemory(
        profile,
        AMDF_MEMORY_FLAG_HOST_VISIBLE | AMDF_MEMORY_FLAG_DEVICE_ADDRESS |
            AMDF_MEMORY_FLAG_QUEUE_STORAGE,
        nullptr,
        AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE |
            AMDF_MEMORY_ACCESS_EXECUTE,
        &memory_result));
    amdf_gpu_umd_host_mapping_result_t mapping_result = {};
    ASSERT_NO_FATAL_FAILURE(MapMemory(profile, &mapping_result));

    auto* words = static_cast<uint32_t*>(mapping_result.pointer);
    auto* source = reinterpret_cast<uint32_t*>(
        static_cast<uint8_t*>(mapping_result.pointer) + kSourceByteOffset);
    auto* target = reinterpret_cast<uint32_t*>(
        static_cast<uint8_t*>(mapping_result.pointer) + kTargetByteOffset);
    constexpr uint32_t kExpected = 0x13579BDFu;
    *source = kExpected;
    *target = ~kExpected;
    const auto commands =
        make_commands(memory_result.device_address + kSourceByteOffset,
                      memory_result.device_address + kTargetByteOffset);
    std::memcpy(words, commands.data(), commands.size() * sizeof(uint32_t));
    ASSERT_NO_FATAL_FAILURE(FlushMemory());

    const amdf_dxg_provider_gpu_kernel_queue_info_t queue_info =
        CreateKernelQueue(command_type);
    ASSERT_NE(queue_, nullptr);
    constexpr uint64_t kProgressValue = 1;
    bool accepted = false;
    ASSERT_EQ(InvokeSilently([&] {
                return amdf_gpu_dxg_adapter_submit_kernel_queue(
                    &device_->logical_device.provider_adapter, queue_,
                    memory_result.device_address,
                    commands.size() * sizeof(uint32_t), kProgressValue,
                    &accepted);
              }),
              AMDF_STATUS_OK);
    ASSERT_TRUE(accepted);
    ASSERT_NO_FATAL_FAILURE(WaitForQueue(queue_info, kProgressValue));
    ASSERT_NO_FATAL_FAILURE(InvalidateMemory());
    EXPECT_EQ(*target, kExpected);

    ASSERT_NO_FATAL_FAILURE(ReleaseQueue());
    ASSERT_NO_FATAL_FAILURE(ReleaseMemory());
  }

  void WaitForQueue(const amdf_dxg_provider_gpu_kernel_queue_info_t& info,
                    uint64_t progress_value) {
    ASSERT_NE(info.progress_fence_pointer, nullptr);
    D3DKMT_WAITFORSYNCHRONIZATIONOBJECTFROMCPU wait = {
        .hDevice = device_->logical_device.native,
        .ObjectCount = 1,
        .ObjectHandleArray = &info.progress_fence_handle,
        .FenceValueArray = &progress_value,
    };
    ASSERT_EQ(InvokeSilently([&] {
                return amdf_linux_dxg_kmt_wait_from_cpu(
                    device_->logical_device.kmt, &wait);
              }),
              0);
    EXPECT_GE(__atomic_load_n(info.progress_fence_pointer, __ATOMIC_ACQUIRE),
              progress_value);
  }

  void ReleaseQueue() {
    ASSERT_NE(queue_, nullptr);
    amdf_dxg_provider_gpu_kernel_queue_t* queue = queue_;
    queue_ = nullptr;
    ASSERT_EQ(InvokeSilently([&] {
                return amdf_gpu_dxg_adapter_destroy_kernel_queue(
                    &device_->logical_device.provider_adapter, queue);
              }),
              AMDF_STATUS_OK);
  }

  void ReleaseMemory() {
    if (mapping_ != nullptr) {
      amdf_gpu_linux_dxg_host_mapping_destroy(mapping_);
      mapping_ = nullptr;
    }
    ASSERT_EQ(InvokeSilently(
                  [&] { return amdf_gpu_linux_dxg_memory_destroy(memory_); }),
              AMDF_STATUS_OK);
    memory_ = nullptr;
  }

  void ReleaseReservation() {
    ASSERT_NE(reservation_, nullptr);
    amdf_gpu_umd_memory_reservation_t* reservation = reservation_;
    reservation_ = nullptr;
    ASSERT_EQ(
        InvokeSilently([&] {
          return amdf_gpu_linux_dxg_memory_reservation_destroy(reservation);
        }),
        AMDF_STATUS_OK);
  }

  amdf_platform_instance_t* instance_ = nullptr;
  amdf_platform_endpoint_t* endpoint_ = nullptr;
  amdf_gpu_umd_device_t* device_ = nullptr;
  amdf_gpu_umd_device_result_t device_result_ = {};
  amdf_gpu_umd_memory_t* memory_ = nullptr;
  amdf_dxg_provider_gpu_kernel_queue_t* queue_ = nullptr;
  amdf_gpu_umd_memory_reservation_t* reservation_ = nullptr;
  amdf_gpu_umd_host_mapping_t* mapping_ = nullptr;
  void* registered_pointer_ = nullptr;
};

TEST_F(LinuxDxgMemorySystemTest, OwnsSystemMemoryAndPublishesHostWrites) {
  const amdf_memory_native_profile_t profile = QueryProfile(0);
  ASSERT_NE(profile.roles & AMDF_MEMORY_PROFILE_ROLE_CREATE, 0u);
  ASSERT_NE(profile.roles & AMDF_MEMORY_PROFILE_ROLE_HOST_MAP, 0u);
  ASSERT_EQ(profile.memory_class, AMDF_MEMORY_CLASS_SYSTEM);

  amdf_gpu_umd_memory_result_t memory_result = {};
  ASSERT_NO_FATAL_FAILURE(PrepareMemory(
      profile, AMDF_MEMORY_FLAG_HOST_VISIBLE | AMDF_MEMORY_FLAG_DEVICE_ADDRESS,
      nullptr, &memory_result));
  amdf_gpu_umd_host_mapping_result_t mapping_result = {};
  ASSERT_NO_FATAL_FAILURE(MapMemory(profile, &mapping_result));
  std::memset(mapping_result.pointer, 0xA5, kByteLength);
  ASSERT_NO_FATAL_FAILURE(FlushMemory());
  ASSERT_NO_FATAL_FAILURE(ReleaseMemory());
}

TEST_F(LinuxDxgMemorySystemTest, OwnsDeviceLocalMemory) {
  const amdf_memory_native_profile_t profile = QueryProfile(1);
  ASSERT_NE(profile.roles & AMDF_MEMORY_PROFILE_ROLE_CREATE, 0u);
  ASSERT_EQ(profile.memory_class, AMDF_MEMORY_CLASS_LOCAL);

  amdf_gpu_umd_memory_result_t memory_result = {};
  ASSERT_NO_FATAL_FAILURE(PrepareMemory(
      profile, AMDF_MEMORY_FLAG_DEVICE_LOCAL | AMDF_MEMORY_FLAG_DEVICE_ADDRESS,
      nullptr, &memory_result));
  ASSERT_NO_FATAL_FAILURE(ReleaseMemory());
}

TEST_F(LinuxDxgMemorySystemTest, RegistersBorrowedHostMemory) {
  const amdf_memory_native_profile_t profile = QueryProfile(2);
  ASSERT_NE(profile.roles & AMDF_MEMORY_PROFILE_ROLE_REGISTER, 0u);
  ASSERT_NE(profile.roles & AMDF_MEMORY_PROFILE_ROLE_HOST_MAP, 0u);
  ASSERT_EQ(profile.memory_class, AMDF_MEMORY_CLASS_SYSTEM);

  ASSERT_EQ(
      amdf_platform_host_memory_allocate(
          kByteLength, profile.registration.registered_host_pointer_alignment,
          &registered_pointer_),
      AMDF_STATUS_OK);
  std::memset(registered_pointer_, 0x5A, kByteLength);
  amdf_gpu_umd_memory_result_t memory_result = {};
  ASSERT_NO_FATAL_FAILURE(PrepareMemory(
      profile, AMDF_MEMORY_FLAG_HOST_VISIBLE | AMDF_MEMORY_FLAG_DEVICE_ADDRESS,
      registered_pointer_, &memory_result));
  amdf_gpu_umd_host_mapping_result_t mapping_result = {};
  ASSERT_NO_FATAL_FAILURE(MapMemory(profile, &mapping_result));
  EXPECT_EQ(mapping_result.pointer, registered_pointer_);
  static_cast<uint8_t*>(mapping_result.pointer)[kByteLength - 1] = 0xC3;
  ASSERT_NO_FATAL_FAILURE(FlushMemory());
  ASSERT_NO_FATAL_FAILURE(ReleaseMemory());
  EXPECT_EQ(static_cast<uint8_t*>(registered_pointer_)[kByteLength - 1], 0xC3);
}

TEST_F(LinuxDxgMemorySystemTest, ExecutesPm4ThroughPrivateKernelQueue) {
  ASSERT_NE(device_->properties.supports_pm4_kernel_queue, 0u);
  ASSERT_NO_FATAL_FAILURE(ExecuteCopyThroughPrivateKernelQueue(
      AMDF_DXG_PROVIDER_GPU_QUEUE_COMMAND_TYPE_PM4, MakePm4CopyData32));
}

TEST_F(LinuxDxgMemorySystemTest, ExecutesSdmaThroughPrivateKernelQueue) {
  if (device_->properties.supports_sdma_kernel_queue == 0) {
    GTEST_SKIP() << "the DXG adapter exposes no immutable SDMA queue";
  }
  amdf_gpu_endpoint_properties_t properties = {};
  ASSERT_TRUE(amdf_gpu_dxg_endpoint_properties_translate(&device_->properties,
                                                         &properties));
  const auto* const family = std::find_if(
      properties.queue_families,
      properties.queue_families + properties.queue_family_count,
      [](const amdf_gpu_queue_family_properties_t& candidate) {
        return candidate.command_type == AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA;
      });
  ASSERT_NE(family, properties.queue_families + properties.queue_family_count);
  const amdf_queue_format_features_t format_features = family->format_features;
  ASSERT_NO_FATAL_FAILURE(ExecuteCopyThroughPrivateKernelQueue(
      AMDF_DXG_PROVIDER_GPU_QUEUE_COMMAND_TYPE_SDMA,
      [format_features](uint64_t source_address, uint64_t target_address) {
        return MakeSdmaCopy32(format_features, source_address, target_address);
      }));
}

TEST_F(LinuxDxgMemorySystemTest, OwnsSparseVirtualMemory) {
  const amdf_memory_native_profile_t profile = QueryProfile(0);
  ASSERT_NE(profile.supported_flags & AMDF_MEMORY_FLAG_MAPPING_SOURCE, 0u);
  amdf_memory_reservation_capabilities_t capabilities = {};
  ASSERT_EQ(amdf_gpu_linux_dxg_query_memory_reservation_capabilities(
                device_, 0, &capabilities),
            AMDF_STATUS_OK);
  ASSERT_GT(capabilities.mapping_granularity, 0u);
  ASSERT_LE(3 * capabilities.mapping_granularity, kByteLength);
  const amdf_memory_access_t access =
      capabilities.required_access | AMDF_MEMORY_ACCESS_WRITE;
  amdf_gpu_umd_memory_result_t memory_result = {};
  ASSERT_NO_FATAL_FAILURE(PrepareMemory(profile,
                                        AMDF_MEMORY_FLAG_MAPPING_SOURCE,
                                        nullptr, access, &memory_result));

  amdf_gpu_umd_device_t* devices[] = {device_};
  ASSERT_EQ(InvokeSilently([&] {
              return amdf_gpu_linux_dxg_memory_reservation_create(
                  amdf_allocator_system(), 1, devices, kByteLength,
                  capabilities.reservation_granularity, 0, &reservation_);
            }),
            AMDF_STATUS_OK);

  uint64_t address = 0;
  uint32_t mapping_access_ordinal = UINT32_MAX;
  uint64_t mapping_granularity = 0;
  amdf_memory_access_t required_access = 0;
  amdf_gpu_linux_dxg_memory_reservation_query_access(
      reservation_, 0, &address, &mapping_access_ordinal, &mapping_granularity,
      &required_access);
  ASSERT_NE(address, 0u);
  EXPECT_EQ(mapping_access_ordinal, 0u);
  ASSERT_EQ(mapping_granularity, capabilities.mapping_granularity);
  EXPECT_EQ(required_access, capabilities.required_access);
  const amdf_gpu_umd_mapping_update_t initial[] = {
      {
          .target_byte_offset = 0,
          .byte_length = mapping_granularity,
          .memory = memory_,
          .source_byte_offset = 0,
          .access = access,
      },
      {
          .target_byte_offset = 2 * mapping_granularity,
          .byte_length = mapping_granularity,
          .memory = memory_,
          .source_byte_offset = mapping_granularity,
          .access = access,
      },
  };
  bool mutation_started = false;
  ASSERT_EQ(InvokeSilently([&] {
              return amdf_gpu_linux_dxg_memory_reservation_update(
                  reservation_, 0, 2, initial, &mutation_started);
            }),
            AMDF_STATUS_OK);
  ASSERT_TRUE(mutation_started);

  const amdf_gpu_umd_mapping_update_t replacement = {
      .target_byte_offset = 0,
      .byte_length = mapping_granularity,
      .memory = memory_,
      .source_byte_offset = 2 * mapping_granularity,
      .access = access,
  };
  mutation_started = false;
  ASSERT_EQ(InvokeSilently([&] {
              return amdf_gpu_linux_dxg_memory_reservation_update(
                  reservation_, 0, 1, &replacement, &mutation_started);
            }),
            AMDF_STATUS_OK);
  ASSERT_TRUE(mutation_started);

  const amdf_gpu_umd_mapping_update_t clear = {
      .target_byte_offset = 2 * mapping_granularity,
      .byte_length = mapping_granularity,
  };
  mutation_started = false;
  ASSERT_EQ(InvokeSilently([&] {
              return amdf_gpu_linux_dxg_memory_reservation_update(
                  reservation_, 0, 1, &clear, &mutation_started);
            }),
            AMDF_STATUS_OK);
  ASSERT_TRUE(mutation_started);
  ASSERT_NO_FATAL_FAILURE(ReleaseReservation());
  ASSERT_NO_FATAL_FAILURE(ReleaseMemory());
}

}  // namespace

// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <dlfcn.h>
#include <errno.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "libamdf/src/allocator.h"
#include "libamdf/src/gpu/umd/device.h"
#include "libamdf/src/gpu/umd/memory.h"
#include "libamdf/src/platform/endpoint.h"
#include "libamdf/src/platform/host_memory.h"
#include "libamdf/src/platform/instance.h"

namespace {

constexpr uint64_t kByteLength = 64 * 1024;

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

bool ModuleIsMapped(const char* path) {
  void* module = dlopen(path, RTLD_NOW | RTLD_NOLOAD);
  if (module == nullptr) {
    return false;
  }
  dlclose(module);
  return true;
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
    ASSERT_EQ(endpoint_info.engine_kind, AMDF_ENGINE_KIND_GPU);

    amdf_platform_instance_lock_native(instance_);
    const amdf_status_t device_status = InvokeSilently([&] {
      return amdf_gpu_umd_device_create(nullptr, endpoint_, host_allocator,
                                        AMDF_NATIVE_LIFETIME_INSTANCE, &device_,
                                        &device_result_);
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
    if (mapping_ != nullptr) {
      amdf_gpu_umd_host_mapping_destroy(mapping_);
      mapping_ = nullptr;
    }
    if (memory_ != nullptr) {
      const amdf_status_t status =
          InvokeSilently([&] { return amdf_gpu_umd_memory_destroy(memory_); });
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
      const amdf_status_t status =
          InvokeSilently([&] { return amdf_gpu_umd_device_destroy(device_); });
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
    const char* sidecar_path = std::getenv("AMDF_DXG_BRIDGE_PATH");
    if (sidecar_path != nullptr && device_ == nullptr) {
      EXPECT_FALSE(ModuleIsMapped(sidecar_path));
    }
  }

  amdf_memory_native_profile_t QueryProfile(uint32_t ordinal) {
    amdf_memory_native_profile_t profile = {};
    EXPECT_EQ(
        amdf_gpu_umd_device_query_memory_profile(device_, ordinal, &profile),
        AMDF_STATUS_OK);
    return profile;
  }

  void PrepareMemory(const amdf_memory_native_profile_t& profile,
                     amdf_memory_flags_t required_flags,
                     void* registered_host_pointer,
                     amdf_gpu_umd_memory_result_t* out_result) {
    const amdf_memory_native_create_info_t create_info = {
        .device_access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
        .required_flags = required_flags,
        .byte_length = kByteLength,
        .minimum_alignment = profile.device_address.minimum_alignment,
        .registered_host_pointer = registered_host_pointer,
    };
    ASSERT_EQ(InvokeSilently([&] {
                return amdf_gpu_umd_memory_prepare(device_, 0, nullptr,
                                                   &profile, &create_info,
                                                   &memory_, out_result);
              }),
              AMDF_STATUS_OK);
    ASSERT_NE(memory_, nullptr);
    EXPECT_EQ(out_result->byte_length, kByteLength);
    EXPECT_EQ(out_result->flags & required_flags, required_flags);
    EXPECT_NE(out_result->device_address, 0u);
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
                return amdf_gpu_umd_memory_map(memory_, &profile.host_mapping,
                                               &map_info, &mapping_,
                                               out_result);
              }),
              AMDF_STATUS_OK);
    ASSERT_NE(mapping_, nullptr);
    ASSERT_NE(out_result->pointer, nullptr);
    EXPECT_EQ(out_result->byte_length, kByteLength);
  }

  void FlushMemory() {
    ASSERT_EQ(InvokeSilently([&] {
                return amdf_gpu_umd_host_mapping_cache_control(
                    mapping_, AMDF_HOST_CACHE_OPERATION_FLUSH, 0, kByteLength);
              }),
              AMDF_STATUS_OK);
  }

  void ReleaseMemory() {
    if (mapping_ != nullptr) {
      amdf_gpu_umd_host_mapping_destroy(mapping_);
      mapping_ = nullptr;
    }
    ASSERT_EQ(
        InvokeSilently([&] { return amdf_gpu_umd_memory_destroy(memory_); }),
        AMDF_STATUS_OK);
    memory_ = nullptr;
  }

  amdf_platform_instance_t* instance_ = nullptr;
  amdf_platform_endpoint_t* endpoint_ = nullptr;
  amdf_gpu_umd_device_t* device_ = nullptr;
  amdf_gpu_umd_device_result_t device_result_ = {};
  amdf_gpu_umd_memory_t* memory_ = nullptr;
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

}  // namespace

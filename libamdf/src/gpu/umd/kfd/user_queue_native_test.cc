// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/kfd/user_queue_native.h"

#include <cerrno>
#include <cstring>

#include "gtest/gtest.h"

namespace {

TEST(KfdUserQueueNativeTest, FailedVmFaultQueryPreservesOutput) {
  amdf_gpu_umd_device_t device = {
      .render_descriptor = -1,
  };
  struct drm_amdgpu_info_gpuvm_fault fault;
  std::memset(&fault, 0xA5, sizeof(fault));
  const struct drm_amdgpu_info_gpuvm_fault original_fault = fault;
  const amdf_gpu_kfd_user_queue_native_api_t* native_api =
      amdf_gpu_kfd_user_queue_default_native_api();
  EXPECT_EQ(native_api->vm_fault_query(native_api->user_data, &device, &fault),
            amdf_make_status(AMDF_STATUS_DOMAIN_ERRNO, EBADF));
  EXPECT_EQ(std::memcmp(&fault, &original_fault, sizeof(fault)), 0);
}

TEST(KfdUserQueueNativeTest, DestroyPreservesNativeErrorDomain) {
  amdf_gpu_kfd_memory_lock_t memory_lock;
  ASSERT_EQ(amdf_gpu_kfd_memory_lock_initialize(&memory_lock), 0);
  amdf_gpu_umd_device_t device = {
      .descriptor = -1,
      .memory_lock = &memory_lock,
  };
  const auto* native_api = amdf_gpu_kfd_user_queue_default_native_api();
  EXPECT_EQ(native_api->queue_destroy(native_api->user_data, &device, 47),
            amdf_make_status(AMDF_STATUS_DOMAIN_ERRNO, EBADF));
  EXPECT_FALSE(memory_lock.occupied);
  amdf_gpu_kfd_memory_lock_deinitialize(&memory_lock);
}

TEST(KfdUserQueueNativeTest,
     FailedCreateReleasesSerializationAndPreservesOutput) {
  amdf_gpu_kfd_memory_lock_t memory_lock;
  ASSERT_EQ(amdf_gpu_kfd_memory_lock_initialize(&memory_lock), 0);
  amdf_gpu_umd_device_t device = {
      .descriptor = -1,
      .memory_lock = &memory_lock,
  };
  kfd_ioctl_create_queue_args arguments;
  std::memset(&arguments, 0xA5, sizeof(arguments));
  const auto original = arguments;
  const auto* native_api = amdf_gpu_kfd_user_queue_default_native_api();
  EXPECT_EQ(
      native_api->queue_create(native_api->user_data, &device, &arguments),
      amdf_make_status(AMDF_STATUS_DOMAIN_ERRNO, EBADF));
  EXPECT_EQ(std::memcmp(&arguments, &original, sizeof(arguments)), 0);
  EXPECT_FALSE(memory_lock.occupied);
  amdf_gpu_kfd_memory_lock_deinitialize(&memory_lock);
}

}  // namespace

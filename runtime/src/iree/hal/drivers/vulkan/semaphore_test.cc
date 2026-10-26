// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/semaphore.h"

#include <atomic>
#include <semaphore>
#include <thread>

#include "iree/async/proactor_platform.h"
#include "iree/async/semaphore.h"
#include "iree/hal/drivers/vulkan/util/libvulkan.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::vulkan {
namespace {

#if !IREE_HAL_VULKAN_LIBVULKAN_STATIC

// Only the external Vulkan calls are replaced. The real HAL semaphore owns
// timeline publication, pending callbacks, failure propagation, and lifetime.
struct NativeSemaphore {
  // Value visible through the native counter query.
  std::atomic<uint64_t> value{0};
  // Number of native signal calls, including terminal failure publication.
  std::atomic<int> signal_count{0};
  // Whether native calls violated strictly increasing signal order.
  std::atomic<bool> regressed{false};
  // Normal signal result; failure-sentinel signals always succeed.
  VkResult signal_result = VK_SUCCESS;
  // Value whose native signal pauses before changing the counter.
  uint64_t paused_value = UINT64_MAX;
  // Handshake announcing entry to the paused native signal.
  std::binary_semaphore signal_entered{0};
  // Handshake allowing that native signal to complete.
  std::binary_semaphore signal_resume{0};
};

static VKAPI_ATTR VkResult VKAPI_CALL CaptureCreateSemaphore(
    VkDevice device, const VkSemaphoreCreateInfo* create_info,
    const VkAllocationCallbacks*, VkSemaphore* out_handle) {
  auto* native = reinterpret_cast<NativeSemaphore*>(device);
  const auto* timeline =
      static_cast<const VkSemaphoreTypeCreateInfo*>(create_info->pNext);
  native->value = timeline->initialValue;
  *out_handle = reinterpret_cast<VkSemaphore>(static_cast<uintptr_t>(1));
  return VK_SUCCESS;
}

static VKAPI_ATTR void VKAPI_CALL
CaptureDestroySemaphore(VkDevice, VkSemaphore, const VkAllocationCallbacks*) {}

static VKAPI_ATTR VkResult VKAPI_CALL CaptureGetSemaphoreCounterValue(
    VkDevice device, VkSemaphore, uint64_t* out_value) {
  *out_value = reinterpret_cast<NativeSemaphore*>(device)->value.load();
  return VK_SUCCESS;
}

static VKAPI_ATTR VkResult VKAPI_CALL CaptureSignalSemaphore(
    VkDevice device, const VkSemaphoreSignalInfo* signal_info) {
  auto* native = reinterpret_cast<NativeSemaphore*>(device);
  ++native->signal_count;
  if (signal_info->value == native->paused_value) {
    native->signal_entered.release();
    native->signal_resume.acquire();
  }
  if (signal_info->value != IREE_HAL_SEMAPHORE_FAILURE_VALUE &&
      native->signal_result != VK_SUCCESS) {
    return native->signal_result;
  }
  const uint64_t previous = native->value.exchange(signal_info->value);
  if (signal_info->value <= previous) {
    native->regressed = true;
  }
  return VK_SUCCESS;
}

static VKAPI_ATTR VkResult VKAPI_CALL CaptureWaitSemaphores(
    VkDevice device, const VkSemaphoreWaitInfo* wait_info, uint64_t timeout) {
  // These tests poll inside a failure callback, before its native failure
  // signal. Model the actual zero-timeout Vulkan result without blocking.
  EXPECT_EQ(timeout, 0u);
  EXPECT_EQ(wait_info->semaphoreCount, 1u);
  const uint64_t value = reinterpret_cast<NativeSemaphore*>(device)->value;
  return value >= wait_info->pValues[0] ? VK_SUCCESS : VK_TIMEOUT;
}

class VulkanSemaphoreTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(iree_async_proactor_create_platform(
        iree_async_proactor_options_default(), iree_allocator_system(),
        &proactor_));
    const iree_hal_vulkan_device_syms_t syms = {
        .vkCreateSemaphore = CaptureCreateSemaphore,
        .vkDestroySemaphore = CaptureDestroySemaphore,
        .vkGetSemaphoreCounterValue = CaptureGetSemaphoreCounterValue,
        .vkSignalSemaphore = CaptureSignalSemaphore,
        .vkWaitSemaphores = CaptureWaitSemaphores,
    };
    IREE_ASSERT_OK(iree_hal_vulkan_semaphore_create(
        reinterpret_cast<iree_hal_vulkan_logical_device_t*>(&native_), &syms,
        reinterpret_cast<VkDevice>(&native_), proactor_, /*initial_value=*/0,
        IREE_HAL_SEMAPHORE_FLAG_NONE, iree_allocator_system(), &semaphore_));
  }

  void TearDown() override {
    iree_hal_semaphore_release(semaphore_);
    iree_async_proactor_release(proactor_);
  }

  // Native dependency state, alive through HAL semaphore destruction.
  NativeSemaphore native_;
  // Real proactor required by the semaphore's async integration contract.
  iree_async_proactor_t* proactor_ = nullptr;
  // Real Vulkan HAL semaphore under test.
  iree_hal_semaphore_t* semaphore_ = nullptr;
};

TEST_F(VulkanSemaphoreTest, QueryDoesNotPublishInFlightHostSignal) {
  native_.paused_value = 1;
  std::thread signal_thread([&]() {
    IREE_EXPECT_OK(iree_hal_semaphore_signal(semaphore_, 1, nullptr));
  });
  native_.signal_entered.acquire();

  uint64_t value = UINT64_MAX;
  IREE_EXPECT_OK(iree_hal_semaphore_query(semaphore_, &value));
  EXPECT_EQ(value, 0u);

  native_.signal_resume.release();
  signal_thread.join();
  IREE_EXPECT_OK(iree_hal_semaphore_query(semaphore_, &value));
  EXPECT_EQ(value, 1u);
}

TEST_F(VulkanSemaphoreTest, FailureWinsBeforeNativeSignalCompletes) {
  native_.paused_value = 1;
  std::thread signal_thread([&]() {
    IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED,
                          iree_hal_semaphore_signal(semaphore_, 1, nullptr));
  });
  native_.signal_entered.acquire();
  std::thread failure_thread([&]() {
    iree_hal_semaphore_fail(semaphore_,
                            iree_status_from_code(IREE_STATUS_CANCELLED));
  });
  auto* async = reinterpret_cast<iree_async_semaphore_t*>(semaphore_);
  while (iree_async_semaphore_query_status(async) == IREE_STATUS_OK) {
    std::this_thread::yield();
  }
  native_.signal_resume.release();
  signal_thread.join();
  failure_thread.join();

  EXPECT_FALSE(native_.regressed);
  EXPECT_EQ(native_.value, IREE_HAL_SEMAPHORE_FAILURE_VALUE);
  uint64_t value = 0;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED,
                        iree_hal_semaphore_query(semaphore_, &value));
}

TEST_F(VulkanSemaphoreTest, FailedCallbackWaitSeesFailureBeforeNativeWakeup) {
  iree_async_semaphore_timepoint_t timepoint = {
      .callback =
          [](void* user_data, iree_async_semaphore_timepoint_t*,
             iree_status_t status) {
            auto* semaphore = static_cast<iree_hal_semaphore_t*>(user_data);
            IREE_EXPECT_STATUS_IS(IREE_STATUS_ABORTED, status);
            IREE_EXPECT_STATUS_IS(
                IREE_STATUS_ABORTED,
                iree_hal_semaphore_wait(semaphore, 1, iree_immediate_timeout(),
                                        IREE_ASYNC_WAIT_FLAG_NONE));
          },
      .user_data = semaphore_,
  };
  IREE_ASSERT_OK(iree_async_semaphore_acquire_timepoint(
      reinterpret_cast<iree_async_semaphore_t*>(semaphore_), 1, &timepoint));
  iree_hal_semaphore_fail(semaphore_,
                          iree_status_from_code(IREE_STATUS_ABORTED));
}

TEST_F(VulkanSemaphoreTest, CompletionCallbackCanSignalAgain) {
  iree_async_semaphore_timepoint_t timepoint = {
      .callback =
          [](void* user_data, iree_async_semaphore_timepoint_t*,
             iree_status_t status) {
            IREE_EXPECT_OK(status);
            IREE_EXPECT_OK(iree_hal_semaphore_signal(
                static_cast<iree_hal_semaphore_t*>(user_data), 2, nullptr));
          },
      .user_data = semaphore_,
  };
  IREE_ASSERT_OK(iree_async_semaphore_acquire_timepoint(
      reinterpret_cast<iree_async_semaphore_t*>(semaphore_), 1, &timepoint));
  IREE_ASSERT_OK(iree_hal_semaphore_signal(semaphore_, 1, nullptr));
  EXPECT_EQ(native_.value, 2u);
  EXPECT_EQ(native_.signal_count, 2);
}

TEST_F(VulkanSemaphoreTest, NativeSignalFailureFailsPendingTimepoint) {
  native_.signal_result = VK_ERROR_DEVICE_LOST;
  int callback_count = 0;
  iree_async_semaphore_timepoint_t timepoint = {
      .callback =
          [](void* user_data, iree_async_semaphore_timepoint_t*,
             iree_status_t status) {
            ++*static_cast<int*>(user_data);
            IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS, status);
          },
      .user_data = &callback_count,
  };
  IREE_ASSERT_OK(iree_async_semaphore_acquire_timepoint(
      reinterpret_cast<iree_async_semaphore_t*>(semaphore_), 1, &timepoint));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS,
                        iree_hal_semaphore_signal(semaphore_, 1, nullptr));
  EXPECT_EQ(callback_count, 1);
  EXPECT_EQ(native_.value, IREE_HAL_SEMAPHORE_FAILURE_VALUE);
  uint64_t value = 0;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS,
                        iree_hal_semaphore_query(semaphore_, &value));
}

TEST_F(VulkanSemaphoreTest, InvalidSignalsPreserveSemaphore) {
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_semaphore_signal(semaphore_, 0, nullptr));
  IREE_ASSERT_OK(iree_hal_semaphore_signal(semaphore_, 2, nullptr));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_semaphore_signal(semaphore_, 2, nullptr));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_hal_semaphore_signal(semaphore_, 1, nullptr));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hal_semaphore_signal(semaphore_, IREE_HAL_SEMAPHORE_FAILURE_VALUE,
                                nullptr));
  EXPECT_EQ(native_.signal_count, 1);
  IREE_ASSERT_OK(iree_hal_semaphore_signal(semaphore_, 3, nullptr));
  EXPECT_EQ(native_.value, 3u);
  EXPECT_FALSE(native_.regressed);
}

TEST_F(VulkanSemaphoreTest, NativeObservationDispatchesPendingTimepoint) {
  int callback_count = 0;
  iree_async_semaphore_timepoint_t timepoint = {
      .callback =
          [](void* user_data, iree_async_semaphore_timepoint_t*,
             iree_status_t status) {
            ++*static_cast<int*>(user_data);
            IREE_EXPECT_OK(status);
          },
      .user_data = &callback_count,
  };
  IREE_ASSERT_OK(iree_async_semaphore_acquire_timepoint(
      reinterpret_cast<iree_async_semaphore_t*>(semaphore_), 1, &timepoint));

  // The public native handle allows externally synchronized native producers.
  native_.value = 1;
  uint64_t value = 0;
  IREE_ASSERT_OK(iree_hal_semaphore_query(semaphore_, &value));
  EXPECT_EQ(value, 1u);
  EXPECT_EQ(callback_count, 1);
  IREE_ASSERT_OK(iree_hal_semaphore_signal(semaphore_, 2, nullptr));
  EXPECT_EQ(native_.value, 2u);
  EXPECT_FALSE(native_.regressed);
}

#endif  // !IREE_HAL_VULKAN_LIBVULKAN_STATIC

}  // namespace
}  // namespace iree::hal::vulkan

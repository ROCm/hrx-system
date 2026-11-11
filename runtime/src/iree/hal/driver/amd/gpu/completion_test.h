// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVER_AMD_GPU_COMPLETION_TEST_H_
#define IREE_HAL_DRIVER_AMD_GPU_COMPLETION_TEST_H_

#include <condition_variable>
#include <functional>
#include <mutex>
#include <vector>

#include "iree/hal/driver/amd/gpu/completion.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amd::gpu::testing {

// Dependency model for the native coalescing event. The actual IREE worker,
// source scheduling, stop protocol, and thread join are the subject under test.
class NativeEvent {
 public:
  amdf_gpu_event_t* handle() {
    return reinterpret_cast<amdf_gpu_event_t*>(this);
  }

  static amdf_status_t AMDF_CALL Wait(amdf_gpu_event_t* event,
                                      uint64_t timeout_nanoseconds) {
    EXPECT_EQ(timeout_nanoseconds, AMDF_TIMEOUT_INFINITE);
    auto* self = reinterpret_cast<NativeEvent*>(event);
    std::unique_lock<std::mutex> lock(self->mutex_);
    ++self->wait_count_;
    self->condition_.notify_all();
    self->condition_.wait(lock, [&] {
      return self->signaled_ || !amdf_status_is_ok(self->wait_status_);
    });
    self->signaled_ = false;
    return self->wait_status_;
  }

  static amdf_status_t AMDF_CALL Signal(amdf_gpu_event_t* event) {
    auto* self = reinterpret_cast<NativeEvent*>(event);
    std::lock_guard<std::mutex> lock(self->mutex_);
    ++self->signal_count_;
    if (amdf_status_is_ok(self->signal_status_)) {
      self->signaled_ = true;
    }
    self->condition_.notify_all();
    return self->signal_status_;
  }

  void AwaitWaitCount(uint32_t count) {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [&] { return wait_count_ >= count; });
  }

  void AwaitSignalCount(uint32_t count) {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [&] { return signal_count_ >= count; });
  }

  void SetWaitStatus(amdf_status_t status) {
    std::lock_guard<std::mutex> lock(mutex_);
    wait_status_ = status;
    condition_.notify_all();
  }

  void SetSignalStatus(amdf_status_t status) {
    std::lock_guard<std::mutex> lock(mutex_);
    signal_status_ = status;
  }

  uint32_t signal_count() {
    std::lock_guard<std::mutex> lock(mutex_);
    return signal_count_;
  }

 private:
  // Serializes dependency event state and test observation.
  std::mutex mutex_;
  // Native wait and controller readiness, without elapsed-time assumptions.
  std::condition_variable condition_;
  // Pending coalesced wake, consumed by the sole waiter.
  bool signaled_ = false;
  // Number of calls entering the native wait boundary.
  uint32_t wait_count_ = 0;
  // Number of attempted host signals, including failures.
  uint32_t signal_count_ = 0;
  // Native failure delivered to the waiting worker, if any.
  amdf_status_t wait_status_ = AMDF_STATUS_OK;
  // Native failure returned by the host wake boundary, if any.
  amdf_status_t signal_status_ = AMDF_STATUS_OK;
};

class CompletionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    gpu_.event_wait = NativeEvent::Wait;
    gpu_.event_signal = NativeEvent::Signal;
    context_.gpu = &gpu_;
    context_.event = event_.handle();
    context_.event_sink = {
        .fn =
            [](void* user_data, const iree_hal_device_event_t* event) {
              auto* self = static_cast<CompletionTest*>(user_data);
              EXPECT_EQ(event->type, IREE_HAL_DEVICE_EVENT_TYPE_DRIVER_FAILURE);
              const auto* payload = reinterpret_cast<
                  const iree_hal_device_driver_failure_event_t*>(
                  event->payload.data);
              self->reports_.push_back(payload->backend_result_code);
            },
        .user_data = this,
    };
  }

  void TearDown() override {
    IREE_EXPECT_OK(iree_hal_amd_gpu_completion_deinitialize(&completion_));
  }

  using Callback = std::function<bool(amdf_status_t)>;
  iree_hal_amd_gpu_completion_source_t Source(Callback& callback) {
    return {
        .fn =
            [](void* user_data, amdf_status_t status,
               iree_hal_amd_gpu_completion_scope_t* scope,
               iree_hal_amd_gpu_completion_wait_t* out_wait,
               iree_hal_amd_gpu_completion_copy_t* out_copy) {
              return (*static_cast<Callback*>(user_data))(status);
            },
        .user_data = &callback,
    };
  }

  using WaitCallback =
      std::function<bool(amdf_status_t, iree_hal_amd_gpu_completion_wait_t*)>;
  iree_hal_amd_gpu_completion_source_t Source(WaitCallback& callback) {
    return {
        .fn =
            [](void* user_data, amdf_status_t status,
               iree_hal_amd_gpu_completion_scope_t* scope,
               iree_hal_amd_gpu_completion_wait_t* out_wait,
               iree_hal_amd_gpu_completion_copy_t* out_copy) {
              return (*static_cast<WaitCallback*>(user_data))(status, out_wait);
            },
        .user_data = &callback,
    };
  }

  using Callout = std::function<bool(iree_hal_amd_gpu_completion_scope_t*)>;
  iree_hal_amd_gpu_completion_source_t Source(Callout& callback) {
    return {
        .fn =
            [](void* user_data, amdf_status_t,
               iree_hal_amd_gpu_completion_scope_t* scope,
               iree_hal_amd_gpu_completion_wait_t*,
               iree_hal_amd_gpu_completion_copy_t*) {
              return (*static_cast<Callout*>(user_data))(scope);
            },
        .user_data = &callback,
    };
  }

  using WaitCallout = std::function<bool(iree_hal_amd_gpu_completion_scope_t*,
                                         iree_hal_amd_gpu_completion_wait_t*)>;
  iree_hal_amd_gpu_completion_source_t Source(WaitCallout& callback) {
    return {
        .fn =
            [](void* user_data, amdf_status_t,
               iree_hal_amd_gpu_completion_scope_t* scope,
               iree_hal_amd_gpu_completion_wait_t* out_wait,
               iree_hal_amd_gpu_completion_copy_t* out_copy) {
              (void)out_copy;
              return (*static_cast<WaitCallout*>(user_data))(scope, out_wait);
            },
        .user_data = &callback,
    };
  }

  // Event dependency outlives the actual worker's final join.
  NativeEvent event_;
  // Native API table with only the dependency boundaries used here.
  amdf_gpu_api_t gpu_ = {};
  // Borrowed context whose native construction is covered by context tests.
  iree_hal_amd_gpu_context_t context_ = {};
  // Actual completion owner, safe for teardown before initialization.
  iree_hal_amd_gpu_completion_t completion_ = {};
  // Native failures reported once by the worker before its source callbacks.
  std::vector<uint64_t> reports_;
};

}  // namespace iree::hal::amd::gpu::testing

#endif  // IREE_HAL_DRIVER_AMD_GPU_COMPLETION_TEST_H_

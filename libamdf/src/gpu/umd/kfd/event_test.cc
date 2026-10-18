// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/event.h"

#include <linux/kfd_ioctl.h>

#include <cstdarg>
#include <cstring>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "libamdf/src/allocator.h"
#include "libamdf/src/gpu/umd/kfd/device.h"
#include "libamdf/src/gpu/umd/kfd/instance.h"
#include "libamdf/src/platform/linux/file.h"

namespace {

enum class WaitResult {
  kSignal,
  kMemoryFault,
  kHardwareFault,
  kTimeout,
  kInvalid
};

// Models the native dependencies of the real event implementation. No GPU
// faults are injected; native exception payloads exercise their consumer here.
struct NativeState {
  // Preparation result returned by the separately tested connection owner.
  amdf_status_t prepare_status = AMDF_STATUS_OK;
  // Whether the next local metadata allocation fails.
  bool fail_allocation = false;
  // Number of live metadata allocations owned by the event.
  uint32_t allocation_count = 0;
  // Native calls observed, including failed calls.
  std::vector<unsigned long> requests;
  // One syscall error returned by the next modeled ioctl.
  int error = 0;
  // Result of the next successful native wait.
  WaitResult wait_result = WaitResult::kSignal;
  // Timeout supplied to the most recent native wait, in milliseconds.
  uint32_t timeout = 0;
  // Durable age expected at entry and advanced by successful signal waits.
  uint64_t age = 1;
};

// Only this test thread's synthetic descriptor uses the native model.
thread_local NativeState* native_state = nullptr;

class KfdEventTest : public ::testing::Test {
 protected:
  void SetUp() override {
    native_state = &native_;
    ASSERT_EQ(pthread_mutex_init(&mutex_, nullptr), 0);
    device_.descriptor = 10000;
    device_.page_size = 4096;
    device_.memory_mutex = &mutex_;
    device_.host_allocator = {
        .user_data = &native_,
        .allocate = [](void* user_data, uint64_t byte_length,
                       uint64_t minimum_alignment) -> void* {
          auto& state = *static_cast<NativeState*>(user_data);
          if (state.fail_allocation) {
            return nullptr;
          }
          const auto allocator = amdf_allocator_system();
          void* allocation = allocator.allocate(allocator.user_data,
                                                byte_length, minimum_alignment);
          if (allocation) {
            ++state.allocation_count;
          }
          return allocation;
        },
        .free =
            [](void* user_data, void* allocation) {
              --static_cast<NativeState*>(user_data)->allocation_count;
              const auto allocator = amdf_allocator_system();
              allocator.free(allocator.user_data, allocation);
            },
    };
  }

  void TearDown() override {
    native_.error = 0;
    if (event_) {
      EXPECT_EQ(amdf_gpu_umd_event_destroy(event_), AMDF_STATUS_OK);
    }
    EXPECT_EQ(native_.allocation_count, 0u);
    EXPECT_EQ(pthread_mutex_destroy(&mutex_), 0);
    native_state = nullptr;
  }

  void Create() {
    ASSERT_EQ(amdf_gpu_umd_event_create(&device_, &event_, &result_),
              AMDF_STATUS_OK);
  }

  // Native dependency state, alive through event teardown.
  NativeState native_;
  // Real connection lock, used only by event creation.
  pthread_mutex_t mutex_;
  // Borrowed device carrying the exact native descriptor and allocator.
  amdf_gpu_umd_device_t device_ = {};
  // Test-owned real implementation, published only after creation succeeds.
  amdf_gpu_umd_event_t* event_ = nullptr;
  // Immutable producer operands returned with event ownership.
  amdf_gpu_umd_event_result_t result_ = {};
};

TEST_F(KfdEventTest, PublishesImmutableNativeOperands) {
  ASSERT_NO_FATAL_FAILURE(Create());
  EXPECT_EQ(result_.format, AMDF_GPU_EVENT_FORMAT_KFD_SIGNAL);
  EXPECT_EQ(result_.format_version, 1u);
  EXPECT_EQ(result_.publication_modes, AMDF_QUEUE_PUBLICATION_MODE_USER);
  EXPECT_EQ(result_.mailbox_address, 0x123456780008u);
  EXPECT_EQ(result_.trigger_data, 23u);
  EXPECT_EQ(result_.interrupt_context, 11u);
}

TEST_F(KfdEventTest, WaitRetainsAgeAndIncludesContextFaultSources) {
  ASSERT_NO_FATAL_FAILURE(Create());
  for (uint32_t i = 0; i < 4; ++i) {
    EXPECT_EQ(amdf_gpu_umd_event_wait(event_, AMDF_TIMEOUT_INFINITE),
              AMDF_STATUS_OK);
    EXPECT_EQ(native_.timeout, UINT32_MAX);
  }
  EXPECT_EQ(native_.age, 5u);
  EXPECT_EQ(native_.allocation_count, 1u);
}

TEST_F(KfdEventTest, HostSignalDoesNotMutateWaiterAge) {
  ASSERT_NO_FATAL_FAILURE(Create());
  EXPECT_EQ(amdf_gpu_umd_event_signal(event_), AMDF_STATUS_OK);
  EXPECT_EQ(native_.age, 1u);
  native_.error = EINTR;
  EXPECT_EQ(amdf_gpu_umd_event_signal(event_), AMDF_STATUS_OK);
  EXPECT_EQ(native_.requests.size(), 3u);
  EXPECT_EQ(amdf_gpu_umd_event_wait(event_, AMDF_TIMEOUT_INFINITE),
            AMDF_STATUS_OK);
}

TEST_F(KfdEventTest, ContextExceptionsReportDeviceLost) {
  ASSERT_NO_FATAL_FAILURE(Create());
  for (auto result : {WaitResult::kMemoryFault, WaitResult::kHardwareFault}) {
    native_.wait_result = result;
    EXPECT_EQ(amdf_gpu_umd_event_wait(event_, AMDF_TIMEOUT_INFINITE),
              amdf_make_api_status(AMDF_STATUS_CODE_DEVICE_LOST));
  }
}

TEST_F(KfdEventTest, InterruptedWaitReturnsForPredicateRecheck) {
  ASSERT_NO_FATAL_FAILURE(Create());
  native_.error = EINTR;
  EXPECT_EQ(amdf_gpu_umd_event_wait(event_, AMDF_TIMEOUT_INFINITE),
            AMDF_STATUS_OK);
  EXPECT_EQ(native_.requests.size(), 1u);
}

TEST_F(KfdEventTest, NativeErrorsArePreserved) {
  ASSERT_NO_FATAL_FAILURE(Create());
  native_.error = EIO;
  EXPECT_EQ(amdf_gpu_umd_event_wait(event_, AMDF_TIMEOUT_INFINITE),
            amdf_linux_error(EIO));
  native_.error = EIO;
  EXPECT_EQ(amdf_gpu_umd_event_signal(event_), amdf_linux_error(EIO));
}

TEST_F(KfdEventTest, FiniteWaitRoundsUpAndCannotBecomeNativeInfinite) {
  ASSERT_NO_FATAL_FAILURE(Create());
  for (const auto& duration : {std::pair<uint64_t, uint32_t>{0, 0},
                               {1, 1},
                               {1000001, 2},
                               {UINT64_MAX - 1, INT32_MAX}}) {
    EXPECT_EQ(amdf_gpu_umd_event_wait(event_, duration.first), AMDF_STATUS_OK);
    EXPECT_EQ(native_.timeout, duration.second);
  }
}

TEST_F(KfdEventTest, EmptyZeroWaitReportsDeadline) {
  ASSERT_NO_FATAL_FAILURE(Create());
  native_.wait_result = WaitResult::kTimeout;
  EXPECT_EQ(amdf_gpu_umd_event_wait(event_, 0),
            amdf_make_api_status(AMDF_STATUS_CODE_DEADLINE_EXCEEDED));
  EXPECT_EQ(native_.requests.size(), 1u);
}

TEST_F(KfdEventTest, ImpossibleNativeWaitResultsReportProtocolError) {
  ASSERT_NO_FATAL_FAILURE(Create());
  for (auto result : {WaitResult::kTimeout, WaitResult::kInvalid}) {
    native_.wait_result = result;
    EXPECT_EQ(amdf_gpu_umd_event_wait(event_, AMDF_TIMEOUT_INFINITE),
              amdf_linux_error(EPROTO));
  }
}

TEST_F(KfdEventTest, FailedDestructionPreservesEventOwner) {
  ASSERT_NO_FATAL_FAILURE(Create());
  native_.error = EIO;
  EXPECT_EQ(amdf_gpu_umd_event_destroy(event_), amdf_linux_error(EIO));
  EXPECT_EQ(native_.allocation_count, 1u);
  EXPECT_EQ(amdf_gpu_umd_event_signal(event_), AMDF_STATUS_OK);
  ASSERT_EQ(amdf_gpu_umd_event_destroy(event_), AMDF_STATUS_OK);
  event_ = nullptr;
  EXPECT_EQ(native_.allocation_count, 0u);
}

TEST_F(KfdEventTest, FailedConstructionPublishesNothing) {
  auto* const sentinel = reinterpret_cast<amdf_gpu_umd_event_t*>(uintptr_t{1});
  amdf_gpu_umd_event_t* output = sentinel;
  amdf_gpu_umd_event_result_t result = {.mailbox_address = 0x1234};
  const auto original = result;
  native_.fail_allocation = true;
  EXPECT_EQ(amdf_gpu_umd_event_create(&device_, &output, &result),
            amdf_make_api_status(AMDF_STATUS_CODE_RESOURCE_EXHAUSTED));
  native_.fail_allocation = false;
  native_.prepare_status = amdf_linux_error(ENOMEM);
  EXPECT_EQ(amdf_gpu_umd_event_create(&device_, &output, &result),
            native_.prepare_status);
  EXPECT_EQ(output, sentinel);
  EXPECT_EQ(std::memcmp(&result, &original, sizeof(result)), 0);
  EXPECT_EQ(native_.allocation_count, 0u);
}

}  // namespace

extern "C" {
int __real_ioctl(int descriptor, unsigned long request, ...);

int __wrap_ioctl(int descriptor, unsigned long request, ...) {
  va_list arguments;
  va_start(arguments, request);
  void* argument = va_arg(arguments, void*);
  va_end(arguments);
  if (native_state == nullptr || descriptor != 10000) {
    return __real_ioctl(descriptor, request, argument);
  }
  auto& state = *native_state;
  state.requests.push_back(request);
  if (state.error != 0) {
    errno = state.error;
    state.error = 0;
    return -1;
  }
  if (request == AMDKFD_IOC_SET_EVENT) {
    EXPECT_EQ(static_cast<kfd_ioctl_set_event_args*>(argument)->event_id, 11u);
    return 0;
  }
  if (request == AMDKFD_IOC_DESTROY_EVENT) {
    EXPECT_EQ(static_cast<kfd_ioctl_destroy_event_args*>(argument)->event_id,
              11u);
    return 0;
  }
  EXPECT_EQ(request, AMDKFD_IOC_WAIT_EVENTS);
  auto& wait = *static_cast<kfd_ioctl_wait_events_args*>(argument);
  EXPECT_EQ(wait.num_events, 3u);
  EXPECT_EQ(wait.wait_for_all, 0u);
  state.timeout = wait.timeout;
  auto* events = reinterpret_cast<kfd_event_data*>(wait.events_ptr);
  EXPECT_EQ(events[0].event_id, 11u);
  EXPECT_EQ(events[1].event_id, 12u);
  EXPECT_EQ(events[2].event_id, 13u);
  EXPECT_EQ(events[0].signal_event_data.last_event_age, state.age);
  EXPECT_EQ(events[1].memory_exception_data.failure.NotPresent, UINT32_MAX);
  EXPECT_EQ(events[2].hw_exception_data.reset_type, UINT32_MAX);
  wait.wait_result = KFD_IOC_WAIT_RESULT_COMPLETE;
  switch (state.wait_result) {
    case WaitResult::kSignal:
      events[0].signal_event_data.last_event_age = ++state.age;
      break;
    case WaitResult::kMemoryFault:
      events[1].memory_exception_data = {};
      break;
    case WaitResult::kHardwareFault:
      events[2].hw_exception_data = {};
      break;
    case WaitResult::kTimeout:
      wait.wait_result = KFD_IOC_WAIT_RESULT_TIMEOUT;
      break;
    case WaitResult::kInvalid:
      wait.wait_result = KFD_IOC_WAIT_RESULT_FAIL;
      break;
  }
  return 0;
}

amdf_status_t __wrap_amdf_gpu_kfd_instance_create_event(
    amdf_gpu_umd_instance_t*, const amdf_gpu_kfd_topology_t*, size_t page_size,
    amdf_gpu_kfd_event_info_t* out_info) {
  EXPECT_EQ(page_size, 4096u);
  if (!amdf_status_is_ok(native_state->prepare_status)) {
    return native_state->prepare_status;
  }
  *out_info = {
      .mailbox_address = 0x123456780008,
      .signal_id = 11,
      .trigger_data = 23,
      .memory_exception_id = 12,
      .hardware_exception_id = 13,
  };
  return AMDF_STATUS_OK;
}
}  // extern "C"

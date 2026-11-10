// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include <atomic>
#include <barrier>
#include <thread>

#include "hrx_internal.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

struct ProfileOutput {
  // Number of backend flushes that entered the shared output concurrently.
  std::atomic<int> active{0};
  // Completed backend flush calls.
  std::atomic<int> calls{0};
  // Dependency error returned by the next flushes until cleared by the test.
  bool fail = false;
};

struct ProfileDevice {
  explicit ProfileDevice(ProfileOutput* output) : output(output) {
    static const iree_hal_device_vtable_t vtable = {
        .destroy = [](iree_hal_device_t*) {},
        .profiling_flush = [](iree_hal_device_t* base) -> iree_status_t {
          auto* output = reinterpret_cast<ProfileDevice*>(base)->output;
          EXPECT_EQ(output->active.fetch_add(1), 0);
          // A sink can yield or block during I/O. Separate devices still share
          // its output stream and cannot enter concurrently.
          std::this_thread::yield();
          ++output->calls;
          EXPECT_EQ(output->active.fetch_sub(1), 1);
          return output->fail ? iree_make_status(IREE_STATUS_DATA_LOSS,
                                                 "profile output failed")
                              : iree_ok_status();
        },
    };
    iree_hal_resource_initialize(&vtable, &resource);
  }

  iree_hal_device_t* get() {
    return reinterpret_cast<iree_hal_device_t*>(this);
  }

  // HAL dependency used only for its profiling flush boundary.
  iree_hal_resource_t resource;
  // Borrowed file-output state shared by all profiled devices.
  ProfileOutput* output;
};

TEST(RuntimeProfileTest, InactiveCaptureDoesNotCallBackend) {
  ProfileOutput output;
  ProfileDevice backend(&output);
  hrx_device_s device = {.hal_device = backend.get()};
  IREE_EXPECT_OK(hrx_runtime_profile_flush(nullptr));
  IREE_EXPECT_OK(hrx_runtime_profile_flush(&device));
  EXPECT_EQ(output.calls, 0);
}

TEST(RuntimeProfileTest, StreamSynchronizationReturnsProfileOutputFailure) {
  ProfileOutput output;
  ProfileDevice backend(&output);
  hrx_device_s device = {.hal_device = backend.get(), .profiling_active = true};
  hrx_queue_s queue = {.device = &device};
  hrx_stream_s stream = {.queue = &queue};
  output.fail = true;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS,
                        hrx_status_to_iree(hrx_stream_synchronize(&stream)));
  output.fail = false;
  IREE_EXPECT_OK(hrx_status_to_iree(hrx_stream_synchronize(&stream)));
  EXPECT_EQ(output.calls, 2);
}

TEST(RuntimeProfileTest, ConcurrentDevicesSerializeSharedProfileOutput) {
  ProfileOutput output;
  ProfileDevice first_backend(&output);
  ProfileDevice second_backend(&output);
  hrx_device_s devices[] = {
      {.hal_device = first_backend.get(), .profiling_active = true},
      {.hal_device = second_backend.get(), .profiling_active = true},
  };
  std::barrier<> start(2);
  auto flush = [&](hrx_device_t device) {
    start.arrive_and_wait();
    for (int i = 0; i < 1000; ++i) {
      IREE_ASSERT_OK(hrx_runtime_profile_flush(device));
    }
  };
  std::thread first(flush, &devices[0]);
  std::thread second(flush, &devices[1]);
  first.join();
  second.join();
  EXPECT_EQ(output.calls, 2000);
}

}  // namespace

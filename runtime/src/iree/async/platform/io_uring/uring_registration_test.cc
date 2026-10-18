// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/platform/io_uring/uring_registration.h"

#include <sys/mman.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cstdint>
#include <thread>

#include "iree/async/platform/io_uring/defs.h"
#include "iree/async/platform/io_uring/uring.h"
#include "iree/base/threading/notification.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

constexpr uint32_t kProbeOpcodeCount = 64;

struct ProbeStorage {
  alignas(iree_io_uring_probe_t)
      uint8_t bytes[sizeof(iree_io_uring_probe_t) +
                    kProbeOpcodeCount * sizeof(iree_io_uring_probe_op_t)];

  iree_io_uring_probe_t* probe() {
    return reinterpret_cast<iree_io_uring_probe_t*>(bytes);
  }
};

struct TestNotification {
  std::atomic<bool> signaled{false};
  iree_notification_t notification;
};

static void InitializeTestNotification(TestNotification* value) {
  iree_notification_initialize(&value->notification);
}

static void DeinitializeTestNotification(TestNotification* value) {
  iree_notification_deinitialize(&value->notification);
}

static bool TestNotificationIsSignaled(void* user_data) {
  auto* value = static_cast<TestNotification*>(user_data);
  return value->signaled.load(std::memory_order_acquire);
}

static void SignalTestNotification(TestNotification* value) {
  value->signaled.store(true, std::memory_order_release);
  iree_notification_post(&value->notification, IREE_ALL_WAITERS);
}

static void AwaitTestNotification(TestNotification* value) {
  iree_notification_await(&value->notification, TestNotificationIsSignaled,
                          value, iree_infinite_timeout());
}

static void WakeRegistrationOwner(void* user_data) {
  SignalTestNotification(static_cast<TestNotification*>(user_data));
}

class IoUringRegistrationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_io_uring_ring_options_t options = iree_io_uring_ring_options_default();
    options.threading_mode = IREE_IO_URING_RING_THREADING_CROSS_THREAD;
    iree_status_t status = iree_io_uring_ring_initialize(options, &ring_);
    if (iree_status_is_unavailable(status)) {
      iree_status_free(status);
      GTEST_SKIP() << "io_uring is unavailable";
    }
    IREE_ASSERT_OK(status);
    ring_initialized_ = true;

    if (!iree_all_bits_set(
            ring_.setup_flags,
            IREE_IORING_SETUP_SINGLE_ISSUER | IREE_IORING_SETUP_R_DISABLED)) {
      GTEST_SKIP() << "kernel lacks cross-thread SINGLE_ISSUER support";
    }
  }

  void TearDown() override {
    if (ring_initialized_) {
      iree_io_uring_ring_deinitialize(&ring_);
    }
  }

  iree_io_uring_ring_t ring_ = {};
  bool ring_initialized_ = false;
};

TEST_F(IoUringRegistrationTest, ExecutesDirectlyBeforeOwnerBinding) {
  ProbeStorage probe_storage = {};
  EXPECT_EQ(
      iree_io_uring_ring_register(&ring_, IREE_IORING_REGISTER_PROBE,
                                  probe_storage.probe(), kProbeOpcodeCount),
      0);
  EXPECT_GT(probe_storage.probe()->ops_len, 0);
}

TEST_F(IoUringRegistrationTest,
       BoundOwnerDrainsQueuedRequestsBeforeDirectCalls) {
  TestNotification owner_ready;
  TestNotification owner_wake;
  InitializeTestNotification(&owner_ready);
  InitializeTestNotification(&owner_wake);
  iree_io_uring_ring_set_registration_wake_callback(
      &ring_, {WakeRegistrationOwner, &owner_wake});

  ProbeStorage queued_probe = {};
  ProbeStorage owner_probe = {};
  std::atomic<int> enable_code{IREE_STATUS_UNKNOWN};
  std::atomic<int> queued_result{-EINPROGRESS};
  std::atomic<int> owner_result{-EINPROGRESS};
  bool queued_probe_was_processed = false;

  std::thread owner([&] {
    iree_status_t status = iree_io_uring_ring_enable(&ring_);
    enable_code.store(iree_status_code(status), std::memory_order_release);
    iree_status_free(status);
    SignalTestNotification(&owner_ready);
    AwaitTestNotification(&owner_wake);

    owner_result.store(
        iree_io_uring_ring_register(&ring_, IREE_IORING_REGISTER_PROBE,
                                    owner_probe.probe(), kProbeOpcodeCount),
        std::memory_order_release);
    queued_probe_was_processed = queued_probe.probe()->ops_len > 0;
    iree_io_uring_ring_end_polling(&ring_);
  });

  AwaitTestNotification(&owner_ready);
  if (enable_code.load(std::memory_order_acquire) != IREE_STATUS_OK) {
    SignalTestNotification(&owner_wake);
    owner.join();
    DeinitializeTestNotification(&owner_wake);
    DeinitializeTestNotification(&owner_ready);
    FAIL() << "failed to enable cross-thread ring";
  }
  std::thread caller([&] {
    queued_result.store(
        iree_io_uring_ring_register(&ring_, IREE_IORING_REGISTER_PROBE,
                                    queued_probe.probe(), kProbeOpcodeCount),
        std::memory_order_release);
  });

  owner.join();
  caller.join();
  EXPECT_EQ(queued_result.load(std::memory_order_acquire), 0);
  EXPECT_EQ(owner_result.load(std::memory_order_acquire), 0);
  EXPECT_TRUE(queued_probe_was_processed);

  DeinitializeTestNotification(&owner_wake);
  DeinitializeTestNotification(&owner_ready);
}

class IoUringRegistrationShutdownTest
    : public IoUringRegistrationTest,
      public ::testing::WithParamInterface<bool> {};

TEST_P(IoUringRegistrationShutdownTest, ReleasesPendingCaller) {
  iree_io_uring_rsrc_register_t table = {
      .nr = 8,
      .flags = IREE_IORING_RSRC_REGISTER_SPARSE,
  };
  ASSERT_EQ(iree_io_uring_ring_register(&ring_, IREE_IORING_REGISTER_BUFFERS2,
                                        &table, sizeof(table)),
            0);
  uint8_t backing[4096] = {};
  struct iovec buffer = {backing, sizeof(backing)};

  TestNotification owner_ready;
  TestNotification request_queued;
  TestNotification close_requested;
  InitializeTestNotification(&owner_ready);
  InitializeTestNotification(&request_queued);
  InitializeTestNotification(&close_requested);
  iree_io_uring_ring_set_registration_wake_callback(
      &ring_, {WakeRegistrationOwner, &request_queued});

  std::atomic<int> enable_code{IREE_STATUS_UNKNOWN};
  std::atomic<int> request_result{-EINPROGRESS};
  std::thread owner([&] {
    iree_status_t status = iree_io_uring_ring_enable(&ring_);
    enable_code.store(iree_status_code(status), std::memory_order_release);
    iree_status_free(status);
    SignalTestNotification(&owner_ready);
    AwaitTestNotification(&close_requested);
    iree_io_uring_ring_end_polling(&ring_);
  });

  AwaitTestNotification(&owner_ready);
  if (enable_code.load(std::memory_order_acquire) != IREE_STATUS_OK) {
    SignalTestNotification(&close_requested);
    owner.join();
    DeinitializeTestNotification(&close_requested);
    DeinitializeTestNotification(&request_queued);
    DeinitializeTestNotification(&owner_ready);
    FAIL() << "failed to enable cross-thread ring";
  }
  std::thread caller([&] {
    ProbeStorage probe_storage = {};
    int result = GetParam() ? iree_io_uring_registration_update_buffers(
                                  &ring_.registration, 0, &buffer, 1)
                            : iree_io_uring_ring_register(
                                  &ring_, IREE_IORING_REGISTER_PROBE,
                                  probe_storage.probe(), kProbeOpcodeCount);
    request_result.store(result, std::memory_order_release);
  });

  AwaitTestNotification(&request_queued);
  SignalTestNotification(&close_requested);
  owner.join();
  caller.join();
  EXPECT_EQ(request_result.load(std::memory_order_acquire), -ESHUTDOWN);

  EXPECT_EQ(iree_io_uring_registration_update_buffers(&ring_.registration, 0,
                                                      &buffer, 1),
            -ESHUTDOWN);

  DeinitializeTestNotification(&close_requested);
  DeinitializeTestNotification(&request_queued);
  DeinitializeTestNotification(&owner_ready);
}

INSTANTIATE_TEST_SUITE_P(Operation, IoUringRegistrationShutdownTest,
                         ::testing::Bool(),
                         [](const ::testing::TestParamInfo<bool>& info) {
                           return info.param ? "PreparedPins" : "Probe";
                         });

TEST_F(IoUringRegistrationTest, PinFailureDoesNotRequestOwnerProgress) {
  TestNotification owner_ready;
  TestNotification close_requested;
  InitializeTestNotification(&owner_ready);
  InitializeTestNotification(&close_requested);
  std::atomic<int> wake_count{0};
  iree_io_uring_ring_set_registration_wake_callback(
      &ring_, {+[](void* value) {
                 static_cast<std::atomic<int>*>(value)->fetch_add(1);
               },
               &wake_count});
  std::atomic<int> enable_code{IREE_STATUS_UNKNOWN};
  std::thread owner([&] {
    iree_status_t status = iree_io_uring_ring_enable(&ring_);
    enable_code.store(iree_status_code(status), std::memory_order_release);
    iree_status_free(status);
    SignalTestNotification(&owner_ready);
    AwaitTestNotification(&close_requested);
    iree_io_uring_ring_end_polling(&ring_);
  });

  AwaitTestNotification(&owner_ready);
  int result = -EINPROGRESS;
  const size_t page_size = static_cast<size_t>(sysconf(_SC_PAGESIZE));
  void* memory =
      mmap(nullptr, page_size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (enable_code.load(std::memory_order_acquire) == IREE_STATUS_OK &&
      memory != MAP_FAILED) {
    // The caller cannot pin an inaccessible mapping. The owner deliberately
    // does not drain registrations: failure must precede any owner handoff.
    struct iovec buffer = {memory, page_size};
    result = iree_io_uring_registration_update_buffers(&ring_.registration, 0,
                                                       &buffer, 1);
  }
  SignalTestNotification(&close_requested);
  owner.join();
  EXPECT_EQ(enable_code.load(std::memory_order_acquire), IREE_STATUS_OK);
  EXPECT_NE(memory, MAP_FAILED);
  EXPECT_EQ(result, -EFAULT);
  EXPECT_EQ(wake_count.load(), 0);
  if (memory != MAP_FAILED) {
    EXPECT_EQ(munmap(memory, page_size), 0);
  }
  DeinitializeTestNotification(&close_requested);
  DeinitializeTestNotification(&owner_ready);
}

}  // namespace

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
#include <vector>

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

// Holds a real SINGLE_ISSUER owner until the test explicitly allows progress.
// Teardown always joins the owner, including after a fatal test assertion.
class HeldRegistrationOwner {
 public:
  explicit HeldRegistrationOwner(iree_io_uring_ring_t* ring,
                                 bool drain_before_shutdown = true)
      : ring_(ring) {
    InitializeTestNotification(&ready_);
    InitializeTestNotification(&release_);
    iree_io_uring_ring_set_registration_wake_callback(
        ring_, {+[](void* user_data) {
                  static_cast<HeldRegistrationOwner*>(user_data)
                      ->wake_count_.fetch_add(1);
                },
                this});
    thread_ = std::thread([this, drain_before_shutdown] {
      iree_status_t status = iree_io_uring_ring_enable(ring_);
      enable_code_ = iree_status_code(status);
      iree_status_free(status);
      SignalTestNotification(&ready_);
      AwaitTestNotification(&release_);
      if (drain_before_shutdown) {
        iree_io_uring_ring_drain_registration_requests(ring_);
      }
      iree_io_uring_ring_end_polling(ring_);
    });
    AwaitTestNotification(&ready_);
  }

  ~HeldRegistrationOwner() {
    Release();
    DeinitializeTestNotification(&release_);
    DeinitializeTestNotification(&ready_);
  }

  iree_status_code_t enable_code() const { return enable_code_; }
  int wake_count() const { return wake_count_.load(); }

  void Release() {
    if (thread_.joinable()) {
      SignalTestNotification(&release_);
      thread_.join();
    }
  }

 private:
  // Borrowed native ring, which outlives this owner.
  iree_io_uring_ring_t* ring_;
  // Publishes the owner binding result to the test caller.
  TestNotification ready_;
  // Explicit permission to drain or shut down; no wall-clock readiness.
  TestNotification release_;
  // Native owner, joined before either notification is destroyed.
  std::thread thread_;
  // Published before ready_ and immutable afterward.
  iree_status_code_t enable_code_ = IREE_STATUS_UNKNOWN;
  // Records actual empty-to-pending wake requests without allowing progress.
  std::atomic<int> wake_count_{0};
};

TEST_F(IoUringRegistrationTest, ExecutesDirectlyBeforeOwnerBinding) {
  ProbeStorage probe_storage = {};
  EXPECT_EQ(
      iree_io_uring_ring_register(&ring_, IREE_IORING_REGISTER_PROBE,
                                  probe_storage.probe(), kProbeOpcodeCount),
      0);
  EXPECT_GT(probe_storage.probe()->ops_len, 0);
}

TEST_F(IoUringRegistrationTest, AsyncInlineCompletionCanFreeRequest) {
  struct Capture {
    // Native request and its argument share the caller-owned lifetime.
    iree_io_uring_registration_request_t request;
    // Actual kernel probe storage, freed by terminal delivery.
    ProbeStorage probe;
    // Caller result remains after this capture is destroyed.
    int* result;
  };
  int result = -EINPROGRESS;
  auto* capture = new Capture{};
  capture->result = &result;
  capture->request = {
      .opcode = IREE_IORING_REGISTER_PROBE,
      .arg = capture->probe.probe(),
      .argument_count = kProbeOpcodeCount,
      .completion = {+[](void* user_data, int result) {
                       auto* capture = static_cast<Capture*>(user_data);
                       *capture->result = result;
                       EXPECT_GT(capture->probe.probe()->ops_len, 0);
                       delete capture;
                     },
                     capture},
  };
  iree_io_uring_registration_submit(&ring_.registration, &capture->request);
  EXPECT_EQ(result, 0);
}

TEST_F(IoUringRegistrationTest,
       AsyncReturnsBeforeOwnerProgressAndSupportsReentrantReuse) {
  struct Capture {
    // Request can be resubmitted or freed from its terminal callback.
    iree_io_uring_registration_request_t request;
    // The kernel writes a probe before each callback.
    ProbeStorage probe;
    // Borrowed registration owner, live until every callback completes.
    iree_io_uring_registration_t* registration;
    // Owner-thread callback ordering, inspected after the owner joins.
    std::vector<int>* order;
    // Owner-thread callback depth, proving asynchronous chains do not recurse.
    int* depth;
    // Identifies this request's next completed invocation.
    int ordinal;
  };
  std::vector<int> order;
  int depth = 0;
  auto completion = +[](void* user_data, int result) {
    auto* capture = static_cast<Capture*>(user_data);
    int* depth = capture->depth;
    EXPECT_EQ(++*depth, 1);
    EXPECT_EQ(result, 0);
    EXPECT_GT(capture->probe.probe()->ops_len, 0);
    capture->order->push_back(capture->ordinal);
    if (capture->ordinal == 1) {
      capture->ordinal = 3;
      capture->probe = {};
      iree_io_uring_registration_submit(capture->registration,
                                        &capture->request);
      // The resubmitted request owns this capture until its next callback.
    } else {
      delete capture;
    }
    --*depth;
  };

  HeldRegistrationOwner owner(&ring_);
  ASSERT_EQ(owner.enable_code(), IREE_STATUS_OK);
  for (int ordinal = 1; ordinal <= 2; ++ordinal) {
    auto* capture = new Capture{};
    capture->registration = &ring_.registration;
    capture->order = &order;
    capture->depth = &depth;
    capture->ordinal = ordinal;
    capture->request = {
        .opcode = IREE_IORING_REGISTER_PROBE,
        .arg = capture->probe.probe(),
        .argument_count = kProbeOpcodeCount,
        .completion = {completion, capture},
    };
    iree_io_uring_registration_submit(&ring_.registration, &capture->request);
  }
  // Both submissions returned while the actual execution owner is held.
  EXPECT_TRUE(order.empty());
  EXPECT_TRUE(iree_io_uring_registration_has_pending(&ring_.registration));
  EXPECT_EQ(owner.wake_count(), 1);
  owner.Release();
  EXPECT_EQ(order, (std::vector<int>{1, 2, 3}));
  EXPECT_EQ(depth, 0);
  EXPECT_FALSE(iree_io_uring_registration_has_pending(&ring_.registration));
}

TEST_F(IoUringRegistrationTest, AsyncCallbackCanExecuteSynchronously) {
  struct Capture {
    // Native request and its argument remain live until the owner joins.
    iree_io_uring_registration_request_t request;
    // Output from the actual kernel probe.
    ProbeStorage probe;
    // Borrowed owner used by the nested synchronous call.
    iree_io_uring_registration_t* registration;
    // Callback ordering, inspected after the owner joins.
    std::vector<int>* order;
    // Identifies which callback issues the synchronous successor.
    int ordinal;
  };
  Capture captures[2] = {};
  std::vector<int> order;
  HeldRegistrationOwner owner(&ring_);
  ASSERT_EQ(owner.enable_code(), IREE_STATUS_OK);
  for (int i = 0; i < 2; ++i) {
    Capture* capture = &captures[i];
    capture->registration = &ring_.registration;
    capture->order = &order;
    capture->ordinal = i + 1;
    capture->request = {
        .opcode = IREE_IORING_REGISTER_PROBE,
        .arg = capture->probe.probe(),
        .argument_count = kProbeOpcodeCount,
        .completion = {+[](void* user_data, int result) {
                         auto* capture = static_cast<Capture*>(user_data);
                         EXPECT_EQ(result, 0);
                         capture->order->push_back(capture->ordinal);
                         if (capture->ordinal == 1) {
                           ProbeStorage nested_probe = {};
                           EXPECT_EQ(
                               iree_io_uring_registration_execute(
                                   capture->registration,
                                   IREE_IORING_REGISTER_PROBE,
                                   nested_probe.probe(), kProbeOpcodeCount),
                               0);
                           EXPECT_GT(nested_probe.probe()->ops_len, 0);
                           capture->order->push_back(3);
                         }
                       },
                       capture},
    };
    iree_io_uring_registration_submit(&ring_.registration, &capture->request);
  }
  owner.Release();
  EXPECT_EQ(order, (std::vector<int>{1, 2, 3}));
}

TEST_F(IoUringRegistrationTest, AsyncShutdownAllowsCallbackReentry) {
  struct Capture {
    // Native request rejected when its held owner closes.
    iree_io_uring_registration_request_t request;
    // Native argument storage must stay live until rejection.
    ProbeStorage probe;
    // Borrowed dispatch state used for reentrant shutdown and submission.
    iree_io_uring_registration_t* registration;
    // Callback results inspected only after joining the owner.
    std::vector<int>* results;
  };
  std::vector<int> results;
  HeldRegistrationOwner owner(&ring_, /*drain_before_shutdown=*/false);
  ASSERT_EQ(owner.enable_code(), IREE_STATUS_OK);
  for (int i = 0; i < 2; ++i) {
    auto* capture = new Capture{};
    capture->registration = &ring_.registration;
    capture->results = &results;
    capture->request = {
        .opcode = IREE_IORING_REGISTER_PROBE,
        .arg = capture->probe.probe(),
        .argument_count = kProbeOpcodeCount,
        .completion = {+[](void* user_data, int result) {
                         auto* capture = static_cast<Capture*>(user_data);
                         capture->results->push_back(result);
                         EXPECT_EQ(capture->probe.probe()->ops_len, 0);
                         iree_io_uring_registration_end_polling(
                             capture->registration);
                         capture->request.completion.fn =
                             +[](void* user_data, int result) {
                               auto* capture = static_cast<Capture*>(user_data);
                               capture->results->push_back(result);
                               delete capture;
                             };
                         iree_io_uring_registration_submit(
                             capture->registration, &capture->request);
                       },
                       capture},
    };
    iree_io_uring_registration_submit(&ring_.registration, &capture->request);
  }
  EXPECT_TRUE(results.empty());
  owner.Release();
  EXPECT_EQ(results,
            (std::vector<int>{-ESHUTDOWN, -ESHUTDOWN, -ESHUTDOWN, -ESHUTDOWN}));
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

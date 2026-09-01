// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/platform/iocp/proactor.h"

#include <atomic>
#include <cstring>
#include <thread>

#include "iree/async/event.h"
#include "iree/async/notification.h"
#include "iree/async/operations/semaphore.h"
#include "iree/async/semaphore.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

struct CompletionState {
  int call_count = 0;
  iree_status_code_t status_code = IREE_STATUS_UNKNOWN;
};

static void RecordCompletion(void* user_data, iree_async_operation_t* operation,
                             iree_status_t status,
                             iree_async_completion_flags_t flags) {
  (void)operation;
  (void)flags;
  auto* state = static_cast<CompletionState*>(user_data);
  ++state->call_count;
  state->status_code = iree_status_code(status);
  iree_status_free(status);
}

// Replaces the component's owned handle with NULL while preserving the same
// completion-port object through a duplicate. Synthetic posts then fail
// through the real PostQueuedCompletionStatus call. Restoring the duplicate
// returns ownership of a valid handle to the component.
static iree_status_t DisableCompletionPosting(
    iree_async_proactor_iocp_t* proactor, HANDLE* out_preserved_port) {
  *out_preserved_port = NULL;
  HANDLE completion_port = (HANDLE)proactor->completion_port.handle;
  HANDLE preserved_port = NULL;
  if (!DuplicateHandle(GetCurrentProcess(), completion_port,
                       GetCurrentProcess(), &preserved_port, 0, FALSE,
                       DUPLICATE_SAME_ACCESS)) {
    DWORD error_code = GetLastError();
    return iree_make_status(iree_status_code_from_win32_error(error_code),
                            "DuplicateHandle failed (error %lu)",
                            (unsigned long)error_code);
  }
  if (!CloseHandle(completion_port)) {
    DWORD error_code = GetLastError();
    iree_status_t status = iree_make_status(
        iree_status_code_from_win32_error(error_code),
        "CloseHandle failed while disabling completion posts (error %lu)",
        (unsigned long)error_code);
    if (!CloseHandle(preserved_port)) {
      DWORD close_error_code = GetLastError();
      status = iree_status_join(
          status,
          iree_make_status(iree_status_code_from_win32_error(close_error_code),
                           "CloseHandle failed while unwinding duplicate "
                           "completion port (error %lu)",
                           (unsigned long)close_error_code));
    }
    return status;
  }
  proactor->completion_port.handle = 0;
  *out_preserved_port = preserved_port;
  return iree_ok_status();
}

static void RestoreCompletionPosting(iree_async_proactor_iocp_t* proactor,
                                     HANDLE preserved_port) {
  proactor->completion_port.handle = (uintptr_t)preserved_port;
}

static void WaitForFallbackCompletionReady(iree_async_iocp_carrier_t* carrier) {
  while (iree_atomic_load(&carrier->fallback_completion_state,
                          iree_memory_order_acquire) !=
         IREE_ASYNC_IOCP_FALLBACK_COMPLETION_READY) {
    std::this_thread::yield();
  }
}

class IocpProactorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_async_proactor_options_t options =
        iree_async_proactor_options_default();
    IREE_ASSERT_OK(iree_async_proactor_create_iocp(
        options, iree_allocator_system(), &proactor_));
  }

  void TearDown() override { iree_async_proactor_release(proactor_); }

  iree_async_proactor_iocp_t* iocp() {
    return iree_async_proactor_iocp_cast(proactor_);
  }

  iree_async_proactor_t* proactor_ = nullptr;
};

TEST_F(IocpProactorTest, DirectPostFailureReleasesRetainedResources) {
  iree_async_notification_t* notification = nullptr;
  IREE_ASSERT_OK(iree_async_notification_create(
      proactor_, IREE_ASYNC_NOTIFICATION_FLAG_NONE, &notification));
  ASSERT_EQ(iree_atomic_ref_count_load(&notification->ref_count), 1);

  CompletionState completion;
  iree_async_notification_signal_operation_t signal_operation;
  std::memset(&signal_operation, 0, sizeof(signal_operation));
  signal_operation.base.type = IREE_ASYNC_OPERATION_TYPE_NOTIFICATION_SIGNAL;
  signal_operation.base.completion_fn = RecordCompletion;
  signal_operation.base.user_data = &completion;
  signal_operation.notification = notification;
  signal_operation.wake_count = 1;

  uint32_t initial_epoch = iree_async_notification_query_epoch(notification);
  HANDLE preserved_port = NULL;
  IREE_ASSERT_OK(DisableCompletionPosting(iocp(), &preserved_port));
  iree_status_t submit_status =
      iree_async_proactor_submit_one(proactor_, &signal_operation.base);
  RestoreCompletionPosting(iocp(), preserved_port);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, submit_status);

  EXPECT_NE(iree_async_notification_query_epoch(notification), initial_epoch);
  EXPECT_EQ(iree_atomic_ref_count_load(&notification->ref_count), 1);
  EXPECT_EQ(completion.call_count, 0);
  EXPECT_EQ(signal_operation.base.next, nullptr);

  iree_async_notification_release(notification);
}

TEST_F(IocpProactorTest, RichStatusPostFailureConsumesStashedStatus) {
  iree_async_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_async_semaphore_create(
      proactor_, /*initial_value=*/10,
      IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY, iree_allocator_system(),
      &semaphore));
  iree_async_semaphore_t* semaphore_storage = semaphore;
  uint64_t non_monotonic_value = 10;

  CompletionState completion;
  iree_async_semaphore_signal_operation_t signal_operation;
  std::memset(&signal_operation, 0, sizeof(signal_operation));
  signal_operation.base.type = IREE_ASYNC_OPERATION_TYPE_SEMAPHORE_SIGNAL;
  signal_operation.base.completion_fn = RecordCompletion;
  signal_operation.base.user_data = &completion;
  signal_operation.semaphores = &semaphore_storage;
  signal_operation.values = &non_monotonic_value;
  signal_operation.count = 1;

  HANDLE preserved_port = NULL;
  IREE_ASSERT_OK(DisableCompletionPosting(iocp(), &preserved_port));
  iree_status_t submit_status =
      iree_async_proactor_submit_one(proactor_, &signal_operation.base);
  RestoreCompletionPosting(iocp(), preserved_port);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, submit_status);

  EXPECT_EQ(iree_async_semaphore_query(semaphore), 10u);
  EXPECT_EQ(completion.call_count, 0);
  EXPECT_EQ(signal_operation.base.next, nullptr);

  iree_async_semaphore_release(semaphore);
}

TEST_F(IocpProactorTest, FailedWakePersistsUntilPoll) {
  HANDLE preserved_port = NULL;
  IREE_ASSERT_OK(DisableCompletionPosting(iocp(), &preserved_port));
  iree_async_proactor_wake(proactor_);

  EXPECT_EQ(iree_atomic_load(&iocp()->completion_port.fallback_wake_pending,
                             iree_memory_order_acquire),
            1);
  RestoreCompletionPosting(iocp(), preserved_port);
  IREE_ASSERT_OK(iree_async_proactor_poll(proactor_, iree_immediate_timeout(),
                                          /*out_completed_count=*/nullptr));
  EXPECT_EQ(iree_atomic_load(&iocp()->completion_port.fallback_wake_pending,
                             iree_memory_order_acquire),
            0);
}

TEST_F(IocpProactorTest, FallbackApcInterruptsBlockedPoll) {
  std::atomic<iree_status_code_t> poll_status{IREE_STATUS_UNKNOWN};
  std::thread poll_thread([&]() {
    iree_status_t status = iree_async_proactor_poll(
        proactor_, iree_infinite_timeout(), /*out_completed_count=*/nullptr);
    poll_status.store(iree_status_code(status), std::memory_order_release);
    iree_status_free(status);
  });

  // Waiting for the durable owner handle ensures wake() must use the APC path
  // instead of relying on a pre-poll pending flag.
  while (iree_atomic_load(&iocp()->completion_port.poll_thread_handle,
                          iree_memory_order_acquire) == 0) {
    std::this_thread::yield();
  }
  iree_async_iocp_completion_port_request_fallback_wake(
      &iocp()->completion_port);
  poll_thread.join();

  EXPECT_EQ(poll_status.load(std::memory_order_acquire), IREE_STATUS_OK);
  EXPECT_EQ(iree_atomic_load(&iocp()->completion_port.fallback_wake_pending,
                             iree_memory_order_acquire),
            0);
}

TEST(IocpLegacyEventWaitTest, FailedCallbackPostDispatchesFromPoll) {
  iree_async_proactor_options_t options = iree_async_proactor_options_default();
  options.allowed_capabilities &=
      ~IREE_ASYNC_PROACTOR_CAPABILITY_WAIT_COMPLETION_PACKET;
  iree_async_proactor_t* proactor = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_create_iocp(
      options, iree_allocator_system(), &proactor));
  iree_async_proactor_iocp_t* iocp = iree_async_proactor_iocp_cast(proactor);

  iree_async_event_t* event = nullptr;
  IREE_ASSERT_OK(iree_async_event_create(proactor, &event));
  CompletionState completion;
  iree_async_event_wait_operation_t wait_operation;
  std::memset(&wait_operation, 0, sizeof(wait_operation));
  wait_operation.base.type = IREE_ASYNC_OPERATION_TYPE_EVENT_WAIT;
  wait_operation.base.completion_fn = RecordCompletion;
  wait_operation.base.user_data = &completion;
  wait_operation.event = event;

  IREE_ASSERT_OK(
      iree_async_proactor_submit_one(proactor, &wait_operation.base));
  IREE_ASSERT_OK(iree_async_proactor_poll(proactor, iree_immediate_timeout(),
                                          /*out_completed_count=*/nullptr));
  ASSERT_EQ(completion.call_count, 0);
  ASSERT_EQ(iree_atomic_load(&iocp->outstanding_carrier_count,
                             iree_memory_order_relaxed),
            1);

  HANDLE preserved_port = NULL;
  IREE_ASSERT_OK(DisableCompletionPosting(iocp, &preserved_port));
  IREE_ASSERT_OK(iree_async_event_set(event));
  iree_async_iocp_carrier_t* carrier = iocp->active_carriers;
  ASSERT_NE(carrier, nullptr);
  WaitForFallbackCompletionReady(carrier);
  RestoreCompletionPosting(iocp, preserved_port);

  iree_host_size_t completed_count = 0;
  IREE_ASSERT_OK(iree_async_proactor_poll(proactor, iree_infinite_timeout(),
                                          &completed_count));

  EXPECT_EQ(completed_count, 1);
  EXPECT_EQ(completion.call_count, 1);
  EXPECT_EQ(completion.status_code, IREE_STATUS_OK);
  EXPECT_EQ(iree_atomic_load(&iocp->outstanding_carrier_count,
                             iree_memory_order_relaxed),
            0);
  EXPECT_EQ(iree_atomic_ref_count_load(&event->ref_count), 1);

  iree_async_event_release(event);
  iree_async_proactor_release(proactor);
}

TEST(IocpLegacyEventWaitTest, SuccessfulCallbackPostReleasesRegistration) {
  iree_async_proactor_options_t options = iree_async_proactor_options_default();
  options.allowed_capabilities &=
      ~IREE_ASYNC_PROACTOR_CAPABILITY_WAIT_COMPLETION_PACKET;
  iree_async_proactor_t* proactor = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_create_iocp(
      options, iree_allocator_system(), &proactor));
  iree_async_proactor_iocp_t* iocp = iree_async_proactor_iocp_cast(proactor);

  iree_async_event_t* event = nullptr;
  IREE_ASSERT_OK(iree_async_event_create(proactor, &event));
  CompletionState completion;
  iree_async_event_wait_operation_t wait_operation;
  std::memset(&wait_operation, 0, sizeof(wait_operation));
  wait_operation.base.type = IREE_ASYNC_OPERATION_TYPE_EVENT_WAIT;
  wait_operation.base.completion_fn = RecordCompletion;
  wait_operation.base.user_data = &completion;
  wait_operation.event = event;

  IREE_ASSERT_OK(
      iree_async_proactor_submit_one(proactor, &wait_operation.base));
  IREE_ASSERT_OK(iree_async_proactor_poll(proactor, iree_immediate_timeout(),
                                          /*out_completed_count=*/nullptr));
  ASSERT_EQ(iree_atomic_load(&iocp->outstanding_carrier_count,
                             iree_memory_order_relaxed),
            1);

  IREE_ASSERT_OK(iree_async_event_set(event));
  iree_host_size_t completed_count = 0;
  while (completion.call_count == 0) {
    iree_host_size_t poll_completed_count = 0;
    IREE_ASSERT_OK(iree_async_proactor_poll(proactor, iree_infinite_timeout(),
                                            &poll_completed_count));
    completed_count += poll_completed_count;
  }

  EXPECT_EQ(completed_count, 1);
  EXPECT_EQ(completion.call_count, 1);
  EXPECT_EQ(completion.status_code, IREE_STATUS_OK);
  EXPECT_EQ(iree_atomic_load(&iocp->outstanding_carrier_count,
                             iree_memory_order_relaxed),
            0);
  EXPECT_EQ(iree_atomic_ref_count_load(&event->ref_count), 1);

  iree_async_event_release(event);
  iree_async_proactor_release(proactor);
}

TEST(IocpLegacyEventWaitTest, CancelledFallbackDispatchesExactlyOnce) {
  iree_async_proactor_options_t options = iree_async_proactor_options_default();
  options.allowed_capabilities &=
      ~IREE_ASYNC_PROACTOR_CAPABILITY_WAIT_COMPLETION_PACKET;
  iree_async_proactor_t* proactor = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_create_iocp(
      options, iree_allocator_system(), &proactor));
  iree_async_proactor_iocp_t* iocp = iree_async_proactor_iocp_cast(proactor);

  iree_async_event_t* event = nullptr;
  IREE_ASSERT_OK(iree_async_event_create(proactor, &event));
  CompletionState completion;
  iree_async_event_wait_operation_t wait_operation;
  std::memset(&wait_operation, 0, sizeof(wait_operation));
  wait_operation.base.type = IREE_ASYNC_OPERATION_TYPE_EVENT_WAIT;
  wait_operation.base.completion_fn = RecordCompletion;
  wait_operation.base.user_data = &completion;
  wait_operation.event = event;

  IREE_ASSERT_OK(
      iree_async_proactor_submit_one(proactor, &wait_operation.base));
  IREE_ASSERT_OK(iree_async_proactor_poll(proactor, iree_immediate_timeout(),
                                          /*out_completed_count=*/nullptr));

  HANDLE preserved_port = NULL;
  IREE_ASSERT_OK(DisableCompletionPosting(iocp, &preserved_port));
  IREE_ASSERT_OK(iree_async_event_set(event));
  iree_async_iocp_carrier_t* carrier = iocp->active_carriers;
  ASSERT_NE(carrier, nullptr);
  WaitForFallbackCompletionReady(carrier);
  IREE_ASSERT_OK(iree_async_proactor_cancel(proactor, &wait_operation.base));
  RestoreCompletionPosting(iocp, preserved_port);

  iree_host_size_t completed_count = 0;
  IREE_ASSERT_OK(iree_async_proactor_poll(proactor, iree_infinite_timeout(),
                                          &completed_count));

  EXPECT_EQ(completed_count, 1);
  EXPECT_EQ(completion.call_count, 1);
  EXPECT_EQ(completion.status_code, IREE_STATUS_CANCELLED);
  EXPECT_EQ(iree_atomic_load(&iocp->outstanding_carrier_count,
                             iree_memory_order_relaxed),
            0);
  EXPECT_EQ(iree_atomic_ref_count_load(&event->ref_count), 1);

  iree_async_event_release(event);
  iree_async_proactor_release(proactor);
}

}  // namespace

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
#include "iree/async/file.h"
#include "iree/async/notification.h"
#include "iree/async/operations/file.h"
#include "iree/async/operations/message.h"
#include "iree/async/operations/net.h"
#include "iree/async/operations/scheduling.h"
#include "iree/async/operations/semaphore.h"
#include "iree/async/semaphore.h"
#include "iree/async/socket.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

struct CompletionState {
  int call_count = 0;
  iree_status_code_t status_code = IREE_STATUS_UNKNOWN;
};

struct MessageState {
  int call_count = 0;
  uint64_t value = 0;
};

struct ControlledAllocator {
  iree_allocator_t delegate = iree_allocator_system();
  int allocations_before_failure = -1;

  static iree_status_t Control(void* self, iree_allocator_command_t command,
                               const void* params, void** inout_ptr) {
    auto* allocator = static_cast<ControlledAllocator*>(self);
    const bool is_allocation = command == IREE_ALLOCATOR_COMMAND_MALLOC ||
                               command == IREE_ALLOCATOR_COMMAND_CALLOC ||
                               command == IREE_ALLOCATOR_COMMAND_REALLOC;
    if (is_allocation && allocator->allocations_before_failure == 0) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "intentional allocation failure");
    }
    if (is_allocation && allocator->allocations_before_failure > 0) {
      --allocator->allocations_before_failure;
    }
    return allocator->delegate.ctl(allocator->delegate.self, command, params,
                                   inout_ptr);
  }

  iree_allocator_t value() { return {.self = this, .ctl = Control}; }
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

static void RecordMessage(iree_async_proactor_t* proactor,
                          uint64_t message_data, void* user_data) {
  (void)proactor;
  auto* state = static_cast<MessageState*>(user_data);
  ++state->call_count;
  state->value = message_data;
}

template <typename T>
static void InitializeOperation(T* operation, iree_async_operation_type_t type,
                                CompletionState* completion) {
  iree_async_operation_zero(&operation->base, sizeof(*operation));
  iree_async_operation_initialize(
      &operation->base, type, IREE_ASYNC_OPERATION_FLAG_NONE,
      completion ? RecordCompletion : nullptr, completion);
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

TEST_F(IocpProactorTest, DirectPostFailureCompletesAcceptedSignal) {
  iree_async_notification_t* notification = nullptr;
  IREE_ASSERT_OK(iree_async_notification_create(
      proactor_, IREE_ASYNC_NOTIFICATION_FLAG_NONE, &notification));
  ASSERT_EQ(iree_atomic_ref_count_load(&notification->ref_count), 1);

  CompletionState completion;
  iree_async_notification_signal_operation_t signal_operation;
  InitializeOperation(&signal_operation,
                      IREE_ASYNC_OPERATION_TYPE_NOTIFICATION_SIGNAL,
                      &completion);
  signal_operation.notification = notification;
  signal_operation.wake_count = 1;

  uint32_t initial_epoch = iree_async_notification_query_epoch(notification);
  HANDLE preserved_port = NULL;
  IREE_ASSERT_OK(DisableCompletionPosting(iocp(), &preserved_port));
  iree_status_t submit_status =
      iree_async_proactor_submit_one(proactor_, &signal_operation.base);
  RestoreCompletionPosting(iocp(), preserved_port);
  IREE_EXPECT_OK(submit_status);

  EXPECT_NE(iree_async_notification_query_epoch(notification), initial_epoch);
  EXPECT_EQ(iree_atomic_ref_count_load(&notification->ref_count), 2);
  EXPECT_EQ(completion.call_count, 0);
  EXPECT_EQ(signal_operation.base.next, nullptr);
  EXPECT_EQ(iree_atomic_load(&iocp()->outstanding_carrier_count,
                             iree_memory_order_relaxed),
            1);

  iree_host_size_t completed_count = 0;
  IREE_ASSERT_OK(iree_async_proactor_poll(proactor_, iree_infinite_timeout(),
                                          &completed_count));
  EXPECT_EQ(completed_count, 1);
  EXPECT_EQ(completion.call_count, 1);
  EXPECT_EQ(completion.status_code, IREE_STATUS_OK);
  EXPECT_EQ(iree_atomic_ref_count_load(&notification->ref_count), 1);
  EXPECT_EQ(iree_atomic_load(&iocp()->outstanding_carrier_count,
                             iree_memory_order_relaxed),
            0);

  iree_async_notification_release(notification);
}

TEST_F(IocpProactorTest, DirectPostFailurePreservesRichStatus) {
  iree_async_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_async_semaphore_create(
      proactor_, /*initial_value=*/10,
      IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY, iree_allocator_system(),
      &semaphore));
  iree_async_semaphore_t* semaphore_storage = semaphore;
  uint64_t non_monotonic_value = 10;

  CompletionState completion;
  iree_async_semaphore_signal_operation_t signal_operation;
  InitializeOperation(&signal_operation,
                      IREE_ASYNC_OPERATION_TYPE_SEMAPHORE_SIGNAL, &completion);
  signal_operation.semaphores = &semaphore_storage;
  signal_operation.values = &non_monotonic_value;
  signal_operation.count = 1;

  HANDLE preserved_port = NULL;
  IREE_ASSERT_OK(DisableCompletionPosting(iocp(), &preserved_port));
  iree_status_t submit_status =
      iree_async_proactor_submit_one(proactor_, &signal_operation.base);
  RestoreCompletionPosting(iocp(), preserved_port);
  IREE_EXPECT_OK(submit_status);

  EXPECT_EQ(iree_async_semaphore_query(semaphore), 10u);
  EXPECT_EQ(completion.call_count, 0);
  EXPECT_EQ(signal_operation.base.next, nullptr);

  iree_host_size_t completed_count = 0;
  IREE_ASSERT_OK(iree_async_proactor_poll(proactor_, iree_infinite_timeout(),
                                          &completed_count));
  EXPECT_EQ(completed_count, 1);
  EXPECT_EQ(completion.call_count, 1);
  EXPECT_EQ(completion.status_code, IREE_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(iree_atomic_load(&iocp()->outstanding_carrier_count,
                             iree_memory_order_relaxed),
            0);

  iree_async_semaphore_release(semaphore);
}

TEST_F(IocpProactorTest, DirectPostFailureCompletesAcceptedSocketClose) {
  iree_async_socket_t* socket = nullptr;
  IREE_ASSERT_OK(iree_async_socket_create(proactor_, IREE_ASYNC_SOCKET_TYPE_TCP,
                                          IREE_ASYNC_SOCKET_OPTION_NONE,
                                          &socket));
  iree_async_socket_retain(socket);

  CompletionState completion;
  iree_async_socket_close_operation_t close_operation;
  InitializeOperation(&close_operation, IREE_ASYNC_OPERATION_TYPE_SOCKET_CLOSE,
                      &completion);
  close_operation.socket = socket;

  HANDLE preserved_port = NULL;
  IREE_ASSERT_OK(DisableCompletionPosting(iocp(), &preserved_port));
  iree_status_t submit_status =
      iree_async_proactor_submit_one(proactor_, &close_operation.base);
  RestoreCompletionPosting(iocp(), preserved_port);
  IREE_EXPECT_OK(submit_status);

  EXPECT_EQ(socket->primitive.type, IREE_ASYNC_PRIMITIVE_TYPE_NONE);
  EXPECT_EQ(socket->state, IREE_ASYNC_SOCKET_STATE_CLOSED);
  EXPECT_EQ(iree_atomic_ref_count_load(&socket->ref_count), 2);
  EXPECT_EQ(completion.call_count, 0);

  iree_host_size_t completed_count = 0;
  IREE_ASSERT_OK(iree_async_proactor_poll(proactor_, iree_infinite_timeout(),
                                          &completed_count));
  EXPECT_EQ(completed_count, 1);
  EXPECT_EQ(completion.call_count, 1);
  EXPECT_EQ(completion.status_code, IREE_STATUS_OK);
  EXPECT_EQ(iree_atomic_ref_count_load(&socket->ref_count), 1);
  EXPECT_EQ(iree_atomic_load(&iocp()->outstanding_carrier_count,
                             iree_memory_order_relaxed),
            0);

  iree_async_socket_release(socket);
}

TEST_F(IocpProactorTest, DirectPostFailureCompletesPlatformSetupFailure) {
  CompletionState completion;
  iree_async_file_open_operation_t open_operation;
  InitializeOperation(&open_operation, IREE_ASYNC_OPERATION_TYPE_FILE_OPEN,
                      &completion);
  open_operation.path = "iree-iocp-invalid-<>.bin";
  open_operation.open_flags = IREE_ASYNC_FILE_OPEN_FLAG_READ;

  HANDLE preserved_port = NULL;
  IREE_ASSERT_OK(DisableCompletionPosting(iocp(), &preserved_port));
  iree_status_t submit_status =
      iree_async_proactor_submit_one(proactor_, &open_operation.base);
  RestoreCompletionPosting(iocp(), preserved_port);
  IREE_EXPECT_OK(submit_status);

  EXPECT_EQ(open_operation.opened_file, nullptr);
  EXPECT_EQ(completion.call_count, 0);
  EXPECT_EQ(iree_atomic_load(&iocp()->outstanding_carrier_count,
                             iree_memory_order_relaxed),
            1);

  iree_host_size_t completed_count = 0;
  IREE_ASSERT_OK(iree_async_proactor_poll(proactor_, iree_infinite_timeout(),
                                          &completed_count));
  EXPECT_EQ(completed_count, 1);
  EXPECT_EQ(completion.call_count, 1);
  EXPECT_NE(completion.status_code, IREE_STATUS_OK);
  EXPECT_EQ(iree_atomic_load(&iocp()->outstanding_carrier_count,
                             iree_memory_order_relaxed),
            0);
}

TEST_F(IocpProactorTest, MalformedTailDoesNotConsumeSocketClose) {
  iree_async_socket_t* socket = nullptr;
  IREE_ASSERT_OK(iree_async_socket_create(proactor_, IREE_ASYNC_SOCKET_TYPE_TCP,
                                          IREE_ASYNC_SOCKET_OPTION_NONE,
                                          &socket));
  const SOCKET native_socket = (SOCKET)socket->primitive.value.win32_handle;

  CompletionState completion;
  iree_async_socket_close_operation_t close_operation;
  InitializeOperation(&close_operation, IREE_ASYNC_OPERATION_TYPE_SOCKET_CLOSE,
                      &completion);
  close_operation.socket = socket;

  iree_async_nop_operation_t malformed_operation;
  InitializeOperation(&malformed_operation, IREE_ASYNC_OPERATION_TYPE_NOP,
                      /*completion=*/nullptr);
  iree_async_operation_t* operations[] = {&close_operation.base,
                                          &malformed_operation.base};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_async_proactor_submit(proactor_,
                                 iree_async_operation_list_make(
                                     operations, IREE_ARRAYSIZE(operations))));

  int socket_type = 0;
  int socket_type_length = sizeof(socket_type);
  EXPECT_EQ(getsockopt(native_socket, SOL_SOCKET, SO_TYPE, (char*)&socket_type,
                       &socket_type_length),
            0);
  EXPECT_EQ(socket->primitive.value.win32_handle, (uintptr_t)native_socket);
  EXPECT_EQ(socket->state, IREE_ASYNC_SOCKET_STATE_CREATED);
  EXPECT_EQ(iree_atomic_ref_count_load(&socket->ref_count), 1);
  EXPECT_EQ(completion.call_count, 0);

  iree_async_socket_retain(socket);
  IREE_ASSERT_OK(
      iree_async_proactor_submit_one(proactor_, &close_operation.base));
  iree_host_size_t completed_count = 0;
  IREE_ASSERT_OK(iree_async_proactor_poll(proactor_, iree_infinite_timeout(),
                                          &completed_count));
  EXPECT_EQ(completed_count, 1);
  EXPECT_EQ(completion.call_count, 1);
  EXPECT_EQ(completion.status_code, IREE_STATUS_OK);
  EXPECT_EQ(socket->primitive.type, IREE_ASYNC_PRIMITIVE_TYPE_NONE);
  EXPECT_EQ(iree_atomic_ref_count_load(&socket->ref_count), 1);
  iree_async_socket_release(socket);
}

TEST(IocpProactorSubmitTest, CarrierAllocationFailureRollsBackCloseAndMessage) {
  ControlledAllocator allocator;
  iree_async_proactor_t* source = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_create_iocp(
      iree_async_proactor_options_default(), allocator.value(), &source));
  iree_async_proactor_t* target = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_create_iocp(
      iree_async_proactor_options_default(), iree_allocator_system(), &target));

  MessageState message_state;
  iree_async_proactor_set_message_callback(
      target, {.fn = RecordMessage, .user_data = &message_state});

  iree_async_socket_t* socket = nullptr;
  IREE_ASSERT_OK(iree_async_socket_create(source, IREE_ASYNC_SOCKET_TYPE_TCP,
                                          IREE_ASYNC_SOCKET_OPTION_NONE,
                                          &socket));
  const SOCKET native_socket = (SOCKET)socket->primitive.value.win32_handle;

  CompletionState completion;
  iree_async_socket_close_operation_t close_operation;
  InitializeOperation(&close_operation, IREE_ASYNC_OPERATION_TYPE_SOCKET_CLOSE,
                      &completion);
  close_operation.socket = socket;

  iree_async_message_operation_t message_operation;
  InitializeOperation(&message_operation, IREE_ASYNC_OPERATION_TYPE_MESSAGE,
                      &completion);
  message_operation.target = target;
  message_operation.message_data = 0xC0FFEE;
  message_operation.message_flags = IREE_ASYNC_MESSAGE_FLAG_NONE;

  iree_async_operation_t* operations[] = {&close_operation.base,
                                          &message_operation.base};
  allocator.allocations_before_failure = 1;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      iree_async_proactor_submit(
          source, iree_async_operation_list_make(operations,
                                                 IREE_ARRAYSIZE(operations))));

  int socket_type = 0;
  int socket_type_length = sizeof(socket_type);
  EXPECT_EQ(getsockopt(native_socket, SOL_SOCKET, SO_TYPE, (char*)&socket_type,
                       &socket_type_length),
            0);
  EXPECT_EQ(socket->primitive.value.win32_handle, (uintptr_t)native_socket);
  EXPECT_EQ(iree_atomic_ref_count_load(&socket->ref_count), 1);
  EXPECT_EQ(message_operation.platform.software.reserved_entry, nullptr);
  EXPECT_EQ(completion.call_count, 0);
  EXPECT_EQ(message_state.call_count, 0);
  EXPECT_EQ(
      iree_atomic_load(
          &iree_async_proactor_iocp_cast(source)->outstanding_carrier_count,
          iree_memory_order_relaxed),
      0);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_DEADLINE_EXCEEDED,
      iree_async_proactor_poll(source, iree_immediate_timeout(), nullptr));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_DEADLINE_EXCEEDED,
      iree_async_proactor_poll(target, iree_immediate_timeout(), nullptr));

  allocator.allocations_before_failure = -1;
  iree_async_socket_retain(socket);
  IREE_ASSERT_OK(iree_async_proactor_submit(
      source,
      iree_async_operation_list_make(operations, IREE_ARRAYSIZE(operations))));
  iree_host_size_t completed_count = 0;
  IREE_ASSERT_OK(iree_async_proactor_poll(source, iree_infinite_timeout(),
                                          &completed_count));
  EXPECT_EQ(completed_count, 2);
  EXPECT_EQ(completion.call_count, 2);
  EXPECT_EQ(completion.status_code, IREE_STATUS_OK);
  EXPECT_EQ(socket->primitive.type, IREE_ASYNC_PRIMITIVE_TYPE_NONE);
  EXPECT_EQ(iree_atomic_ref_count_load(&socket->ref_count), 1);

  IREE_ASSERT_OK(iree_async_proactor_poll(target, iree_infinite_timeout(),
                                          &completed_count));
  EXPECT_EQ(message_state.call_count, 1);
  EXPECT_EQ(message_state.value, 0xC0FFEEu);

  iree_async_socket_release(socket);
  iree_async_proactor_release(target);
  iree_async_proactor_release(source);
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

TEST(IocpEventSourceTest, RequiresWaitCompletionPackets) {
  iree_async_proactor_options_t options = iree_async_proactor_options_default();
  options.allowed_capabilities &=
      ~IREE_ASYNC_PROACTOR_CAPABILITY_WAIT_COMPLETION_PACKET;
  iree_async_proactor_t* proactor = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_create_iocp(
      options, iree_allocator_system(), &proactor));

  iree_async_event_t* event = nullptr;
  IREE_ASSERT_OK(iree_async_event_create(proactor, &event));
  iree_async_event_source_callback_t callback = {
      +[](void*, iree_async_event_source_t*, iree_async_poll_events_t) {},
      nullptr,
  };
  iree_async_event_source_t* source = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNAVAILABLE,
      iree_async_proactor_register_event_source(
          proactor, event->native.wait_primitive, callback, &source));
  EXPECT_EQ(source, nullptr);

  iree_async_event_release(event);
  iree_async_proactor_release(proactor);
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
  iree_async_event_set(event);
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

  iree_async_event_set(event);
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
  iree_async_event_set(event);
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

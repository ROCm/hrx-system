// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <arpa/inet.h>
#include <fcntl.h>
#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "iree/async/api.h"
#include "iree/async/operations/futex.h"
#include "iree/async/operations/message.h"
#include "iree/async/platform/posix/proactor.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

struct ProactorDeleter {
  void operator()(iree_async_proactor_t* proactor) const {
    iree_async_proactor_release(proactor);
  }
};
using ProactorPtr = std::unique_ptr<iree_async_proactor_t, ProactorDeleter>;

struct SocketDeleter {
  void operator()(iree_async_socket_t* socket) const {
    iree_async_socket_release(socket);
  }
};
using SocketPtr = std::unique_ptr<iree_async_socket_t, SocketDeleter>;

class ScopedFd {
 public:
  ScopedFd() = default;
  explicit ScopedFd(int fd) : fd_(fd) {}
  ~ScopedFd() { Reset(); }

  ScopedFd(const ScopedFd&) = delete;
  ScopedFd& operator=(const ScopedFd&) = delete;

  int get() const { return fd_; }

  int Release() {
    int fd = fd_;
    fd_ = -1;
    return fd;
  }

  void Reset(int fd = -1) {
    if (fd_ >= 0) {
      close(fd_);
    }
    fd_ = fd;
  }

 private:
  int fd_ = -1;
};

struct CompletionTracker {
  int call_count = 0;
  std::vector<iree_status_code_t> status_codes;

  static void Callback(void* user_data, iree_async_operation_t* operation,
                       iree_status_t status,
                       iree_async_completion_flags_t flags) {
    (void)operation;
    (void)flags;
    auto* tracker = static_cast<CompletionTracker*>(user_data);
    ++tracker->call_count;
    tracker->status_codes.push_back(iree_status_code(status));
    iree_status_free(status);
  }
};

struct MessageTracker {
  std::vector<uint64_t> values;

  static void Callback(iree_async_proactor_t* proactor, uint64_t message_data,
                       void* user_data) {
    (void)proactor;
    static_cast<MessageTracker*>(user_data)->values.push_back(message_data);
  }
};

template <typename T>
void InitializeOperation(T* operation, iree_async_operation_type_t type,
                         CompletionTracker* tracker) {
  iree_async_operation_zero(&operation->base, sizeof(*operation));
  iree_async_operation_initialize(&operation->base, type,
                                  IREE_ASYNC_OPERATION_FLAG_NONE,
                                  CompletionTracker::Callback, tracker);
}

void InitializeNotificationSignal(
    iree_async_notification_signal_operation_t* operation,
    iree_async_notification_t* notification, CompletionTracker* tracker) {
  InitializeOperation(operation, IREE_ASYNC_OPERATION_TYPE_NOTIFICATION_SIGNAL,
                      tracker);
  operation->notification = notification;
  operation->wake_count = 1;
}

void InitializeMessage(iree_async_message_operation_t* operation,
                       iree_async_proactor_t* target, uint64_t message_data,
                       iree_async_message_flags_t message_flags,
                       CompletionTracker* tracker) {
  iree_async_operation_zero(&operation->base, sizeof(*operation));
  iree_async_operation_initialize(
      &operation->base, IREE_ASYNC_OPERATION_TYPE_MESSAGE,
      IREE_ASYNC_OPERATION_FLAG_NONE,
      tracker ? CompletionTracker::Callback : nullptr, tracker);
  operation->target = target;
  operation->message_data = message_data;
  operation->message_flags = message_flags;
}

iree_status_t CreateManagedSocketPair(iree_async_proactor_t* proactor,
                                      SocketPtr* out_socket,
                                      ScopedFd* out_peer_fd) {
  int socket_fds[2] = {-1, -1};
  if (socketpair(AF_UNIX, SOCK_STREAM, 0, socket_fds) != 0) {
    return iree_make_status(iree_status_code_from_errno(errno),
                            "socketpair failed");
  }
  ScopedFd managed_fd(socket_fds[0]);
  ScopedFd peer_fd(socket_fds[1]);

  iree_async_socket_t* socket = nullptr;
  IREE_RETURN_IF_ERROR(iree_async_socket_import(
      proactor, iree_async_primitive_from_fd(managed_fd.get()),
      IREE_ASYNC_SOCKET_TYPE_UNIX_STREAM, IREE_ASYNC_SOCKET_FLAG_NONE,
      &socket));
  managed_fd.Release();
  out_socket->reset(socket);
  out_peer_fd->Reset(peer_fd.Release());
  return iree_ok_status();
}

iree_status_t CreateManagedUdpSender(iree_async_proactor_t* proactor,
                                     SocketPtr* out_sender,
                                     ScopedFd* out_receiver_fd,
                                     iree_async_address_t* out_destination) {
  ScopedFd sender_fd(socket(AF_INET, SOCK_DGRAM, 0));
  if (sender_fd.get() < 0) {
    return iree_make_status(iree_status_code_from_errno(errno),
                            "sender socket creation failed");
  }
  ScopedFd receiver_fd(socket(AF_INET, SOCK_DGRAM, 0));
  if (receiver_fd.get() < 0) {
    return iree_make_status(iree_status_code_from_errno(errno),
                            "receiver socket creation failed");
  }

  sockaddr_in receiver_address = {
      .sin_family = AF_INET,
  };
  receiver_address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  receiver_address.sin_port = 0;
  if (bind(receiver_fd.get(),
           reinterpret_cast<const sockaddr*>(&receiver_address),
           sizeof(receiver_address)) != 0) {
    return iree_make_status(iree_status_code_from_errno(errno),
                            "receiver bind failed");
  }
  socklen_t receiver_address_length = sizeof(receiver_address);
  if (getsockname(receiver_fd.get(),
                  reinterpret_cast<sockaddr*>(&receiver_address),
                  &receiver_address_length) != 0) {
    return iree_make_status(iree_status_code_from_errno(errno),
                            "receiver getsockname failed");
  }
  IREE_RETURN_IF_ERROR(iree_async_address_from_ipv4(
      IREE_SV("127.0.0.1"), ntohs(receiver_address.sin_port), out_destination));

  iree_async_socket_t* sender = nullptr;
  IREE_RETURN_IF_ERROR(iree_async_socket_import(
      proactor, iree_async_primitive_from_fd(sender_fd.get()),
      IREE_ASYNC_SOCKET_TYPE_UDP, IREE_ASYNC_SOCKET_FLAG_NONE, &sender));
  sender_fd.Release();
  out_sender->reset(sender);
  out_receiver_fd->Reset(receiver_fd.Release());
  return iree_ok_status();
}

class PosixProactorSubmitTest : public ::testing::Test {
 protected:
  static constexpr iree_host_size_t kReadyPoolCapacity = 1;
  static constexpr iree_host_size_t kCompletionPoolCapacity =
      kReadyPoolCapacity + IREE_ASYNC_POSIX_DEFAULT_WORKER_COUNT;

  void SetUp() override {
    iree_async_proactor_options_t options =
        iree_async_proactor_options_default();
    options.max_concurrent_operations = kReadyPoolCapacity;
    options.message_pool_capacity = 1;
    IREE_ASSERT_OK(iree_async_proactor_create_posix(
        options, iree_allocator_system(), &proactor_));
    ASSERT_EQ(posix_proactor()->completion_pool.capacity,
              kCompletionPoolCapacity);
  }

  void TearDown() override { iree_async_proactor_release(proactor_); }

  iree_async_proactor_posix_t* posix_proactor() {
    return iree_async_proactor_posix_cast(proactor_);
  }

  void PollUntilCallbacks(iree_async_proactor_t* proactor,
                          CompletionTracker* tracker, int expected_call_count) {
    while (tracker->call_count < expected_call_count) {
      iree_host_size_t completed_count = 0;
      IREE_ASSERT_OK(iree_async_proactor_poll(proactor, iree_infinite_timeout(),
                                              &completed_count));
    }
  }

  void PollUntilMessages(iree_async_proactor_t* proactor,
                         MessageTracker* tracker,
                         iree_host_size_t expected_message_count) {
    while (tracker->values.size() < expected_message_count) {
      iree_host_size_t completed_count = 0;
      IREE_ASSERT_OK(iree_async_proactor_poll(proactor, iree_infinite_timeout(),
                                              &completed_count));
    }
  }

  void ExpectQuiescent(iree_async_proactor_t* proactor) {
    iree_host_size_t completed_count = 0;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_DEADLINE_EXCEEDED,
        iree_async_proactor_poll(proactor, iree_immediate_timeout(),
                                 &completed_count));
    EXPECT_EQ(completed_count, 0u);
  }

  void ExpectSuccessfulCompletions(const CompletionTracker& tracker,
                                   int expected_call_count) {
    EXPECT_EQ(tracker.call_count, expected_call_count);
    ASSERT_EQ(tracker.status_codes.size(),
              static_cast<iree_host_size_t>(expected_call_count));
    for (iree_status_code_t status_code : tracker.status_codes) {
      EXPECT_EQ(status_code, IREE_STATUS_OK);
    }
  }

  iree_async_proactor_t* proactor_ = nullptr;
};

TEST_F(PosixProactorSubmitTest,
       CompletionExhaustionPreservesStreamAndNotificationState) {
  SocketPtr socket;
  ScopedFd peer_fd;
  IREE_ASSERT_OK(CreateManagedSocketPair(proactor_, &socket, &peer_fd));

  iree_async_notification_t* notification = nullptr;
  IREE_ASSERT_OK(iree_async_notification_create(
      proactor_, IREE_ASYNC_NOTIFICATION_FLAG_NONE, &notification));
  std::unique_ptr<iree_async_notification_t,
                  decltype(&iree_async_notification_release)>
      notification_owner(notification, iree_async_notification_release);
  const uint32_t initial_epoch =
      iree_async_notification_query_epoch(notification);

  CompletionTracker tracker;
  char payload = 's';
  iree_async_span_t payload_span =
      iree_async_span_from_ptr(&payload, sizeof(payload));
  iree_async_socket_send_operation_t send;
  InitializeOperation(&send, IREE_ASYNC_OPERATION_TYPE_SOCKET_SEND, &tracker);
  send.socket = socket.get();
  send.buffers = iree_async_span_list_make(&payload_span, 1);

  std::array<iree_async_notification_signal_operation_t,
             kCompletionPoolCapacity>
      signals;
  std::array<iree_async_operation_t*, kCompletionPoolCapacity + 1> operations;
  operations[0] = &send.base;
  for (iree_host_size_t i = 0; i < signals.size(); ++i) {
    InitializeNotificationSignal(&signals[i], notification, &tracker);
    operations[i + 1] = &signals[i].base;
  }

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      iree_async_proactor_submit(
          proactor_, iree_async_operation_list_make(operations.data(),
                                                    operations.size())));
  EXPECT_EQ(tracker.call_count, 0);
  EXPECT_EQ(iree_async_notification_query_epoch(notification), initial_epoch);
  char received_payload = 0;
  EXPECT_EQ(recv(peer_fd.get(), &received_payload, sizeof(received_payload),
                 MSG_DONTWAIT),
            -1);
  EXPECT_TRUE(errno == EAGAIN || errno == EWOULDBLOCK);
  ExpectQuiescent(proactor_);

  IREE_ASSERT_OK(iree_async_proactor_submit(
      proactor_, iree_async_operation_list_make(operations.data(),
                                                kCompletionPoolCapacity)));
  EXPECT_EQ(recv(peer_fd.get(), &received_payload, sizeof(received_payload), 0),
            1);
  EXPECT_EQ(received_payload, payload);
  EXPECT_EQ(iree_async_notification_query_epoch(notification),
            initial_epoch + kCompletionPoolCapacity - 1);
  PollUntilCallbacks(proactor_, &tracker, kCompletionPoolCapacity);
  ExpectSuccessfulCompletions(tracker, kCompletionPoolCapacity);
}

TEST_F(PosixProactorSubmitTest, CancellationSurvivesInlineCallbackMapGrowth) {
  constexpr size_t kWaitCount = 16;
  // Both layouts use eight native descriptors per wait, creating collisions
  // without exceeding a process's small default descriptor limit.
#if defined(IREE_PLATFORM_LINUX)
  constexpr size_t kEventsPerWait = 7;
#else
  constexpr size_t kEventsPerWait = 3;
#endif  // IREE_PLATFORM_LINUX
  constexpr size_t kEventCount = kWaitCount * kEventsPerWait;
  struct EventDeleter {
    void operator()(iree_async_event_t* event) const {
      iree_async_event_release(event);
    }
  };
  struct State {
    // Proactor that owns all waits and registrations.
    iree_async_proactor_t* proactor;
    // Completion results from all cancelled waits.
    CompletionTracker tracker;
    // Caller-owned events kept alive through source unregistration.
    std::array<std::unique_ptr<iree_async_event_t, EventDeleter>, kEventCount>
        events;
    // Sources registered by the first inline cancellation callback.
    std::array<iree_async_event_source_t*, kEventCount> sources = {};
    // Whether that callback has registered the sources.
    bool registered_sources = false;
  } state = {proactor_};
  struct WaitState {
    // Shared results and callback-owned registrations.
    State* state;
    // Caller reference dropped in the wait's terminal callback.
    std::unique_ptr<iree_async_notification_t,
                    decltype(&iree_async_notification_release)>
        notification = {nullptr, iree_async_notification_release};
  } wait_states[kWaitCount] = {};
  std::array<iree_async_notification_wait_operation_t, kWaitCount> waits = {};
  for (size_t i = 0; i < waits.size(); ++i) {
    wait_states[i].state = &state;
    iree_async_notification_t* notification = nullptr;
    IREE_ASSERT_OK(iree_async_notification_create(
        proactor_, IREE_ASYNC_NOTIFICATION_FLAG_NONE, &notification));
    wait_states[i].notification.reset(notification);
    // Interleave live native handles so the initial map contains collisions
    // rather than only consecutive descriptor keys.
    for (size_t j = 0; j < kEventsPerWait; ++j) {
      iree_async_event_t* event = nullptr;
      IREE_ASSERT_OK(iree_async_event_create(proactor_, &event));
      state.events[i * kEventsPerWait + j].reset(event);
    }
  }
  for (size_t i = 0; i < waits.size(); ++i) {
    InitializeOperation(&waits[i], IREE_ASYNC_OPERATION_TYPE_NOTIFICATION_WAIT,
                        &state.tracker);
    waits[i].notification = wait_states[i].notification.get();
    waits[i].base.user_data = &wait_states[i];
    waits[i].base.completion_fn = +[](void* user_data,
                                      iree_async_operation_t* operation,
                                      iree_status_t status,
                                      iree_async_completion_flags_t flags) {
      auto* wait_state = static_cast<WaitState*>(user_data);
      auto* state = wait_state->state;
      CompletionTracker::Callback(&state->tracker, operation, status, flags);
      wait_state->notification.reset();
      if (!state->registered_sources) {
        state->registered_sources = true;
        iree_async_event_source_callback_t callback = {
            +[](void*, iree_async_event_source_t*, iree_async_poll_events_t) {},
            nullptr};
        for (size_t i = 0; i < state->events.size(); ++i) {
          IREE_EXPECT_OK(iree_async_proactor_register_event_source(
              state->proactor, state->events[i]->native.wait_primitive,
              callback, &state->sources[i]));
        }
      }
    };
    IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &waits[i].base));
  }
  iree_async_proactor_wake(proactor_);
  IREE_ASSERT_OK(
      iree_async_proactor_poll(proactor_, iree_infinite_timeout(), nullptr));
  const iree_host_size_t initial_bucket_count =
      posix_proactor()->fd_map.bucket_count;
  for (auto& wait : waits) {
    IREE_ASSERT_OK(iree_async_proactor_cancel(proactor_, &wait.base));
    IREE_ASSERT_OK(iree_async_proactor_cancel(proactor_, &wait.base));
  }
  PollUntilCallbacks(proactor_, &state.tracker, kWaitCount);
  EXPECT_GT(posix_proactor()->fd_map.bucket_count, initial_bucket_count);
  EXPECT_EQ(state.tracker.call_count, kWaitCount);
  for (iree_status_code_t code : state.tracker.status_codes) {
    EXPECT_EQ(code, IREE_STATUS_CANCELLED);
  }
  for (auto* source : state.sources) {
    if (source) {
      bool unregistered = false;
      iree_async_proactor_unregister_event_source(
          proactor_, source,
          {+[](void* context) { *static_cast<bool*>(context) = true; },
           &unregistered});
      EXPECT_TRUE(unregistered);
    }
  }
  iree_async_proactor_wake(proactor_);
  IREE_ASSERT_OK(
      iree_async_proactor_poll(proactor_, iree_infinite_timeout(), nullptr));
  EXPECT_EQ(iree_atomic_load(&posix_proactor()->pending_cancellations,
                             iree_memory_order_acquire),
            0);
  ExpectQuiescent(proactor_);
}

TEST_F(PosixProactorSubmitTest,
       ReadinessCompletionReturnsNotificationOwnership) {
  constexpr size_t kWaitCount = kCompletionPoolCapacity + 1;
  CompletionTracker tracker;
  struct WaitState {
    // Combined terminal completion results.
    CompletionTracker* tracker;
    // Caller reference dropped by terminal completion.
    std::unique_ptr<iree_async_notification_t,
                    decltype(&iree_async_notification_release)>
        notification = {nullptr, iree_async_notification_release};
  } states[kWaitCount] = {};
  std::array<iree_async_notification_wait_operation_t, kWaitCount> waits = {};
  for (size_t i = 0; i < waits.size(); ++i) {
    states[i].tracker = &tracker;
    iree_async_notification_t* notification = nullptr;
    IREE_ASSERT_OK(iree_async_notification_create(
        proactor_, IREE_ASYNC_NOTIFICATION_FLAG_NONE, &notification));
    states[i].notification.reset(notification);
  }
  for (size_t i = 0; i < waits.size(); ++i) {
    InitializeOperation(&waits[i], IREE_ASYNC_OPERATION_TYPE_NOTIFICATION_WAIT,
                        &tracker);
    waits[i].notification = states[i].notification.get();
    waits[i].base.user_data = &states[i];
    waits[i].base.completion_fn =
        +[](void* user_data, iree_async_operation_t* operation,
            iree_status_t status, iree_async_completion_flags_t flags) {
          auto* state = static_cast<WaitState*>(user_data);
          CompletionTracker::Callback(state->tracker, operation, status, flags);
          state->notification.reset();
        };
    IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &waits[i].base));
  }
  iree_async_proactor_wake(proactor_);
  IREE_ASSERT_OK(
      iree_async_proactor_poll(proactor_, iree_infinite_timeout(), nullptr));
  for (auto& state : states) {
    iree_async_notification_signal(state.notification.get(), 1);
  }
  PollUntilCallbacks(proactor_, &tracker, kWaitCount);
  ExpectSuccessfulCompletions(tracker, kWaitCount);
}

TEST_F(PosixProactorSubmitTest, SharedRelayCapturesRegistrationEpoch) {
  if (!iree_async_notification_native_is_supported()) {
    GTEST_SKIP();
  }
  iree_notification_state_t state = {};
  iree_notification_state_initialize(&state);
  iree_async_notification_native_t native = {};
  IREE_ASSERT_OK(iree_async_notification_native_initialize(&state, &native));
  for (int i = 0; i < 7; ++i) {
    iree_async_notification_native_signal(&native, 1);
  }
  iree_async_notification_t* source = nullptr;
  iree_async_notification_t* sink = nullptr;
  IREE_ASSERT_OK(
      iree_async_notification_create_shared(proactor_, &native, &source));
  IREE_ASSERT_OK(iree_async_notification_create(
      proactor_, IREE_ASYNC_NOTIFICATION_FLAG_NONE, &sink));

  // The native wake remains pending, but its epoch predates registration.
  iree_async_notification_signal(source, 1);
  iree_async_relay_t* relay = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_register_relay(
      proactor_, iree_async_relay_source_from_notification(source),
      iree_async_relay_sink_signal_notification(sink, 1),
      IREE_ASYNC_RELAY_FLAG_PERSISTENT, iree_async_relay_error_callback_none(),
      &relay));
  IREE_ASSERT_OK(
      iree_async_proactor_poll(proactor_, iree_infinite_timeout(), nullptr));
  EXPECT_EQ(iree_async_notification_query_epoch(sink), 0u);

  iree_async_notification_signal(source, 1);
  IREE_ASSERT_OK(
      iree_async_proactor_poll(proactor_, iree_infinite_timeout(), nullptr));
  EXPECT_EQ(iree_async_notification_query_epoch(sink), 1u);
  iree_async_proactor_unregister_relay(
      proactor_, relay, iree_async_relay_unregistered_callback_none());
  iree_async_notification_release(sink);
  iree_async_notification_release(source);
  iree_async_notification_native_deinitialize(&native);
}

TEST_F(PosixProactorSubmitTest, ValidationFailurePrecedesEagerSend) {
  SocketPtr socket;
  ScopedFd peer_fd;
  IREE_ASSERT_OK(CreateManagedSocketPair(proactor_, &socket, &peer_fd));

  CompletionTracker tracker;
  char payload = 'v';
  iree_async_span_t payload_span =
      iree_async_span_from_ptr(&payload, sizeof(payload));
  iree_async_socket_send_operation_t send;
  InitializeOperation(&send, IREE_ASYNC_OPERATION_TYPE_SOCKET_SEND, &tracker);
  send.socket = socket.get();
  send.buffers = iree_async_span_list_make(&payload_span, 1);

  iree_async_futex_wake_operation_t unsupported;
  InitializeOperation(&unsupported, IREE_ASYNC_OPERATION_TYPE_FUTEX_WAKE,
                      &tracker);
  uint32_t futex_word = 0;
  unsupported.futex_address = &futex_word;
  unsupported.wake_count = 1;
  unsupported.futex_flags = IREE_ASYNC_FUTEX_SIZE_U32;

  iree_async_operation_t* operations[] = {&send.base, &unsupported.base};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNIMPLEMENTED,
      iree_async_proactor_submit(proactor_,
                                 iree_async_operation_list_make(
                                     operations, IREE_ARRAYSIZE(operations))));
  EXPECT_EQ(tracker.call_count, 0);
  char received_payload = 0;
  EXPECT_EQ(recv(peer_fd.get(), &received_payload, sizeof(received_payload),
                 MSG_DONTWAIT),
            -1);
  EXPECT_TRUE(errno == EAGAIN || errno == EWOULDBLOCK);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &send.base));
  EXPECT_EQ(recv(peer_fd.get(), &received_payload, sizeof(received_payload), 0),
            1);
  EXPECT_EQ(received_payload, payload);
  PollUntilCallbacks(proactor_, &tracker, 1);
  ExpectSuccessfulCompletions(tracker, 1);
}

TEST_F(PosixProactorSubmitTest, CompletionExhaustionDoesNotConsumeSocketClose) {
  SocketPtr socket;
  ScopedFd peer_fd;
  IREE_ASSERT_OK(CreateManagedSocketPair(proactor_, &socket, &peer_fd));
  const int socket_fd = socket->primitive.value.fd;

  iree_async_notification_t* notification = nullptr;
  IREE_ASSERT_OK(iree_async_notification_create(
      proactor_, IREE_ASYNC_NOTIFICATION_FLAG_NONE, &notification));
  std::unique_ptr<iree_async_notification_t,
                  decltype(&iree_async_notification_release)>
      notification_owner(notification, iree_async_notification_release);
  const uint32_t initial_epoch =
      iree_async_notification_query_epoch(notification);

  CompletionTracker tracker;
  iree_async_socket_close_operation_t close_operation;
  InitializeOperation(&close_operation, IREE_ASYNC_OPERATION_TYPE_SOCKET_CLOSE,
                      &tracker);
  close_operation.socket = socket.get();

  std::array<iree_async_notification_signal_operation_t,
             kCompletionPoolCapacity>
      signals;
  std::array<iree_async_operation_t*, kCompletionPoolCapacity + 1> operations;
  operations[0] = &close_operation.base;
  for (iree_host_size_t i = 0; i < signals.size(); ++i) {
    InitializeNotificationSignal(&signals[i], notification, &tracker);
    operations[i + 1] = &signals[i].base;
  }

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      iree_async_proactor_submit(
          proactor_, iree_async_operation_list_make(operations.data(),
                                                    operations.size())));
  EXPECT_NE(fcntl(socket_fd, F_GETFD), -1);
  EXPECT_EQ(iree_async_notification_query_epoch(notification), initial_epoch);
  EXPECT_EQ(tracker.call_count, 0);
  ExpectQuiescent(proactor_);

  IREE_ASSERT_OK(
      iree_async_proactor_submit_one(proactor_, &close_operation.base));
  socket.release();
  EXPECT_EQ(fcntl(socket_fd, F_GETFD), -1);
  EXPECT_EQ(errno, EBADF);
  PollUntilCallbacks(proactor_, &tracker, 1);
  ExpectSuccessfulCompletions(tracker, 1);
}

TEST_F(PosixProactorSubmitTest,
       CompletionExhaustionPreservesDatagramAndNotificationState) {
  SocketPtr sender;
  ScopedFd receiver_fd;
  iree_async_address_t destination;
  IREE_ASSERT_OK(
      CreateManagedUdpSender(proactor_, &sender, &receiver_fd, &destination));

  iree_async_notification_t* notification = nullptr;
  IREE_ASSERT_OK(iree_async_notification_create(
      proactor_, IREE_ASYNC_NOTIFICATION_FLAG_NONE, &notification));
  std::unique_ptr<iree_async_notification_t,
                  decltype(&iree_async_notification_release)>
      notification_owner(notification, iree_async_notification_release);
  const uint32_t initial_epoch =
      iree_async_notification_query_epoch(notification);

  CompletionTracker tracker;
  char payload = 'd';
  iree_async_span_t payload_span =
      iree_async_span_from_ptr(&payload, sizeof(payload));
  iree_async_socket_sendto_operation_t sendto_operation;
  InitializeOperation(&sendto_operation,
                      IREE_ASYNC_OPERATION_TYPE_SOCKET_SENDTO, &tracker);
  sendto_operation.socket = sender.get();
  sendto_operation.buffers = iree_async_span_list_make(&payload_span, 1);
  sendto_operation.destination = destination;

  std::array<iree_async_notification_signal_operation_t,
             kCompletionPoolCapacity>
      signals;
  std::array<iree_async_operation_t*, kCompletionPoolCapacity + 1> operations;
  operations[0] = &sendto_operation.base;
  for (iree_host_size_t i = 0; i < signals.size(); ++i) {
    InitializeNotificationSignal(&signals[i], notification, &tracker);
    operations[i + 1] = &signals[i].base;
  }

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      iree_async_proactor_submit(
          proactor_, iree_async_operation_list_make(operations.data(),
                                                    operations.size())));
  EXPECT_EQ(tracker.call_count, 0);
  EXPECT_EQ(iree_async_notification_query_epoch(notification), initial_epoch);
  char received_payload = 0;
  EXPECT_EQ(recv(receiver_fd.get(), &received_payload, sizeof(received_payload),
                 MSG_DONTWAIT),
            -1);
  EXPECT_TRUE(errno == EAGAIN || errno == EWOULDBLOCK);
  ExpectQuiescent(proactor_);

  IREE_ASSERT_OK(
      iree_async_proactor_submit_one(proactor_, &sendto_operation.base));
  EXPECT_EQ(
      recv(receiver_fd.get(), &received_payload, sizeof(received_payload), 0),
      1);
  EXPECT_EQ(received_payload, payload);
  PollUntilCallbacks(proactor_, &tracker, 1);
  ExpectSuccessfulCompletions(tracker, 1);
}

TEST_F(PosixProactorSubmitTest, MessagePoolExhaustionDoesNotPublishPrefix) {
  iree_async_proactor_options_t target_options =
      iree_async_proactor_options_default();
  target_options.max_concurrent_operations = 1;
  target_options.message_pool_capacity = 1;
  iree_async_proactor_t* target_raw = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_create_posix(
      target_options, iree_allocator_system(), &target_raw));
  ProactorPtr target(target_raw);

  MessageTracker message_tracker;
  iree_async_proactor_set_message_callback(
      target.get(), {MessageTracker::Callback, &message_tracker});

  iree_async_message_operation_t messages[2];
  InitializeMessage(&messages[0], target.get(), 0xA11CE,
                    IREE_ASYNC_MESSAGE_FLAG_SKIP_SOURCE_COMPLETION, nullptr);
  InitializeMessage(&messages[1], target.get(), 0xBAD,
                    IREE_ASYNC_MESSAGE_FLAG_SKIP_SOURCE_COMPLETION, nullptr);
  iree_async_operation_t* operations[] = {&messages[0].base, &messages[1].base};

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      iree_async_proactor_submit(proactor_,
                                 iree_async_operation_list_make(
                                     operations, IREE_ARRAYSIZE(operations))));
  ExpectQuiescent(proactor_);
  ExpectQuiescent(target.get());
  EXPECT_TRUE(message_tracker.values.empty());

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &messages[0].base));
  PollUntilMessages(target.get(), &message_tracker, 1);
  ASSERT_EQ(message_tracker.values.size(), 1u);
  EXPECT_EQ(message_tracker.values[0], 0xA11CEu);
}

TEST_F(PosixProactorSubmitTest,
       SourceExhaustionReleasesTargetMessageReservation) {
  iree_async_proactor_options_t target_options =
      iree_async_proactor_options_default();
  target_options.max_concurrent_operations = 1;
  target_options.message_pool_capacity = 1;
  iree_async_proactor_t* target_raw = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_create_posix(
      target_options, iree_allocator_system(), &target_raw));
  ProactorPtr target(target_raw);

  MessageTracker message_tracker;
  iree_async_proactor_set_message_callback(
      target.get(), {MessageTracker::Callback, &message_tracker});

  iree_async_notification_t* notification = nullptr;
  IREE_ASSERT_OK(iree_async_notification_create(
      proactor_, IREE_ASYNC_NOTIFICATION_FLAG_NONE, &notification));
  std::unique_ptr<iree_async_notification_t,
                  decltype(&iree_async_notification_release)>
      notification_owner(notification, iree_async_notification_release);
  const uint32_t initial_epoch =
      iree_async_notification_query_epoch(notification);

  CompletionTracker tracker;
  std::array<iree_async_notification_signal_operation_t,
             kCompletionPoolCapacity>
      signals;
  std::array<iree_async_operation_t*, kCompletionPoolCapacity + 1> operations;
  for (iree_host_size_t i = 0; i < signals.size(); ++i) {
    InitializeNotificationSignal(&signals[i], notification, &tracker);
    operations[i] = &signals[i].base;
  }
  iree_async_message_operation_t message;
  InitializeMessage(&message, target.get(), 0xC0FFEE,
                    IREE_ASYNC_MESSAGE_FLAG_NONE, &tracker);
  operations[kCompletionPoolCapacity] = &message.base;

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      iree_async_proactor_submit(
          proactor_, iree_async_operation_list_make(operations.data(),
                                                    operations.size())));
  EXPECT_EQ(tracker.call_count, 0);
  EXPECT_EQ(iree_async_notification_query_epoch(notification), initial_epoch);
  ExpectQuiescent(proactor_);
  ExpectQuiescent(target.get());
  EXPECT_TRUE(message_tracker.values.empty());

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &message.base));
  PollUntilCallbacks(proactor_, &tracker, 1);
  PollUntilMessages(target.get(), &message_tracker, 1);
  ExpectSuccessfulCompletions(tracker, 1);
  ASSERT_EQ(message_tracker.values.size(), 1u);
  EXPECT_EQ(message_tracker.values[0], 0xC0FFEEu);
}

}  // namespace

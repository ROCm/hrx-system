// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/notification_native.h"

#include <stdio.h>
#include <stdlib.h>

#include <string>
#include <thread>
#include <vector>

#include "iree/async/notification.h"
#include "iree/async/operations/scheduling.h"
#include "iree/async/proactor_platform.h"
#include "iree/async/util/local_stream.h"
#include "iree/base/internal/shm.h"
#include "iree/testing/coordinated_test.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

#if !defined(IREE_PLATFORM_WINDOWS)
#include <fcntl.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#endif

namespace {

// Child failures must affect the role's exit code, not a parent gtest object.
void Check(bool condition, const char* expression, int line) {
  if (!condition) {
    fprintf(stderr, "notification role failed at line %d: %s\n", line,
            expression);
    abort();
  }
}
#define ROLE_CHECK(expression) Check((expression), #expression, __LINE__)

void CheckStatus(iree_status_t status) {
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    abort();
  }
}

struct SharedState {
  // Native notification state, initialized only by the exporting process.
  iree_notification_state_t notification;
  // Ordinary data acquired by every observer through notification publication.
  uint32_t payload;
  // Observer round admitted after its poll owner stops setup progress.
  iree_atomic_int32_t idle_round;
};

constexpr uint32_t kHandleCount =
    1 + IREE_ASYNC_NOTIFICATION_NATIVE_HANDLE_COUNT;
constexpr uint32_t kRounds = 64;
constexpr uint32_t kWaiterCount = 4;

struct Transfer {
  // Set once by the actual completion callback on the polling owner.
  bool done = false;

  iree_async_local_stream_callback_t callback() {
    return {+[](void* user_data, iree_status_t status) {
              CheckStatus(status);
              *static_cast<bool*>(user_data) = true;
            },
            &done};
  }

  void Wait(iree_async_proactor_t* proactor) {
    while (!done) {
      CheckStatus(
          iree_async_proactor_poll(proactor, iree_infinite_timeout(), nullptr));
    }
  }
};

void WaitForEnrollment(iree_notification_state_t* state, uint32_t count) {
  while ((uint32_t)iree_atomic_load(&state->value, iree_memory_order_acquire) !=
         count) {
    std::this_thread::yield();
  }
}

struct Peer {
  // Owned executor used for setup transfer and async observer progress.
  iree_async_proactor_t* proactor = nullptr;
  // Owned native local channel, borrowed by stream.
  iree_async_primitive_t channel = {};
  // Owned resource-transfer helper.
  iree_async_local_stream_t* stream = nullptr;

  void Initialize() {
    CheckStatus(iree_async_proactor_create_platform(
        iree_async_proactor_options_default(), iree_allocator_system(),
        &proactor));
  }

  void OpenStream() {
    CheckStatus(iree_async_local_stream_create(
        proactor, channel, kHandleCount, iree_allocator_system(), &stream));
  }

  void Close() {
    bool retired = false;
    CheckStatus(iree_async_local_stream_deactivate(
        stream,
        {+[](void* user_data) { *static_cast<bool*>(user_data) = true; },
         &retired}));
    while (!retired) {
      CheckStatus(
          iree_async_proactor_poll(proactor, iree_infinite_timeout(), nullptr));
    }
    iree_async_local_stream_destroy(stream);
    iree_async_primitive_close(&channel);
    iree_async_proactor_release(proactor);
  }
};

#if defined(IREE_PLATFORM_WINDOWS)
std::string PipeName(const char* directory) {
  std::string name = directory;
  return name.substr(name.find_last_of("\\/") + 1) + "-notification";
}
#else
sockaddr_un SocketAddress(const char* directory) {
  sockaddr_un address = {.sun_family = AF_UNIX};
  std::string path = std::string(directory) + "/notification.sock";
  ROLE_CHECK(path.size() < sizeof(address.sun_path));
  memcpy(address.sun_path, path.c_str(), path.size() + 1);
  return address;
}
#endif

int Publisher(int, char**, const char* directory) {
  Peer peer;
  peer.Initialize();
#if defined(IREE_PLATFORM_WINDOWS)
  std::string name = PipeName(directory);
  CheckStatus(iree_async_local_stream_pipe_create(
      iree_make_string_view(name.data(), name.size()), 1,
      IREE_ASYNC_LOCAL_STREAM_PIPE_FLAG_FIRST_INSTANCE, &peer.channel));
  peer.OpenStream();
  iree_coordinated_test_signal_ready(directory);
  Transfer accepted;
  CheckStatus(
      iree_async_local_stream_pipe_accept(peer.stream, accepted.callback()));
  accepted.Wait(peer.proactor);
#else
  int listener = socket(AF_UNIX, SOCK_STREAM, 0);
  ROLE_CHECK(listener >= 0);
  sockaddr_un address = SocketAddress(directory);
  ROLE_CHECK(bind(listener, (sockaddr*)&address, sizeof(address)) == 0);
  ROLE_CHECK(listen(listener, 1) == 0);
  iree_coordinated_test_signal_ready(directory);
  int descriptor = accept(listener, nullptr, nullptr);
  ROLE_CHECK(descriptor >= 0);
  ROLE_CHECK(close(listener) == 0);
  ROLE_CHECK(fcntl(descriptor, F_SETFL, O_NONBLOCK) == 0);
  peer.channel = iree_async_primitive_from_fd(descriptor);
  peer.OpenStream();
#endif

  iree_shm_mapping_t mapping = {};
  CheckStatus(iree_shm_create(nullptr, sizeof(SharedState), &mapping));
  auto* state = static_cast<SharedState*>(mapping.base);
  iree_notification_state_initialize(&state->notification);
  iree_atomic_store(&state->idle_round, 0, iree_memory_order_relaxed);
  iree_async_notification_native_t native = {};
  CheckStatus(
      iree_async_notification_native_initialize(&state->notification, &native));
  // Import must preserve publication preceding the native resource transfer.
  state->payload = 100;
  iree_async_notification_native_signal(&native, IREE_ALL_WAITERS);
  iree_async_primitive_t handles[kHandleCount];
#if defined(IREE_PLATFORM_WINDOWS)
  handles[0] = iree_async_primitive_from_win32_handle(mapping.handle.value);
#else
  handles[0] = iree_async_primitive_from_fd((int)mapping.handle.value);
#endif
  iree_async_notification_native_export(&native, handles + 1);
  const uint8_t offer = 1;
  Transfer offered;
  CheckStatus(iree_async_local_stream_send(
      peer.stream, iree_make_const_byte_span(&offer, 1), kHandleCount, handles,
      offered.callback()));
  offered.Wait(peer.proactor);

  for (uint32_t round = 0; round < kRounds; ++round) {
    uint32_t ready = UINT32_MAX;
    Transfer enrolled;
    CheckStatus(iree_async_local_stream_receive(
        peer.stream, iree_make_byte_span(&ready, sizeof(ready)), 0, nullptr,
        enrolled.callback()));
    enrolled.Wait(peer.proactor);
    ROLE_CHECK(ready == round);
    while (iree_atomic_load(&state->idle_round, iree_memory_order_acquire) !=
           (int32_t)(round + 1)) {
      std::this_thread::yield();
    }
    ROLE_CHECK((uint32_t)iree_atomic_load(&state->notification.value,
                                          iree_memory_order_acquire) ==
               kWaiterCount);
    state->payload = 101 + round;
    iree_async_notification_native_signal(&native, IREE_ALL_WAITERS);
  }
  uint8_t finished = 0;
  Transfer joined;
  CheckStatus(iree_async_local_stream_receive(peer.stream,
                                              iree_make_byte_span(&finished, 1),
                                              0, nullptr, joined.callback()));
  joined.Wait(peer.proactor);
  ROLE_CHECK(finished == 1);
  ROLE_CHECK((uint32_t)iree_atomic_load(&state->notification.value,
                                        iree_memory_order_acquire) == 0);
  iree_async_notification_native_deinitialize(&native);
  iree_shm_close(&mapping);
  peer.Close();
  return 0;
}

int Observer(int, char**, const char* directory) {
  Peer peer;
  peer.Initialize();
#if defined(IREE_PLATFORM_WINDOWS)
  std::string name = PipeName(directory);
  CheckStatus(iree_async_local_stream_pipe_open(
      iree_make_string_view(name.data(), name.size()), &peer.channel));
#else
  int descriptor = socket(AF_UNIX, SOCK_STREAM, 0);
  ROLE_CHECK(descriptor >= 0);
  sockaddr_un address = SocketAddress(directory);
  ROLE_CHECK(connect(descriptor, (sockaddr*)&address, sizeof(address)) == 0);
  ROLE_CHECK(fcntl(descriptor, F_SETFL, O_NONBLOCK) == 0);
  peer.channel = iree_async_primitive_from_fd(descriptor);
#endif
  peer.OpenStream();
  uint8_t offer = 0;
  iree_async_primitive_t handles[kHandleCount] = {};
  Transfer offered;
  CheckStatus(iree_async_local_stream_receive(
      peer.stream, iree_make_byte_span(&offer, 1), kHandleCount, handles,
      offered.callback()));
  offered.Wait(peer.proactor);
  ROLE_CHECK(offer == 1);
  iree_shm_handle_t mapping_handle =
      {};  // NOLINT(iree-cpp-designated-initializer) -- Assignment conversion
           // differs from list initialization.
#if defined(IREE_PLATFORM_WINDOWS)
  mapping_handle.value = handles[0].value.win32_handle;
#else
  mapping_handle.value = handles[0].value.fd;
#endif
  iree_shm_mapping_t mapping = {};
  CheckStatus(iree_shm_open_handle(
      mapping_handle, iree_shm_required_size(sizeof(SharedState)), &mapping));
  iree_async_primitive_close(&handles[0]);
  auto* state = static_cast<SharedState*>(mapping.base);
  iree_async_notification_native_t native = {};
  CheckStatus(iree_async_notification_native_import(&state->notification,
                                                    handles + 1, &native));
  for (auto handle : handles) {
    ROLE_CHECK(iree_async_primitive_is_none(handle));
  }
  ROLE_CHECK(iree_notification_state_query_epoch(&state->notification) == 1);
  ROLE_CHECK(state->payload == 100);
  iree_async_notification_t* notification = nullptr;
  CheckStatus(iree_async_notification_create_shared(peer.proactor, &native,
                                                    &notification));

  for (uint32_t round = 0; round < kRounds; ++round) {
    bool uses_async_wait = (round % 2) != 0;
    uint32_t token = iree_async_notification_query_epoch(notification);
    std::vector<std::thread> waiters;
    for (uint32_t i = 0; i < kWaiterCount; ++i) {
      waiters.emplace_back([&] {
        ROLE_CHECK(iree_async_notification_native_wait_for_token(
            &native, token, iree_infinite_timeout()));
        ROLE_CHECK(state->payload == 101 + round);
      });
    }
    bool completed = false;
    iree_async_notification_wait_operation_t wait = {};
    iree_async_operation_initialize(
        &wait.base, IREE_ASYNC_OPERATION_TYPE_NOTIFICATION_WAIT,
        IREE_ASYNC_OPERATION_FLAG_NONE,
        +[](void* user_data, iree_async_operation_t*, iree_status_t status,
            iree_async_completion_flags_t) {
          CheckStatus(status);
          *static_cast<bool*>(user_data) = true;
        },
        &completed);
    wait.notification = notification;
    wait.wait_flags = IREE_ASYNC_NOTIFICATION_WAIT_FLAG_USE_WAIT_TOKEN;
    wait.wait_token = token;
    if (uses_async_wait) {
      CheckStatus(iree_async_proactor_submit_one(peer.proactor, &wait.base));
    }
    WaitForEnrollment(&state->notification, kWaiterCount);
    Transfer ready;
    CheckStatus(iree_async_local_stream_send(
        peer.stream, iree_make_const_byte_span(&round, sizeof(round)), 0,
        nullptr, ready.callback()));
    ready.Wait(peer.proactor);
    // Publication cannot occur before this handoff. Blocking workers must
    // finish while the proactor is idle, even with an admitted async observer.
    iree_atomic_store(&state->idle_round, (int32_t)(round + 1),
                      iree_memory_order_release);
    for (auto& waiter : waiters) {
      waiter.join();
    }
    while (uses_async_wait && !completed) {
      CheckStatus(iree_async_proactor_poll(peer.proactor,
                                           iree_infinite_timeout(), nullptr));
    }
    ROLE_CHECK(state->payload == 101 + round);
    ROLE_CHECK(iree_async_notification_query_epoch(notification) == token + 1);
  }
  const uint8_t finished = 1;
  Transfer joined;
  CheckStatus(iree_async_local_stream_send(
      peer.stream, iree_make_const_byte_span(&finished, 1), 0, nullptr,
      joined.callback()));
  joined.Wait(peer.proactor);
  iree_async_notification_release(notification);
  peer.Close();
  iree_async_notification_native_deinitialize(&native);
  iree_shm_close(&mapping);
  return 0;
}

const iree_test_role_t kRoles[] = {
    {"publisher", Publisher, true},
    {"observer", Observer, false},
};
const iree_coordinated_test_config_t kConfig = {kRoles, IREE_ARRAYSIZE(kRoles)};
IREE_COORDINATED_TEST_REGISTER(kConfig);

TEST(NativeNotificationTest, AvailabilityMatchesConstruction) {
#if defined(IREE_PLATFORM_WINDOWS) || defined(IREE_PLATFORM_LINUX)
  EXPECT_EQ(iree_async_notification_native_is_supported(),
            iree_atomic_int64_is_lock_free());
#endif
  iree_notification_state_t state = {};
  iree_notification_state_initialize(&state);
  iree_async_notification_native_t native = {};
  iree_status_t status =
      iree_async_notification_native_initialize(&state, &native);
  if (iree_async_notification_native_is_supported()) {
    IREE_EXPECT_OK(status);
  } else {
    IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE, status);
    EXPECT_EQ(native.state, nullptr);
    EXPECT_TRUE(
        iree_async_primitive_is_none(native.async_event.wait_primitive));
    EXPECT_TRUE(
        iree_async_primitive_is_none(native.async_event.signal_primitive));
  }
  iree_async_notification_native_deinitialize(&native);
}

TEST(NativeNotificationTest, CrossProcessSynchronousAndAsyncPublication) {
  if (!iree_async_notification_native_is_supported()) {
    GTEST_SKIP();
  }
  EXPECT_EQ(iree_coordinated_test_run(iree_coordinated_test_argc(),
                                      iree_coordinated_test_argv(), &kConfig),
            0);
}

TEST(NativeNotificationTest, TimeoutDoesNotStrandOtherWaiter) {
  if (!iree_async_notification_native_is_supported()) {
    GTEST_SKIP();
  }
  iree_notification_state_t state = {};
  iree_notification_state_initialize(&state);
  iree_async_notification_native_t native = {};
  IREE_ASSERT_OK(iree_async_notification_native_initialize(&state, &native));
  std::thread timed([&] {
    EXPECT_FALSE(iree_async_notification_native_wait_for_token(
        &native, 0, iree_make_timeout_ms(10)));
  });
  std::thread indefinite([&] {
    EXPECT_TRUE(iree_async_notification_native_wait_for_token(
        &native, 0, iree_infinite_timeout()));
  });
  timed.join();
  WaitForEnrollment(&state, 1);
  iree_async_notification_native_signal(&native, IREE_ALL_WAITERS);
  indefinite.join();
  EXPECT_EQ((uint32_t)iree_atomic_load(&state.value, iree_memory_order_acquire),
            0u);
  iree_async_notification_native_deinitialize(&native);
}

}  // namespace

// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdio.h>
#include <string.h>

#include <string>

#include "iree/async/proactor_platform.h"
#include "iree/async/util/local_stream.h"
#include "iree/base/internal/shm.h"
#include "iree/testing/coordinated_test.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

#if defined(IREE_PLATFORM_WINDOWS)
#include <windows.h>
#else
#include <errno.h>
#include <fcntl.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#if defined(IREE_PLATFORM_APPLE)
#include <sys/event.h>
#else
#include <poll.h>
#include <sys/syscall.h>

// Older libc headers can omit pidfd_open even when the running kernel has it.
// IREE's Linux architectures use 434, with an additional tag for the x32 ABI.
#ifndef SYS_pidfd_open
#if defined(__NR_pidfd_open)
#define SYS_pidfd_open __NR_pidfd_open
#elif defined(__x86_64__) && defined(__ILP32__)
#define SYS_pidfd_open (__X32_SYSCALL_BIT + 434)
#else
#define SYS_pidfd_open 434
#endif
#endif
#endif
#endif

namespace {

constexpr size_t kMappingSize = 4096;

// Child roles cannot use fatal gtest assertions as a process exit result. A
// failure terminates the role and is reported by the coordinating parent.
void Check(bool condition, const char* expression, int line) {
  if (!condition) {
    fprintf(stderr, "local stream role failed at line %d: %s\n", line,
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

struct Transfer {
  // Terminal ownership witness updated by the actual proactor callback.
  bool done = false;
  // Result owned until the role checks it.
  iree_status_t status = iree_ok_status();

  iree_async_local_stream_callback_t callback() {
    return {+[](void* user_data, iree_status_t status) {
              auto* transfer = static_cast<Transfer*>(user_data);
              ROLE_CHECK(!transfer->done);
              transfer->status = status;
              transfer->done = true;
            },
            this};
  }

  void Wait(iree_async_proactor_t* proactor) {
    while (!done) {
      CheckStatus(
          iree_async_proactor_poll(proactor, iree_infinite_timeout(), nullptr));
    }
    CheckStatus(status);
    status = iree_ok_status();
  }
};

void CloseStream(iree_async_proactor_t* proactor,
                 iree_async_local_stream_t* stream) {
  struct Join {
    // Helper released only from its final join callback.
    iree_async_local_stream_t* stream;
    // External lifetime witness.
    bool done = false;
  } join{stream};
  CheckStatus(iree_async_local_stream_deactivate(
      stream, {+[](void* user_data) {
                 auto* join = static_cast<Join*>(user_data);
                 iree_async_local_stream_destroy(join->stream);
                 join->done = true;
               },
               &join}));
  while (!join.done) {
    CheckStatus(
        iree_async_proactor_poll(proactor, iree_infinite_timeout(), nullptr));
  }
}

#if defined(IREE_PLATFORM_WINDOWS)
std::string PipeName(const char* directory) {
  std::string name = directory;
  return name.substr(name.find_last_of("\\/") + 1) + "-stream";
}
#else
sockaddr_un SocketAddress(const char* directory) {
  sockaddr_un address = {.sun_family = AF_UNIX};
  std::string path = std::string(directory) + "/carrier.sock";
  ROLE_CHECK(path.size() < sizeof(address.sun_path));
  memcpy(address.sun_path, path.c_str(), path.size() + 1);
  return address;
}
#endif

int Exporter(int, char**, const char* directory) {
  iree_async_proactor_t* proactor = nullptr;
  CheckStatus(
      iree_async_proactor_create_platform(iree_async_proactor_options_default(),
                                          iree_allocator_system(), &proactor));
  iree_async_primitive_t channel = {};
#if defined(IREE_PLATFORM_WINDOWS)
  std::string name = PipeName(directory);
  CheckStatus(iree_async_local_stream_pipe_create(
      iree_make_string_view(name.data(), name.size()), 1,
      IREE_ASYNC_LOCAL_STREAM_PIPE_FLAG_FIRST_INSTANCE, &channel));
  iree_coordinated_test_signal_ready(directory);
#else
  int listener = socket(AF_UNIX, SOCK_STREAM, 0);
  ROLE_CHECK(listener >= 0);
  sockaddr_un address = SocketAddress(directory);
  ROLE_CHECK(bind(listener, (sockaddr*)&address, sizeof(address)) == 0);
  ROLE_CHECK(listen(listener, 1) == 0);
  iree_coordinated_test_signal_ready(directory);
  // Blocking acquisition is test fixture setup; the production transfer helper
  // borrows a connected nonblocking descriptor, as from async socket accept.
  int descriptor = accept(listener, nullptr, nullptr);
  ROLE_CHECK(descriptor >= 0);
  ROLE_CHECK(close(listener) == 0);
  ROLE_CHECK(fcntl(descriptor, F_SETFL, O_NONBLOCK) == 0);
  channel = iree_async_primitive_from_fd(descriptor);
#endif
  iree_async_local_stream_t* stream = nullptr;
  CheckStatus(iree_async_local_stream_create(proactor, channel, 3,
                                             iree_allocator_system(), &stream));
#if defined(IREE_PLATFORM_WINDOWS)
  Transfer accepted;
  CheckStatus(iree_async_local_stream_pipe_accept(stream, accepted.callback()));
  accepted.Wait(proactor);
#endif

  // These objects are created only in the independent exporter, after spawn.
  // Neither the launcher nor the importer can inherit or reopen them by name.
  iree_async_primitive_t resources[3] = {};
  iree_shm_mapping_t mapping = {};
  CheckStatus(iree_shm_create(nullptr, kMappingSize, &mapping));
#if defined(IREE_PLATFORM_WINDOWS)
  uint64_t process_id = GetCurrentProcessId();
  resources[0] = iree_async_primitive_from_win32_handle(mapping.handle.value);
  for (int i = 1; i < 3; ++i) {
    HANDLE event = CreateEventW(nullptr, FALSE, TRUE, nullptr);
    ROLE_CHECK(event != nullptr);
    resources[i] = iree_async_primitive_from_win32_handle((uintptr_t)event);
  }
#else
  uint64_t process_id = getpid();
  resources[0] = iree_async_primitive_from_fd((int)mapping.handle.value);
  for (int i = 1; i < 3; ++i) {
    int descriptors[2];
    ROLE_CHECK(pipe(descriptors) == 0);
    resources[i] = iree_async_primitive_from_fd(descriptors[0]);
    char value = (char)i;
    ROLE_CHECK(write(descriptors[1], &value, 1) == 1);
    ROLE_CHECK(close(descriptors[1]) == 0);
  }
#endif
  memset(mapping.base, 0xA7, mapping.size);
  Transfer offered;
  CheckStatus(iree_async_local_stream_send(
      stream, iree_make_const_byte_span(&process_id, sizeof(process_id)), 3,
      resources, offered.callback()));
  offered.Wait(proactor);
  uint8_t accept = 0;
  Transfer acknowledged;
  CheckStatus(
      iree_async_local_stream_receive(stream, iree_make_byte_span(&accept, 1),
                                      0, nullptr, acknowledged.callback()));
  acknowledged.Wait(proactor);
  ROLE_CHECK(accept == 0xAC);
  const uint8_t ready = 0xEA;
  Transfer published;
  CheckStatus(iree_async_local_stream_send(stream,
                                           iree_make_const_byte_span(&ready, 1),
                                           0, nullptr, published.callback()));
  published.Wait(proactor);
  CloseStream(proactor, stream);
  iree_async_primitive_close(&channel);
  // The first export is borrowed from the mapping; the wake resources are
  // owned standalone handles.
  iree_shm_close(&mapping);
  for (int i = 1; i < 3; ++i) {
    iree_async_primitive_close(&resources[i]);
  }
  iree_async_proactor_release(proactor);
  return 0;
}

int Importer(int, char**, const char* directory) {
  iree_async_proactor_t* proactor = nullptr;
  CheckStatus(
      iree_async_proactor_create_platform(iree_async_proactor_options_default(),
                                          iree_allocator_system(), &proactor));
  iree_async_primitive_t channel = {};
#if defined(IREE_PLATFORM_WINDOWS)
  std::string name = PipeName(directory);
  CheckStatus(iree_async_local_stream_pipe_open(
      iree_make_string_view(name.data(), name.size()), &channel));
#else
  int descriptor = socket(AF_UNIX, SOCK_STREAM, 0);
  ROLE_CHECK(descriptor >= 0);
  sockaddr_un address = SocketAddress(directory);
  ROLE_CHECK(connect(descriptor, (sockaddr*)&address, sizeof(address)) == 0);
  ROLE_CHECK(fcntl(descriptor, F_SETFL, O_NONBLOCK) == 0);
  channel = iree_async_primitive_from_fd(descriptor);
#endif
  iree_async_local_stream_t* stream = nullptr;
  CheckStatus(iree_async_local_stream_create(proactor, channel, 3,
                                             iree_allocator_system(), &stream));
  uint64_t process_id = 0;
  iree_async_primitive_t resources[3] = {};
  Transfer offered;
  CheckStatus(iree_async_local_stream_receive(
      stream, iree_make_byte_span(&process_id, sizeof(process_id)), 3,
      resources, offered.callback()));
  offered.Wait(proactor);

  // The PID is used only to prove process exit, never as resource-import
  // authority. Pin an exit observation before ACK lets the exporter exit.
#if defined(IREE_PLATFORM_WINDOWS)
  HANDLE process = OpenProcess(SYNCHRONIZE, FALSE, (DWORD)process_id);
  ROLE_CHECK(process != nullptr);
#elif defined(IREE_PLATFORM_APPLE)
  int process = kqueue();
  ROLE_CHECK(process >= 0);
  struct kevent change;
  EV_SET(&change, process_id, EVFILT_PROC, EV_ADD | EV_ONESHOT, NOTE_EXIT, 0,
         nullptr);
  ROLE_CHECK(kevent(process, &change, 1, nullptr, 0, nullptr) == 0);
#else
  int process = (int)syscall(SYS_pidfd_open, (pid_t)process_id, 0);
  ROLE_CHECK(process >= 0);
#endif
  const uint8_t accept = 0xAC;
  Transfer acknowledged;
  CheckStatus(iree_async_local_stream_send(
      stream, iree_make_const_byte_span(&accept, 1), 0, nullptr,
      acknowledged.callback()));
  acknowledged.Wait(proactor);
  uint8_t ready = 0;
  Transfer published;
  CheckStatus(iree_async_local_stream_receive(stream,
                                              iree_make_byte_span(&ready, 1), 0,
                                              nullptr, published.callback()));
  published.Wait(proactor);
  ROLE_CHECK(ready == 0xEA);

  // Native process-exit primitives, not an elapsed-time guess, establish that
  // no exporter resource or mapping can keep the imported objects alive.
#if defined(IREE_PLATFORM_WINDOWS)
  ROLE_CHECK(WaitForSingleObject(process, INFINITE) == WAIT_OBJECT_0);
  ROLE_CHECK(CloseHandle(process));
#elif defined(IREE_PLATFORM_APPLE)
  struct kevent event;
  ROLE_CHECK(kevent(process, nullptr, 0, &event, 1, nullptr) == 1);
  ROLE_CHECK((event.fflags & NOTE_EXIT) != 0);
  ROLE_CHECK(close(process) == 0);
#else
  pollfd event = {process, POLLIN, 0};
  int result = 0;
  do {
    result = poll(&event, 1, -1);
  } while (result < 0 && errno == EINTR);
  ROLE_CHECK(result == 1 && (event.revents & POLLIN));
  ROLE_CHECK(close(process) == 0);
#endif

  iree_shm_handle_t memory = {};
#if defined(IREE_PLATFORM_WINDOWS)
  memory.value = resources[0].value.win32_handle;
  for (int i = 1; i < 3; ++i) {
    HANDLE event = (HANDLE)resources[i].value.win32_handle;
    ROLE_CHECK(WaitForSingleObject(event, 0) == WAIT_OBJECT_0);
    ROLE_CHECK(SetEvent(event));
    ROLE_CHECK(WaitForSingleObject(event, 0) == WAIT_OBJECT_0);
  }
#else
  memory.value = resources[0].value.fd;
  for (int i = 1; i < 3; ++i) {
    char value = 0;
    ROLE_CHECK(read(resources[i].value.fd, &value, 1) == 1);
    ROLE_CHECK(value == i);
  }
#endif
  iree_shm_mapping_t mapping = {};
  CheckStatus(iree_shm_open_handle(memory, iree_shm_required_size(kMappingSize),
                                   &mapping));
  for (size_t i = 0; i < mapping.size; ++i) {
    ROLE_CHECK(((uint8_t*)mapping.base)[i] == 0xA7);
  }
  memset(mapping.base, 0xB9, mapping.size);
  iree_shm_close(&mapping);
  uint8_t unexpected = 0;
  Transfer eof;
  CheckStatus(iree_async_local_stream_receive(
      stream, iree_make_byte_span(&unexpected, 1), 0, nullptr, eof.callback()));
  while (!eof.done) {
    CheckStatus(
        iree_async_proactor_poll(proactor, iree_infinite_timeout(), nullptr));
  }
  ROLE_CHECK(iree_status_is_out_of_range(eof.status));
  iree_status_free(eof.status);
  CloseStream(proactor, stream);
  iree_async_primitive_close(&channel);
  for (auto& resource : resources) {
    iree_async_primitive_close(&resource);
  }
  iree_async_proactor_release(proactor);
  return 0;
}

const iree_test_role_t kRoles[] = {
    {"exporter", Exporter, true},
    {"importer", Importer, false},
};
const iree_coordinated_test_config_t kConfig = {kRoles, IREE_ARRAYSIZE(kRoles)};
IREE_COORDINATED_TEST_REGISTER(kConfig);

TEST(LocalStreamProcessTest, ResourcesSurviveExporterExit) {
  EXPECT_EQ(iree_coordinated_test_run(iree_coordinated_test_argc(),
                                      iree_coordinated_test_argv(), &kConfig),
            0);
}

}  // namespace
